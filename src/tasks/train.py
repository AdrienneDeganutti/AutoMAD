from __future__ import absolute_import, division, print_function

import os
import sys
import pickle as pkl
import pandas as pd

pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
import json
import os.path as op
import time
import wandb

from apex import amp
from apex.parallel import DistributedDataParallel as DDP
from src.configs.config import (
    basic_check_arguments,
    restore_training_settings,
    shared_configs,
)
from src.datasets.vl_dataloader import make_data_loader
from src.tasks.eval import Evaluation
from src.transformers import GPT2LMHeadModel
from src.modeling.model_ad import VideoCaptionModel
from src.modeling.caption_tokenizer import TokenizerHandler
from src.solver import AdamW, WarmupLinearLR
from src.utils.comm import dist_init, get_rank, get_world_size, is_main_process
from src.utils.load_save import TrainingRestorer, TrainingSaver
from src.utils.logger import LOGGER as logger
from src.utils.logger import TB_LOGGER, add_log_to_file
from src.utils.metric_logger import MetricLogger
from src.utils.miscellaneous import (NoOp, mkdir, set_seed)

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from datetime import timedelta


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data  # argmax

    #Workaround:
    new_labels = labels[:, 1:]
    apend_token = 50256
    add = apend_token * torch.ones((new_labels.size(0), 1), dtype=torch.long).to(args.device)
    new_labels = torch.cat([new_labels, add], dim=1)
    
    correct_preds = (logits == new_labels)
    return correct_preds


def mixed_precision_init(args, VTmodel, GPTmodel):
    max_global_step = args.max_global_step

    if args.freeze_lm:
        VTmodel = DDP(VTmodel)
        vt_parameters = VTmodel.parameters()
        optimizer = AdamW(vt_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    
    else:
        VTmodel = DDP(VTmodel)
        GPTmodel = DDP(GPTmodel)
    
        combined_parameters = list(VTmodel.parameters()) + list(GPTmodel.parameters())
        models = [VTmodel, GPTmodel]

        optimizer = AdamW(combined_parameters, lr=args.learning_rate, eps=args.adam_epsilon)

    if args.scheduler == "warmup_linear":
        scheduler = WarmupLinearLR(optimizer,
                                   max_global_step,
                                   warmup_ratio=args.warmup_ratio)
    else:
        scheduler = CosineAnnealingLR(optimizer, T_max=max_global_step)


    if args.mixed_precision_method == "apex":
        # opt_level is O0, Apex will run as fp32
        if args.freeze_lm:
            VTmodel, optimizer = amp.initialize(VTmodel, optimizer, enabled=True, opt_level=f'O{args.amp_opt_level}')
        else:
            [VTmodel, GPTmodel], optimizer = amp.initialize(models, optimizer, enabled=True, opt_level=f'O{args.amp_opt_level}')

    return args, VTmodel, GPTmodel, optimizer, scheduler



def train(args, train_dataloader, val_dataloader, test_dataloader, VTmodel, GPTmodel, tokenizer,
          training_saver, optimizer, scheduler):
    
    # Initialize wandb
    if args.rank == 0:
        wandb.init(project="Auto-MAD", name="full-batch-size-1-ep100", settings=wandb.Settings(_service_wait=300))

    meters = MetricLogger(delimiter='  ')
    max_iter = args.max_iter
    max_global_step = args.max_global_step
    global_iters_per_epoch = args.global_iters_per_epoch

    VTmodel.train()
    GPTmodel.train()

    if args.evaluate_during_training:
        evaluation = Evaluation(args, VTmodel, GPTmodel, tokenizer, val_dataloader, test_dataloader)
        val_log = []
        test_log = []
        val_best_score = 0
        test_best_score = 0
    
    start_training_time = time.time()
    end = time.time()


    if args.restore_ratio > 0:
        restorer = TrainingRestorer(args, VTmodel, optimizer)
        global_step = restorer.global_step
    else:
        global_step = 0

    #TB_LOGGER.global_step = global_step
    if not is_main_process() or args.restore_ratio <= 0:
        restorer = NoOp()

    training_saver.save_args(args)
    training_saver.save_tokenizer(tokenizer)

    if get_world_size() > 1:
        dist.barrier()


    for iteration, (img_ID, caption, visual_frame) in enumerate(train_dataloader):
        iteration += 1
        data_time = time.time() - end

        VTmodel.zero_grad()
        GPTmodel.zero_grad()

        # Tokenize and pad caption:
        tokenized_caption = tokenizer.tokenize_caption(args, caption)

        # Pass visual features only through transformer:
        visual_frame = visual_frame.to(args.device)
        prefix_vector = VTmodel(visual_frame, img_ID)   #prefix_vector, context_embed = model(inputs['img_feats'], encoded)
        
        # Token shift workaround:
        add = -100 * torch.ones(prefix_vector.shape[0], 1, prefix_vector.shape[2]).to(args.device)  # Shape [batch_size, 1, feature_size]
        prefix_vector = torch.cat([add, prefix_vector], dim=1)

        #Print shape of each input
        if iteration == 1:
            logger.info(f'ID: {img_ID}')
            logger.info(f'visual features = {prefix_vector.shape}')
            logger.info(f'padded caption = {tokenized_caption.shape}')

        outputs = GPTmodel(inputs_embeds=prefix_vector, labels=tokenized_caption)
        loss, logits = outputs[:2]
            
        batch_score = compute_score_with_logits(logits, tokenized_caption)
        batch_acc = torch.mean(batch_score.float())


        loss_dict = {
            'loss': float(loss),
            'acc': float(batch_acc)
        }

        backward_now = iteration % args.gradient_accumulation_steps == 0
        
        # backward pass
        with amp.scale_loss(loss, optimizer, delay_unscale=not backward_now) as scaled_loss:
            scaled_loss.backward()

        if backward_now:
            global_step += 1

            # Gradient clipping:
            if args.max_grad_norm != -1:        
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    amp.master_params(optimizer), args.max_grad_norm)
            
            optimizer.step()            
            scheduler.step()
            restorer.step()
        
        batch_time = time.time() - end
            
        if backward_now:
            if global_step % args.logging_steps == 0 or global_step == max_global_step:
                if 'time_info' in meters.meters:
                    avg_time = meters.meters['time_info']['compute'].global_avg
                    eta_seconds = avg_time * (max_iter - iteration)
                    eta_string = str(timedelta(seconds=int(eta_seconds)))
                else:
                    eta_string = 'Unknown'
                eta_seconds = batch_time * (max_iter - iteration)
                eta_string = str(timedelta(seconds=int(eta_seconds)))
                memory = torch.cuda.max_memory_allocated()
                logger.info(meters.delimiter.join([
                    f"eta: {eta_string}",
                    f"iter: {iteration}",
                    f"global_step: {global_step}",
                    f"{meters}",
                    f"learning rate: {'{:.2e}'.format(optimizer.param_groups[0]['lr'])}",
                    f"max mem: {memory:.0f}",
                ]))
            
            if global_step % global_iters_per_epoch == 0:
                epoch = global_step // global_iters_per_epoch
                if args.rank == 0:
                    wandb.log({"Gradient Norm": grad_norm,
                                "Training Loss": loss_dict['loss'],
                                "Training Accuracy": loss_dict['acc'],
                            }, step=epoch)


            if (args.save_steps > 0 and global_step % args.save_steps == 0
                ) or global_step == max_global_step or global_step == 1:
                epoch = global_step // global_iters_per_epoch

                checkpoint_dir = op.join(args.output_dir, 'checkpoint-{}-{}'.format(epoch, global_step))
                if get_world_size() > 1:
                    dist.barrier()

                if epoch % 9 == 0:
                    training_saver.save_model(checkpoint_dir, global_step, VTmodel, optimizer, model_name='VTmodel')
                    training_saver.save_model(checkpoint_dir, global_step, GPTmodel, optimizer, model_name='GPTmodel')

                if get_world_size() > 1:
                    dist.barrier()

                if args.evaluate_during_training:
                    logger.info(f"Perform evaluation at iteration {iteration}, global_step {global_step}")
                    val_best_score, test_best_score = evaluation.evaluate(checkpoint_dir, epoch, val_log, 
                                                                          test_log, iteration, val_best_score, test_best_score)

        if iteration > 2:
            meters.update(batch_time=batch_time, data_time=data_time)
        end = time.time()
        

    total_training_time = time.time() - start_training_time
    total_time_str = str(timedelta(seconds=total_training_time))
    logger.info(f'Total training time: {total_time_str} ({(total_training_time / max_iter):.4f} s / iter)')

    # Finish the Weights & Biases run
    wandb.finish()

    return checkpoint_dir


def get_custom_args(base_config):
    parser = base_config.parser
    parser.add_argument('--max_num_frames', type=int, default=8)
    parser.add_argument('--backbone_coef_lr', type=float, default=0.001)
    parser.add_argument('--loss_sparse_w', type=float, default=0)
    parser.add_argument('--test_video_fname', type=str, default='None')
    args = base_config.parse_args()
    return args


def main(args):

    # global training_saver
    args.device = torch.device(args.device)
    # Setup CUDA, GPU & distributed training
    dist_init(args)
    basic_check_arguments(args)
    mkdir(args.output_dir)
    logger.info(f"creating output_dir at: {args.output_dir}")
    set_seed(args.seed, args.num_gpus)

    if args.mixed_precision_method == "apex":
        fp16_trainning = f"apex O{args.amp_opt_level}"
    else:
        fp16_trainning = None

    logger.info("device: {}, n_gpu: {}, rank: {}, "
                "16-bits training: {}".format(args.device, args.num_gpus,
                                              get_rank(), fp16_trainning))

    if not is_main_process():
        logger.disabled = True
        training_saver = NoOp()
    else:
        training_saver = TrainingSaver(args.output_dir)
        TB_LOGGER.create(op.join(args.output_dir, 'log'))
        add_log_to_file(op.join(args.output_dir, 'log', "log.txt"))

    logger.info(f"Pytorch version is: {torch.__version__}")
    logger.info(f"Cuda version is: {torch.version.cuda}")
    logger.info(f"cuDNN version is : {torch.backends.cudnn.version()}")

    tokenizer = TokenizerHandler()
    VTmodel = VideoCaptionModel(num_latents=args.max_seq_length)
    GPTmodel = GPT2LMHeadModel.from_pretrained("gpt2")

    VTmodel.to(args.device)
    GPTmodel.to(args.device)

    if args.do_train:
        args = restore_training_settings(args)
        train_dataloader = make_data_loader(args,
                                            args.train_yaml,
                                            args.distributed,
                                            is_train=True)
        val_dataloader = make_data_loader(args,
                                          args.val_yaml,
                                          args.distributed,
                                          is_train=False)
        
        test_dataloader = make_data_loader(args,
                                          args.test_yaml,
                                          args.distributed,
                                          is_train=False)

        args.max_iter = len(train_dataloader)
        args.max_global_step = args.max_iter // args.gradient_accumulation_steps
        args.global_iters_per_epoch = args.max_global_step // args.num_train_epochs
        args.save_steps = args.global_iters_per_epoch * 3

        args, VTmodel, GPTmodel, optimizer, scheduler = mixed_precision_init(
            args, VTmodel, GPTmodel)
        
        train(args, train_dataloader, val_dataloader, test_dataloader, VTmodel, GPTmodel, 
              tokenizer, training_saver, optimizer, scheduler)


    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = get_custom_args(shared_configs)
    main(args)