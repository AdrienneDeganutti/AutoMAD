from __future__ import absolute_import, division, print_function

import os
import sys
import h5py
import pickle as pkl
import pandas as pd
import csv

pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
import json
import os
import os.path as op
import time
import wandb

from apex import amp
import tensorflow as tf
from apex.parallel import DistributedDataParallel as DDP
from src.configs.config import (
    basic_check_arguments,
    restore_training_settings,
    shared_configs,
)
from src.datasets.vl_dataloader import make_data_loader
from src.evalcap.utils_caption_evaluate import evaluate_on_coco_caption
from src.datasets.MAD_dataloader import build_dataset
from src.modeling.gpt_utils import generate_beam, generate_greedy
from src.transformers import GPT2LMHeadModel, GPT2Model
from src.modeling.model_ad import VideoCaptionModel
from src.transformers import GPT2Tokenizer
from src.solver import AdamW, WarmupLinearLR
from src.utils.tsv_file_ops import reorder_tsv_keys, tsv_writer
from src.utils.comm import dist_init, get_rank, get_world_size, is_main_process
from src.utils.load_save import TrainingRestorer, TrainingSaver
from src.utils.logger import LOGGER as logger
from src.utils.logger import TB_LOGGER, RunningMeter, add_log_to_file
from src.utils.metric_logger import MetricLogger
from src.utils.miscellaneous import (
    NoOp,
    concat_tsv_files,
    delete_tsv_files,
    mkdir,
    set_seed,
    str_to_bool,
)
import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch import nn
from torch.utils.data import DataLoader
from datetime import timedelta
from tqdm import tqdm


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data  # argmax
    return logits == labels


def mixed_precision_init(args, VTmodel, GPTmodel):
    max_iter = args.max_iter
    max_global_step = args.max_global_step

    if args.distributed:
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
        [VTmodel, GPTmodel], optimizer = amp.initialize(models, optimizer, enabled=True, opt_level=f'O{args.amp_opt_level}')

    return args, VTmodel, GPTmodel, optimizer, scheduler



def train(args, train_dataloader, val_dataloader, encoded, model, GPTmodel, tokenizer,
          training_saver, optimizer, scheduler):

    # Initialize wandb
    if args.rank == 0:
        wandb.init(project="Auto-MAD", name="Debugging")


    encoded = encoded.to(args.device)
    meters = MetricLogger(delimiter='  ')
    max_iter = args.max_iter
    max_global_step = args.max_global_step
    global_iters_per_epoch = args.global_iters_per_epoch

    eval_log = []
    best_score = 0
    start_training_time = time.time()
    end = time.time()
    log_start = time.time()
    running_loss = RunningMeter('train_loss')
    running_batch_acc = RunningMeter('train_batch_acc')

    if args.restore_ratio > 0:
        restorer = TrainingRestorer(args, model, optimizer)
        global_step = restorer.global_step
    else:
        global_step = 0

    TB_LOGGER.global_step = global_step
    if not is_main_process() or args.restore_ratio <= 0:
        restorer = NoOp()

    training_saver.save_args(args)
    training_saver.save_tokenizer(tokenizer)

    train_loss_dict = {}


    for iteration, (img_keys, batch, meta_data) in enumerate(train_dataloader):
        iteration += 1
        data_time = time.time() - end
        batch = tuple(t.to(args.device) for t in batch)

        model.train()
        GPTmodel.train()

        inputs = {
            'input_ids': batch[0],
            'token_type_ids': batch[1],
            'img_feats': batch[2],
            'input_token_ids': batch[3],
            'output_token_ids': batch[4],
        }

        #Print shape of each input
        if iteration == 1:
            for k, v in inputs.items():
                logger.info(f'{k} = {v.shape}')


        model.zero_grad()
        GPTmodel.zero_grad()
        # Pass visual features and text through transformer:
        #prefix_vector, context_embed = model(inputs['img_feats'], encoded)
            
        # Pass visual features only through transformer:
        prefix_vector = model(inputs['img_feats'])

        add = -100*torch.ones(1,1,768)
        add = add.to(args.device)
        prefix_vector = torch.cat((prefix_vector, add), dim=1)

        
        outputs = GPTmodel(inputs_embeds=prefix_vector, labels=encoded)
        #outputs = GPTmodel(args, inputs_embeds=prefix_vector, labels=inputs['input_ids'])
        loss, logits = outputs[:2]


        pred = torch.max(logits, -1)[1].data  # argmax
        batch_score = pred == encoded
        pred_amended = pred[0][:-1]
        encoded_amended = encoded[1:]
        test_score = pred_amended == encoded_amended
            
        #batch_score = compute_score_with_logits(logits, encoded)
        #batch_score = compute_score_with_logits(logits, inputs['output_token_ids'])
        batch_acc = torch.mean(test_score.float())


        #for element in pred:
        #    if tokenizer.decode(element).startswith(' '):
        #        element = [element[0]].strip()


        loss_dict = {
            'loss': loss,
            'acc': batch_acc
        }

        train_loss_dict[iteration] = float(loss_dict['loss']), float(loss_dict['acc'])

        backward_now = iteration % args.gradient_accumulation_steps == 0
        if backward_now:
            global_step += 1
            # backward pass
            loss.backward()
            optimizer.step()
            scheduler.step()

            #Add weights and biases logging
            if args.rank == 0:
                wandb.log({
                    "Training Loss": loss_dict['loss'],
                    "Accuracy": loss_dict['acc'],
                }, step=global_step)


    total_training_time = time.time() - start_training_time
    total_time_str = str(timedelta(seconds=total_training_time))
    logger.info(f'Total training time: {total_time_str} ({(total_training_time / max_iter):.4f} s / iter)')

    # Finish the Weights & Biases run
    wandb.finish()

    #output text

    prediction_NLP = tokenizer.decode(pred[0])
    print(prediction_NLP)


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

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = VideoCaptionModel()
    GPTmodel = GPT2LMHeadModel.from_pretrained("gpt2")

    model.to(args.device)
    GPTmodel.to(args.device)

    if args.do_train:
        args = restore_training_settings(args)
        train_dataloader = make_data_loader(args,
                                            args.train_yaml,
                                            tokenizer,
                                            args.distributed,
                                            is_train=True)
        val_dataloader = make_data_loader(args,
                                          args.val_yaml,
                                          tokenizer,
                                          args.distributed,
                                          is_train=False)

        args.max_iter = len(train_dataloader)
        args.max_global_step = args.max_iter // args.gradient_accumulation_steps
        args.global_iters_per_epoch = args.max_global_step // args.num_train_epochs
        args.save_steps = args.global_iters_per_epoch * 3

        caption = train_dataloader.dataset.caption_on_memory[0,0]
        encoded = tokenizer.encode(caption, return_tensors='pt', add_prefix_space=True, add_bos_token=True)

        add = torch.tensor([-100])
        encoded = torch.cat((add, encoded[0]), dim=0)


        args, model, GPTmodel, optimizer, scheduler = mixed_precision_init(
            args, model, GPTmodel)
        model.to(args.device)
        GPTmodel.to(args.device)
        train(args, train_dataloader, val_dataloader, encoded, model, GPTmodel,
              tokenizer, training_saver, optimizer, scheduler)


    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = get_custom_args(shared_configs)
    main(args)