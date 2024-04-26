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
from torch import nn
from torch.utils.data import DataLoader
from datetime import timedelta
from tqdm import tqdm


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data  # argmax
    return logits == labels


def mixed_precision_init(args, model, GPTmodel):
    max_iter = args.max_iter
    max_global_step = args.max_global_step
    global_iters_per_epoch = args.global_iters_per_epoch

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']

    decay_param_tp = [(n, p) for n, p in param_optimizer
                      if not any(nd in n for nd in no_decay)]
    no_decay_param_tp = [(n, p) for n, p in param_optimizer
                         if any(nd in n for nd in no_decay)]

    decay_swin_param_tp = [(n, p) for n, p in decay_param_tp if "swin." in n]
    decay_bert_param_tp = [(n, p) for n, p in decay_param_tp
                           if "swin." not in n]

    no_decay_swin_param_tp = [(n, p) for n, p in no_decay_param_tp
                              if "swin." in n]
    no_decay_bert_param_tp = [(n, p) for n, p in no_decay_param_tp
                              if "swin." not in n]

    weight_decay = 0.2
    coef_lr = args.backbone_coef_lr
    optimizer_grouped_parameters = [{
        'params': [p for n, p in decay_swin_param_tp],
        'weight_decay':
        weight_decay,
        'lr':
        args.learning_rate * coef_lr
    }, {
        'params': [p for n, p in decay_bert_param_tp],
        'weight_decay':
        weight_decay
    }, {
        'params': [p for n, p in no_decay_swin_param_tp],
        'weight_decay':
        0.0,
        'lr':
        args.learning_rate * coef_lr
    }, {
        'params': [p for n, p in no_decay_bert_param_tp],
        'weight_decay':
        0.0
    }]

    optimizer = AdamW(optimizer_grouped_parameters,
                          lr=args.learning_rate,
                          eps=args.adam_epsilon)
    if args.scheduler == "warmup_linear":
        scheduler = WarmupLinearLR(optimizer,
                                   max_global_step,
                                   warmup_ratio=args.warmup_ratio)
    else:
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                    step_size=int(max_iter /
                                                                  2.0),
                                                    gamma=0.1)

    if args.mixed_precision_method == "apex":
        # opt_level is O0, Apex will run as fp32
        model, optimizer = amp.initialize(model,
                                          optimizer,
                                          enabled=True,
                                          opt_level=f'O{args.amp_opt_level}')
        if args.distributed:  #
            model = DDP(model)
            #model = VideoCaptionModel()
            #GPTmodel = GPT2LMHeadModel.from_pretrained("gpt2")
            #GPTconfiguration = GPTmodel.config
            GPTmodel = DDP(GPTmodel)
    return args, model, GPTmodel, optimizer, scheduler


def get_predict_file(output_dir, args, data_yaml_file):
    cc = ['pred']
    # example data_yaml_file: datasets/coco_caption/test.yaml
    data = data_yaml_file.split('/')[-2]
    if data != 'coco_caption':
        cc.append(data)
    cc.append(op.splitext(op.basename(data_yaml_file))[0])
    cc.append('beam{}'.format(args.num_beams))
    cc.append('max{}'.format(args.max_gen_length))
    if args.num_keep_best != 1:
        cc.append('best{}'.format(args.num_keep_best))
    if args.output_hidden_states:
        cc.append('hidden')
    return op.join(output_dir, '{}.tsv'.format('.'.join(cc)))


def get_evaluate_file(predict_file):
    assert predict_file.endswith('.tsv')
    return op.splitext(predict_file)[0] + '.eval.json'


def test(args, test_dataloader, model, tokenizer, predict_file):

    cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
        tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token,
        tokenizer.pad_token, tokenizer.mask_token, '.'])
    world_size = get_world_size()
    if world_size == 1:
        cache_file = predict_file
    else:
        # local_rank would not work for cross-node distributed training
        cache_file = op.splitext(predict_file)[0] + '_{}_{}'.format(
            get_rank(), world_size) + op.splitext(predict_file)[1]

    model.eval()

    def gen_rows():
        time_meter = 0
        # restore existing results for long running inference tasks
        exist_key2pred = {}
        tmp_file = cache_file + '.tmp.copy'
        if op.isfile(tmp_file):
            with open(tmp_file, 'r') as fp:
                for line in fp:
                    parts = line.strip().split('\t')
                    if len(parts) == 2:
                        exist_key2pred[parts[0]] = parts[1]

        with torch.no_grad():
            for step, (img_keys, batch, meta_data) in tqdm(enumerate(test_dataloader)):
                # torch.cuda.empty_cache()
                is_exist = True
                for k in img_keys:
                    if k not in exist_key2pred:
                        is_exist = False
                        break
                if is_exist:
                    for k in img_keys:
                        yield k, exist_key2pred[k]
                    continue
                batch = tuple(t.to(args.device) for t in batch)
                inputs = {
                    'is_decode': True,
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'img_feats': batch[3],
                    'audio_feat': batch[4],
                    'masked_pos': batch[5],
                    'input_token_ids': batch[6],
                    'output_token_ids': batch[7],
                    'do_sample': False,
                    'bos_token_id': cls_token_id,
                    'pad_token_id': pad_token_id,
                    'eos_token_ids': [sep_token_id],
                    'mask_token_id': mask_token_id,
                    # for adding od labels
                    'add_od_labels': args.add_od_labels,
                    'od_labels_start_posid': args.max_seq_a_length,
                    # hyperparameters of beam search
                    'max_length': args.max_gen_length,
                    'num_beams': args.num_beams,
                    "temperature": args.temperature,
                    "top_k": args.top_k,
                    "top_p": args.top_p,
                    "repetition_penalty": args.repetition_penalty,
                    "length_penalty": args.length_penalty,
                    "num_return_sequences": args.num_return_sequences,
                    "num_keep_best": args.num_keep_best,
                }

                tic = time.time()
                # captions, logprobs

                outputs = model(**inputs)
                time_meter += time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                all_confs = torch.exp(outputs[1])

                for img_key, caps, confs in zip(img_keys, all_caps, all_confs):
                    res = []
                    for cap, conf in zip(caps, confs):
                        cap = tokenizer.decode(cap.tolist(),
                                               skip_special_tokens=True)
                        res.append({'caption': cap, 'conf': conf.item()})
                    if isinstance(img_key, torch.Tensor):
                        img_key = img_key.item()
                    yield img_key, json.dumps(res)

        logger.info(
            f"Inference model computing time: {(time_meter / (step+1))} seconds per batch"
        )

    tsv_writer(gen_rows(), cache_file)
    if world_size > 1:
        dist.barrier()
    if world_size > 1 and is_main_process():
        cache_files = [op.splitext(predict_file)[0] + '_{}_{}'.format(i, world_size) + \
            op.splitext(predict_file)[1] for i in range(world_size)]
        concat_tsv_files(cache_files, predict_file)
        delete_tsv_files(cache_files)
        reorder_tsv_keys(predict_file, test_dataloader.dataset.image_keys,
                         predict_file)
    if world_size > 1:
        dist.barrier()


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
        #pred_amended = pred[0][:-1]
        #encoded_amended = encoded[1:]
        #test_score = pred_amended == encoded_amended
            
        #batch_score = compute_score_with_logits(logits, encoded)
        #batch_score = compute_score_with_logits(logits, inputs['output_token_ids'])
        batch_acc = torch.mean(batch_score.float())


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

    transform = nn.Linear(50257, 768)
    transform.to(args.device)
    reshaped_logits = logits.view(-1, 50257)
    transformed_logits = transform(reshaped_logits)
    transformed_logits = transformed_logits.view(1, 38, 768)
    greedy = generate_greedy(model, tokenizer, embed=transformed_logits)
    print(greedy)


    return checkpoint_dir


def evaluate(args, val_dataloader, model, tokenizer, output_dir):
    predict_file = get_predict_file(output_dir, args,
                                    val_dataloader.dataset.yaml_file)
    test(args, val_dataloader, model, tokenizer, predict_file)

    if get_world_size() > 1:
        dist.barrier()
    evaluate_file = get_evaluate_file(predict_file)
    if is_main_process():
        caption_file = val_dataloader.dataset.get_caption_file_in_coco_format()
        data = val_dataloader.dataset.yaml_file.split('/')[-2]
        result = evaluate_on_coco_caption(predict_file,
                                          caption_file,
                                          outfile=evaluate_file)
        logger.info(f'evaluation result: {str(result)}')
        logger.info(f'evaluation result saved to {evaluate_file}')
    if get_world_size() > 1:
        dist.barrier()
    return evaluate_file


def get_custom_args(base_config):
    parser = base_config.parser
    parser.add_argument('--max_num_frames', type=int, default=8)
    #parser.add_argument('--img_res', type=int, default=224)
    parser.add_argument("--grid_feat",
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=True)
    parser.add_argument("--att_mode",
                        type=str,
                        default='default',
                        help="default, full")
    parser.add_argument("--lambda_",
                        type=float,
                        default=0.5,
                        help="lambda_ loss")
    parser.add_argument("--pretrained_2d",
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--freeze_backbone',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--use_checkpoint',
                        type=str_to_bool,
                        nargs='?',
                        const=True,
                        default=False)
    parser.add_argument('--backbone_coef_lr', type=float, default=0.001)
    parser.add_argument('--loss_sparse_w', type=float, default=0)
    parser.add_argument('--resume_checkpoint', type=str, default='None')
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