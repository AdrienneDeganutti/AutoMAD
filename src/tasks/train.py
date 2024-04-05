from __future__ import absolute_import, division, print_function

import os
import sys

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
from apex.parallel import DistributedDataParallel as DDP
from src.configs.config import (
    basic_check_arguments,
    restore_training_settings,
    shared_configs,
)
from src.datasets.vl_dataloader import make_data_loader
from src.modeling.model_ad import VideoCaptionModel
from src.transformers import GPT2Tokenizer
from src.solver import AdamW, WarmupLinearLR
from src.utils.comm import dist_init, get_rank, get_world_size, is_main_process
from src.utils.load_save import TrainingRestorer, TrainingSaver
from src.utils.logger import LOGGER as logger
from src.utils.logger import TB_LOGGER, RunningMeter, add_log_to_file
from src.utils.metric_logger import MetricLogger
from src.utils.miscellaneous import (
    NoOp,
    mkdir,
    set_seed,
    str_to_bool,
)
import torch
import torch.distributed as dist
from datetime import timedelta


def compute_score_with_logits(logits, labels):
    logits = torch.max(logits, -1)[1].data  # argmax
    return logits == labels

def mixed_precision_init(args, model):
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

    if args.mixed_precision_method == "fairscale":
        from fairscale.optim.oss import OSS
        optimizer = OSS(params=optimizer_grouped_parameters,
                        optim=AdamW,
                        lr=args.learning_rate,
                        eps=args.adam_epsilon)
    else:
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
            #model = DDP(model)
            model = VideoCaptionModel()
    return args, model, optimizer, scheduler


def train(args, train_dataloader, val_dataloader, model, tokenizer,
          training_saver, optimizer, scheduler):


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


    for iteration, (img_keys, batch, meta_data) in enumerate(train_dataloader):
        iteration += 1
        data_time = time.time() - end
        batch = tuple(t.to(args.device) for t in batch)

        inputs = {
            'input_ids': batch[0],
            'token_type_ids': batch[1],
            'img_feats': batch[2],
            'input_token_ids': batch[3],
            'output_token_ids': batch[4],
        }

        if iteration == 1:
            for k, v in inputs.items():
                logger.info(f'{k} = {v.shape}')
        
        from src.modeling.gpt_utils import generate_beam, generate_greedy
        prefix_vector = model(batch[2])
        greedy_search = generate_greedy(model, tokenizer, embed=prefix_vector)  #greedy search text generation
        print(greedy_search)
        beam_search = generate_beam(model, tokenizer, embed=prefix_vector)      #beam search text generation
        print(beam_search)

        end = time.time()
        batch_time = time.time() - end


    total_training_time = time.time() - start_training_time
    total_time_str = str(timedelta(seconds=total_training_time))
    logger.info(
        f'Total training time: {total_time_str} ({(total_training_time / max_iter):.4f} s / iter)'
    )

    return



def check_arguments(args):
    # shared basic checks
    
    basic_check_arguments(args)
    # additional sanity check:
    #args.max_img_seq_length = int(
     #   (args.max_num_frames / 2) * (int(args.img_res) / 32) *
    #    (int(args.img_res) / 32)) + 473

    if args.freeze_backbone or args.backbone_coef_lr == 0:
        args.backbone_coef_lr = 0
        args.freeze_backbone = True

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
    check_arguments(args)
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

    model.to(args.device)

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

        args, model, optimizer, scheduler = mixed_precision_init(
            args, model)
        model.to(args.device)
        train(args, train_dataloader, val_dataloader, model,
              tokenizer, training_saver, optimizer, scheduler)


    if args.distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = get_custom_args(shared_configs)
    main(args)