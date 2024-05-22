from __future__ import absolute_import, division, print_function

import os
import sys

pythonpath = os.path.abspath(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
print(pythonpath)
sys.path.insert(0, pythonpath)
import io
import json
import os.path as op
import time

from PIL import Image
import cv2
import numpy as np
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import CLIPModel, CLIPProcessor
from src.transformers import GPT2LMHeadModel
from src.modeling.model_ad import VideoCaptionModel
from src.modeling.caption_tokenizer import TokenizerHandler
from src.configs.config import basic_check_arguments, shared_configs
from src.datasets.caption_tensorizer import build_tensorizer
from src.utils.comm import dist_init, get_rank, get_world_size, is_main_process
from src.utils.logger import LOGGER as logger
from src.utils.logger import TB_LOGGER, RunningMeter, add_log_to_file
from src.utils.miscellaneous import mkdir, set_seed, str_to_bool
import torch

def uniform_subsample(tensor, num_samples):
    """
    Uniformly subsample 'num_samples' frames from the tensor.
    """
    total_frames = tensor.size(0)
    if total_frames < num_samples:
        raise ValueError(f"Cannot subsample {num_samples} frames from {total_frames} frames.")
    indices = torch.linspace(0, total_frames - 1, num_samples).long()
    return tensor[indices]


def _online_video_decode(args, video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
        frames.append(frame)
    cap.release()
    
    return frames


def _transforms(args, frames, num_frames):

    CLIPmodel = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(args.device)

    preprocess = Compose([
        Resize(224),
        CenterCrop(224),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Apply preprocessing and move to the correct device
    frames = [preprocess(Image.fromarray(frame)) for frame in frames]

    # Stack frames into a single tensor (B, T, C, H, W) -> (T, C, H, W)
    frames_tensor = torch.stack(frames)

     # Subsample frames
    frames_tensor = uniform_subsample(frames_tensor, num_frames).to(args.device)

    # Extract features using the CLIP model
    with torch.no_grad():
        features = CLIPmodel.get_image_features(frames_tensor)
    
    # Reshape features to add batch dimension
    features = features.unsqueeze(0)  # Shape: (1, num_frames, 512)

    return features


def check_arguments(args):
    # shared basic checks
    basic_check_arguments(args)

    args.backbone_coef_lr = 0


def update_existing_config_for_inference(args):
    ''' load Vit & LM args for evaluation and inference 
    '''
    assert args.do_test or args.do_eval
    model_dir = args.eval_model_dir
    checkpoint = args.checkpoint_dir

    json_path = op.join(model_dir, 'log', 'args.json')
         
    with open(json_path, 'r') as f:
        json_data = json.load(f)

    from easydict import EasyDict
    train_args = EasyDict(json_data)

    train_args.eval_model_dir = args.eval_model_dir
    train_args.resume_LMcheckpoint = args.eval_model_dir + checkpoint + 'GPTmodel.bin'
    train_args.resume_VTcheckpoint = args.eval_model_dir + checkpoint + 'VTmodel.bin'
    train_args.do_train = False
    train_args.do_eval = True
    train_args.do_test = True
    train_args.val_yaml = args.val_yaml
    train_args.test_video_fname = args.test_video_fname
    return train_args


def batch_inference(args, video_path, VTmodel, GPTmodel, tokenizer,
                    tensorizer):

    tokenizer = tokenizer.tokenizer

    #cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
    #    tokenizer.convert_tokens_to_ids([tokenizer.cls_token, tokenizer.sep_token,
    #    tokenizer.pad_token, tokenizer.mask_token, '.'])

    VTmodel.float()
    GPTmodel.float()
    VTmodel.eval()
    GPTmodel.eval()

    for video in os.listdir(video_path):
        if video.split('.')[-1] == 'mp4':
            v_path = os.path.join(video_path, video)
            logger.info(f"\n")
            logger.info(f"Load video: {v_path}")

            frames = _online_video_decode(args, v_path)
            preproc_frames = _transforms(args, frames, num_frames=8)
            
            preproc_frames = preproc_frames.to(args.device)
            
            with torch.no_grad():

                tic = time.time()
                prefix_vector = VTmodel(preproc_frames, '')
                outputs = GPTmodel(inputs_embeds=prefix_vector, )

                time_meter = time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                #all_confs = torch.exp(outputs[1])

                for caps in all_caps:
                    cap = torch.max(caps, -1)[1].data
                    cap = tokenizer.decode(cap.tolist(),
                                               skip_special_tokens=True)
                    logger.info(f"Prediction: {cap}")
                    #logger.info(f"Conf: {conf.item()}")

            logger.info(
                f"Inference model computing time: {time_meter} seconds")


def get_custom_args(base_config):
    parser = base_config.parser
    parser.add_argument('--max_num_frames', type=int, default=32)
    parser.add_argument('--backbone_coef_lr', type=float, default=0.001)
    parser.add_argument('--loss_sparse_w', type=float, default=0)
    parser.add_argument('--resume_checkpoint', type=str, default='None')
    parser.add_argument('--test_video_fname', type=str, default='None')
    parser.add_argument('--test_features', type=str, default='None')
    args = base_config.parse_args()
    return args

def freeze_models(VTmodel, GPTmodel):
    for param in VTmodel.parameters():
        param.requires_grad = False
    for param in GPTmodel.parameters():
        param.requires_grad = False
    
    #Verify the paramers are frozen:
    for name, param in VTmodel.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")
    for name, param in GPTmodel.named_parameters():
        print(f"{name}: requires_grad={param.requires_grad}")


def main(args):
    
    args = update_existing_config_for_inference(args)
    
    # global training_saver
    args.device = torch.device(args.device)
    
    # Setup CUDA, GPU & distributed training(?)
    #dist_init(args)
    check_arguments(args)
    set_seed(args.seed, args.num_gpus)

    if not is_main_process():
        logger.disabled = True

    logger.info(f"Pytorch version is: {torch.__version__}")
    logger.info(f"Cuda version is: {torch.version.cuda}")
    logger.info(f"cuDNN version is : {torch.backends.cudnn.version()}")

    # Get Vision Transformer, LLM and Tokenizer:
    VTmodel = VideoCaptionModel(num_latents=args.max_seq_length)
    GPTmodel = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = TokenizerHandler()

    # Freeze models
    freeze_models(VTmodel, GPTmodel)

    # load weights for inference
    logger.info(f"Loading LM from checkpoint {args.resume_LMcheckpoint}")
    logger.info(f"Loading Vision Transformer from checkpoint {args.resume_VTcheckpoint}")
    cpu_device = torch.device('cpu')
    LM_pretrained_model = torch.load(args.resume_LMcheckpoint,
                                  map_location=cpu_device)
    VT_pretrained_model = torch.load(args.resume_VTcheckpoint,
                                  map_location=cpu_device)

    if isinstance(LM_pretrained_model, dict):
        LM_rst = GPTmodel.load_state_dict(LM_pretrained_model, strict=True)
    else:
        LM_rst = GPTmodel.load_state_dict(LM_pretrained_model.state_dict(), strict=True)
    
    logger.info(f'Result of loading LM weights: {LM_rst}')

    if isinstance(VT_pretrained_model, dict):
        VT_rst = VTmodel.load_state_dict(VT_pretrained_model, strict=True)
    else:
        VT_rst = GPTmodel.load_state_dict(VT_pretrained_model.state_dict(), strict=True)

    logger.info(f'Result of loading VT weights: {VT_rst}')

    GPTmodel.to(args.device)
    VTmodel.to(args.device)
    GPTmodel.eval()
    VTmodel.eval()

    tensorizer = build_tensorizer(args, tokenizer, is_train=False)
    batch_inference(args, args.test_video_fname,
                    VTmodel, GPTmodel, tokenizer, tensorizer)


if __name__ == "__main__":
    shared_configs.shared_video_captioning_config(cbs=True, scst=True)
    args = get_custom_args(shared_configs)
    main(args)