import torch
import json
import time
import torch.distributed as dist
import os.path as op

from tqdm import tqdm
from src.utils.logger import LOGGER as logger
from src.evalcap.utils_caption_evaluate import evaluate_on_coco_caption
from src.utils.comm import get_rank, get_world_size, is_main_process
from src.utils.miscellaneous import concat_tsv_files, delete_tsv_files
from src.utils.tsv_file_ops import reorder_tsv_keys, tsv_writer


def evaluate(args, val_dataloader, VTmodel, GPTmodel, tokenizer, output_dir):
    
    predict_file = get_predict_file(output_dir, args, val_dataloader.dataset.yaml_file)
    test(args, val_dataloader, VTmodel, GPTmodel, tokenizer, predict_file)

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



def test(args, test_dataloader, VTmodel, GPTmodel, tokenizer, predict_file):

    tokenizer = tokenizer.tokenizer

    #cls_token_id, sep_token_id, pad_token_id, mask_token_id, period_token_id = \
     #   tokenizer._convert_token_to_id([tokenizer.cls_token, tokenizer.sep_token,
    #    tokenizer.pad_token, tokenizer.mask_token, '.'])
    world_size = get_world_size()
    if world_size == 1:
        cache_file = predict_file
    else:
        # local_rank would not work for cross-node distributed training
        cache_file = op.splitext(predict_file)[0] + '_{}_{}'.format(
            get_rank(), world_size) + op.splitext(predict_file)[1]

    VTmodel.eval()
    GPTmodel.eval()

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
            for step, (img_keys, caption, visual_frame) in tqdm(enumerate(test_dataloader)):
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

                tic = time.time()

                visual_frame = visual_frame.to(args.device)
                prefix_vector = VTmodel(visual_frame, img_keys)
                outputs = GPTmodel(inputs_embeds=prefix_vector)

                time_meter += time.time() - tic
                all_caps = outputs[0]  # batch_size * num_keep_best * max_len
                #all_confs = torch.exp(outputs[1])

                for img_key, caps in zip(img_keys, all_caps):
                    res = []
                    
                    caps = torch.max(caps, -1)[1].data
                    cap = tokenizer.decode(caps)
                    res.append({'caption': cap})
                    
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