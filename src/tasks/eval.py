import torch
import json
import time
import wandb
import torch.distributed as dist
import os.path as op

from tqdm import tqdm
from src.evalcap.coco_caption.pycocotools.coco import COCO
from src.evalcap.coco_caption.pycocoevalcap.eval import COCOEvalCap
from src.utils.logger import LOGGER as logger
from src.evalcap.utils_caption_evaluate import evaluate_on_coco_caption
from src.utils.comm import get_rank, get_world_size, is_main_process
from src.utils.miscellaneous import concat_tsv_files, delete_tsv_files
from src.utils.tsv_file_ops import reorder_tsv_keys, tsv_writer


class Evaluation:
    def __init__(self, args, VTmodel, GPTmodel, tokenizer, val_dataloader, test_dataloader):
        self.args = args
        self.VTmodel = VTmodel
        self.GPTmodel = GPTmodel
        self.tokenizer = tokenizer
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader


    def evaluate(self, checkpoint_dir, epoch, val_log, test_log, iteration, val_best_score, test_best_score):
    
        val_predict_file = self._get_predict_file(self.val_dataloader.dataset.yaml_file, checkpoint_dir)
        test_predict_file = self._get_predict_file(self.test_dataloader.dataset.yaml_file, checkpoint_dir)
        self._test(self.val_dataloader, val_predict_file, epoch)
        self._test(self.test_dataloader, test_predict_file, epoch)

        if get_world_size() > 1:
            dist.barrier()
    
        val_evaluate_file = self._get_evaluate_file(val_predict_file)
        test_evaluate_file = self._get_evaluate_file(test_predict_file)

        if is_main_process():
            val_caption_file = self.val_dataloader.dataset.get_caption_file_in_coco_format()
            val_result = evaluate_on_coco_caption(val_predict_file, val_caption_file, outfile=val_evaluate_file)
        
            logger.info(f'Validation result: {str(val_result)}')
            logger.info(f'Validation result saved to {val_evaluate_file}')

            test_caption_file = self.test_dataloader.dataset.get_caption_file_in_coco_format()
            test_result = evaluate_on_coco_caption(test_predict_file, test_caption_file, outfile=test_evaluate_file)

            logger.info(f'Evaluation result: {str(test_result)}')
            logger.info(f'Evaluation result saved to {test_evaluate_file}')
    
        if get_world_size() > 1:
            dist.barrier()
            
        if is_main_process():
            with open(val_evaluate_file, 'r') as f:
                val_res = json.load(f)
            val_best_score = max(val_best_score, val_res['CIDEr'])
            val_res['epoch'] = epoch
            val_res['iteration'] = iteration
            val_res['best_CIDEr'] = val_best_score
            val_log.append(val_res)
            with open(op.join(self.args.output_dir, self.args.val_yaml.replace('/', '_') + 'val_logs.json'), 'w') as f:
                json.dump(val_log, f)

            with open(test_evaluate_file, 'r') as f:
                test_res = json.load(f)
            test_best_score = max(test_best_score, test_res['CIDEr'])
            test_res['epoch'] = epoch
            test_res['iteration'] = iteration
            test_res['best_CIDEr'] = test_best_score
            test_log.append(test_res)
            with open(op.join(self.args.output_dir, self.args.test_yaml.replace('/', '_') + 'test_logs.json'), 'w') as f:
                json.dump(test_log, f)
        
        if get_world_size() > 1:
            dist.barrier()

        if is_main_process():
            wandb.log({"Validation Accuracy (CIDEr)": val_res['CIDEr'],
                       "Testing Accuracy (CIDEr)": test_res['CIDEr']}, step=epoch)
        
            #Evaluate individual captions for Validation split:
            val_file = (self.args.val_yaml.split('.')[0]).split('/')[1]
            json_predictions_dir = op.join(checkpoint_dir, f'pred.metadata.{val_file}.beam1.max75_coco_format.json')
            self.evaluate_individual_captions(val_caption_file, json_predictions_dir)

            #Evaluate individual captions for Testing split:
            test_file = (self.args.test_yaml.split('.')[0]).split('/')[1]
            test_json_predictions_dir = op.join(checkpoint_dir, f'pred.metadata.{test_file}.beam1.max75_coco_format.json')
            self.evaluate_individual_captions(test_caption_file, test_json_predictions_dir)
        
        if get_world_size() > 1:
            dist.barrier()

        return val_best_score, test_best_score


    def _get_predict_file(self, data_yaml_file, checkpoint_dir):
        cc = ['pred']
        data = data_yaml_file.split('/')[-2]
        
        if data != 'coco_caption':
            cc.append(data)
    
        cc.append(op.splitext(op.basename(data_yaml_file))[0])
        cc.append('beam{}'.format(self.args.num_beams))
        cc.append('max{}'.format(self.args.max_gen_length))
    
        if self.args.num_keep_best != 1:
            cc.append('best{}'.format(self.args.num_keep_best))
        if self.args.output_hidden_states:
            cc.append('hidden')
    
        return op.join(checkpoint_dir, '{}.tsv'.format('.'.join(cc)))


    def _get_evaluate_file(self, predict_file):
        assert predict_file.endswith('.tsv')
        return op.splitext(predict_file)[0] + '.eval.json'



    def _test(self, dataloader, predict_file, epoch):

        world_size = get_world_size()
        cache_file = predict_file if world_size == 1 else f"{op.splitext(predict_file)[0]}_{get_rank()}_{world_size}{op.splitext(predict_file)[1]}"
        type = dataloader.dataset.cap_file.split('.')[0]

        self.VTmodel.eval()
        self.GPTmodel.eval()

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
                for step, (img_keys, caption, visual_frame) in tqdm(enumerate(dataloader)):
                    is_exist = all(k in exist_key2pred for k in img_keys)
                    if is_exist:
                        for k in img_keys:
                            yield k, exist_key2pred[k]
                        continue

                    tic = time.time()

                    visual_frame = visual_frame.to(self.args.device)
                    prefix_vector = self.VTmodel(visual_frame, img_keys)

                    # Token shift workaround:
                    add = -100 * torch.ones(prefix_vector.shape[0], 1, prefix_vector.shape[2]).to(self.args.device)  # Shape [batch_size, 1, feature_size]
                    prefix_vector = torch.cat([add, prefix_vector], dim=1)

                    outputs = self.GPTmodel(inputs_embeds=prefix_vector, )

                    time_meter += time.time() - tic
                    all_caps = outputs[0]  # batch_size * num_keep_best * max_len

                    # To compute accuracy:
                    #tokenized_caption = self.tokenizer.tokenize_caption_for_eval(self.args, caption)
                    #batch_accuracy = torch.tensor([]).to(self.args.device)

                    for img_key, caps in zip(img_keys, all_caps):
                        res = []
                        caps = torch.max(caps, -1)[1].data
                        cap = self.tokenizer.tokenizer.decode(caps, skip_special_tokens=True)
                        res.append({'caption': cap})
                    
                        if isinstance(img_key, torch.Tensor):
                            img_key = img_key.item()
                        yield img_key, json.dumps(res)

                        #Compute batch accuracy:
                        #append_token = torch.tensor([50256], dtype=torch.long).to(self.args.device)
                        #caps = torch.cat([append_token, caps], dim=0)

                        #correct_preds = (caps == GT).float()
                        #batch_accuracy = torch.cat((batch_accuracy, correct_preds))
                
                    if world_size > 1:
                        dist.barrier()
                
                    #batch_acc = torch.mean(batch_accuracy)
                    #batch_acc = float(batch_acc)
                    
                    #if type == 'val':
                    #    if self.args.rank == 0:
                    #        wandb.log({"Validation Accuracy": batch_acc}, step=epoch)
                    #if type == 'test':
                    #    if self.args.rank == 0:
                    #        wandb.log({"Test Accuracy": batch_acc}, step=epoch)


            logger.info(f"Inference model computing time: {(time_meter / (step+1))} seconds per batch")

        tsv_writer(gen_rows(), cache_file)
    
        if world_size > 1:
            dist.barrier()
        if world_size > 1 and is_main_process():
            cache_files = [
                f"{op.splitext(predict_file)[0]}_{i}_{world_size}{op.splitext(predict_file)[1]}"
                for i in range(world_size)
            ]
            concat_tsv_files(cache_files, predict_file)
            delete_tsv_files(cache_files)
            reorder_tsv_keys(predict_file, dataloader.dataset.image_keys, predict_file)
        if world_size > 1:
            dist.barrier()


    def evaluate_individual_captions(self, ground_truth, predictions_file):
        logger.info('Beginning Caption-level evaluation...')

        coco = COCO(ground_truth)
        cocoRes = coco.loadRes(predictions_file)

        img_ids = cocoRes.getImgIds()

        caption_results = {}
        scores = []
        for img_id in img_ids:
            cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')
            cocoEval.params['image_id'] = [img_id]
            cocoEval.evaluate()

            caption_results[img_id] = cocoEval.evalImgs[0]
            scores.append((img_id, cocoEval.evalImgs[0]['CIDEr']))  # Collecting CIDER scores
        
        with open(predictions_file, 'r') as f:
            caption_predictions = json.load(f)
            
        for caption in caption_predictions:
            img_id = caption['image_id']
            if img_id in caption_results:
                caption['results'] = caption_results[img_id]

        # Calculate top-3 and lowest-3 img_IDs based on CIDEr scores
        scores.sort(key=lambda x: x[1], reverse=True)
        top_3 = scores[:3]
        lowest_3 = scores[-3:]

        output_data = {
            "predictions": caption_predictions,
            "top_3_highest_rated": [img_id for img_id, _ in top_3],
            "lowest_3_rated": [img_id for img_id, _ in lowest_3]
        }


        output_file_path = predictions_file.replace('.beam1.max75_coco_format.json', '.caption_metrics.json')
        with open(output_file_path, 'w') as f:
            json.dump(output_data, f, indent=4)
        
        logger.info(f'\nUpdated predictions with individual scores saved to {output_file_path}')