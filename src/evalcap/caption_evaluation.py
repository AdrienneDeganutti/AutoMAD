import json
from coco_caption.pycocotools.coco import COCO
from coco_caption.pycocoevalcap.eval import COCOEvalCap

input_file = 'AutoMAD/output/full-Batch-Size-8-ep100/checkpoint-84-642012/pred.metadata.test_8frames.beam1.max75_coco_format.json'
gt_file = 'AutoMAD/datasets/metadata/test.caption_coco_format.json'
output_file = 'AutoMAD/output/caption-scores.json'

coco = COCO(gt_file)
cocoRes = coco.loadRes(input_file)

img_ids = cocoRes.getImgIds()

caption_results = {}
for img_id in img_ids:
    cocoEval = COCOEvalCap(coco, cocoRes, 'corpus')
    cocoEval.params['image_id'] = [img_id]
    cocoEval.evaluate()

    caption_results[img_id] = cocoEval.evalImgs[0]

with open(input_file, 'r') as f:
    input_predictions = json.load(f)

for prediction in input_predictions:
    img_id = prediction['image_id']
    if img_id in caption_results:
        prediction['results'] = caption_results[img_id]

with open(output_file, 'w') as f:
    json.dump(input_predictions, f, indent=4)