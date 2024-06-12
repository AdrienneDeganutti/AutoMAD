python \
    src/tasks/inference.py \
    --eval_model_dir output/full-Batch-Size-8-ep100/ \
    --checkpoint_dir checkpoint-81-619083/ \
    --test_video_fname datasets/test/ \
    --test_features datasets/8_frames_tsv/train_segmented_features.tsv \
    --test_yaml metadata/train_8frames.yaml \
    --do_lower_case \
    --do_test \