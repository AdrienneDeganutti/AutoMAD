python \
    src/tasks/inference.py \
    --eval_LMmodel_dir output/full-dataset-30ep/ \
    --eval_ViTmodel_dir output/full-dataset-30ep/ \
    --test_video_fname datasets/test/video_file.mp4 \
    --test_features /datasets/test/8_frames_test_features.tsv \
    --do_lower_case \
    --do_test \