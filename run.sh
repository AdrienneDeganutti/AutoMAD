## Multiple GPUs ##
torchrun --nproc_per_node=1 \
    ./src/tasks/train.py \
    --config ./src/configs/8frames_default.json \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --num_train_epochs 20 \
    --learning_rate 0.0001\
    --max_num_frames 8 \
    --backbone_coef_lr 0.05 \
    --loss_sparse_w 0.5 \
    --output_dir ./output/1clip-frozen-lm/ \