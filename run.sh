## Single GPU ##
#python \
#    ./src/tasks/train.py \
#    --config ./src/configs/favd_32frm_default.json \
#    --per_gpu_train_batch_size 2 \
#    --per_gpu_eval_batch_size 2 \
#    --num_train_epochs 150 \
#    --learning_rate 0.0001 \
#    --max_num_frames 32 \
#    --backbone_coef_lr 0.05 \
#    --learn_mask_enabled \
#    --loss_sparse_w 0.5 \
#    --lambda_ 0.1 \
#    --output_dir ./output/favd_default \

## Multiple GPUs ##
torchrun --nproc_per_node=1 \
    ./src/tasks/train.py \
    --config ./src/configs/8frames_default.json \
    --per_gpu_train_batch_size 1 \
    --per_gpu_eval_batch_size 1 \
    --num_train_epochs 10 \
    --learning_rate 0.0001\
    --max_num_frames 8 \
    --backbone_coef_lr 0.05 \
    --loss_sparse_w 0.5 \
    --lambda_ 0.1 \
    --output_dir ./output/ \

## Multiple nodes ##
#torchrun --nproc_per_node=8 \
#    --master_addr= \
#    --master_port= \
#    --nnodes= \
#    --node_rank= \
#    --config ./src/configs/favd_32frm_default.json \
#    --per_gpu_train_batch_size 2 \
#    --per_gpu_eval_batch_size 2 \
#    --num_train_epochs 150 \
#    --learning_rate 0.0001 \
#    --max_num_frames 32 \
#    --backbone_coef_lr 0.05 \
#    --learn_mask_enabled \
#    --loss_sparse_w 0.5 \
#    --lambda_ 0.1 \
#    --output_dir ./output/favd_default \