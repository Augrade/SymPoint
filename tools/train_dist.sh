#!/usr/bin/env bash

export PYTHONPATH=./
GPUS=8

# For remapped classes training
# To resume training, add: --resume /path/to/checkpoint.pth
OMP_NUM_THREADS=$GPUS /home/kai/miniconda3/envs/spv1/bin/torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/train.py \
	configs/svg/svg_pointT_remapped.yaml \
	--dist \
	--exp_name remapped_classes \
	--sync_bn \
	--resume /home/kai/SymPoint/work_dirs/svg/svg_pointT_remapped/remapped_classes/latest.pth

# For training without railing (uncomment to use)
# OMP_NUM_THREADS=$GPUS /home/kai/miniconda3/envs/spv1/bin/torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/train.py \
# 	configs/svg/svg_pointT_no_railing.yaml \
# 	--dist \
# 	--exp_name no_railing \
# 	--sync_bn

# For original training (uncomment to use)
# OMP_NUM_THREADS=$GPUS /home/kai/miniconda3/envs/spv1/bin/torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/train.py \
# 	configs/svg/svg_pointT.yaml \
# 	--dist \
# 	--exp_name baseline_original \
# 	--sync_bn
