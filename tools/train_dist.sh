#!/usr/bin/env bash

export PYTHONPATH=./
GPUS=1

# For remapped classes training
OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/train.py \
	configs/svg/svg_pointT_remapped.yaml \
	--dist \
	--exp_name remapped_classes \
	--sync_bn

# For training without railing (uncomment to use)
# OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/train.py \
# 	configs/svg/svg_pointT_no_railing.yaml \
# 	--dist \
# 	--exp_name no_railing \
# 	--sync_bn

# For original training (uncomment to use)
# OMP_NUM_THREADS=$GPUS torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/train.py \
# 	configs/svg/svg_pointT.yaml \
# 	--dist \
# 	--exp_name baseline_original \
# 	--sync_bn
