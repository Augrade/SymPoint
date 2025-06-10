#!/usr/bin/env bash


export PYTHONPATH=./
GPUS=1

OMP_NUM_THREADS=$GPUS /home/kai/miniconda3/envs/spv1/bin/torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/train.py \
	--dist configs/svg/svg_pointT.yaml  \
	--exp_name baseline_nclsw_grelu \
	--sync_bn
