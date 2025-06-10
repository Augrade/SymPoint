#!/usr/bin/env bash

export PYTHONPATH=./
GPUS=1
workdir=./work_dirs/svg/svg_pointT/baseline_nclsw_grelu
OMP_NUM_THREADS=$GPUS /home/kai/miniconda3/envs/spv1/bin/torchrun --nproc_per_node=$GPUS --master_port=$((RANDOM + 10000)) tools/test.py \
	 $workdir/svg_pointT.yaml  $workdir/best.pth --dist
