#!/usr/bin/env bash

export PYTHONPATH=./

# For remapped classes training (non-distributed)
python tools/train.py configs/svg/svg_pointT_remapped.yaml --exp_name remapped_classes_simple

# For training without railing (uncomment to use)
# python tools/train.py configs/svg/svg_pointT_no_railing.yaml --exp_name no_railing_simple

# For original training (uncomment to use) 
# python tools/train.py configs/svg/svg_pointT.yaml --exp_name baseline_original_simple