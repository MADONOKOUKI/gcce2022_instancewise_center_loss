#!/usr/bin/env bash
# Table I of the paper: accuracy with N=2 augmented views, mean +- std over three runs.
# 3 datasets x 3 models x 3 augmentations x 5 regularizations x 3 seeds = 405 training runs.
#
#   CUB_ROOT=/path/to/CUB_200_2011 bash scripts/reproduce_table1.sh [extra train.py flags]
#   python scripts/summarize_runs.py runs          # mean +- std per setting
#
# CIFAR-10/100 download automatically; CUB-200-2011 must be downloaded by hand
# (https://www.vision.caltech.edu/datasets/cub_200_2011/), see the README.
set -euo pipefail
cd "$(dirname "$0")/.."
CUB_ROOT=${CUB_ROOT:-./data/CUB_200_2011}

for dataset in cifar10 cifar100 cub200; do
  root=./data
  if [ "$dataset" = cub200 ]; then root=$CUB_ROOT; fi
  for model in resnet18 resnext wideresnet; do
    lambda_ic=0.5                       # ResNeXt-29 8x32d and WideResNet-28-10
    if [ "$model" = resnet18 ]; then lambda_ic=0.7; fi
    for aug in standard cutout autoaug; do   # flip and cropping, Cutout, AutoAugment
      for method in baseline center_loss contrastive_center_loss triplet_loss proposed; do
        for seed in 1 2 3; do
          python train.py --dataset "$dataset" --data-root "$root" --model "$model" --augmentation "$aug" \
            --method "$method" --num-views 2 --lambda-ic "$lambda_ic" --seed "$seed" "$@"
        done
      done
    done
  done
done
