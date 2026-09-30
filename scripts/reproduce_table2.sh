#!/usr/bin/env bash
# Table II of the paper: ResNet-18 with N=3 augmented views (lambda_IC = 0.7), no
# regularization vs. the JS-divergence of AugMix vs. the instance-wise center loss,
# mean +- std over three runs. 2 datasets x 4 augmentations x 3 methods x 3 seeds = 72 runs.
#
#   bash scripts/reproduce_table2.sh [extra train.py flags]
#   python scripts/summarize_runs.py runs
set -euo pipefail
cd "$(dirname "$0")/.."

for dataset in cifar10 cifar100; do
  for aug in standard cutout autoaug augmix; do   # flip and cropping, Cutout, AutoAugment, AugMix
    for method in baseline jsd proposed; do       # jsd: CE on the clean view + 12 x JSD(clean, view 1, view 2)
      for seed in 1 2 3; do
        python train.py --dataset "$dataset" --model resnet18 --augmentation "$aug" \
          --method "$method" --num-views 3 --lambda-ic 0.7 --seed "$seed" "$@"
      done
    done
  done
done
