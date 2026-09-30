#!/usr/bin/env bash
# Table III of the paper: ResNet-18 with AutoAugment trained on 100, 500 or 1000 CIFAR-10
# images (the same number per class), N=2 views, mean +- std over three runs.
# 3 subset sizes x 5 regularizations x 3 seeds = 45 runs. The seed also selects the subset,
# so all methods of one seed see the same images.
#
#   bash scripts/reproduce_table3.sh [extra train.py flags]
#   python scripts/summarize_runs.py runs
set -euo pipefail
cd "$(dirname "$0")/.."

for per_class in 10 50 100; do                    # 100 / 500 / 1000 training images
  for method in baseline center_loss contrastive_center_loss triplet_loss proposed; do
    for seed in 1 2 3; do
      python train.py --dataset cifar10 --subset-per-class "$per_class" --model resnet18 --augmentation autoaug \
        --method "$method" --num-views 2 --lambda-ic 0.7 --seed "$seed" "$@"
    done
  done
done
