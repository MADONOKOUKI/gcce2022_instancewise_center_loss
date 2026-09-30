#!/usr/bin/env bash
# Table IV of the paper: distance of the instance-wise center loss (L1, Huber, KL, L2)
# with lambda_IC = 0.1, ResNet-18, flip and cropping, compared with no regularization.
# The paper does not name the dataset of this table, so pass it as the first argument.
#
#   bash scripts/reproduce_table4.sh <cifar10|cifar100|cub200> [extra train.py flags]
#   python scripts/summarize_runs.py runs
set -euo pipefail
cd "$(dirname "$0")/.."
DATASET=${1:?usage: bash scripts/reproduce_table4.sh <dataset> [extra train.py flags]}
shift

for seed in 1 2 3; do
  python train.py --dataset "$DATASET" --model resnet18 --augmentation standard --method baseline \
    --num-views 2 --seed "$seed" "$@"
  for distance in l1 huber kl mse; do             # mse = L2 (the default)
    python train.py --dataset "$DATASET" --model resnet18 --augmentation standard --method proposed \
      --distance "$distance" --lambda-ic 0.1 --num-views 2 --seed "$seed" "$@"
  done
done
