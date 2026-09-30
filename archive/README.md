# Archive: original research code and logs

This folder keeps the original research code of *Instance-wise Center Loss for Efficient
Training of Deep Convolutional Neural Networks* (IEEE GCCE 2022) for reference and
reproducibility. It is **unmaintained**: it needs a 2021-era environment (Python 3.6,
PyTorch 1.8, hydra-core 1.0, mlflow) and CUDA GPUs, and contains hard-coded cluster paths.
For new work use the package and `train.py` at the repository root, which reimplement the
same method (see the main [README](../README.md)). Nothing here is imported by the new code;
the tests in `tests/` only load some of these files to check that the rewrite gives the same
results.

The folder has two parts.

## 1. `release_2023/`: the code published with the paper

The code first published in this repository (November 2023), moved here unchanged.

| file | purpose |
|---|---|
| `train.py` | hydra + mlflow training script (`Solver`): K augmented views per image, the proposed loss and the comparison methods, 4-GPU `DataParallel` |
| `train_mask.py` | an identical copy of `train.py` |
| `config/` | hydra configs: `train.yaml` (method), `dataset/`, `model/`, `augmentation/` |
| `loss/` | center loss and contrastive-center loss |
| `dataset/` | modified torchvision datasets that return ten transformed copies of every image; `cifar.py` is left in the small-data setting of Table III (10 images of each of the classes 0-9) |
| `augmentation/` | Cutout, AutoAugment, AugMix (RandAugment was installed from ildoonet/pytorch-randaugment) |
| `models/` | WideResNet, ResNet-18, ResNeXt-29, DenseNet-BC-100, Shake-Shake-26 |
| `visualization.py` | t-SNE plot of test-set outputs of a saved model |
| `train.sh`, `requirement.txt`, `mlruns/` | cluster job script, pinned environment, empty mlflow store |
| `README.md` | the original README |

Mapping from `config/train.yaml` to the new command line:

| release setting | new command |
|---|---|
| `regularization: True`, `regularization_loss_function: MSE` (or `L1`, `Hubor`, `KL`) | `python train.py --method proposed --distance mse` (or `l1`, `huber`, `kl`) |
| `num_ensemble_imgs: 2`, `alpha_rate: 0.5` | `--num-views 2 --lambda-ic 0.5` (the paper, and the new default: 0.7 for ResNet-18, 0.5 otherwise) |
| `regularization: False`, `competitive_method: baseline` / `center_loss` / `contrastive_center_loss` / `triplet_loss` / `augmix` | `--method baseline` / `center_loss` / `contrastive_center_loss` / `triplet_loss` / `jsd` (`augmix` is accepted too) |
| `defaults: dataset / model / augmentation` | `--dataset`, `--model`, `--augmentation` (same names) |
| `cutout.length`, `cutout.n_holes`, `randaug.N`, `randaug.M` | `--cutout-length`, `--cutout-holes`, `--randaug-n`, `--randaug-m` |
| `config/model/*.yaml` (`optim.*`) | `--epochs`, `--lr`, `--weight-decay`, `--milestones`, `--gamma`, `--nesterov`; the defaults follow the paper (weight decay 5e-4 for every model, whereas the release used 1e-4 except for WideResNet) |
| `model: resnext` | `--model resnext --resnext-base-width 64` (the release's 8x64d; the paper and the default use 8x32d) |
| `data.batch_size: 128` x 4 GPUs | `--batch-size 512 --data-parallel` (the paper and the default use 256) |
| `visualization.py` | not ported (it needs scikit-learn); `best.pt` written by `train.py` holds the weights |

Where this release and the paper differ (hyper-parameters, ResNeXt width, the KL variant, the
training subset left in `dataset/cifar.py`), the new code follows the paper; release bugs
(STL-10 labels `-1` passed to the cross-entropy, contrastive-center-loss centres never
registered as parameters, hard-coded centre-loss feature size, unused seed) are fixed or
reproduced behind a flag. See "Notes on the paper, the released code and this
implementation" in the main README.

## 2. Earlier exploratory scripts and their logs

The top-level `main*.py` scripts and their `exec*.sh` job scripts are earlier experiments from
the development of the method. All of them train `WideResNet(depth=28, num_classes=100)` on
CIFAR-100 for 200 epochs (SGD, learning rate 0.1, momentum 0.9, weight decay 5e-4, decay x0.1 at
epochs 60/120/180, batch 64; batch 128 in `main_avg_randaug.py`) and print
`===> BEST ACC. PERFORMANCE: x%` (the best test accuracy over the epochs) at the end. `--num_imgs`
is the number of augmented views K.

| script | objective (per view, averaged over the K views) | closest new command |
|---|---|---|
| `main.py` | cross-entropy on one view (torchvision CIFAR-100) | `--method baseline --num-views 1` |
| `main_avg_nomse.py` | cross-entropy on each of the K views | `--method baseline --num-views K` |
| `main_avg_detach.py` | CE + MSE(view logits, detached mean of the views) | `--method proposed --distance mse --lambda-ic 0.5` (half of this loss) |
| `main_avg.py` (current version) | CE + KL(softmax(mean) \|\| softmax(view)), no stop-gradient | `--method proposed --distance kl --lambda-ic 0.5` (half of this loss) |
| `main_avg_randaug.py` | RandAugment(N, M); `--mean 1` adds MSE(view logits, mean of the views) | `--augmentation randaug --randaug-n N --randaug-m M --method proposed --lambda-ic 0.5` (half of this loss) |
| `main_avg_sim.py` | CE + negative cosine similarity to the detached mean (SimSiam-style) | not ported |
| `main_avg_sep.py` | CE + MSE between a second head and the detached mean | not ported |
| `main_avg_post.py` | CE + MSE to a mean weighted by the probability of the true class | not ported |
| `main_avg_em.py` | alternating CE step and MSE-to-mean step | not ported |
| `main_em.py`, `main_instance*.py` | MSE to class-wise or per-image running centres kept in memory | not ported |

(The scripts were edited between runs, as their commented-out lines show, so the objective of
an older log cannot always be read from the current version of its script.)

### Logs and `results_summary.csv`

`results/`, `results_exp1/`, `results_exp2/` and `results_exp3/wideresnet28-2/` hold 93 raw
training logs (about 2.2 GB of progress-bar output). The file names follow the `exec*.sh` scripts:
`<script>_num_<K>.txt` is `--num_imgs K`, and `<prefix>_<K>_<N>_<M>_<mean>[_tag].txt` is
`--num_imgs K --N N --M M --mean mean` (`exec_avg_randaug.sh`). The 24 `main_noavg_randaug_*`
logs were written by a script that is not in this archive.

[`results_summary.csv`](results_summary.csv) summarises every log (regenerate it with
`python scripts/summarize_archived_logs.py` from the repository root):

| column | meaning |
|---|---|
| `path`, `experiment_folder`, `run_name` | the log file |
| `script`, `script_archived` | the script that wrote it (from the file name) and whether it is in this folder |
| `setting`, `num_views` | the arguments encoded in the file name |
| `epochs_total`, `epochs_completed` | epochs planned and epochs with a complete test pass |
| `train_images_per_epoch` | training images seen in the first epoch (50,000 in every log that has one) |
| `best_acc_reported` | the `BEST ACC. PERFORMANCE` value printed at the end (empty if the run did not finish) |
| `best_test_acc_parsed` | the maximum of the per-epoch test accuracies (equal to the reported value for all 80 finished runs) |
| `last_epoch_test_acc` | the test accuracy after the last completed epoch |
| `status` | `complete` (80), `incomplete` (9) or `no epoch finished` (4) |

These logs document the development of the method; they are not the experiments reported in
the paper.
