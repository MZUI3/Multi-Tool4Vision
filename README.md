# Multi-Tool4Vision

English | [한국어](README.ko.md)

A PyTorch hub for training, evaluating and inspecting 136 image classification models in 53 families. It is built for a single GPU with 16 GB or less and needs only `torch` and `torchvision`.

## Installation

`pip install -r requirements.txt` installs `torch>=2.3.0`, `torchvision>=0.18.0` and `torchinfo>=1.8.0`. torchinfo is used only by `--mode summary`; without it, that mode prints the model and an install hint.

## Quick start

```bash
git clone https://github.com/MZUI3/Mutli-Tool4Vision.git
cd Mutli-Tool4Vision
pip install -r requirements.txt
python main.py --model resnet18 --dataset cifar10 --epochs 10
python main.py --mode eval --model mobilevit --checkpoint-path checkpoints/mobilevit_cifar10_best.pth
```

The repository tracks a copy of CIFAR-10 in `data/cifar-10-batches-py/`, so CIFAR-10 runs need no download. It also tracks `checkpoints/mobilevit_cifar10_best.pth`, a MobileViT checkpoint for CIFAR-10. `.gitignore` ignores everything else under `data/`, plus `checkpoints/*_last.pth` and `*.pth.tmp`; new `*_best.pth` files are not ignored.

## Usage

**list** prints every model by category and family, with its paper, then the datasets. `--category` limits it to `cnn`, `lightweight` or `transformer`.

```bash
python main.py --mode list --category transformer
```

**train** is the default mode. It validates after every epoch and saves checkpoints.

```bash
python main.py --model vit_tiny --dataset cifar10 --optimizer adamw --lr 1e-3 --warmup-epochs 5 --amp
```

**eval** runs one validation pass with the weights from `--checkpoint-path`. Use the `--model`, `--dataset` and `--resize` the checkpoint was trained with.

```bash
python main.py --mode eval --model resnet18 --dataset cifar10 --checkpoint-path checkpoints/resnet18_cifar10_best.pth
```

**print** shows the module tree and **summary** adds a torchinfo table. Neither loads data, but `imagefolder` needs `<data-root>/train` or `--num-classes` for the class count.

```bash
python main.py --mode summary --model convnext_tiny --dataset imagenet
```

**run.sh** wraps `main.py` with positional arguments (defaults: `mobilevit cifar10 ./checkpoints 20 64 cpu none 2`) and always adds `--download-data`. `parallel` is `none`, `dp` or `ddp` (`true` also means ddp). For ddp the script runs `torchrun` if it is on the PATH, otherwise `launch_ddp.py`, and does not pass `device`. Arguments after the eighth go to `main.py`; `bash run.sh --help` prints the full help.

```bash
bash run.sh [model] [dataset] [save_dir] [epochs] [batch_size] [device] [parallel] [nproc_per_node] [extra main.py args...]
bash run.sh resnet50 imagenet ./checkpoints 90 64 cuda ddp 4 --data-root /data/imagenet --amp
```

**Checkpoints.** Training writes `<model>_<dataset>_last.pth` to `--save-dir` (default `./checkpoints`) after every epoch, and `<model>_<dataset>_best.pth` whenever validation accuracy improves. Each file holds the weights, the EMA weights (with `--ema`), the optimizer, scheduler and AMP scaler state, the epoch, the best accuracy and the options. `--checkpoint-path` on its own loads only the weights; plain state dicts and DP/DDP `module.` prefixes also load. With `--resume` it also restores the optimizer, scheduler, scaler, epoch and best accuracy, and training continues up to `--epochs`, for example `python main.py --model resnet18 --epochs 50 --checkpoint-path checkpoints/resnet18_cifar10_last.pth --resume`. A changed `--epochs` reshapes the rest of the schedule. The EMA state is restored whenever `--ema` is on, with or without `--resume`; if the checkpoint has none, the EMA starts from the loaded weights. In eval mode `--ema` evaluates the EMA weights.

**Python.** `img_size` selects the stem (see Models).

```python
from models import create_model
model = create_model("resnet18", num_classes=10, img_size=32)
```

## Models

| Category | Implemented here: 67 models in `models/<category>/` | Wrapped from torchvision: 69 models in `models/torchvision_zoo.py` |
|---|---|---|
| `cnn`: 18 families, 54 models | ConvMixer, DLA, DPN, PreAct ResNet, PyramidNet, RepVGG, Res2Net, Xception | AlexNet, VGG, GoogLeNet, Inception v3, ResNet, ResNeXt, Wide ResNet, RegNet, ConvNeXt, DenseNet |
| `lightweight`: 17 families, 41 models | MobileNetV1, ShuffleNetV1, GhostNet, MobileOne, SENet, CBAM, ECA-Net, SKNet, ResNeSt, Coordinate Attention | MobileNetV2, MobileNetV3, ShuffleNetV2, SqueezeNet, MnasNet, EfficientNet, EfficientNetV2 |
| `transformer`: 18 families, 41 models | ViT (`vit_tiny`, `vit_small`), DeiT, CaiT, CCT, CvT, CoAtNet, BoTNet, PVTv2, MobileViT, MobileViTv2, PoolFormer, EfficientFormer, MLP-Mixer, ResMLP, gMLP | ViT (`vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32`), Swin Transformer, Swin Transformer V2, MaxViT |

- The torchvision builders are called with `weights=None`, so every model trains from scratch. `mobilevit` and `exmobilevit` are compact variants written for this project, not the paper's configurations. RepVGG and MobileOne train with multi-branch blocks; `models.reparameterize_model(model)` folds them into single convolutions for inference. `main.py` does not call it.
- Names ignore case, spaces, `_`, `-` and `.`, so `ResNet-18` finds `resnet18`. `mobilenet` is an alias for `mobilenet_v2` and `senet` for `se_resnet18`. An unknown name fails with up to three close matches.
- Small inputs: at 64px or less, most models switch to small-image settings so that a 32px image is not reduced to 1×1 before the last stage. These are a stride-1 first convolution without stem max-pooling, 2×2 patches for ConvNeXt, Swin and ConvMixer, and a 512-wide classifier for VGG. ViT, DeiT, CaiT, MLP-Mixer, ResMLP and gMLP use 4×4 patches below 128px and 16×16 otherwise. The torchvision ViTs use 4/8 instead of 16/32 at 64px or less. `mobilevit`, `exmobilevit`, CCT, AlexNet, Inception v3 and MaxViT keep their layers.
- Input size: without `--resize`, the dataset's native size is raised to the model's minimum and rounded up to its required multiple, and images are resized to match. On CIFAR-10, `alexnet` runs at 64px, `inception_v3` at 75px and `maxvit_t` at 224px. A `--resize` value the model cannot take is an error.

## Datasets

| `--dataset` | Classes | Native size | Data |
|---|---|---|---|
| `cifar10`, `cifar100` | 10, 100 | 32 | `--download-data` |
| `stl10` | 10 | 96 | `--download-data` |
| `food101`, `flowers102`, `oxfordpets` | 101, 102, 37 | 224 | `--download-data` |
| `imagenet` | 1000 | 224 | ILSVRC tar files + devkit in `--data-root`, or extracted `train/<wnid>/` and `val/<wnid>/` |
| `imagefolder` | from folders | 224 | `<data-root>/train/<class>/*` and `<data-root>/val/<class>/*` |

Data is read from `--data-root` (default `./data`), for example `python main.py --model resnet50 --dataset imagefolder --data-root ./my_images`. The torchvision datasets validate on their test split. ImageNet and `imagefolder` use `val/`, and for `imagefolder` every class folder must exist in both splits. Names ignore case, spaces, `_` and `-`. `pets` and `oxford_pets` mean `oxfordpets`; `folder` and `custom` mean `imagefolder`.

## Training options

| Group | Flags (default) | Effect |
|---|---|---|
| Multi-scale | `--multi-scale` | Each training batch uses a random resolution, and its batch size is scaled by (base/size)² to keep memory about constant. Validation stays at `--resize`. The model is test-run at the smallest and largest size first. |
| | `--ms-min-scale 0.625`, `--ms-max-scale 1.25`, `--ms-num-scales 5`, `--ms-sizes`, `--ms-fixed-batch` | Resolutions as fractions of `--resize`, rounded to multiples of 8 (4 below 64px) and kept at or above the model minimum, or an explicit list such as `160,192,224,256,288`. `--ms-fixed-batch` keeps one batch size. |
| Augmentation | `--auto-augment none`, `--random-erase 0` | `rand`, `trivial` or `autoaugment` on top of the basic augmentation (padded crop, flip and rotation for CIFAR and STL-10; RandomResizedCrop and flip for the rest); Random Erasing probability. |
| | `--mixup 0`, `--cutmix 0`, `--mix-prob 1.0`, `--mix-switch-prob 0.5` | Mixup and CutMix alpha (0 is off), chance to mix a batch, chance to pick CutMix when both are on. |
| Regularization | `--label-smoothing 0`, `--ema`, `--ema-decay 0.9998` | Label smoothing for cross-entropy. Weight EMA, used for validation and `*_best.pth` selection. |
| Optimization | `--optimizer sgd`, `--lr 0.01`, `--momentum 0.9`, `--weight-decay 1e-4`, `--grad-clip` | `sgd` (Nesterov when momentum > 0) or `adamw`. Biases, norm weights and position/class tokens get no weight decay. `--grad-clip` clips the gradient norm. |
| | `--epochs 10`, `--scheduler cosine`, `--warmup-epochs 0`, `--min-lr 0`, `--step-size 30`, `--gamma 0.1` | Per-epoch schedule: linear warmup, then `cosine` down to `--min-lr`, `step` (× `--gamma` every `--step-size` epochs) or `none` (constant). |
| Hardware | `--device auto`, `--parallel none`, `--ddp`, `--amp` | `auto`, `cpu`, `cuda` or `cuda:N`; `none`, `dp` or `ddp`, where `--ddp` is short for `--parallel ddp`. `--amp`: float16 autocast and GradScaler, CUDA only. |
| | `--gpu-ids`, `--sync-bn`, `--backend`, `--master-addr 127.0.0.1`, `--master-port 29500` | GPUs for dp; SyncBatchNorm, backend and rendezvous for ddp. |
| Checkpoints | `--save-dir ./checkpoints`, `--checkpoint-path`, `--resume` | See Usage. |
| Sampling | `--batch-size 64`, `--num-workers 4`, `--sampler default` | Batch size for training and validation; data-loading workers. `weighted` samples classes evenly, with replacement, and is ignored under ddp unless `--multi-scale` is on. |
| | `--seed 42`, `--limit-batches` | Seed, plus the rank under ddp. `--limit-batches N` uses only the first N batches per epoch, for debugging. |

A MobileViT-style recipe on the tracked CIFAR-10 copy:

```bash
python main.py --model mobilevit --dataset cifar10 --epochs 100 --batch-size 128 \
  --multi-scale --ema --mixup 0.2 --cutmix 1.0 --auto-augment rand --label-smoothing 0.1 \
  --optimizer adamw --lr 2e-3 --weight-decay 0.01 --warmup-epochs 5 --amp
```

- `--multi-scale`: batches at 32, 36 or 40px with 128, 101 or 82 images. The smaller default scales are clamped to the model's 32px minimum. Validation runs at 32px.
- `--mixup 0.2 --cutmix 1.0 --auto-augment rand`: every batch is mixed with Mixup or CutMix, picked at random, and RandAugment runs on top of crop, flip and rotation.
- `--label-smoothing 0.1 --ema`: softened targets; validation and `*_best.pth` use the averaged weights.
- `--optimizer adamw --lr 2e-3 --weight-decay 0.01 --warmup-epochs 5 --amp`: AdamW, 5 warmup epochs, then the default cosine decay, with mixed precision on CUDA. On CPU, `--amp` is turned off with a note.

## Multi-GPU

```bash
python main.py --model resnet50 --parallel dp --gpu-ids 0,1   # one process, nn.DataParallel
torchrun --nproc_per_node=2 main.py --model resnet50          # DDP, one process per GPU
python launch_ddp.py --nproc 2 main.py --model resnet50       # DDP without torchrun
```

- `--batch-size` is the batch per step on a single device. Under `dp` it is the total batch, split across the GPUs. Under ddp it is the batch per process, so the global batch is the process count × `--batch-size`. With `--multi-scale` it is the batch at `--resize`, and all ranks use the same resolution sequence.
- `dp` needs two or more CUDA devices and otherwise runs on one device; `--gpu-ids` defaults to all visible GPUs. A torchrun launch selects ddp automatically, and each rank uses the GPU of its local rank (choose GPUs with `CUDA_VISIBLE_DEVICES`). With `--device cpu` or without CUDA, the ranks run on CPU with gloo instead of NCCL. Only rank 0 logs and saves, and validation is split across ranks without padding.
- `launch_ddp.py` starts the processes itself with `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR` and `MASTER_PORT` set. Use it where `torchrun` is missing or broken, such as Windows builds without libuv.
- The hardware flags and `utils/hwaccel.py` are the same as in [Mutli-Tool4RL](https://github.com/MZUI3/Mutli-Tool4RL).

## Project structure

```
├── main.py              # entry point: list / train / eval / summary / print
├── trainer.py           # train_one_epoch, validate
├── launch_ddp.py        # torchrun stand-in
├── run.sh               # positional-argument wrapper around main.py
├── options/parser.py    # command-line flags
├── dataset/             # info.py (names, classes, sizes), dataset.py, transform.py, multiscale.py, sampler.py, data_loaders.py
├── models/              # registry.py, torchvision_zoo.py, layers.py, resnet_base.py
│   └── cnn/ lightweight/ transformer/   # hand-written models; most files hold one family
├── utils/               # hwaccel.py (device, DP/DDP, AMP), checkpoint.py, optim.py, ema.py, mixup.py
├── data/cifar-10-batches-py/               # tracked CIFAR-10 copy
└── checkpoints/mobilevit_cifar10_best.pth  # tracked checkpoint
```

## Adding a model

Create a file in the category's folder, for example `models/cnn/`. Decorate a builder that takes `num_classes` and `img_size`, then add the new module to the import list in that folder's `__init__.py`.

```python
import torch.nn as nn

from ..layers import ConvBNAct, is_small_input
from ..registry import register_model


@register_model(category="cnn", family="MyNet", paper="Author et al., Year")
def mynet(num_classes=1000, img_size=224):
    stride = 1 if is_small_input(img_size) else 2
    return nn.Sequential(ConvBNAct(3, 32, 3, stride), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, num_classes))
```

The name defaults to the function name, and `category` must be `cnn`, `lightweight` or `transformer`. Optional `min_input_size` (default 32), `input_size_divisor` and `aliases` feed the input-size rules and name lookup. A name that collides with an existing one after normalization raises an error.
