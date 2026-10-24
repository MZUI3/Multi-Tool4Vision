# Multi-Tool4Vision

## Overview

이미지 분류 모델 136개(53개 계열)를 하나의 PyTorch 코드로 학습하는 저장소입니다. 개인 GPU(16GB 이하)에서 돌리는 것을 기준으로 했고, 외부 모델 라이브러리 없이 torch와 torchvision만 씁니다.
절반 정도의 계열(MobileViT, PVT, CoAtNet, RepVGG, GhostNet, SKNet, CaiT, CCT, ...)은 논문을 보고 직접 구현했고 나머지는 torchvision 구현을 그대로 등록했습니다.

학습 쪽에는 MobileViT 논문의 멀티스케일 샘플러, EMA, Mixup/CutMix, RandAugment, DP/DDP가 들어 있습니다.

## Usage

```bash
pip install -r requirements.txt

python main.py --mode list                                                     # 모델 목록
python main.py --model mobilevit --dataset cifar10 --download-data --epochs 20  # 학습
python main.py --model mobilevit --dataset cifar10 --mode eval --checkpoint-path checkpoints/mobilevit_cifar10_best.pth
python main.py --model resnet50 --dataset imagefolder --data-root ./my_images   # train/<class>/, val/<class>/ 폴더
```

MobileViT 레시피:

```bash
python main.py --model mobilevit --dataset cifar10 --download-data --epochs 100 --batch-size 128 \
  --multi-scale --ema --mixup 0.2 --cutmix 1.0 --auto-augment rand --label-smoothing 0.1 \
  --optimizer adamw --lr 2e-3 --weight-decay 0.01 --warmup-epochs 5 --amp
```

멀티 GPU:

```bash
python main.py --model resnet50 --dataset cifar100 --parallel dp --gpu-ids 0,1
torchrun --nproc_per_node=2 main.py --model resnet50 --dataset cifar100
python launch_ddp.py --nproc 2 main.py --model resnet50 --dataset cifar100       # torchrun이 안 될 때
```

모델을 코드에서 바로 쓸 수도 있습니다.

```python
import torch
from models import create_model

model = create_model("mobilevit", num_classes=10, img_size=32)
out = model(torch.randn(1, 3, 32, 32))
```

## Models

`python main.py --mode list`로 이름과 논문을 볼 수 있습니다. 이름은 대소문자와 `_`, `-`를 구분하지 않습니다.

- **CNN**: ResNet, PreAct ResNet, ResNeXt, Wide ResNet, DenseNet, DPN, Res2Net, PyramidNet, DLA, Xception, RegNet, ConvNeXt, RepVGG, ConvMixer, VGG, AlexNet, GoogLeNet, Inception v3
- **경량 / 채널 어텐션**: MobileNet v1/v2/v3, ShuffleNet v1/v2, SqueezeNet, MnasNet, EfficientNet, EfficientNetV2, GhostNet, MobileOne, SENet, CBAM, ECA-Net, SKNet, ResNeSt, Coordinate Attention
- **Transformer / MLP**: MobileViT, MobileViTv2, ViT, DeiT, CaiT, Swin, Swin V2, MaxViT, PVTv2, CvT, PoolFormer, EfficientFormer, CoAtNet, BoTNet, CCT, MLP-Mixer, ResMLP, gMLP

64px 이하 입력에서는 stem을 CIFAR용으로 바꿉니다(stride-1 conv, 2x2 패치, 4px ViT 패치). 모델이 더 큰 입력을 요구하면 자동으로 키웁니다.

## Datasets

`cifar10`, `cifar100`, `stl10`, `imagenet`, `food101`, `flowers102`, `oxford_pets`, `imagefolder`.
`imagefolder`는 `<data-root>/train/<class>/`, `<data-root>/val/<class>/` 구조면 됩니다.

## Options

| 옵션 | 내용 |
|---|---|
| `--multi-scale` | 배치마다 해상도를 바꾸고 배치 크기를 `(기본/해상도)^2`배로 조절 (`--ms-sizes`, `--ms-min-scale`, `--ms-max-scale`) |
| `--ema`, `--mixup`, `--cutmix`, `--auto-augment`, `--random-erase`, `--label-smoothing` | 정규화 / 증강 |
| `--optimizer sgd\|adamw`, `--scheduler cosine\|step\|none`, `--warmup-epochs`, `--grad-clip`, `--amp` | 최적화 |
| `--parallel none\|dp\|ddp`, `--device`, `--gpu-ids`, `--sync-bn` | 하드웨어 |
| `--checkpoint-path`, `--resume`, `--save-dir` | 체크포인트. `--resume`은 optimizer, scheduler, EMA까지 복원 |
| `--sampler weighted` | 클래스 불균형용 가중 샘플링 |

하드웨어 가속 코드(`utils/hwaccel.py`)는 [Mutli-Tool4RL](https://github.com/MZUI3/Mutli-Tool4RL)과 같은 파일입니다.

## Structure

```
main.py                 진입점
trainer.py              학습 / 검증 루프
models/                 registry.py + cnn/ lightweight/ transformer/ + torchvision_zoo.py
dataset/                데이터셋, transform, 멀티스케일 샘플러
utils/                  hwaccel.py (DP/DDP/AMP), ema.py, mixup.py, optim.py, checkpoint.py
launch_ddp.py           torchrun 대체 런처
```

새 모델은 `models/<카테고리>/`에 파일을 만들고 `@register_model`을 붙이면 됩니다.

```python
from ..registry import register_model

@register_model(category="cnn", family="MyNet", paper="Me et al., 2026")
def mynet(num_classes=1000, img_size=224):
    return MyNet(num_classes, img_size)
```
