# Multi-Tool4Vision

[English](README.md) | 한국어

이미지 분류 모델 136개(53개 계열)를 학습·평가하고 구조를 확인하는 PyTorch 저장소입니다. 16GB 이하 GPU 한 장을 기준으로 만들었고, `torch`와 `torchvision`만 있으면 됩니다.

## 설치

`pip install -r requirements.txt`로 `torch>=2.3.0`, `torchvision>=0.18.0`, `torchinfo>=1.8.0`을 설치합니다. torchinfo는 `--mode summary`에서만 씁니다. 없으면 이 모드는 모델과 설치 안내만 출력합니다.

## 빠른 시작

```bash
git clone https://github.com/MZUI3/Mutli-Tool4Vision.git
cd Mutli-Tool4Vision
pip install -r requirements.txt
python main.py --model resnet18 --dataset cifar10 --epochs 10
python main.py --mode eval --model mobilevit --checkpoint-path checkpoints/mobilevit_cifar10_best.pth
```

저장소의 `data/cifar-10-batches-py/`에 CIFAR-10 사본이 들어 있어서 CIFAR-10은 따로 내려받지 않아도 됩니다. CIFAR-10용 MobileViT 체크포인트 `checkpoints/mobilevit_cifar10_best.pth`도 들어 있습니다. `.gitignore`는 `data/` 아래의 나머지와 `checkpoints/*_last.pth`, `*.pth.tmp`를 무시합니다. 새로 생기는 `*_best.pth` 파일은 무시하지 않습니다.

## 사용법

**list**는 모든 모델을 카테고리와 계열별로 논문과 함께 출력하고, 이어서 데이터셋을 출력합니다. `--category`로 `cnn`, `lightweight`, `transformer` 중 하나만 볼 수 있습니다.

```bash
python main.py --mode list --category transformer
```

**train**이 기본 모드입니다. 에폭마다 검증하고 체크포인트를 저장합니다.

```bash
python main.py --model vit_tiny --dataset cifar10 --optimizer adamw --lr 1e-3 --warmup-epochs 5 --amp
```

**eval**은 `--checkpoint-path`의 가중치로 검증 세트를 한 번 돌립니다. `--model`, `--dataset`, `--resize`는 체크포인트를 학습할 때와 같게 주세요.

```bash
python main.py --mode eval --model resnet18 --dataset cifar10 --checkpoint-path checkpoints/resnet18_cifar10_best.pth
```

**print**는 모듈 트리를 보여 주고, **summary**는 여기에 torchinfo 표를 더합니다. 둘 다 데이터를 불러오지 않습니다. 다만 `imagefolder`는 클래스 수를 알아야 하므로 `<data-root>/train`이나 `--num-classes`가 필요합니다.

```bash
python main.py --mode summary --model convnext_tiny --dataset imagenet
```

**run.sh**는 `main.py`를 위치 인자로 감싸고(기본값: `mobilevit cifar10 ./checkpoints 20 64 cpu none 2`), 항상 `--download-data`를 붙입니다. `parallel`은 `none`, `dp`, `ddp` 중 하나이고, `true`도 ddp로 처리합니다. ddp일 때는 PATH에 `torchrun`이 있으면 `torchrun`을, 없으면 `launch_ddp.py`를 실행하며 `device`는 넘기지 않습니다. 여덟 번째 인자 뒤의 인자는 `main.py`로 넘어갑니다. 전체 도움말은 `bash run.sh --help`로 볼 수 있습니다.

```bash
bash run.sh [model] [dataset] [save_dir] [epochs] [batch_size] [device] [parallel] [nproc_per_node] [extra main.py args...]
bash run.sh resnet50 imagenet ./checkpoints 90 64 cuda ddp 4 --data-root /data/imagenet --amp
```

**체크포인트.** 학습 중에는 에폭마다 `--save-dir`(기본값 `./checkpoints`)에 `<model>_<dataset>_last.pth`를 쓰고, 검증 정확도가 오를 때마다 `<model>_<dataset>_best.pth`를 씁니다. 각 파일에는 가중치, EMA 가중치(`--ema`일 때), 옵티마이저·스케줄러·AMP 스케일러 상태, 에폭, 최고 정확도, 옵션이 들어 있습니다. `--checkpoint-path`만 주면 가중치만 불러옵니다. 일반 state dict와 DP/DDP의 `module.` 접두사가 붙은 가중치도 불러올 수 있습니다. `--resume`을 함께 주면 옵티마이저, 스케줄러, 스케일러, 에폭, 최고 정확도까지 복원하고 `--epochs`까지 이어서 학습합니다. 예: `python main.py --model resnet18 --epochs 50 --checkpoint-path checkpoints/resnet18_cifar10_last.pth --resume`. `--epochs`를 바꾸면 남은 스케줄도 그에 맞게 바뀝니다. EMA 상태는 `--resume` 여부와 관계없이 `--ema`를 켜면 복원하고, 체크포인트에 EMA 상태가 없으면 불러온 가중치에서 EMA를 시작합니다. eval 모드에서 `--ema`를 주면 EMA 가중치로 평가합니다.

**Python.** `img_size`에 따라 stem이 정해집니다(모델 절 참고).

```python
from models import create_model
model = create_model("resnet18", num_classes=10, img_size=32)
```

## 모델

| 카테고리 | 직접 구현: `models/<category>/`의 67개 | torchvision 래핑: `models/torchvision_zoo.py`의 69개 |
|---|---|---|
| `cnn`: 18개 계열, 54개 모델 | ConvMixer, DLA, DPN, PreAct ResNet, PyramidNet, RepVGG, Res2Net, Xception | AlexNet, VGG, GoogLeNet, Inception v3, ResNet, ResNeXt, Wide ResNet, RegNet, ConvNeXt, DenseNet |
| `lightweight`: 17개 계열, 41개 모델 | MobileNetV1, ShuffleNetV1, GhostNet, MobileOne, SENet, CBAM, ECA-Net, SKNet, ResNeSt, Coordinate Attention | MobileNetV2, MobileNetV3, ShuffleNetV2, SqueezeNet, MnasNet, EfficientNet, EfficientNetV2 |
| `transformer`: 18개 계열, 41개 모델 | ViT (`vit_tiny`, `vit_small`), DeiT, CaiT, CCT, CvT, CoAtNet, BoTNet, PVTv2, MobileViT, MobileViTv2, PoolFormer, EfficientFormer, MLP-Mixer, ResMLP, gMLP | ViT (`vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32`), Swin Transformer, Swin Transformer V2, MaxViT |

- torchvision 빌더는 `weights=None`으로 호출하므로 모든 모델을 처음부터 학습합니다. `mobilevit`는 이 프로젝트용으로 작게 만든 변형이며 논문의 설정과 다릅니다. `exmobilevit`는 같은 `mobilevit` 백본에 ExShortcut(MobileViT 블록 입출력을 1×1 합성곱과 global average pooling을 거쳐 분류기 입력에 이어 붙임)만 더한 모델로, CIFAR-10에서 파라미터가 4.4% 늘어납니다. RepVGG와 MobileOne은 여러 갈래의 블록으로 학습하고, `models.reparameterize_model(model)`로 추론용 단일 합성곱으로 합칠 수 있습니다. `main.py`는 이 함수를 호출하지 않습니다.
- 이름은 대소문자, 공백, `_`, `-`, `.`을 무시하므로 `ResNet-18`로도 `resnet18`을 찾습니다. `mobilenet`은 `mobilenet_v2`, `senet`은 `se_resnet18`의 별칭입니다. 없는 이름을 주면 비슷한 이름을 최대 세 개까지 보여 주고 오류를 냅니다.
- 작은 입력: 64px 이하에서는 32px 이미지가 마지막 단계 전에 1×1로 줄어들지 않도록 대부분의 모델이 작은 이미지용 설정으로 바뀝니다. 첫 합성곱을 stride 1로 바꾸고 stem의 max-pooling을 빼며, ConvNeXt, Swin, ConvMixer는 2×2 패치를, VGG는 폭 512의 분류기를 씁니다. ViT, DeiT, CaiT, MLP-Mixer, ResMLP, gMLP는 128px 미만에서 4×4 패치를, 그 이상에서 16×16 패치를 씁니다. torchvision ViT는 64px 이하에서 16/32 대신 4/8 패치를 씁니다. `mobilevit`, `exmobilevit`, CCT, AlexNet, Inception v3, MaxViT는 구조를 바꾸지 않습니다.
- 입력 크기: `--resize`를 주지 않으면 데이터셋의 기본 크기를 모델의 최소 크기까지 올리고 필요한 배수로 올림한 뒤, 이미지를 그 크기로 리사이즈합니다. CIFAR-10에서 `alexnet`은 64px, `inception_v3`는 75px, `maxvit_t`는 224px로 돌아갑니다. 모델이 받을 수 없는 `--resize` 값을 주면 오류가 납니다.

## 데이터셋

| `--dataset` | 클래스 | 기본 크기 | 데이터 |
|---|---|---|---|
| `cifar10`, `cifar100` | 10, 100 | 32 | `--download-data` |
| `stl10` | 10 | 96 | `--download-data` |
| `food101`, `flowers102`, `oxfordpets` | 101, 102, 37 | 224 | `--download-data` |
| `imagenet` | 1000 | 224 | `--data-root`에 ILSVRC tar 파일과 devkit을 두거나, 압축을 푼 `train/<wnid>/`와 `val/<wnid>/` |
| `imagefolder` | 폴더에서 읽음 | 224 | `<data-root>/train/<class>/*`와 `<data-root>/val/<class>/*` |

데이터는 `--data-root`(기본값 `./data`)에서 읽습니다. 예: `python main.py --model resnet50 --dataset imagefolder --data-root ./my_images`. torchvision 데이터셋은 test 분할로 검증합니다. ImageNet과 `imagefolder`는 `val/`을 쓰고, `imagefolder`는 모든 클래스 폴더가 두 분할에 모두 있어야 합니다. 이름은 대소문자, 공백, `_`, `-`를 무시합니다. `pets`와 `oxford_pets`는 `oxfordpets`, `folder`와 `custom`은 `imagefolder`로 처리합니다.

## 학습 옵션

| 분류 | 플래그(기본값) | 효과 |
|---|---|---|
| 멀티스케일 | `--multi-scale` | 학습 배치마다 해상도를 무작위로 고르고, 메모리가 거의 일정하도록 배치 크기를 (기준 크기/해상도)²배로 조절합니다. 검증은 `--resize`에서 합니다. 학습 전에 가장 작은 크기와 가장 큰 크기로 모델을 한 번씩 돌려 봅니다. |
| | `--ms-min-scale 0.625`, `--ms-max-scale 1.25`, `--ms-num-scales 5`, `--ms-sizes`, `--ms-fixed-batch` | 해상도는 `--resize`에 대한 비율이며, 8의 배수(64px 미만은 4의 배수)로 반올림하고 모델의 최소 크기 이상으로 맞춥니다. `160,192,224,256,288`처럼 목록을 직접 줄 수도 있습니다. `--ms-fixed-batch`는 배치 크기를 하나로 고정합니다. |
| 증강 | `--auto-augment none`, `--random-erase 0` | 기본 증강(CIFAR와 STL-10은 패딩 크롭·좌우 뒤집기·회전, 나머지는 RandomResizedCrop·좌우 뒤집기) 위에 `rand`, `trivial`, `autoaugment`를 더합니다. `--random-erase`는 Random Erasing 확률입니다. |
| | `--mixup 0`, `--cutmix 0`, `--mix-prob 1.0`, `--mix-switch-prob 0.5` | Mixup과 CutMix의 alpha(0이면 끔), 배치를 섞을 확률, 둘 다 켰을 때 CutMix를 고를 확률입니다. |
| 정규화 | `--label-smoothing 0`, `--ema`, `--ema-decay 0.9998` | 교차 엔트로피의 라벨 스무딩입니다. 가중치 EMA는 검증과 `*_best.pth` 선택에 쓰입니다. |
| 최적화 | `--optimizer sgd`, `--lr 0.01`, `--momentum 0.9`, `--weight-decay 1e-4`, `--grad-clip` | `sgd`(momentum > 0이면 Nesterov) 또는 `adamw`입니다. bias, 정규화 가중치, 위치·클래스 토큰에는 weight decay를 주지 않습니다. `--grad-clip`은 그래디언트 노름을 잘라 냅니다. |
| | `--epochs 10`, `--scheduler cosine`, `--warmup-epochs 0`, `--min-lr 0`, `--step-size 30`, `--gamma 0.1` | 에폭 단위 스케줄입니다. 선형 워밍업 뒤 `cosine`(`--min-lr`까지 감소), `step`(`--step-size` 에폭마다 × `--gamma`), `none`(고정) 중 하나를 씁니다. |
| 하드웨어 | `--device auto`, `--parallel none`, `--ddp`, `--amp` | 장치는 `auto`, `cpu`, `cuda`, `cuda:N`, 병렬 방식은 `none`, `dp`, `ddp`입니다. `--ddp`는 `--parallel ddp`와 같습니다. `--amp`는 float16 autocast와 GradScaler를 쓰며 CUDA 전용입니다. |
| | `--gpu-ids`, `--sync-bn`, `--backend`, `--master-addr 127.0.0.1`, `--master-port 29500` | dp에서 쓸 GPU, ddp의 SyncBatchNorm·백엔드·랑데부 주소입니다. |
| 체크포인트 | `--save-dir ./checkpoints`, `--checkpoint-path`, `--resume` | 사용법 절 참고. |
| 샘플링 | `--batch-size 64`, `--num-workers 4`, `--sampler default` | 학습과 검증의 배치 크기, 데이터 로딩 워커 수입니다. `weighted`는 클래스가 고르게 나오도록 복원 추출하며, `--multi-scale`이 꺼져 있으면 ddp에서 무시됩니다. |
| | `--seed 42`, `--limit-batches` | 시드입니다(ddp에서는 랭크를 더함). `--limit-batches N`은 디버깅용으로 에폭마다 처음 N개 배치만 씁니다. |

저장소의 CIFAR-10 사본으로 돌리는 MobileViT식 레시피:

```bash
python main.py --model mobilevit --dataset cifar10 --epochs 100 --batch-size 128 \
  --multi-scale --ema --mixup 0.2 --cutmix 1.0 --auto-augment rand --label-smoothing 0.1 \
  --optimizer adamw --lr 2e-3 --weight-decay 0.01 --warmup-epochs 5 --amp
```

- `--multi-scale`: 배치를 32, 36, 40px로 뽑고, 배치 크기는 각각 128, 101, 82장입니다. 더 작은 기본 스케일은 모델의 최소 크기인 32px로 올라갑니다. 검증은 32px에서 합니다.
- `--mixup 0.2 --cutmix 1.0 --auto-augment rand`: 모든 배치를 Mixup과 CutMix 중 무작위로 고른 쪽으로 섞고, 크롭·뒤집기·회전 위에 RandAugment를 더합니다.
- `--label-smoothing 0.1 --ema`: 타깃을 부드럽게 하고, 검증과 `*_best.pth`에는 평균 가중치를 씁니다.
- `--optimizer adamw --lr 2e-3 --weight-decay 0.01 --warmup-epochs 5 --amp`: AdamW로 5에폭 워밍업한 뒤 기본값인 cosine으로 줄이고, CUDA에서는 혼합 정밀도로 학습합니다. CPU에서는 안내 메시지를 남기고 `--amp`를 끕니다.

## 멀티 GPU

```bash
python main.py --model resnet50 --parallel dp --gpu-ids 0,1   # 프로세스 하나, nn.DataParallel
torchrun --nproc_per_node=2 main.py --model resnet50          # DDP, GPU마다 프로세스 하나
python launch_ddp.py --nproc 2 main.py --model resnet50       # torchrun 없이 DDP
```

- `--batch-size`는 장치 하나일 때 스텝당 배치입니다. `dp`에서는 여러 GPU에 나눠지는 전체 배치이고, ddp에서는 프로세스당 배치라서 전체 배치는 프로세스 수 × `--batch-size`입니다. `--multi-scale`에서는 `--resize` 해상도 기준의 배치이고, 모든 랭크가 같은 해상도 순서를 씁니다.
- `dp`는 CUDA 장치가 두 개 이상 필요하며, 아니면 장치 하나로 돌아갑니다. `--gpu-ids`의 기본값은 보이는 모든 GPU입니다. torchrun으로 실행하면 ddp가 자동으로 선택되고, 각 랭크는 자기 로컬 랭크의 GPU를 씁니다(GPU는 `CUDA_VISIBLE_DEVICES`로 고릅니다). `--device cpu`이거나 CUDA가 없으면 NCCL 대신 gloo로 CPU에서 돌아갑니다. 로그 출력과 저장은 랭크 0만 하고, 검증 데이터는 패딩 없이 랭크들에 나눕니다.
- `launch_ddp.py`는 `RANK`, `WORLD_SIZE`, `LOCAL_RANK`, `MASTER_ADDR`, `MASTER_PORT`를 설정해 프로세스를 직접 띄웁니다. libuv 없이 빌드된 Windows 버전처럼 `torchrun`이 없거나 동작하지 않는 환경에서 쓰세요.
- 하드웨어 플래그와 `utils/hwaccel.py`는 [Mutli-Tool4RL](https://github.com/MZUI3/Mutli-Tool4RL)과 같습니다.

## 프로젝트 구조

```
├── main.py              # 진입점: list / train / eval / summary / print
├── trainer.py           # train_one_epoch, validate
├── launch_ddp.py        # torchrun 대체 런처
├── run.sh               # main.py를 위치 인자로 감싼 스크립트
├── options/parser.py    # 명령줄 플래그
├── dataset/             # info.py(이름, 클래스, 크기), dataset.py, transform.py, multiscale.py, sampler.py, data_loaders.py
├── models/              # registry.py, torchvision_zoo.py, layers.py, resnet_base.py
│   └── cnn/ lightweight/ transformer/   # 직접 구현한 모델, 대부분 파일 하나에 계열 하나
├── utils/               # hwaccel.py(장치, DP/DDP, AMP), checkpoint.py, optim.py, ema.py, mixup.py
├── data/cifar-10-batches-py/               # 저장소에 포함된 CIFAR-10 사본
└── checkpoints/mobilevit_cifar10_best.pth  # 저장소에 포함된 체크포인트
```

## 모델 추가

카테고리 폴더(예: `models/cnn/`)에 파일을 만듭니다. `num_classes`와 `img_size`를 받는 빌더에 데코레이터를 붙이고, 그 폴더 `__init__.py`의 import 목록에 새 모듈을 추가합니다.

```python
import torch.nn as nn

from ..layers import ConvBNAct, is_small_input
from ..registry import register_model


@register_model(category="cnn", family="MyNet", paper="Author et al., Year")
def mynet(num_classes=1000, img_size=224):
    stride = 1 if is_small_input(img_size) else 2
    return nn.Sequential(ConvBNAct(3, 32, 3, stride), nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(32, num_classes))
```

이름은 기본적으로 함수 이름을 쓰고, `category`는 `cnn`, `lightweight`, `transformer` 중 하나여야 합니다. 선택 인자 `min_input_size`(기본값 32), `input_size_divisor`, `aliases`는 입력 크기 규칙과 이름 검색에 쓰입니다. 정규화한 이름이 기존 이름과 겹치면 오류가 납니다.
