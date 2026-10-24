# Multi-Tool4Vision

이미지 분류 모델을 한곳에 모아 놓고 `--model 이름`만 바꿔가며 학습할 수 있게 만든 PyTorch 코드입니다.
개인 GPU(16GB 이하)에서 돌리는 걸 기준으로 잡았고, 논문에 나오는 모델 상당수는 직접 구현했습니다.
현재 53개 계열 136개 모델이 등록되어 있습니다.

```bash
pip install -r requirements.txt
python main.py --mode list                                                  # 모델 목록
python main.py --model mobilevit --dataset cifar10 --download-data --epochs 20
```

## 할 수 있는 것

- 모델 136개를 같은 CLI로 학습 / 평가 / 구조 출력 (`--mode train|eval|print|summary|list`)
- 데이터셋: CIFAR-10/100, STL-10, ImageNet, Food-101, Flowers-102, Oxford Pets, 그리고 폴더 구조만 맞추면 되는 `imagefolder`
- 학습 기법: MobileViT 논문의 멀티스케일 샘플러, EMA, Mixup/CutMix, RandAugment/TrivialAugment/AutoAugment, Random Erasing, label smoothing, warmup + cosine/step
- 멀티 GPU: `nn.DataParallel`, `DistributedDataParallel`(torchrun 또는 `launch_ddp.py`), SyncBN, AMP
- 체크포인트 저장/재개 (`--resume`로 optimizer, scheduler, EMA까지 복원)
- 외부 모델 라이브러리(timm 등) 없이 torch + torchvision만 씁니다.

## 설치

Python 3.8+, PyTorch 2.3+, torchvision 0.18+. `--mode summary`를 쓰려면 torchinfo가 추가로 필요합니다.

```bash
pip install -r requirements.txt
```

GPU가 없어도 전부 돌아갑니다 (`--device cpu`). DDP도 CPU + gloo로 테스트할 수 있습니다.

## 사용법

기본 학습:

```bash
python main.py --model resnet18 --dataset cifar10 --download-data --epochs 20
```

Transformer 계열은 AdamW + warmup + AMP 조합이 잘 맞습니다:

```bash
python main.py --model deit_tiny --dataset cifar10 --download-data --epochs 100 \
  --optimizer adamw --lr 1e-3 --weight-decay 0.05 --warmup-epochs 5 --label-smoothing 0.1 --amp
```

MobileViT 논문 레시피에 가깝게 (멀티스케일 + EMA + Mixup/CutMix + RandAugment):

```bash
python main.py --model mobilevit --dataset cifar10 --download-data --epochs 100 --batch-size 128 \
  --multi-scale --ema --mixup 0.2 --cutmix 1.0 --auto-augment rand --random-erase 0.25 --label-smoothing 0.1 \
  --optimizer adamw --lr 2e-3 --weight-decay 0.01 --warmup-epochs 5 --amp
```

모델 구조만 확인 (데이터셋을 읽지 않음):

```bash
python main.py --model mobilevit --dataset cifar10 --mode print
python main.py --model mobilevit --dataset cifar10 --mode summary --resize 224
```

체크포인트는 `<save-dir>/<model>_<dataset>_best.pth`, `..._last.pth`로 저장됩니다.

```bash
# 가중치만 불러와서 이어서 학습 / fine-tune
python main.py --model mobilevit --dataset cifar10 --checkpoint-path ./checkpoints/mobilevit_cifar10_best.pth
# 중단된 학습 재개 (optimizer, scheduler, epoch, EMA 복원)
python main.py --model mobilevit --dataset cifar10 --epochs 20 --checkpoint-path ./checkpoints/mobilevit_cifar10_last.pth --resume
# 평가만
python main.py --model mobilevit --dataset cifar10 --mode eval --checkpoint-path ./checkpoints/mobilevit_cifar10_best.pth
```

ImageNet이나 자기 이미지 폴더:

```bash
python main.py --model resnet50 --dataset imagenet --data-root /data/imagenet --epochs 90 --batch-size 256 --lr 0.1 --amp
python main.py --model efficientnet_b0 --dataset imagefolder --data-root ./my_images --epochs 30 --amp
```

`run.sh`로도 실행할 수 있습니다. 인자 순서는 `model dataset save_dir epochs batch_size device parallel nproc_per_node`이고, 그 뒤는 `main.py`에 그대로 넘어갑니다.

```bash
./run.sh mobilevit cifar10 ./checkpoints 20 64 cpu
./run.sh resnet50 imagenet ./checkpoints 90 64 cuda ddp 4 --data-root /data/imagenet --amp
```

## 지원 모델

🔧 직접 구현, 📦 torchvision 구현 사용. 전부 `weights=None`으로 처음부터 학습합니다.
모델 이름은 대소문자, `_`, `-`를 구분하지 않고(`ResNet-18` = `resnet18`), 오타를 내면 비슷한 이름을 알려줍니다.

### CNN (`cnn`)

| 계열 | 논문 | 모델 이름 |
|---|---|---|
| ResNet 📦 | He et al., 2016 | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` |
| DenseNet 📦 | Huang et al., 2017 | `densenet121`, `densenet169`, `densenet201` |
| AlexNet 📦 | Krizhevsky et al., 2012 | `alexnet` |
| VGG 📦 | Simonyan & Zisserman, 2014 | `vgg11`, `vgg13`, `vgg16`, `vgg19` (+ `_bn` 버전) |
| GoogLeNet 📦 | Szegedy et al., 2015 | `googlenet` |
| Inception v3 📦 | Szegedy et al., 2016 | `inception_v3` |
| PreAct ResNet 🔧 | He et al., 2016 | `preact_resnet18`, `preact_resnet34`, `preact_resnet50` |
| ResNeXt 📦 | Xie et al., 2017 | `resnext50_32x4d`, `resnext101_32x8d`, `resnext101_64x4d` |
| Wide ResNet 📦 | Zagoruyko & Komodakis, 2016 | `wide_resnet50_2`, `wide_resnet101_2` |
| RegNet 📦 | Radosavovic et al., 2020 | `regnet_x_400mf`, `regnet_x_800mf`, `regnet_x_1_6gf`, `regnet_x_3_2gf`, `regnet_y_400mf`, `regnet_y_800mf`, `regnet_y_1_6gf`, `regnet_y_3_2gf`, `regnet_y_8gf` |
| ConvNeXt 📦 | Liu et al., 2022 | `convnext_tiny`, `convnext_small`, `convnext_base`, `convnext_large` |
| Res2Net 🔧 | Gao et al., 2019 | `res2net50_26w_4s`, `res2net101_26w_4s` |
| DPN 🔧 | Chen et al., 2017 | `dpn26`, `dpn92` |
| PyramidNet 🔧 | Han et al., 2017 | `pyramidnet110_a48`, `pyramidnet164_a270` |
| RepVGG 🔧 | Ding et al., 2021 | `repvgg_a0`, `repvgg_a1`, `repvgg_b0` |
| DLA 🔧 | Yu et al., 2018 | `dla34` |
| Xception 🔧 | Chollet, 2017 | `xception` |
| ConvMixer 🔧 | Trockman & Kolter, 2022 | `convmixer_256_8`, `convmixer_512_16`, `convmixer_768_32` |

### 경량 모델 / 채널 어텐션 (`lightweight`)

| 계열 | 논문 | 모델 이름 |
|---|---|---|
| MobileNetV2 📦 | Sandler et al., 2018 | `mobilenet_v2` (별칭 `mobilenet`) |
| SENet 🔧 | Hu et al., 2018 | `se_resnet18` (별칭 `senet`), `se_resnet50` |
| MobileNetV1 🔧 | Howard et al., 2017 | `mobilenet_v1`, `mobilenet_v1_0_5` |
| MobileNetV3 📦 | Howard et al., 2019 | `mobilenet_v3_small`, `mobilenet_v3_large` |
| ShuffleNetV1 🔧 | Zhang et al., 2018 | `shufflenet_v1_g3`, `shufflenet_v1_g8` |
| ShuffleNetV2 📦 | Ma et al., 2018 | `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x1_5`, `shufflenet_v2_x2_0` |
| SqueezeNet 📦 | Iandola et al., 2016 | `squeezenet1_0`, `squeezenet1_1` |
| MnasNet 📦 | Tan et al., 2019 | `mnasnet0_5`, `mnasnet0_75`, `mnasnet1_0`, `mnasnet1_3` |
| EfficientNet 📦 | Tan & Le, 2019 | `efficientnet_b0` ~ `efficientnet_b4` |
| EfficientNetV2 📦 | Tan & Le, 2021 | `efficientnet_v2_s`, `efficientnet_v2_m`, `efficientnet_v2_l` |
| GhostNet 🔧 | Han et al., 2020 | `ghostnet_1_0`, `ghostnet_1_3` |
| CBAM 🔧 | Woo et al., 2018 | `cbam_resnet18`, `cbam_resnet50` |
| ECA-Net 🔧 | Wang et al., 2020 | `eca_resnet18`, `eca_resnet50` |
| SKNet 🔧 | Li et al., 2019 | `sknet50`, `sknet101` |
| ResNeSt 🔧 | Zhang et al., 2020 | `resnest50`, `resnest101` |
| MobileOne 🔧 | Vasu et al., 2023 | `mobileone_s0`, `mobileone_s1`, `mobileone_s4` |
| Coordinate Attention 🔧 | Hou et al., 2021 | `ca_mobilenet_v2` |

### Transformer / MLP (`transformer`)

| 계열 | 논문 | 모델 이름 |
|---|---|---|
| MobileViT 🔧 | Mehta & Rastegari, 2022 | `mobilevit`, `exmobilevit` |
| ViT 🔧📦 | Dosovitskiy et al., 2021 | `vit_tiny`, `vit_small` 🔧 / `vit_b_16`, `vit_b_32`, `vit_l_16`, `vit_l_32` 📦 |
| DeiT 🔧 | Touvron et al., 2021 | `deit_tiny`, `deit_small` |
| Swin Transformer 📦 | Liu et al., 2021 | `swin_t`, `swin_s`, `swin_b` |
| Swin Transformer V2 📦 | Liu et al., 2022 | `swin_v2_t`, `swin_v2_s`, `swin_v2_b` |
| MaxViT 📦 | Tu et al., 2022 | `maxvit_t` (입력 224의 배수만 가능) |
| CaiT 🔧 | Touvron et al., 2021 | `cait_xxs24`, `cait_xs24` |
| PVTv2 🔧 | Wang et al., 2022 | `pvt_v2_b0`, `pvt_v2_b1`, `pvt_v2_b2` |
| CvT 🔧 | Wu et al., 2021 | `cvt_13`, `cvt_21` |
| PoolFormer 🔧 | Yu et al., 2022 | `poolformer_s12`, `poolformer_s24` |
| EfficientFormer 🔧 | Li et al., 2022 | `efficientformer_l1`, `efficientformer_l3` |
| MLP-Mixer 🔧 | Tolstikhin et al., 2021 | `mixer_s`, `mixer_b` |
| ResMLP 🔧 | Touvron et al., 2021 | `resmlp_12`, `resmlp_24` |
| gMLP 🔧 | Liu et al., 2021 | `gmlp_ti`, `gmlp_s` |
| CoAtNet 🔧 | Dai et al., 2021 | `coatnet_0`, `coatnet_1` |
| MobileViTv2 🔧 | Mehta & Rastegari, 2022 | `mobilevitv2_050`, `mobilevitv2_100` |
| CCT 🔧 | Hassani et al., 2021 | `cct_7_3x1` (CIFAR용), `cct_14_7x2` (ImageNet용) |
| BoTNet 🔧 | Srinivas et al., 2021 | `botnet50` |

### 작은 입력 처리

CIFAR처럼 32px 입력을 ImageNet용 stem에 그대로 넣으면 마지막 stage 전에 특징맵이 1x1이 되어버립니다.
그래서 64px 이하 입력에서는 직접 구현한 모델이든 torchvision 모델이든 stem을 바꿉니다.

- ResNet/ResNeXt/WRN/DenseNet/GoogLeNet: 3x3 stride-1 stem, 첫 풀링 제거
- MobileNet/EfficientNet/MnasNet/ShuffleNet/RegNet/SqueezeNet: 첫 conv stride 1
- ConvNeXt/Swin: 2x2 패치 stem
- VGG: 4096짜리 분류기 대신 512 → 클래스
- ViT/DeiT/CaiT/MLP-Mixer 계열: 패치 크기 16 → 4 (torchvision ViT-B는 4/8)

128px 이상이면 원래 구조 그대로입니다. `--resize`를 안 주면 데이터셋 기본 해상도를 쓰고, 모델이 더 큰 입력을 요구하면(`inception_v3` 75, `alexnet` 64, `maxvit_t` 224) 자동으로 키웁니다. 직접 준 `--resize`가 모델 조건에 안 맞으면 바로 에러가 납니다.

### 구현 메모

- DeiT는 teacher 없이 학습하므로 distillation head도 정답 라벨로 학습하고, 두 head 평균을 예측으로 씁니다.
- RepVGG / MobileOne은 학습 후 `models.reparameterize_model(model)`로 추론용 단일 conv 구조로 바꿀 수 있습니다.
- 원 논문과 다르게 단순화한 부분: EfficientFormer는 attention bias 대신 일반 MHSA, CoAtNet은 MBConv/Transformer 블록의 정규화 위치, BoTNet은 상대 위치 대신 학습형 2D 위치 임베딩.

## 데이터셋

| 이름 | 클래스 | 기본 해상도 | 증강 | 준비 |
|---|---|---|---|---|
| `cifar10`, `cifar100` | 10 / 100 | 32 | pad-crop | `--download-data` |
| `stl10` | 10 | 96 | pad-crop | `--download-data` |
| `imagenet` | 1000 | 224 | RandomResizedCrop | `<data-root>`에 ILSVRC tar + devkit, 또는 압축 푼 `train/<wnid>/`, `val/<wnid>/` |
| `food101`, `flowers102`, `oxford_pets` | 101 / 102 / 37 | 224 | RandomResizedCrop | `--download-data` |
| `imagefolder` (별칭 `folder`) | 폴더 수로 결정 | 224 | RandomResizedCrop | `<data-root>/train/<class>/*.jpg`, `<data-root>/val/<class>/*.jpg` |

32px, 96px처럼 이미 잘려 있는 작은 이미지는 pad-crop + flip + 회전을, 크기가 제각각인 큰 이미지는 RandomResizedCrop(학습) / Resize + CenterCrop(검증)을 씁니다. 정규화 mean/std도 데이터셋마다 다릅니다.

클래스 불균형이 심하면 `--sampler weighted`로 클래스별 가중 샘플링을 할 수 있습니다. DDP에서는 `DistributedSampler`로 대체되고 경고만 나옵니다.

## 학습 기법

| 옵션 | 내용 |
|---|---|
| `--multi-scale` | MobileViT의 멀티스케일 샘플러. 배치마다 해상도를 무작위로 고르고 배치 크기를 `(기본 해상도/해상도)^2`배로 바꿔서 GPU 메모리를 일정하게 유지합니다. 해상도 집합은 기본 `--resize`의 0.625~1.25배 5단계 (`--ms-min-scale`, `--ms-max-scale`, `--ms-num-scales`, 또는 `--ms-sizes 160,192,224,256,288`). `--ms-fixed-batch`면 배치 크기 고정. |
| `--ema`, `--ema-decay` | 가중치 EMA. 검증과 `*_best.pth`는 EMA 가중치를 쓰고, `--mode eval --ema`는 EMA 가중치를 불러옵니다. |
| `--mixup`, `--cutmix` | Mixup / CutMix alpha. `--mix-prob`, `--mix-switch-prob`로 적용 확률과 둘 사이 전환 확률을 조절합니다. 정확도는 원래 라벨 기준으로 잽니다. |
| `--auto-augment` | `rand` (RandAugment), `trivial` (TrivialAugmentWide), `autoaugment` (CIFAR10 / ImageNet 정책) |
| `--random-erase` | Random Erasing 확률 |
| `--label-smoothing` | label smoothing |
| `--limit-batches N` | 디버그용. epoch당 N 배치만 돕니다. |

멀티스케일은 해상도가 바뀌어도 되는 모델에서만 됩니다. positional embedding 크기가 고정인 모델(ViT, DeiT, CaiT, MLP-Mixer/ResMLP/gMLP, CCT, CoAtNet, Swin, MaxViT)은 학습 전에 최소/최대 해상도로 forward를 한 번 돌려보고 안 되면 에러를 냅니다.
DDP에서는 모든 rank가 같은 해상도 순서와 같은 배치 수를 쓰도록 샘플러를 짰습니다.

## 멀티 GPU

`utils/hwaccel.py` 한 파일이 디바이스 선택, DP, DDP, AMP를 담당합니다. 이 파일은 RL 쪽 저장소([Mutli-Tool4RL](https://github.com/MZUI3/Mutli-Tool4RL))의 `rlzoo/common/hwaccel.py`와 같은 파일이라 양쪽 옵션이 같습니다.

| 모드 | 실행 | `--batch-size` 의미 |
|---|---|---|
| `none` | `python main.py` | 전체 배치 |
| `dp` | `python main.py --parallel dp [--gpu-ids 0,1]` | 전체 배치를 GPU 수로 나눔 |
| `ddp` | `torchrun --nproc_per_node=N main.py` 또는 `python launch_ddp.py --nproc N main.py` | 프로세스당 배치 |

```bash
python main.py --model resnet50 --dataset cifar100 --download-data --parallel dp --gpu-ids 0,1 --batch-size 256 --amp
torchrun --nproc_per_node=2 main.py --model mobilevit --dataset cifar10 --download-data --epochs 20 --batch-size 32
python launch_ddp.py --nproc 2 main.py --model mobilevit --dataset cifar10 --download-data --epochs 20 --batch-size 32
python launch_ddp.py --nproc 2 main.py --model resnet18 --dataset cifar10 --device cpu --epochs 1   # CPU에서 DDP 동작 확인
```

- torchrun으로 띄우면 `--parallel ddp`는 자동으로 켜집니다.
- `launch_ddp.py`는 torchrun 대신 쓰는 런처입니다. Windows용 PyTorch 휠이 libuv 없이 빌드되어 torchrun의 rendezvous가 실패하는 경우가 있어서 넣었습니다. RANK/WORLD_SIZE 등 같은 환경 변수를 설정해 N개 프로세스를 띄울 뿐입니다.
- DP는 CUDA GPU가 2장 미만이면 그냥 `none`으로 돌아갑니다.
- DDP 검증은 패딩 없이 rank별로 나눠 돌리고 손실과 정답 수를 all-reduce하므로 단일 GPU와 같은 값이 나옵니다.
- 체크포인트는 rank 0만, DP/DDP 래퍼를 벗긴 상태로 저장합니다.
- `--sync-bn`은 DDP + CUDA에서만 의미가 있습니다. `--amp`는 CUDA에서만 켜지고 CPU에서는 경고 후 무시됩니다.

## 옵션 정리

- `--optimizer`: `sgd` (기본, nesterov) 또는 `adamw`. bias, norm, 토큰 파라미터에는 weight decay를 안 겁니다.
- `--lr`, `--momentum`, `--weight-decay`
- `--scheduler`: `cosine` (기본), `step`, `none`. `--warmup-epochs`, `--min-lr`, `--step-size`, `--gamma`
- `--grad-clip`: gradient norm clipping
- `--amp`: mixed precision (CUDA 전용)
- `--num-workers`, `--resize`, `--seed`, `--save-dir`
- `--device`, `--parallel`, `--gpu-ids`, `--sync-bn`, `--backend`, `--master-addr`, `--master-port`

## 코드 구조

```
main.py                 진입점 (list / print / summary / train / eval)
trainer.py              train_one_epoch, validate
launch_ddp.py           torchrun 대체 런처
run.sh                  쉘 래퍼
options/parser.py       CLI 옵션
dataset/
  info.py               데이터셋 메타데이터 (클래스 수, 해상도, mean/std, 증강 종류)
  dataset.py            torchvision 데이터셋 로더
  multiscale.py         멀티스케일 배치 샘플러, 해상도별 transform 래퍼
  transform.py, sampler.py, data_loaders.py
models/
  registry.py           @register_model, create_model, list_models, 입력 크기 검증
  layers.py             ConvBNAct, SqueezeExcite, DropPath, LayerScale, 재파라미터화 등 공용 블록
  resnet_base.py        SE/CBAM/ECA/SK/ResNeSt/Res2Net/BoTNet이 공유하는 ResNet 골격
  torchvision_zoo.py    torchvision 모델 등록 + 작은 입력용 stem 교체
  cnn/ lightweight/ transformer/   직접 구현한 모델
utils/
  hwaccel.py            디바이스 / DP / DDP / AMP (RL 저장소와 동일 파일)
  ema.py, mixup.py, optim.py, checkpoint.py
```

새 모델은 `models/<카테고리>/`에 파일을 만들고 `__init__.py`에서 import하면 등록됩니다.

```python
from ..registry import register_model

@register_model(category="cnn", family="MyNet", paper="Me et al., 2026", min_input_size=32)
def mynet_s(num_classes=1000, img_size=224):
    return MyNet(num_classes=num_classes, img_size=img_size)
```
