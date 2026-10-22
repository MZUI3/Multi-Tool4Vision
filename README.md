## 이 프로젝트는 개인용 그래픽 카드(<16GB) 기반으로 모델 학습을 위한 종합 허브입니다

모든 모델은 하나의 레지스트리에 등록되어 있어 `--model <이름>` 하나로 학습 · 평가 · 요약을 할 수 있습니다.
전체 목록은 다음 명령으로 확인할 수 있습니다.

```bash
python main.py --mode list                         # 전체
python main.py --mode list --category transformer  # 카테고리별 (cnn, lightweight, transformer)
```

### 지원 모델 리스트 (50 계열 / 109 모델)

표기: 🔧 = 이 저장소에서 직접 구현, 📦 = torchvision 구현 사용 (모두 `weights=None`, 처음부터 학습)

*CNN & Residual based* (`cnn`)

| 계열 | 논문 | 모델 이름 |
|---|---|---|
| ResNet 📦 | He et al., 2016 | `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152` |
| DenseNet 📦 | Huang et al., 2017 | `densenet121`, `densenet169`, `densenet201` |
| AlexNet 📦 | Krizhevsky et al., 2012 | `alexnet` |
| VGG 📦 | Simonyan & Zisserman, 2014 | `vgg11_bn`, `vgg13_bn`, `vgg16`, `vgg16_bn`, `vgg19_bn` |
| GoogLeNet 📦 | Szegedy et al., 2015 | `googlenet` |
| Inception v3 📦 | Szegedy et al., 2016 | `inception_v3` |
| PreAct ResNet 🔧 | He et al., 2016 | `preact_resnet18`, `preact_resnet34`, `preact_resnet50` |
| ResNeXt 📦 | Xie et al., 2017 | `resnext50_32x4d`, `resnext101_32x8d` |
| Wide ResNet 📦 | Zagoruyko & Komodakis, 2016 | `wide_resnet50_2`, `wide_resnet101_2` |
| RegNet 📦 | Radosavovic et al., 2020 | `regnet_x_400mf`, `regnet_x_800mf`, `regnet_y_400mf`, `regnet_y_800mf`, `regnet_y_1_6gf` |
| ConvNeXt 📦 | Liu et al., 2022 | `convnext_tiny`, `convnext_small` |
| Res2Net 🔧 | Gao et al., 2019 | `res2net50_26w_4s`, `res2net101_26w_4s` |
| DPN 🔧 | Chen et al., 2017 | `dpn26`, `dpn92` |
| PyramidNet 🔧 | Han et al., 2017 | `pyramidnet110_a48`, `pyramidnet164_a270` |
| RepVGG 🔧 | Ding et al., 2021 | `repvgg_a0`, `repvgg_a1`, `repvgg_b0` |
| DLA 🔧 | Yu et al., 2018 | `dla34` |
| Xception 🔧 | Chollet, 2017 | `xception` |

*Lightweight & Feature map Squeeze (channel attention)* (`lightweight`)

| 계열 | 논문 | 모델 이름 |
|---|---|---|
| MobileNetV2 📦 | Sandler et al., 2018 | `mobilenet_v2` (별칭 `mobilenet`) |
| SENet 🔧 | Hu et al., 2018 | `se_resnet18` (별칭 `senet`), `se_resnet50` |
| MobileNetV1 🔧 | Howard et al., 2017 | `mobilenet_v1`, `mobilenet_v1_0_5` |
| MobileNetV3 📦 | Howard et al., 2019 | `mobilenet_v3_small`, `mobilenet_v3_large` |
| ShuffleNetV1 🔧 | Zhang et al., 2018 | `shufflenet_v1_g3`, `shufflenet_v1_g8` |
| ShuffleNetV2 📦 | Ma et al., 2018 | `shufflenet_v2_x0_5`, `shufflenet_v2_x1_0`, `shufflenet_v2_x2_0` |
| SqueezeNet 📦 | Iandola et al., 2016 | `squeezenet1_0`, `squeezenet1_1` |
| MnasNet 📦 | Tan et al., 2019 | `mnasnet0_5`, `mnasnet1_0` |
| EfficientNet 📦 | Tan & Le, 2019 | `efficientnet_b0` ~ `efficientnet_b3` |
| EfficientNetV2 📦 | Tan & Le, 2021 | `efficientnet_v2_s` |
| GhostNet 🔧 | Han et al., 2020 | `ghostnet_1_0`, `ghostnet_1_3` |
| CBAM 🔧 | Woo et al., 2018 | `cbam_resnet18`, `cbam_resnet50` |
| ECA-Net 🔧 | Wang et al., 2020 | `eca_resnet18`, `eca_resnet50` |
| SKNet 🔧 | Li et al., 2019 | `sknet50`, `sknet101` |
| ResNeSt 🔧 | Zhang et al., 2020 | `resnest50`, `resnest101` |
| MobileOne 🔧 | Vasu et al., 2023 | `mobileone_s0`, `mobileone_s1`, `mobileone_s4` |
| Coordinate Attention 🔧 | Hou et al., 2021 | `ca_mobilenet_v2` |

*Transformer & MLP based* (`transformer`)

| 계열 | 논문 | 모델 이름 |
|---|---|---|
| MobileViT 🔧 | Mehta & Rastegari, 2022 | `mobilevit`, `exmobilevit` |
| ViT 🔧📦 | Dosovitskiy et al., 2021 | `vit_tiny`, `vit_small` 🔧 / `vit_b_16`, `vit_b_32` 📦 |
| DeiT 🔧 | Touvron et al., 2021 | `deit_tiny`, `deit_small` |
| Swin Transformer 📦 | Liu et al., 2021 | `swin_t`, `swin_s` |
| Swin Transformer V2 📦 | Liu et al., 2022 | `swin_v2_t`, `swin_v2_s` |
| MaxViT 📦 | Tu et al., 2022 | `maxvit_t` (입력 224의 배수만 가능) |
| CaiT 🔧 | Touvron et al., 2021 | `cait_xxs24`, `cait_xs24` |
| PVTv2 🔧 | Wang et al., 2022 | `pvt_v2_b0`, `pvt_v2_b1`, `pvt_v2_b2` |
| CvT 🔧 | Wu et al., 2021 | `cvt_13`, `cvt_21` |
| PoolFormer 🔧 | Yu et al., 2022 | `poolformer_s12`, `poolformer_s24` |
| EfficientFormer 🔧 | Li et al., 2022 | `efficientformer_l1`, `efficientformer_l3` |
| MLP-Mixer 🔧 | Tolstikhin et al., 2021 | `mixer_s`, `mixer_b` |
| CoAtNet 🔧 | Dai et al., 2021 | `coatnet_0`, `coatnet_1` |
| MobileViTv2 🔧 | Mehta & Rastegari, 2022 | `mobilevitv2_050`, `mobilevitv2_100` |
| CCT 🔧 | Hassani et al., 2021 | `cct_7_3x1` (CIFAR용), `cct_14_7x2` (ImageNet용) |
| BoTNet 🔧 | Srinivas et al., 2021 | `botnet50` |

모델 이름은 대소문자 · `_` · `-`를 구분하지 않습니다 (`ResNet-18` = `resnet18`). 오타를 내면 비슷한 이름을 제안합니다.

#### 구현 메모

- 직접 구현한 CNN은 입력이 64px 이하(CIFAR 등)이면 stride 1 stem을 사용해 특징맵이 너무 일찍 1x1로 줄어들지 않게 합니다.
- ViT / DeiT / CaiT / MLP-Mixer의 패치 크기는 입력이 128px 이상이면 16, 그보다 작으면 4로 자동 선택됩니다.
- DeiT는 teacher 없이 학습하므로 distillation head도 정답 라벨로 학습하고, 두 head의 평균을 예측으로 사용합니다.
- RepVGG / MobileOne은 학습 후 `models.reparameterize_model(model)`로 추론용 단일 conv 구조로 변환할 수 있습니다.
- 일부 구현은 원 논문보다 단순화했습니다: EfficientFormer의 attention bias 대신 일반 MHSA, CoAtNet은 MBConv/Transformer 블록의 세부 정규화 위치, BoTNet은 상대 위치 대신 학습형 2D 위치 임베딩(content-position) 사용.

### 프로젝트 구조

```
main.py                 # 진입점: list / print / summary / train / eval
trainer.py              # train_one_epoch, validate (AMP, grad clip, DDP metric reduce)
options/parser.py       # CLI 옵션
dataset/
  info.py               # 데이터셋 메타데이터(클래스 수, 기본 해상도, mean/std) 단일 관리
  dataset.py, transform.py, sampler.py, data_loaders.py
models/
  registry.py           # @register_model, create_model, list_models, 입력 크기 검증
  layers.py             # ConvBNAct, SqueezeExcite, DropPath, LayerScale, 재파라미터화 도구 등 공용 블록
  resnet_base.py        # SE/CBAM/ECA/SK/ResNeSt/Res2Net/BoTNet이 공유하는 ResNet 골격
  torchvision_zoo.py    # torchvision 모델 등록
  cnn/  lightweight/  transformer/   # 직접 구현한 모델
utils/
  distributed.py, optim.py (optimizer/scheduler), checkpoint.py (저장/불러오기/재개)
```

#### 새 모델 추가하기

`models/<카테고리>/` 아래에 파일을 만들고 `__init__.py`에서 import하면 끝입니다.

```python
from ..registry import register_model

@register_model(category="cnn", family="MyNet", paper="Me et al., 2026", min_input_size=32)
def mynet_s(num_classes=1000, img_size=224):
    return MyNet(num_classes=num_classes, img_size=img_size)
```

### 요구 사항

- Python 3.8 이상
- PyTorch 2.3 이상
- torchvision 0.18 이상
- (선택) torchinfo

설치 예시:

```bash
pip install -r requirements.txt
```

### 사용 방법

기본 실행:

```bash
python main.py --model mobilevit --dataset cifar10 --download-data --epochs 20
```

Transformer 계열은 AdamW + warmup + AMP 조합을 권장합니다:

```bash
python main.py --model deit_tiny --dataset cifar10 --download-data --epochs 100 \
  --optimizer adamw --lr 1e-3 --weight-decay 0.05 --warmup-epochs 5 --label-smoothing 0.1 --amp
```

모델 출력 / 요약 확인 (데이터셋을 불러오지 않습니다):

```bash
python main.py --model mobilevit --dataset cifar10 --mode print
python main.py --model mobilevit --dataset cifar10 --mode summary --resize 224
```

체크포인트 저장 경로 (`<model>_<dataset>_best.pth`, `<model>_<dataset>_last.pth`):

```bash
python main.py --model mobilevit --dataset cifar10 --download-data --epochs 20 --save-dir ./checkpoints
```

체크포인트 불러오기 (가중치만 불러와 이어서 학습 / fine-tune):

```bash
python main.py --model mobilevit --dataset cifar10 --checkpoint-path ./checkpoints/mobilevit_cifar10_best.pth
```

중단된 학습 재개 (optimizer · scheduler · epoch까지 복원):

```bash
python main.py --model mobilevit --dataset cifar10 --epochs 20 --checkpoint-path ./checkpoints/mobilevit_cifar10_last.pth --resume
```

체크포인트 평가만 하기:

```bash
python main.py --model mobilevit --dataset cifar10 --mode eval --checkpoint-path ./checkpoints/mobilevit_cifar10_best.pth
```

쉘 스크립트로 실행:

```bash
./run.sh mobilevit cifar10 ./checkpoints 20 64 cpu
```

`run.sh` 인자는 순서대로 다음과 같습니다:

1. model: 모델 이름 (`python main.py --mode list` 참고)
2. dataset: 데이터셋 이름 (`cifar10`, `cifar100`, `imagenet`)
3. save_dir: 체크포인트 저장 디렉터리
4. epochs: 학습 epoch 수
5. batch_size: 배치 크기
6. device: 실행 디바이스 (`cpu`, `cuda`, `cuda:0`)
7. ddp: `ddp` 또는 `true`로 설정해 distributed training 사용
8. nproc_per_node: `torchrun`에 전달할 프로세스 / GPU 개수
9. 이후 인자: `main.py`에 그대로 전달 (예: `--optimizer adamw --amp`)

### 예시

`MobileViT`를 CIFAR-10으로 학습하기:

```bash
./run.sh mobilevit cifar10 ./checkpoints 20 64 cpu
```

`ExMobileViT`를 CIFAR-100으로 학습하기:

```bash
./run.sh exmobilevit cifar100 ./checkpoints 30 64 cuda:0
```

### 입력 크기

`--resize`를 지정하지 않으면 데이터셋 기본 해상도(CIFAR 32, ImageNet 224)를 사용합니다.
모델이 더 큰 입력을 요구하면 자동으로 키웁니다 (예: `inception_v3`는 75, `alexnet`은 64, `maxvit_t`는 224).
직접 지정한 `--resize`가 모델 조건에 맞지 않으면 바로 에러를 냅니다.

### 샘플러 옵션

학습 샘플링은 기본 `default` 외에 클래스 불균형 처리용 `weighted` 샘플러를 지원합니다.

```bash
python main.py --model mobilevit --dataset cifar10 --download-data --sampler weighted
```

DDP 모드에서는 자동으로 `DistributedSampler`가 사용됩니다 (`weighted`는 무시되고 경고가 출력됩니다).

### DDP (Distributed Data Parallel)

멀티 GPU 환경에서는 `torchrun`을 사용합니다.

```bash
torchrun --nproc_per_node=2 main.py --model mobilevit --dataset cifar10 --download-data --ddp --epochs 20 --batch-size 32
```

또는 `run.sh`를 사용하여 DDP를 실행할 수 있습니다.

```bash
./run.sh mobilevit cifar10 ./checkpoints 20 32 cuda:0 ddp 2
```

이 경우 `--batch-size`는 각 프로세스당 배치 크기입니다. `--sync-bn`으로 SyncBatchNorm을 켤 수 있습니다 (CUDA 전용).

### 추가 옵션

- `--optimizer`: `sgd`(기본, nesterov) 또는 `adamw`. bias / norm / 토큰 파라미터에는 weight decay를 적용하지 않습니다
- `--lr`, `--momentum`, `--weight-decay`: 학습률, SGD 모멘텀, 가중치 감쇠
- `--scheduler`: `cosine`(기본), `step`, `none` / `--warmup-epochs`, `--min-lr`, `--step-size`, `--gamma`
- `--label-smoothing`: 라벨 스무딩
- `--grad-clip`: gradient norm clipping
- `--amp`: mixed precision (CUDA 전용, 16GB 이하 GPU에서 메모리 절약)
- `--num-workers`: 데이터 로더 워커 수
- `--resize`: 입력 이미지 크기
- `--master-addr`, `--master-port`: DDP 마스터 주소와 포트
- `--backend`: DDP 백엔드 (`nccl` 또는 `gloo`)
