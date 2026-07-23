## 이 프로젝트는 개인용 그래픽 카드(<16GB) 기반으로 모델 학습을 위한 종합 허브입니다

### 지원 모델 리스트

*Residual based*
- ResNet
- DenseNet

*Feature map Squeeze*
- MobileNet
- SeNet

*Transformer based*
- MobileViT
- ExMobileViT

### 요구 사항

- Python 3.8 이상
- PyTorch
- torchvision
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

모델 출력 확인:

```bash
python main.py --model mobilevit --dataset cifar10 --mode print
```

모델 요약 확인:

```bash
python main.py --model mobilevit --dataset cifar10 --mode summary --resize 224
```

체크포인트 저장 경로:

```bash
python main.py --model mobilevit --dataset cifar10 --download-data --epochs 20 --save-dir ./checkpoints
```

체크포인트 불러오기:

```bash
python main.py --model mobilevit --dataset cifar10 --checkpoint-path ./checkpoints/mobilevit_cifar10_best.pth
```

쉘 스크립트로 실행:

```bash
./run.sh mobilevit cifar10 ./checkpoints 20 64 cpu
```

`run.sh` 인자는 순서대로 다음과 같습니다:

1. model: 모델 이름 (`resnet18`, `mobilenet`, `densenet121`, `mobilevit`, `exmobilevit`)
2. dataset: 데이터셋 이름 (`cifar10`, `cifar100`, `imagenet`)
3. save_dir: 체크포인트 저장 디렉터리
4. epochs: 학습 epoch 수
5. batch_size: 배치 크기
6. device: 실행 디바이스 (`cpu`, `cuda`, `cuda:0`)
7. ddp: `ddp` 또는 `true`로 설정해 distributed training 사용
8. nproc_per_node: `torchrun`에 전달할 프로세스 / GPU 개수

### 예시

`MobileViT`를 CIFAR-10으로 학습하기:

```bash
./run.sh mobilevit cifar10 ./checkpoints 20 64 cpu
```

`ExMobileViT`를 CIFAR-100으로 학습하기:

```bash
./run.sh exmobilevit cifar100 ./checkpoints 30 64 cuda:0
```

### 샘플러 옵션

학습 샘플링은 기본 `default` 외에 클래스 불균형 처리용 `weighted` 샘플러를 지원합니다.

```bash
python main.py --model mobilevit --dataset cifar10 --download-data --sampler weighted
```

DDP 모드에서는 자동으로 `DistributedSampler`가 사용됩니다.

### DDP (Distributed Data Parallel)

멀티 GPU 환경에서는 `torchrun`을 사용합니다.

```bash
torchrun --nproc_per_node=2 main.py --model mobilevit --dataset cifar10 --download-data --ddp --epochs 20 --batch-size 32
```

또는 `run.sh`를 사용하여 DDP를 실행할 수 있습니다.

```bash
./run.sh mobilevit cifar10 ./checkpoints 20 32 cuda:0 ddp 2
```

이 경우 `--batch-size`는 각 프로세스당 배치 크기입니다.

### 추가 옵션

- `--lr`: 학습률
- `--momentum`: SGD 모멘텀
- `--weight-decay`: 가중치 감쇠
- `--num-workers`: 데이터 로더 워커 수
- `--resize`: 입력 이미지 크기
- `--master-addr`, `--master-port`: DDP 마스터 주소와 포트
- `--backend`: DDP 백엔드 (`nccl` 또는 `gloo`)
