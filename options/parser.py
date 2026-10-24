# Import Basic Lib
import argparse
import json

from utils.hwaccel import AccelConfig, add_accel_args


def build_parser():
    parser = argparse.ArgumentParser("Personal_Model_HUB")

    # Model / data
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture (see `--mode list`)")
    parser.add_argument("--category", type=str, default=None, help="Filter `--mode list` by category (cnn, lightweight, transformer)")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        help="Dataset name: cifar10, cifar100, stl10, imagenet, food101, flowers102, oxford_pets, "
        "or imagefolder (<data-root>/train/<class>/*, <data-root>/val/<class>/*)",
    )
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--num-classes", type=int, default=None, help="Override the class count (needed for imagefolder in print/summary mode)")
    parser.add_argument("--download-data", action="store_true", help="Download dataset if needed")
    parser.add_argument(
        "--resize",
        type=int,
        default=None,
        help="Input size. Default: dataset native size (32 for CIFAR, 224 for ImageNet), raised to the model minimum",
    )
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and validation (per process under DDP)")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes for data loading")
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["default", "weighted"],
        default="default",
        help="Sampler strategy for training data",
    )

    # Augmentation / regularisation (MobileViT-style training recipe)
    aug = parser.add_argument_group("augmentation & regularisation")
    aug.add_argument(
        "--multi-scale",
        action="store_true",
        help="MobileViT multi-scale sampler: every training batch uses a random resolution from --ms-sizes and the "
        "batch size is scaled by (base/size)^2 so GPU memory stays constant",
    )
    aug.add_argument("--ms-sizes", type=str, default=None, help="Explicit resolutions for --multi-scale, e.g. 160,192,224,256,288")
    aug.add_argument("--ms-min-scale", type=float, default=0.625, help="Smallest multi-scale resolution as a fraction of --resize (MobileViT: 160/256)")
    aug.add_argument("--ms-max-scale", type=float, default=1.25, help="Largest multi-scale resolution as a fraction of --resize (MobileViT: 320/256)")
    aug.add_argument("--ms-num-scales", type=int, default=5, help="Number of resolutions between min and max scale")
    aug.add_argument("--ms-fixed-batch", action="store_true", help="Keep the batch size constant across resolutions (default: scale it)")
    aug.add_argument(
        "--auto-augment",
        type=str,
        choices=["none", "rand", "trivial", "autoaugment"],
        default="none",
        help="RandAugment / TrivialAugmentWide / AutoAugment policy on top of the basic augmentation",
    )
    aug.add_argument("--random-erase", type=float, default=0.0, help="Random Erasing probability (0 = off)")
    aug.add_argument("--mixup", type=float, default=0.0, help="Mixup alpha (0 = off)")
    aug.add_argument("--cutmix", type=float, default=0.0, help="CutMix alpha (0 = off)")
    aug.add_argument("--mix-prob", type=float, default=1.0, help="Probability of applying mixup/cutmix to a batch")
    aug.add_argument("--mix-switch-prob", type=float, default=0.5, help="Probability of picking cutmix over mixup when both are on")
    aug.add_argument("--ema", action="store_true", help="Keep an exponential moving average of the weights and validate/save it")
    aug.add_argument("--ema-decay", type=float, default=0.9998, help="EMA decay (per optimizer step)")
    aug.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for cross entropy")

    # Optimisation
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--optimizer", type=str, choices=["sgd", "adamw"], default="sgd", help="Optimizer")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay (not applied to norm/bias params)")
    parser.add_argument("--scheduler", type=str, choices=["none", "cosine", "step"], default="cosine", help="LR scheduler")
    parser.add_argument("--warmup-epochs", type=int, default=0, help="Linear warmup epochs before the scheduler")
    parser.add_argument("--step-size", type=int, default=30, help="Epoch interval for the step scheduler")
    parser.add_argument("--gamma", type=float, default=0.1, help="Decay factor for the step scheduler")
    parser.add_argument("--min-lr", type=float, default=0.0, help="Final learning rate for the cosine scheduler")
    parser.add_argument("--grad-clip", type=float, default=None, help="Clip gradient norm to this value")

    # Runtime
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "eval", "summary", "print", "list"],
        default="train",
        help="train | eval (validate a checkpoint) | summary | print | list (registered models)",
    )
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to load model weights from")
    parser.add_argument("--resume", action="store_true", help="Also restore optimizer/scheduler/epoch from --checkpoint-path")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Path to save checkpoints")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--limit-batches", type=int, default=None, help="Debug: use only the first N train / val batches per epoch")

    # Hardware acceleration: --device / --parallel / --gpu-ids / --sync-bn / --backend / --amp ...
    # (identical flags and implementation in the RL hub, see utils/hwaccel.py)
    add_accel_args(parser, amp=True)
    return parser


def get_opts(args=None):
    parser = build_parser()
    opts = parser.parse_args(args)
    if opts.resume and not opts.checkpoint_path:
        parser.error("--resume requires --checkpoint-path")
    accel = AccelConfig.from_namespace(opts)
    opts.parallel = accel.parallel
    opts.ddp = accel.parallel == "ddp"
    if opts.gpu_ids and opts.parallel != "dp":
        parser.error("--gpu-ids is only used with --parallel dp (use CUDA_VISIBLE_DEVICES for ddp)")
    if opts.ms_sizes:
        opts.ms_sizes = [int(s) for s in opts.ms_sizes.split(",") if s.strip()]
    if opts.multi_scale and opts.ms_num_scales < 1:
        parser.error("--ms-num-scales must be >= 1")
    return opts


def format_opts(opts: argparse.Namespace) -> str:
    return "Input args: " + json.dumps(vars(opts), indent=4, default=str)
