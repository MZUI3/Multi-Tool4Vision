# Import Basic Lib
import argparse
import json
import torch


def build_parser():
    parser = argparse.ArgumentParser("Personal_Model_HUB")

    # Model / data
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture (see `--mode list`)")
    parser.add_argument("--category", type=str, default=None, help="Filter `--mode list` by category (cnn, lightweight, transformer)")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name (cifar10, cifar100, imagenet)")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root directory")
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
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing for cross entropy")
    parser.add_argument("--grad-clip", type=float, default=None, help="Clip gradient norm to this value")
    parser.add_argument("--amp", action="store_true", help="Mixed precision training (CUDA only)")

    # Runtime
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
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

    # Distributed
    parser.add_argument("--ddp", action="store_true", help="Enable distributed data parallel training")
    parser.add_argument("--sync-bn", action="store_true", help="Convert BatchNorm to SyncBatchNorm under DDP")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--rank", type=int, default=0, help="Global rank for distributed training")
    parser.add_argument("--world-size", type=int, default=1, help="Number of processes for distributed training")
    parser.add_argument("--backend", type=str, default=None, help="Distributed backend for DDP (nccl or gloo)")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1", help="DDP master address")
    parser.add_argument("--master-port", type=int, default=29500, help="DDP master port")
    return parser


def get_opts(args=None):
    parser = build_parser()
    opts = parser.parse_args(args)
    if opts.resume and not opts.checkpoint_path:
        parser.error("--resume requires --checkpoint-path")
    return opts


def format_opts(opts: argparse.Namespace) -> str:
    return "Input args: " + json.dumps(vars(opts), indent=4)
