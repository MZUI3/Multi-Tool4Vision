# Import Basic Lib
import argparse
import json
import torch


def build_parser():
    parser = argparse.ArgumentParser("Personal_Model_HUB")
    parser.add_argument("--model", type=str, default="resnet18", help="Model architecture (resnet18, mobilenet, densenet121, mobilevit, exmobilevit)")
    parser.add_argument("--dataset", type=str, default="cifar10", help="Dataset name (cifar10, cifar100, imagenet)")
    parser.add_argument("--data-root", type=str, default="./data", help="Dataset root directory")
    parser.add_argument("--download-data", action="store_true", help="Download dataset if needed")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for training and validation")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of worker processes for data loading")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--momentum", type=float, default=0.9, help="Momentum for SGD optimizer")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay for optimizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training")
    parser.add_argument("--mode", type=str, choices=["train", "summary", "print"], default="train", help="Run mode")
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to load checkpoint from")
    parser.add_argument("--save-dir", type=str, default="./checkpoints", help="Path to save checkpoints")
    parser.add_argument("--resize", type=int, default=224, help="Input size for images")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--ddp", action="store_true", help="Enable distributed data parallel training")
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--rank", type=int, default=0, help="Global rank for distributed training")
    parser.add_argument("--world-size", type=int, default=1, help="Number of processes for distributed training")
    parser.add_argument("--backend", type=str, default=None, help="Distributed backend for DDP (nccl or gloo)")
    parser.add_argument("--master-addr", type=str, default="127.0.0.1", help="DDP master address")
    parser.add_argument("--master-port", type=int, default=29500, help="DDP master port")
    parser.add_argument(
        "--sampler",
        type=str,
        choices=["default", "weighted"],
        default="default",
        help="Sampler strategy for training data",
    )
    return parser


def get_opts(stdin):
    parser = build_parser()
    opts = parser.parse_args()
    opts.__class__.__repr__ = lambda x: 'Input args: ' + json.dumps(x.__dict__, indent=4)
    return opts
