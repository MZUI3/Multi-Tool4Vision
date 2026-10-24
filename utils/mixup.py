"""Mixup (Zhang et al., 2018) and CutMix (Yun et al., 2019) with soft targets.

The returned targets are one-hot probability vectors, so the usual ``nn.CrossEntropyLoss`` (which accepts
class probabilities and still applies ``label_smoothing``) can be used unchanged.
"""
import math
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


def _rand_bbox(height: int, width: int, lam: float) -> Tuple[int, int, int, int]:
    ratio = math.sqrt(1.0 - lam)
    cut_h, cut_w = int(height * ratio), int(width * ratio)
    cy, cx = np.random.randint(height), np.random.randint(width)
    y1, y2 = np.clip(cy - cut_h // 2, 0, height), np.clip(cy + cut_h // 2, 0, height)
    x1, x2 = np.clip(cx - cut_w // 2, 0, width), np.clip(cx + cut_w // 2, 0, width)
    return y1, y2, x1, x2


class Mixup:
    """Batch-level Mixup / CutMix. Each call mixes the batch with its reversed copy."""

    def __init__(
        self,
        mixup_alpha: float = 0.0,
        cutmix_alpha: float = 0.0,
        prob: float = 1.0,
        switch_prob: float = 0.5,
        num_classes: int = 1000,
    ):
        self.mixup_alpha = float(mixup_alpha)
        self.cutmix_alpha = float(cutmix_alpha)
        self.prob = float(prob)
        self.switch_prob = float(switch_prob)
        self.num_classes = int(num_classes)

    @property
    def enabled(self) -> bool:
        return self.mixup_alpha > 0 or self.cutmix_alpha > 0

    def _params(self) -> Tuple[float, bool]:
        """(lambda, use_cutmix)."""
        if np.random.rand() > self.prob:
            return 1.0, False
        if self.mixup_alpha > 0 and self.cutmix_alpha > 0:
            use_cutmix = np.random.rand() < self.switch_prob
        else:
            use_cutmix = self.cutmix_alpha > 0
        alpha = self.cutmix_alpha if use_cutmix else self.mixup_alpha
        return float(np.random.beta(alpha, alpha)), use_cutmix

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if x.size(0) % 2 == 1:
            # the reversed-batch pairing needs an even batch; mixing an odd batch pairs the middle sample with itself
            pass
        lam, use_cutmix = self._params()
        y_onehot = F.one_hot(y, self.num_classes).to(x.dtype)
        if lam >= 1.0:
            return x, y_onehot
        if use_cutmix:
            y1, y2, x1, x2 = _rand_bbox(x.size(2), x.size(3), lam)
            x = x.clone()
            x[:, :, y1:y2, x1:x2] = x.flip(0)[:, :, y1:y2, x1:x2]
            lam = 1.0 - (y2 - y1) * (x2 - x1) / float(x.size(2) * x.size(3))
        else:
            x = x * lam + x.flip(0) * (1.0 - lam)
        targets = y_onehot * lam + y_onehot.flip(0) * (1.0 - lam)
        return x, targets


def build_mixup(opts, num_classes: int) -> Optional[Mixup]:
    mix = Mixup(opts.mixup, opts.cutmix, opts.mix_prob, opts.mix_switch_prob, num_classes)
    return mix if mix.enabled else None
