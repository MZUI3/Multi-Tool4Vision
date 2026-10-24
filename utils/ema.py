"""Exponential moving average of model weights (Polyak averaging), as used by the MobileViT recipe."""
from copy import deepcopy

import torch
from torch import nn

from utils.hwaccel import unwrap_model


class ModelEMA:
    """Keeps ``module``, a frozen copy of the model whose weights track an EMA of the trained weights.

    The decay ramps up from 0 at the first step (``(1 + n) / (10 + n)``, timm-style) so early epochs are
    not dominated by the random initialisation.
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998, warmup: bool = True):
        self.module = deepcopy(unwrap_model(model)).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = float(decay)
        self.warmup = warmup
        self.updates = 0

    def current_decay(self) -> float:
        if not self.warmup:
            return self.decay
        return min(self.decay, (1.0 + self.updates) / (10.0 + self.updates))

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        self.updates += 1
        d = self.current_decay()
        source = unwrap_model(model).state_dict()
        for key, ema_value in self.module.state_dict().items():
            value = source[key]
            if ema_value.dtype.is_floating_point:
                ema_value.mul_(d).add_(value.detach().to(ema_value.dtype), alpha=1.0 - d)
            else:
                ema_value.copy_(value)

    def state_dict(self) -> dict:
        return {"module": self.module.state_dict(), "updates": self.updates, "decay": self.decay}

    def load_state_dict(self, state: dict) -> None:
        self.module.load_state_dict(state["module"])
        self.updates = int(state.get("updates", 0))
