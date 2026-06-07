"""
EMA (Exponential Moving Average) of model weights.
Filename kept as gan_module.py for compatibility with the original layout,
but the contents have been swapped to EMA which actually helps test accuracy.

EMA tracks a smoothed copy of model weights. The smoothed model usually
generalizes a bit better than the raw one — easy +0.3-1.0% on classification.
"""
import copy
import torch
import torch.nn as nn


class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.9995):
        self.module = copy.deepcopy(model).eval()
        for p in self.module.parameters():
            p.requires_grad_(False)
        self.decay = decay

    @torch.no_grad()
    def update(self, model: nn.Module):
        msd = model.state_dict()
        for k, v in self.module.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(msd[k].detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(msd[k])

    def state_dict(self):
        return self.module.state_dict()

    def load_state_dict(self, sd):
        self.module.load_state_dict(sd)
