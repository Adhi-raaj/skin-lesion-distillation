

"""
Models module initialization.
Exports model builders and utilities.
"""

from .auxiliary_modules import build_model, RareDiseaseClassifier
from .ssl_module import FocalLoss, build_criterion, mixup_data, cutmix_data, mixed_criterion
from .gan_module import ModelEMA
from .distillation_module import DistillationLoss, DistillationMetrics
from .active_learning import UncertaintySampler, get_sampler

__all__ = [
    # Auxiliary
    "build_model",
    "RareDiseaseClassifier",
    # SSL
    "FocalLoss",
    "build_criterion",
    "mixup_data",
    "cutmix_data",
    "mixed_criterion",
    # GAN/EMA
    "ModelEMA",
    # Distillation
    "DistillationLoss",
    "DistillationMetrics",
    # Active Learning
    "UncertaintySampler",
    "get_sampler",
]
