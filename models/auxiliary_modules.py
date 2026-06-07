"""
Main classifier: pretrained backbone (timm) + custom head with dropout.
Targets HAM10000 7-class problem but flexible via num_classes.
"""

import torch
import torch.nn as nn
import timm


class RareDiseaseClassifier(nn.Module):
    def __init__(
        self,
        backbone: str = "efficientnet_b3",
        num_classes: int = 7,
        pretrained: bool = True,
        dropout: float = 0.4,
    ):
        super().__init__()

        self.backbone_name = backbone

        # Create backbone WITHOUT forcing global pooling
        self.backbone = timm.create_model(
            backbone,
            pretrained=pretrained,
            num_classes=0,
        )

        feat_dim = self.backbone.num_features

        # Classification head
        self.head = nn.Sequential(
            nn.BatchNorm1d(feat_dim),
            nn.Dropout(dropout),

            nn.Linear(feat_dim, 512),
            nn.GELU(),

            nn.BatchNorm1d(512),
            nn.Dropout(dropout * 0.75),

            nn.Linear(512, num_classes),
        )

    def forward(self, x):

        # Extract features
        f = self.backbone.forward_features(x)

        # Some backbones output [B,C,H,W]
        # Some output [B,C]
        if len(f.shape) == 4:
            f = torch.mean(f, dim=(2, 3))  # Global Average Pooling

        return self.head(f)

    def extract_features(self, x):

        f = self.backbone.forward_features(x)

        if len(f.shape) == 4:
            f = torch.mean(f, dim=(2, 3))

        return f


def build_model(cfg) -> nn.Module:
    return RareDiseaseClassifier(
        backbone=cfg.BACKBONE,
        num_classes=cfg.NUM_CLASSES,
        pretrained=cfg.PRETRAINED,
        dropout=cfg.DROPOUT,
    )