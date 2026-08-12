"""
Model architectures for the 6-output regression task (Cd, Cu, Ni, Mn, Fe, Zn).

Two options are provided:

- CustomCNN: a small conv net trained from scratch. Fewer parameters,
  lower overfitting risk -- worth defaulting to given there are only
  36 truly-independent images behind the 650 augmented ones.

- ResNet18Backbone: ImageNet-pretrained transfer learning. Can help if
  the custom CNN underfits, but note the pretrained weights were learned
  on RGB statistics, not HSV -- it's used here as a generic feature
  extractor, which works reasonably in practice but isn't a perfect
  match for the input domain.
"""

import torch
import torch.nn as nn
import torchvision.models as tvm

import config


class CustomCNN(nn.Module):
    def __init__(self, n_outputs=6):
        super().__init__()

        def block(c_in, c_out):
            return nn.Sequential(
                nn.Conv2d(c_in, c_out, 3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            block(3, 16),
            block(16, 32),
            block(32, 64),
            block(64, 128),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


class ResNet18Backbone(nn.Module):
    def __init__(self, n_outputs=6, freeze_until=6):
        super().__init__()
        backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        layers = list(backbone.children())[:-2]  # drop avgpool + fc
        self.features = nn.Sequential(*layers)

        for i, child in enumerate(self.features.children()):
            if i < freeze_until:
                for p in child.parameters():
                    p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


def build_model(backbone=None, n_outputs=6):
    backbone = backbone or config.BACKBONE
    if backbone == "custom":
        return CustomCNN(n_outputs)
    elif backbone == "resnet18":
        return ResNet18Backbone(n_outputs)
    else:
        raise ValueError(f"Unknown backbone '{backbone}'")
