"""
Model architectures for the 6-output regression task (Cd, Cu, Ni, Mn, Fe, Zn).

Four options are provided:

- CustomCNN: a small conv net trained from scratch. Fewest parameters,
  lowest overfitting risk -- worth defaulting to given there are only
  ~10 truly-independent physical samples behind the 650 augmented images
  (see README: the 36 "groups" collapse to ~10 independent sites once
  horizon/orientation redundancy is accounted for).

- ResNet18Backbone / ResNet50Backbone: ImageNet-pretrained transfer
  learning. Can help if the custom CNN underfits, but note the pretrained
  weights were learned on RGB statistics, not HSV -- used here as a
  generic feature extractor, which works reasonably in practice but isn't
  a perfect match for the input domain. ResNet50 has ~25x more parameters
  than the custom CNN and is noticeably slower per epoch on CPU; with
  only ~10 independent sites it's also the most prone to overfitting of
  the four options, so treat it as an experiment, not a safe default.

- DenseNet121Backbone: ImageNet-pretrained, fewer parameters than
  ResNet50 but more memory/compute-hungry per parameter due to dense
  (concatenation-heavy) connectivity -- expect it to be slower per epoch
  than either ResNet option on CPU. Same HSV-vs-RGB caveat as the ResNets.

All transfer-learning backbones freeze their earlier layers by default
(see freeze_until on each class) and only fine-tune the deeper layers
plus the regression head, to reduce overfitting risk on this small
dataset.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
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
            block(128, 256)
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, n_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


class _ResNetBackbone(nn.Module):
    """
    Shared implementation for ResNet-family transfer-learning backbones.
    Subclasses just supply which torchvision constructor/weights/output
    channel count to use -- ResNet18 and ResNet50 have identical
    top-level structure (conv1, bn1, relu, maxpool, layer1..4, avgpool,
    fc), so freeze_until means the same thing (an index into those 8
    top-level children) for both.
    """

    def __init__(self, builder, weights, out_channels, n_outputs=6, freeze_until=6):
        super().__init__()
        backbone = builder(weights=weights)
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
            nn.Linear(out_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)


class ResNet18Backbone(_ResNetBackbone):
    def __init__(self, n_outputs=6, freeze_until=6):
        super().__init__(
            tvm.resnet18, tvm.ResNet18_Weights.IMAGENET1K_V1,
            out_channels=512, n_outputs=n_outputs, freeze_until=freeze_until,
        )


class ResNet50Backbone(_ResNetBackbone):
    def __init__(self, n_outputs=6, freeze_until=6):
        # IMAGENET1K_V2 is torchvision's improved-recipe weights for resnet50
        super().__init__(
            tvm.resnet50, tvm.ResNet50_Weights.IMAGENET1K_V2,
            out_channels=2048, n_outputs=n_outputs, freeze_until=freeze_until,
        )


class DenseNet121Backbone(nn.Module):
    """
    DenseNet121's top-level structure is just `features` (a Sequential of
    conv0/norm0/relu0/pool0, then 4 denseblock/transition pairs, ending in
    a final batchnorm norm5) and `classifier` -- no separate avgpool/fc to
    strip off like ResNet. torchvision's reference DenseNet.forward()
    applies a ReLU to the features output before pooling (norm5 has no
    activation baked in), which we replicate here for correctness.
    """

    def __init__(self, n_outputs=6, freeze_until=6):
        super().__init__()
        backbone = tvm.densenet121(weights=tvm.DenseNet121_Weights.IMAGENET1K_V1)
        self.features = backbone.features  # already excludes the classifier
        out_channels = backbone.classifier.in_features  # 1024 for densenet121

        for i, child in enumerate(self.features.children()):
            if i < freeze_until:
                for p in child.parameters():
                    p.requires_grad = False

        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(out_channels, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, n_outputs),
        )

    def forward(self, x):
        x = self.features(x)
        x = F.relu(x, inplace=True)  # matches torchvision's DenseNet.forward()
        x = self.pool(x)
        return self.head(x)


def build_model(backbone=None, n_outputs=6):
    backbone = backbone or config.BACKBONE
    if backbone == "custom":
        return CustomCNN(n_outputs)
    elif backbone == "resnet18":
        return ResNet18Backbone(n_outputs)
    elif backbone == "resnet50":
        return ResNet50Backbone(n_outputs)
    elif backbone == "densenet121":
        return DenseNet121Backbone(n_outputs)
    else:
        raise ValueError(
            f"Unknown backbone '{backbone}'. Expected one of: "
            "'custom', 'resnet18', 'resnet50', 'densenet121'."
        )