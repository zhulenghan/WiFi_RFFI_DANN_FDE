import torch
from torch import nn
from collections import OrderedDict
from tllib.modules.classifier import Classifier as ClassifierBase
from typing import Tuple, Optional, List, Dict


class L2Norm(nn.Module):
    def __init__(self, eps=1e-10):
        super(L2Norm, self).__init__()
        self.eps = eps

    def forward(self, x):
        norm = torch.sqrt(torch.sum(x ** 2, dim=1, keepdim=True) + self.eps)
        x = x / norm
        return x


class AlexNetFFT(nn.Module):
    def __init__(self, num_classes: int = 5, dropout: float = 0.5, l2_norm=False) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(2, 64, kernel_size=(5, 5), stride=2, padding=(0, 1)),
            nn.ReLU(inplace=True),
            nn.Flatten(start_dim=2),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Flatten(start_dim=1),
            nn.Dropout(p=dropout),
            nn.Linear(6 * 256, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(512, 512))

        if l2_norm:
            self.features.add_module("l2_norm", L2Norm())

        self.classifier = nn.Linear(512, num_classes)

        self._initialize_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)


def linear2_dr2_bn(input_size, output_size):
    return nn.Sequential(OrderedDict([
        ('dr1', nn.Dropout(0.5)),
        ('d1', nn.Linear(input_size, 2048)),
        ('bn1', nn.BatchNorm1d(2048)),
        ('relu1', nn.ReLU()),
        ('dr2', nn.Dropout(0.5)),
        ('d2', nn.Linear(2048, output_size))
    ]))

def linear2_dr2(input_size, output_size):
    return nn.Sequential(OrderedDict([
        ('dr1', nn.Dropout(0.5)),
        ('d1', nn.Linear(input_size, 2048)),
        ('relu1', nn.ReLU()),
        ('dr2', nn.Dropout(0.5)),
        ('d2', nn.Linear(2048, output_size))
    ]))


def linear3_bn2_v1(input_size, output_size):
    return nn.Sequential(OrderedDict([
        ('d1', nn.Linear(input_size, 3072)),
        ('bn1', nn.BatchNorm1d(3072)),
        ('relu1', nn.ReLU()),
        ('d2', nn.Linear(3072, 2048)),
        ('bn2', nn.BatchNorm1d(2048)),
        ('relu2', nn.ReLU()),
        ('d3', nn.Linear(2048, output_size))
    ]))


def linear3_bn2_v2(input_size, output_size):
    return nn.Sequential(OrderedDict([
        ('d1', nn.Linear(input_size, 1024)),
        ('bn1', nn.BatchNorm1d(1024)),
        ('relu1', nn.ReLU()),
        ('d2', nn.Linear(1024, 1024)),
        ('bn2', nn.BatchNorm1d(1024)),
        ('relu2', nn.ReLU()),
        ('d3', nn.Linear(1024, output_size))
    ]))


def linear3_dr2_v2(input_size, output_size):
    return nn.Sequential(OrderedDict([
        ('d1', nn.Linear(input_size, 4096)),
        ('dr1', nn.Dropout(0.5)),
        ('relu1', nn.ReLU()),
        ('d2', nn.Linear(4096, 4096)),
        ('dr2', nn.Dropout(0.5)),
        ('relu2', nn.ReLU()),
        ('d3', nn.Linear(4096, output_size))
    ]))


def linear2_bn(input_size, output_size):
    return nn.Sequential(OrderedDict([
        ('d1', nn.Linear(input_size, 100)),
        ('bn1', nn.BatchNorm1d(100)),
        ('relu1', nn.ReLU(True)),
        ('d2', nn.Linear(100, output_size))
    ]))


class AlexNet1D(nn.Module):
    def __init__(self, num_classes: int = 1000, dropout: float = 0.5) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
            nn.Conv1d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool1d(6)
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class RFFI(ClassifierBase):
    def __init__(self, num_classes: int, finetune=True, l2_norm=False, **kwargs):
        alex = AlexNetFFT(num_classes=num_classes, l2_norm=l2_norm)
        backbone = alex.features
        head = linear2_bn(512, num_classes)
        super(RFFI, self).__init__(backbone=backbone, num_classes=num_classes, out_features=512, head=head,
                                   finetune=finetune, **kwargs)
