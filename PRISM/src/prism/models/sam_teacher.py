from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class SAMTeacherProxy(nn.Module):
    def __init__(self, out_dim: int = 256, input_size: int = 1024) -> None:
        super().__init__()
        self.input_size = input_size
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_dim, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(inplace=True),
        )
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor, target_hw: Tuple[int, int]) -> torch.Tensor:
        x_up = F.interpolate(x, size=(self.input_size, self.input_size), mode="bilinear", align_corners=False)
        feat = self.backbone(x_up)
        feat = F.interpolate(feat, size=target_hw, mode="bilinear", align_corners=False)
        return feat
