import torch
import torch.nn as nn


class SubgroupHead(nn.Module):
    def __init__(self, in_dim: int = 512, fusion_dim: int = 512, num_classes: int = 2) -> None:
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Linear(in_dim, fusion_dim),
            nn.GELU(),
            nn.Dropout(0.2),
        )
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, graph_feat: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.fusion(graph_feat))
