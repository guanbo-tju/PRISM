import math

import torch
import torch.nn as nn


class LightweightViTEncoder(nn.Module):
    def __init__(
        self,
        image_size: int = 270,
        patch_size: int = 15,
        in_channels: int = 3,
        embed_dim: int = 768,
        depth: int = 8,
        num_heads: int = 8,
    ) -> None:
        super().__init__()
        self.patch_embed = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        tokens = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(b, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)
        tokens = tokens + self.pos_embed[:, : tokens.size(1)]
        tokens = self.encoder(tokens)
        tokens = self.norm(tokens)
        spatial = tokens[:, 1:, :]
        h = int(math.sqrt(spatial.size(1)))
        spatial = spatial.transpose(1, 2).reshape(b, -1, h, h)
        return spatial
