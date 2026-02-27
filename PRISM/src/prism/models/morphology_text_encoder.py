import hashlib
from typing import List

import torch
import torch.nn as nn


class MorphologyTextEncoder(nn.Module):
    def __init__(self, out_dim: int = 128) -> None:
        super().__init__()
        self.out_dim = out_dim

    def _hash_to_vector(self, text: str, device: torch.device) -> torch.Tensor:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vals = torch.tensor(list(digest), dtype=torch.float32, device=device)
        vals = vals / 255.0
        repeat = (self.out_dim + vals.numel() - 1) // vals.numel()
        vals = vals.repeat(repeat)[: self.out_dim]
        return vals * 2 - 1

    def forward(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        vectors = [self._hash_to_vector(p, device=device) for p in prompts]
        return torch.stack(vectors, dim=0)
