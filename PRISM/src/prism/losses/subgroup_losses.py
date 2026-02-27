import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.25,
    gamma: float = 2.0,
) -> torch.Tensor:
    ce = F.cross_entropy(logits, target, reduction="none")
    pt = torch.exp(-ce)
    return (alpha * (1 - pt).pow(gamma) * ce).mean()
