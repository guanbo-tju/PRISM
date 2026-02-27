import torch
import torch.nn.functional as F


class PhysicsConstraintModule:
    def spatial_consistency(self, node_probs: torch.Tensor, edge_index: torch.Tensor, edge_weight: torch.Tensor | None = None) -> torch.Tensor:
        if edge_index.numel() == 0:
            return torch.zeros((), device=node_probs.device)
        src, dst = edge_index
        diff = node_probs[src] - node_probs[dst]
        if edge_weight is None:
            return (diff.pow(2).sum(dim=1)).mean()
        return (edge_weight * diff.pow(2).sum(dim=1)).mean()

    def observation_consistency(self, p: torch.Tensor, p_aug: torch.Tensor) -> torch.Tensor:
        p = p.clamp(1e-7, 1 - 1e-7)
        p_aug = p_aug.clamp(1e-7, 1 - 1e-7)
        return F.kl_div(p.log(), p_aug, reduction="batchmean")
