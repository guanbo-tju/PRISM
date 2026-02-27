from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn.functional as F

from .morphology_text_encoder import MorphologyTextEncoder


@dataclass
class GraphData:
    x: torch.Tensor
    edge_index: torch.Tensor
    edge_attr: torch.Tensor
    centers: torch.Tensor


class GraphBuilder:
    def __init__(
        self,
        knn_k: int = 8,
        morph_text_dim: int = 128,
        position_dim: int = 64,
    ) -> None:
        self.knn_k = knn_k
        self.text_encoder = MorphologyTextEncoder(out_dim=morph_text_dim)
        self.position_dim = position_dim

    def _nucleus_stats(self, instance_mask: torch.Tensor) -> Tuple[List[int], torch.Tensor, torch.Tensor, torch.Tensor]:
        ids = torch.unique(instance_mask)
        ids = ids[ids > 0]
        centers, areas, irregularities = [], [], []

        for nucleus_id in ids.tolist():
            pos = (instance_mask == nucleus_id).nonzero(as_tuple=False)
            if pos.numel() == 0:
                continue
            yx = pos.float()
            center = yx.mean(dim=0)
            area = pos.size(0)
            y_min, x_min = yx.min(dim=0).values
            y_max, x_max = yx.max(dim=0).values
            box_area = max((y_max - y_min + 1) * (x_max - x_min + 1), 1.0)
            irregularity = float(area) / float(box_area)
            centers.append(center.flip(0))
            areas.append(float(area))
            irregularities.append(float(irregularity))

        if not centers:
            return [], torch.empty(0, 2), torch.empty(0), torch.empty(0)
        return (
            ids.tolist(),
            torch.stack(centers, dim=0),
            torch.tensor(areas, dtype=torch.float32),
            torch.tensor(irregularities, dtype=torch.float32),
        )

    def _position_embedding(self, centers: torch.Tensor, h: int, w: int) -> torch.Tensor:
        if centers.numel() == 0:
            return torch.empty(0, self.position_dim, device=centers.device)
        xy = centers.clone()
        xy[:, 0] = xy[:, 0] / max(w - 1, 1)
        xy[:, 1] = xy[:, 1] / max(h - 1, 1)
        half = self.position_dim // 4
        freqs = torch.arange(half, device=centers.device).float() + 1
        pe = []
        for i in range(2):
            v = xy[:, i : i + 1]
            pe.append(torch.sin(2 * torch.pi * freqs * v))
            pe.append(torch.cos(2 * torch.pi * freqs * v))
        return torch.cat(pe, dim=1)

    def _knn_edges(self, centers: torch.Tensor) -> torch.Tensor:
        n = centers.size(0)
        if n <= 1:
            return torch.zeros(2, 0, dtype=torch.long, device=centers.device)
        dist = torch.cdist(centers, centers)
        dist.fill_diagonal_(1e8)
        k = min(self.knn_k, n - 1)
        nn_idx = dist.topk(k, largest=False).indices
        src = torch.arange(n, device=centers.device).unsqueeze(1).expand(n, k).reshape(-1)
        dst = nn_idx.reshape(-1)
        return torch.stack([src, dst], dim=0)

    def build(self, feature_map: torch.Tensor, instance_mask: torch.Tensor) -> GraphData:
        device = feature_map.device
        _, c, h, w = feature_map.shape
        ids, centers, areas, irregularities = self._nucleus_stats(instance_mask)
        if len(ids) == 0:
            x = feature_map.mean(dim=(2, 3))
            edge_index = torch.zeros(2, 0, dtype=torch.long, device=device)
            edge_attr = torch.zeros(0, 3, device=device)
            return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr, centers=torch.zeros(1, 2, device=device))

        centers = centers.to(device)
        areas = areas.to(device)
        irregularities = irregularities.to(device)

        grid = centers.clone()
        grid[:, 0] = grid[:, 0] / max(w - 1, 1) * 2 - 1
        grid[:, 1] = grid[:, 1] / max(h - 1, 1) * 2 - 1
        sampled = F.grid_sample(
            feature_map,
            grid.view(1, -1, 1, 2),
            mode="bilinear",
            align_corners=True,
        )
        local_feat = sampled.squeeze(0).squeeze(-1).transpose(0, 1)

        position_feat = self._position_embedding(centers, h=h, w=w)
        prompts = []
        for area, irr, center in zip(areas.tolist(), irregularities.tolist(), centers.tolist()):
            shape = "round" if irr > 0.7 else "irregular"
            diameter = area ** 0.5
            prompts.append(
                f"a label of {shape} cell nucleus located at ({center[0]:.1f},{center[1]:.1f}) with {diameter:.1f} diameter"
            )
        text_feat = self.text_encoder(prompts, device=device)

        x = torch.cat([local_feat, position_feat, text_feat, areas.unsqueeze(1), irregularities.unsqueeze(1)], dim=1)
        edge_index = self._knn_edges(centers)
        if edge_index.numel() == 0:
            edge_attr = torch.zeros(0, 3, device=device)
        else:
            src, dst = edge_index
            delta = centers[dst] - centers[src]
            dist = torch.norm(delta, dim=1, keepdim=True)
            edge_attr = torch.cat([delta, dist], dim=1)
        return GraphData(x=x, edge_index=edge_index, edge_attr=edge_attr, centers=centers)
