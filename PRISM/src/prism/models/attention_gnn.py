import torch
import torch.nn as nn
import torch.nn.functional as F


class EdgeAwareAttentionLayer(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.node_proj = nn.Linear(in_dim, out_dim)
        self.edge_proj = nn.Linear(edge_dim, out_dim)
        self.q = nn.Linear(out_dim, out_dim, bias=False)
        self.k = nn.Linear(out_dim, out_dim, bias=False)
        self.v = nn.Linear(out_dim, out_dim, bias=False)
        self.mlp = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        if x.size(0) == 0:
            return x
        h = self.node_proj(x)
        if edge_index.numel() == 0:
            return self.norm(h + self.mlp(h))

        src, dst = edge_index
        e = self.edge_proj(edge_attr)
        q = self.q(h[dst])
        k = self.k(h[src] + e)
        v = self.v(h[src] + e)

        attn = (q * k).sum(dim=1) / (h.size(1) ** 0.5)
        attn = torch.exp(attn - attn.max())
        denom = torch.zeros(h.size(0), device=h.device).index_add(0, dst, attn)
        alpha = attn / (denom[dst] + 1e-6)

        msg = v * alpha.unsqueeze(1)
        out = torch.zeros_like(h).index_add(0, dst, msg)
        out = self.norm(out + h)
        out = self.norm(out + self.mlp(out))
        return out


class EdgeAwareAttentionGNN(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 256, out_dim: int = 512, edge_dim: int = 3, layers: int = 3) -> None:
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.layers = nn.ModuleList(
            [EdgeAwareAttentionLayer(hidden_dim, hidden_dim, edge_dim=edge_dim) for _ in range(layers)]
        )
        self.graph_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.input_proj(x)
        for layer in self.layers:
            h = layer(h, edge_index, edge_attr)
        if h.size(0) == 0:
            return torch.zeros(1, self.graph_head[-1].out_features, device=x.device)
        pooled = torch.cat([h.mean(dim=0), h.max(dim=0).values], dim=0).unsqueeze(0)
        return self.graph_head(pooled)
