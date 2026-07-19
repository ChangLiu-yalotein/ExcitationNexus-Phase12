from __future__ import annotations

import torch
from torch import nn


class SharedInvariantBackbone(nn.Module):
    """Small distance-only invariant backbone shared by both admission baselines."""
    def __init__(self, hidden_dim: int = 48, num_rbf: int = 16, layers: int = 2, cutoff: float = 5.0):
        super().__init__()
        self.hidden_dim, self.cutoff = hidden_dim, cutoff
        self.z_embedding = nn.Embedding(100, hidden_dim)
        self.role_embedding = nn.Embedding(3, hidden_dim)
        self.register_buffer("rbf_centers", torch.linspace(0.0, cutoff, num_rbf))
        self.rbf_gamma = 10.0 / cutoff ** 2
        self.messages = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim + num_rbf, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(layers)
        ])
        self.updates = nn.ModuleList([
            nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim))
            for _ in range(layers)
        ])

    @staticmethod
    def pool_mean(x: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        out = x.new_zeros((n_graphs, x.shape[-1])); count = x.new_zeros((n_graphs, 1))
        if len(x):
            out.index_add_(0, batch, x); count.index_add_(0, batch, x.new_ones((len(x), 1)))
        return out / count.clamp_min(1.0)

    def forward(self, z: torch.Tensor, pos: torch.Tensor, role: torch.Tensor,
                edge_index: torch.Tensor, batch: torch.Tensor, n_graphs: int) -> torch.Tensor:
        if len(z) == 0:
            return pos.new_zeros((n_graphs, self.hidden_dim))
        x = self.z_embedding(z) + self.role_embedding(role)
        if edge_index.numel():
            src, dst = edge_index
            distance = (pos[src] - pos[dst]).norm(dim=-1)
            rbf = torch.exp(-self.rbf_gamma * (distance[:, None] - self.rbf_centers[None, :]) ** 2)
        for message, update in zip(self.messages, self.updates):
            aggregate = x.new_zeros(x.shape)
            if edge_index.numel():
                msg = message(torch.cat([x[src], rbf], dim=-1)); aggregate.index_add_(0, dst, msg)
            x = x + update(torch.cat([x, aggregate], dim=-1))
        return self.pool_mean(x, batch, n_graphs)


class M3MergedModel(nn.Module):
    def __init__(self, hidden_dim: int = 48, head_width: int = 128, num_rbf: int = 16,
                 layers: int = 2, cutoff: float = 5.0):
        super().__init__()
        self.backbone = SharedInvariantBackbone(hidden_dim, num_rbf, layers, cutoff)
        self.head = nn.Sequential(nn.Linear(hidden_dim + 3, head_width), nn.SiLU(), nn.Linear(head_width, 1))

    def forward(self, batch) -> torch.Tensor:
        n_graphs = int(batch.num_graphs)
        pooled = self.backbone(batch.z, batch.pos, batch.role, batch.edge_index, batch.batch, n_graphs)
        presence = torch.stack([(batch.role.eq(role)).new_zeros(n_graphs, dtype=torch.float32)
                                for role in range(3)], dim=-1).to(pooled.device)
        for role in range(3):
            mask = batch.role.eq(role)
            if mask.any():
                presence[:, role].index_fill_(0, batch.batch[mask].unique(), 1.0)
        return self.head(torch.cat([pooled, presence], dim=-1)).squeeze(-1)
