from __future__ import annotations

import torch
from torch import nn

from .merged_3d import SharedInvariantBackbone


class M3DAUSharedModel(nn.Module):
    def __init__(self, hidden_dim: int = 48, head_width: int = 44, num_rbf: int = 16,
                 layers: int = 2, cutoff: float = 5.0):
        super().__init__()
        self.backbone = SharedInvariantBackbone(hidden_dim, num_rbf, layers, cutoff)
        self.head = nn.Sequential(nn.Linear(3 * hidden_dim + 3, head_width), nn.SiLU(), nn.Linear(head_width, 1))

    def _role_subgraph(self, batch, role_value: int, n_graphs: int) -> tuple[torch.Tensor, torch.Tensor]:
        mask = batch.role.eq(role_value)
        if not mask.any():
            return batch.pos.new_zeros((n_graphs, self.backbone.hidden_dim)), mask.new_zeros(n_graphs, dtype=torch.float32)
        old_to_new = torch.full((batch.num_nodes,), -1, device=batch.z.device, dtype=torch.long)
        old_to_new[mask] = torch.arange(int(mask.sum()), device=batch.z.device)
        edge_mask = mask[batch.edge_index[0]] & mask[batch.edge_index[1]]
        edges = old_to_new[batch.edge_index[:, edge_mask]]
        pooled = self.backbone(batch.z[mask], batch.pos[mask], batch.role[mask], edges,
                               batch.batch[mask], n_graphs)
        presence = mask.new_zeros(n_graphs, dtype=torch.float32)
        presence.index_fill_(0, batch.batch[mask].unique(), 1.0)
        return pooled, presence

    def forward(self, batch) -> torch.Tensor:
        n_graphs = int(batch.num_graphs)
        pooled, presence = [], []
        # Three explicit subgraph calls share this model's one backbone parameter set.
        for role in range(3):
            embedding, exists = self._role_subgraph(batch, role, n_graphs)
            pooled.append(embedding); presence.append(exists)
        return self.head(torch.cat([*pooled, torch.stack(presence, dim=-1)], dim=-1)).squeeze(-1)
