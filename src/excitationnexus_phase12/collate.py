from __future__ import annotations

import torch
from torch_geometric.data import Batch


def collate_phase12(samples):
    if not samples:
        raise ValueError("empty batch")
    if isinstance(samples[0], dict):
        out = {}
        for key in samples[0]:
            values = [x[key] for x in samples]
            out[key] = torch.stack(values) if torch.is_tensor(values[0]) else values
        return out
    return Batch.from_data_list(samples)
