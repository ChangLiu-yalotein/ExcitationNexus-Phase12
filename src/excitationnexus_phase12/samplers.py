from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .collate import collate_phase12


def make_dataloader(dataset, *, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0):
    generator=torch.Generator(); generator.manual_seed(int(seed))
    return DataLoader(dataset,batch_size=batch_size,shuffle=shuffle,generator=generator,
                      num_workers=num_workers,collate_fn=collate_phase12,drop_last=False)
