from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .graph_builder import _directed_radius_graph


ALLOWED_PARTITIONS = {"train", "val", "test"}


def graph_content_hash(z: np.ndarray, pos: np.ndarray, role: np.ndarray,
                       bonds: np.ndarray, bond_types: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in (np.asarray(z, dtype=np.int16), np.asarray(pos, dtype=np.float64),
                  np.asarray(role, dtype=np.int8), np.asarray(bonds, dtype=np.int32),
                  np.asarray(bond_types, dtype=np.int8)):
        digest.update(array.shape.__repr__().encode())
        digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


class DFTGraphCache:
    def __init__(self, cache_path: str | Path, registry_path: str | Path):
        payload = np.load(cache_path, allow_pickle=False)
        self.z, self.pos, self.role = payload["z"], payload["pos"], payload["role"]
        self.registry = pd.read_parquet(registry_path).sort_values("cache_index").reset_index(drop=True)
        if len(self.registry) != 15016 or not self.registry.molecule_id.is_unique:
            raise ValueError("DFT graph registry identity failure")

    def graph(self, cache_index: int, *, cutoff: float, max_neighbors: int) -> Data:
        row = self.registry.iloc[int(cache_index)]
        start, end = int(row.atom_offset_start), int(row.atom_offset_end)
        pos = torch.from_numpy(self.pos[start:end].copy()).float()
        role = torch.from_numpy(self.role[start:end].copy()).long()
        z = torch.from_numpy(self.z[start:end].copy()).long()
        return Data(
            z=z, pos=pos, role=role, donor_mask=role.eq(0), acceptor_mask=role.eq(1),
            unknown_mask=role.eq(2), edge_index=_directed_radius_graph(pos, cutoff, max_neighbors),
            num_nodes=len(z), molecule_id=str(row.molecule_id),
        )


class Gate1B2GraphDataset(Dataset):
    def __init__(self, cache: DFTGraphCache, manifest: pd.DataFrame, *, partition: str,
                 targets: pd.DataFrame | None = None, cutoff: float = 5.0, max_neighbors: int = 32):
        if partition not in ALLOWED_PARTITIONS:
            raise ValueError("partition must be explicit train/val/test; quarantine is forbidden")
        frame = manifest.loc[manifest.partition.eq(partition)].merge(
            cache.registry[["molecule_id", "cache_index"]], on="molecule_id", validate="one_to_one")
        if targets is not None:
            frame = frame.merge(targets, on="molecule_id", validate="one_to_one")
        self.frame = frame.sort_values("molecule_id", kind="mergesort").reset_index(drop=True)
        self.cache, self.cutoff, self.max_neighbors = cache, cutoff, max_neighbors

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Data:
        row = self.frame.iloc[int(index)]
        graph = self.cache.graph(int(row.cache_index), cutoff=self.cutoff, max_neighbors=self.max_neighbors)
        graph.group_weight = torch.tensor(float(row.group_weight), dtype=torch.float32)
        graph.structure_group_id_v1 = str(row.structure_group_id_v1)
        if "target" in row:
            graph.target = torch.tensor(float(row.target), dtype=torch.float32)
        return graph
