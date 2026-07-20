from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as pads
import torch
from torch.utils.data import Dataset
from torch_geometric.data import Data

from .dft_graph_dataset import DFTGraphCache
from .edge_cache import ShardedEdgeCache


MODEL_PARTITIONS = {"train", "val"}


def read_targets(table_path: str | Path, target_column: str, molecule_ids: list[str], *,
                 requested_partition: str, allow_test: bool = False) -> pd.DataFrame:
    if requested_partition == "test" and not allow_test:
        raise PermissionError("TEST_TARGET_FIREWALL_LOCKED")
    if requested_partition not in {"train", "val", "test"}:
        raise PermissionError("quarantine/buffer target access is forbidden")
    dataset = pads.dataset(str(table_path), format="parquet")
    return dataset.to_table(
        columns=["molecule_id", target_column],
        filter=pads.field("molecule_id").isin([str(x) for x in molecule_ids]),
    ).to_pandas().rename(columns={target_column: "target"})


class Gate1B3GraphDataset(Dataset):
    def __init__(self, graph_cache: DFTGraphCache, edge_cache: ShardedEdgeCache,
                 manifest: pd.DataFrame, *, partition: str, targets: pd.DataFrame):
        if partition not in MODEL_PARTITIONS:
            raise PermissionError("Gate1B3 formal Dataset permits only explicit train/val")
        subset = manifest.loc[manifest.partition.eq(partition)].copy()
        frame = subset.merge(
            graph_cache.registry[["molecule_id", "cache_index", "graph_content_sha256"]],
            on="molecule_id", validate="one_to_one",
        ).merge(targets, on="molecule_id", validate="one_to_one")
        if len(frame) != len(subset) or frame.target.isna().any():
            raise ValueError("partition target/graph join failure")
        self.frame = frame.sort_values("molecule_id", kind="mergesort").reset_index(drop=True)
        self.graph_cache, self.edge_cache = graph_cache, edge_cache

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Data:
        row = self.frame.iloc[int(index)]
        source = self.graph_cache.registry.iloc[int(row.cache_index)]
        start, end = int(source.atom_offset_start), int(source.atom_offset_end)
        role = torch.from_numpy(self.graph_cache.role[start:end].copy()).long()
        graph = Data(
            z=torch.from_numpy(self.graph_cache.z[start:end].copy()).long(),
            pos=torch.from_numpy(self.graph_cache.pos[start:end].copy()).float(),
            role=role,
            donor_mask=role.eq(0), acceptor_mask=role.eq(1), unknown_mask=role.eq(2),
            edge_index=self.edge_cache.edge_index(str(row.molecule_id)),
            num_nodes=end - start, molecule_id=str(row.molecule_id),
            structure_group_id_v1=str(row.structure_group_id_v1),
            group_weight=torch.tensor(float(row.group_weight), dtype=torch.float32),
            target=torch.tensor(float(row.target), dtype=torch.float32),
        )
        return graph


def target_blind_subset(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    keys = frame.molecule_id.astype(str).map(lambda x: hashlib.sha256(x.encode()).hexdigest())
    return frame.assign(_key=keys).sort_values("_key").head(int(n)).drop(columns="_key")

