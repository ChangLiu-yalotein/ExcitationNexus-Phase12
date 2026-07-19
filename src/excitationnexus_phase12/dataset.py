from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from .contracts import ALLOWED_PARTITIONS, TaskGraph, allowed_scalar_fields
from .graph_builder import build_graph_from_json, raw_paths


def load_bound_table(table_path: str | Path, manifest_path: str | Path) -> pd.DataFrame:
    table = pd.read_parquet(table_path)
    manifest = pd.read_csv(manifest_path)
    if len(manifest) != 15016 or not manifest.molecule_id.is_unique:
        raise ValueError("manifest identity contract failed")
    joined = manifest.merge(table, on="molecule_id", how="left", validate="one_to_one",
                            suffixes=("", "_table"), sort=False)
    if len(joined) != len(manifest) or joined.canonical_smiles.isna().any():
        raise ValueError("1:1 table join failed")
    return joined


class Phase12Dataset(Dataset):
    def __init__(self, frame: pd.DataFrame, *, partition: str, view: str,
                 raw_root: str | Path, task_graph: TaskGraph,
                 pm6_dipole_enabled: bool = False, cutoff: float = 5.0,
                 max_neighbors: int = 32, target_stats: dict | None = None):
        if partition not in ALLOWED_PARTITIONS:
            raise ValueError(f"partition must be explicit train/val/test, got {partition}")
        self.frame = frame.loc[frame.partition.eq(partition)].copy().sort_values(
            "molecule_id", kind="mergesort").reset_index(drop=True)
        self.partition, self.view, self.raw_root = partition, view, Path(raw_root)
        self.task_graph = task_graph
        self.scalar_fields = allowed_scalar_fields(view, pm6_dipole_enabled)
        self.cutoff, self.max_neighbors = cutoff, max_neighbors
        self.target_stats = target_stats or {}

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[int(idx)]
        target_values, masks = [], []
        for task in self.task_graph.all_reportable_tasks:
            value = row.get(task, np.nan)
            valid = bool(pd.notna(value))
            if valid and task in self.target_stats:
                st = self.target_stats[task]
                value = (float(value) - st["mean"]) / st["std"]
            target_values.append(float(value) if valid else 0.0); masks.append(valid)
        scalars = [float(row[x]) for x in self.scalar_fields]
        common = dict(
            molecule_id=str(row.molecule_id), partition=str(row.partition),
            structure_group_id_v1=str(row.structure_group_id_v1),
            donor_structure_group_id_v1=str(row.donor_structure_group_id_v1),
            acceptor_structure_group_id_v1=str(row.acceptor_structure_group_id_v1),
            pair_group_id_v1=str(row.pair_group_id_v1),
            role_aware_group_id_v1=str(row.role_aware_group_id_v1),
            group_weight=torch.tensor(float(row.group_weight), dtype=torch.float32),
            scalar_inputs=torch.tensor(scalars, dtype=torch.float32),
            targets=torch.tensor(target_values, dtype=torch.float32),
            target_mask=torch.tensor(masks, dtype=torch.bool),
        )
        if self.view in {"table_only", "tier0_2d"}:
            return common
        structure, sidecar = raw_paths(self.raw_root, str(row.molecule_id), self.view)
        graph = build_graph_from_json(structure, sidecar, cutoff=self.cutoff,
                                      max_neighbors=self.max_neighbors)
        for key, value in common.items():
            setattr(graph, key, value)
        return graph

    def subset_by_indices(self, indices: Sequence[int]) -> "Phase12Dataset":
        obj = object.__new__(Phase12Dataset)
        obj.__dict__ = self.__dict__.copy()
        obj.frame = self.frame.iloc[list(indices)].reset_index(drop=True)
        return obj
