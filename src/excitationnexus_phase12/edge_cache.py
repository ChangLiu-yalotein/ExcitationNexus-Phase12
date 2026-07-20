from __future__ import annotations

import hashlib
import json
from collections import OrderedDict
from pathlib import Path

import numpy as np
import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class ShardedEdgeCache:
    """Read-only, hash-verified lookup for target-free radius edges."""

    def __init__(self, registry_path: str | Path, *, max_open_shards: int = 32,
                 verify_hashes: bool = True):
        self.registry_path = Path(registry_path)
        self.registry = json.loads(self.registry_path.read_text())
        if self.registry.get("schema_version") != "gate1b3_radius_edges_v1":
            raise ValueError("unsupported edge-cache schema")
        self.root = self.registry_path.parent.parent / self.registry["cache_root"]
        self.records = self.registry["records"]
        if len(self.records) != 15016 or len(set(self.records)) != 15016:
            raise ValueError("edge-cache record identity failure")
        self.shards = {item["file"]: item for item in self.registry["shards"]}
        self.max_open_shards = int(max_open_shards)
        self._loaded: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()
        if verify_hashes:
            for name, item in self.shards.items():
                path = self.root / name
                if not path.is_file() or sha256_file(path) != item["sha256"]:
                    raise ValueError(f"edge-cache shard hash failure: {name}")

    def edge_index(self, molecule_id: str) -> torch.Tensor:
        record = self.records[str(molecule_id)]
        name = record["shard"]
        if name not in self._loaded:
            payload = np.load(self.root / name, allow_pickle=False)
            self._loaded[name] = {key: payload[key] for key in payload.files}
            self._loaded.move_to_end(name)
            while len(self._loaded) > self.max_open_shards:
                self._loaded.popitem(last=False)
        arrays = self._loaded[name]
        start, end = int(record["edge_start"]), int(record["edge_end"])
        edge = np.stack((arrays["edge_src"][start:end], arrays["edge_dst"][start:end]))
        if edge.size and (edge.min() < 0 or edge.max() >= int(record["atom_count"])):
            raise ValueError(f"edge-cache atom-order failure: {molecule_id}")
        return torch.from_numpy(edge.astype(np.int64, copy=False)).contiguous()

