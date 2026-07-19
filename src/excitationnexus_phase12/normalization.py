from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def weighted_stats(values, weights, *, min_count: int = 1, eps: float = 1e-8):
    x = np.asarray(values, dtype=float); w = np.asarray(weights, dtype=float)
    valid = np.isfinite(x) & np.isfinite(w) & (w > 0)
    x, w = x[valid], w[valid]
    if len(x) < min_count or w.sum() <= 0:
        return {"count": int(len(x)), "effective_weight": float(w.sum()),
                "mean": 0.0, "std": 1.0, "fallback": "INSUFFICIENT_VALID_DATA"}
    mean = float(np.sum(w * x) / np.sum(w))
    var = float(np.sum(w * (x - mean) ** 2) / np.sum(w))
    if not np.isfinite(var) or var <= eps * eps:
        return {"count": int(len(x)), "effective_weight": float(w.sum()),
                "mean": mean, "std": 1.0, "fallback": "ZERO_OR_TINY_STD"}
    return {"count": int(len(x)), "effective_weight": float(w.sum()),
            "mean": mean, "std": float(np.sqrt(var)), "fallback": None}


def fit_train_only_normalization(frame: pd.DataFrame, input_fields, target_fields,
                                 *, manifest_sha256: str, table_sha256: str) -> dict:
    train = frame.loc[frame.partition.eq("train")]
    if train.empty:
        raise ValueError("empty train partition")
    weights = train.group_weight.to_numpy(float)
    result = {
        "scope": "partition=train only", "weighting": "group_weight",
        "manifest_sha256": manifest_sha256, "table_sha256": table_sha256,
        "train_records": int(len(train)),
        "train_effective_structure_weight": float(weights.sum()),
        "inputs": {}, "targets": {},
    }
    for field in input_fields:
        result["inputs"][field] = weighted_stats(train[field], weights)
    for field in target_fields:
        result["targets"][field] = weighted_stats(train[field], weights)
    return result


def save_normalization(stats: dict, path: str | Path) -> None:
    Path(path).write_text(json.dumps(stats, indent=2, sort_keys=True) + "\n")


def normalize_tensor(value, stats):
    return (value - stats["mean"]) / stats["std"]


def inverse_tensor(value, stats):
    return value * stats["std"] + stats["mean"]
