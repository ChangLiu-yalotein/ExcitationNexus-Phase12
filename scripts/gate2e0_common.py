#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with resolve(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_json_sha(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: str | Path) -> dict:
    return json.loads(resolve(path).read_text())


def write_json(path: str | Path, value: object) -> None:
    output = resolve(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    return float(np.sum(values * weights) / np.sum(weights))


def weighted_corr(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    if len(x) < 2 or np.sum(w) <= 0:
        return float("nan")
    mx, my = weighted_mean(x, w), weighted_mean(y, w)
    vx = np.sum(w * (x - mx) ** 2)
    vy = np.sum(w * (y - my) ** 2)
    if vx <= 0 or vy <= 0:
        return float("nan")
    return float(np.sum(w * (x - mx) * (y - my)) / np.sqrt(vx * vy))


def weighted_spearman(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    return weighted_corr(rankdata(x), rankdata(y), w)


def standardized_linear_residual_rmse(x: np.ndarray, y: np.ndarray, w: np.ndarray) -> float:
    mx, my = weighted_mean(x, w), weighted_mean(y, w)
    sx = np.sqrt(weighted_mean((x - mx) ** 2, w))
    sy = np.sqrt(weighted_mean((y - my) ** 2, w))
    if sx == 0 or sy == 0:
        return float("nan")
    zx, zy = (x - mx) / sx, (y - my) / sy
    design = np.column_stack([np.ones(len(zx)), zx])
    root_w = np.sqrt(w)
    beta = np.linalg.lstsq(design * root_w[:, None], zy * root_w, rcond=None)[0]
    residual = zy - design @ beta
    return float(np.sqrt(weighted_mean(residual**2, w)))


def load_config() -> dict:
    return read_json("configs/gate2e0_multitask_target_audit_v1.json")


def load_protocol_aux(config: dict, protocol: str, partition: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    spec = config["protocols"][protocol]
    manifest = pd.read_csv(resolve(spec["manifest"]))
    if sha256(spec["manifest"]) != spec["sha256"]:
        raise RuntimeError(f"manifest hash mismatch: {protocol}")
    selected = manifest.loc[manifest.partition.eq(partition)].copy()
    path = resolve(config["local_output_directory"]) / f"{protocol}_{partition}_aux_labels.parquet"
    labels = pd.read_parquet(path)
    if not labels.molecule_id.is_unique or set(labels.molecule_id) != set(selected.molecule_id):
        raise RuntimeError(f"protocol-local auxiliary label mismatch: {protocol}/{partition}")
    joined = selected.merge(labels, on="molecule_id", validate="one_to_one")
    return joined, manifest


def load_primary_labels(config: dict) -> pd.DataFrame:
    target = config["primary"]
    pieces = []
    for item in (config["primary_artifacts"]["iid_train_val"], config["primary_artifacts"]["validation_union"]):
        if sha256(item["path"]) != item["sha256"]:
            raise RuntimeError("frozen primary artifact hash mismatch")
        frame = pd.read_parquet(resolve(item["path"]))
        if "target" in frame.columns:
            frame = frame.rename(columns={"target": target})
        pieces.append(frame[["molecule_id", target]])
    registry = read_json(config["primary_artifacts"]["train_supplement_registry"])
    for item in registry["protocols"].values():
        if sha256(item["artifact_path"]) != item["artifact_sha256"]:
            raise RuntimeError("frozen train supplement hash mismatch")
        pieces.append(pd.read_parquet(resolve(item["artifact_path"]))[["molecule_id", target]])
    full = pd.concat(pieces, ignore_index=True)
    spread = full.groupby("molecule_id")[target].agg(["min", "max"])
    if ((spread["max"] - spread["min"]).abs() > 1e-12).any():
        raise RuntimeError("inconsistent frozen primary labels")
    return full.drop_duplicates("molecule_id")
