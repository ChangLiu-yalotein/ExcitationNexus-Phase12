#!/usr/bin/env python3
"""Read-only audit of the frozen historical B2-1 seed42 assets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from ase.db import connect
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ordered_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def db_ids(path: Path) -> list[str]:
    return [str(getattr(row, "name", getattr(row, "sid", row.id))) for row in connect(path).select()]


def metrics(frame: pd.DataFrame) -> dict[str, float | int]:
    truth = frame["target_energy"].to_numpy(dtype=np.float64)
    pred = frame["pred_energy"].to_numpy(dtype=np.float64)
    return {
        "mae": float(mean_absolute_error(truth, pred)),
        "rmse": float(np.sqrt(mean_squared_error(truth, pred))),
        "r2": float(r2_score(truth, pred)),
        "n_records": int(len(frame)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    assets = {name: Path(path) for name, path in config["historical_assets"].items()}

    observed = {name: sha256(assets[name]) for name in config["expected_sha256"]}
    expected = config["expected_sha256"]
    if observed != expected:
        raise RuntimeError(f"historical hash mismatch: {observed}")

    ids = {part: db_ids(assets[f"{part}_db"]) for part in ("train", "val", "test")}
    counts = {part: len(values) for part, values in ids.items()}
    if counts != {"train": 5120, "val": 1098, "test": 1098}:
        raise RuntimeError(f"historical DB count mismatch: {counts}")
    if any(len(values) != len(set(values)) for values in ids.values()):
        raise RuntimeError("duplicate SID inside a historical DB partition")
    if any(set(ids[a]) & set(ids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
        raise RuntimeError("historical DB partitions overlap")
    if len(set(sum(ids.values(), []))) != 7316:
        raise RuntimeError("historical B2-1 protocol does not cover 7,316 unique SIDs")

    prediction = pd.read_csv(assets["historical_predictions"])
    if prediction["sid"].astype(str).tolist() != ids["test"]:
        raise RuntimeError("historical prediction order differs from test DB order")
    observed_metrics = metrics(prediction)
    if abs(observed_metrics["mae"] - config["historical_result"]["sid_vector_recomputed_test_mae"]) > 1e-12:
        raise RuntimeError(f"historical SID-vector metric mismatch: {observed_metrics}")
    without_sid = pd.read_csv(assets["historical_predictions_without_sid"])
    without_sid_metrics = metrics(without_sid)
    if abs(without_sid_metrics["mae"] - config["historical_result"]["test_mae"]) > 1e-7:
        raise RuntimeError(f"historical original-vector metric mismatch: {without_sid_metrics}")

    layer_g = json.loads(assets["layer_g_manifest"].read_text())
    layer_counts = {part: len(layer_g[part]) for part in ("train", "val", "test")}
    if layer_counts != {"train": 5118, "val": 1098, "test": 1097}:
        raise RuntimeError(f"Layer G count mismatch: {layer_counts}")
    set_db = {part: set(values) for part, values in ids.items()}
    set_g = {part: set(layer_g[part]) for part in layer_counts}
    db_only = {part: sorted(set_db[part] - set_g[part]) for part in layer_counts}
    g_only = {part: sorted(set_g[part] - set_db[part]) for part in layer_counts}

    evidence = {
        "status": "ASSET_AUDIT_PASS",
        "observed_sha256": observed,
        "historical_db_counts": counts,
        "historical_db_order_sha256": {part: ordered_hash(values) for part, values in ids.items()},
        "historical_db_unique_total": 7316,
        "layer_g_counts": layer_counts,
        "b21_db_minus_layer_g": db_only,
        "layer_g_minus_b21_db": g_only,
        "historical_prediction_order_matches_test_db": True,
        "historical_sid_vector_metrics": observed_metrics,
        "historical_original_vector_metrics": without_sid_metrics,
        "historical_vector_max_abs_prediction_delta": float(
            np.max(np.abs(prediction["pred_energy"].to_numpy() - without_sid["pred_energy"].to_numpy()))
        ),
        "historical_vector_policy": "Original 09:11 metrics are the run-level result; the later SID vector is the paired-comparison asset. Both are retained.",
        "parameter_count_from_checkpoint": 1075318,
        "checkpoint_selection": {
            "artifact": "best_checkpoint.pt",
            "historical_val_metric": 0.07783559350755767,
            "historical_val_batches": 138,
            "known_limitation": "B2_1Trainer.validate averages batch means equally; the final two-record batch is overweighted.",
            "test_used_for_checkpoint_selection": False,
        },
        "formal_protocol_note": "Strict B2-1 reproduction uses the original 7316 ASE DBs. Cheap pairing uses the 1097-SID intersection only.",
        "final673_accessed": False,
        "new15016_accessed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
