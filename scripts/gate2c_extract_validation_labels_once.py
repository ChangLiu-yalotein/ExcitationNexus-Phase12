#!/usr/bin/env python3
"""Authorized one-time extraction of validation labels only."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/gate2c_validation_extraction_v1.json"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    config = json.loads(CONFIG.read_text())
    table_path = resolve(config["table"])
    if sha256(table_path) != config["table_sha256"]:
        raise RuntimeError("BLOCKED_VALIDATION_EXTRACTION_INTEGRITY: table hash mismatch")
    output = resolve(config["output"])
    registry_path = ROOT / "data_registry/gate2c_validation_label_registry.json"
    if output.exists() or registry_path.exists():
        raise RuntimeError("validation labels were already extracted; second source-table read is fail-closed")
    validation_sets: dict[str, set[str]] = {}; test_sets: dict[str, set[str]] = {}
    manifest_evidence = {}
    for protocol, (relative, expected_hash, expected_count) in config["manifests"].items():
        path = resolve(relative)
        if sha256(path) != expected_hash:
            raise RuntimeError(f"BLOCKED_VALIDATION_EXTRACTION_INTEGRITY: {protocol} manifest hash")
        frame = pd.read_csv(path)
        val = set(frame.loc[frame.partition.eq("val"), "molecule_id"].astype(str))
        test = set(frame.loc[frame.partition.eq("test"), "molecule_id"].astype(str))
        if len(val) != expected_count or val & test:
            raise RuntimeError(f"BLOCKED_VALIDATION_EXTRACTION_INTEGRITY: {protocol} val/test boundary")
        if frame.molecule_id.nunique() != 15016:
            raise RuntimeError(f"BLOCKED_VALIDATION_EXTRACTION_INTEGRITY: {protocol} manifest coverage")
        validation_sets[protocol], test_sets[protocol] = val, test
        manifest_evidence[protocol] = {"sha256": expected_hash, "validation_count": len(val), "test_count": len(test), "protocol_local_val_test_overlap": 0}
    union_ids = sorted(set().union(*validation_sets.values()))
    global_test_union = set().union(*test_sets.values())
    # Cross-protocol reuse is recorded only as an aggregate; protocol-local overlap is zero.
    cross_protocol_overlap = len(set(union_ids) & global_test_union)
    dataset = ds.dataset(str(table_path), format="parquet")
    target = config["columns"][1]
    extracted = dataset.to_table(columns=config["columns"], filter=ds.field("molecule_id").isin(union_ids)).to_pandas()
    if list(extracted.columns) != config["columns"] or len(extracted) != len(union_ids):
        raise RuntimeError("BLOCKED_VALIDATION_EXTRACTION_INTEGRITY: union row/column mismatch")
    if extracted.molecule_id.nunique() != len(extracted) or extracted[target].isna().any() or not np.isfinite(extracted[target]).all():
        raise RuntimeError("BLOCKED_VALIDATION_EXTRACTION_INTEGRITY: duplicate/missing/nonfinite label")
    for protocol, ids in validation_sets.items():
        joined = pd.DataFrame({"molecule_id": sorted(ids)}).merge(extracted, on="molecule_id", validate="one_to_one")
        if len(joined) != len(ids):
            raise RuntimeError(f"BLOCKED_VALIDATION_EXTRACTION_INTEGRITY: {protocol} join")
    output.parent.mkdir(parents=True, exist_ok=True)
    extracted.sort_values("molecule_id").to_parquet(output, index=False)
    registry = {
        "status": "AUTHORIZED_MINIMAL_VALIDATION_LABEL_EXTRACTION_DONE",
        "authorization": "AUTHORIZED_MINIMAL_VALIDATION_LABEL_EXTRACTION",
        "blocker_resolved": "BLOCKED_MISSING_CALIBRATION_LABEL_ARTIFACT",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "source_table_sha256": config["table_sha256"], "source_columns_read": config["columns"], "arrow_reads": 1,
        "union_validation_count": len(union_ids), "per_protocol": manifest_evidence,
        "cross_protocol_validation_vs_test_union_overlap_aggregate": cross_protocol_overlap,
        "cross_protocol_note": "Allowed protocol reuse; every protocol has zero overlap between its own validation and test IDs.",
        "artifact_path": str(output), "artifact_sha256": sha256(output), "artifact_git_policy": "LOCAL_IGNORED",
        "test_rows_requested_as_test": 0, "buffer_rows_requested": 0, "quarantine_rows_requested": 0,
        "other_columns_read": 0, "final673_accessed": False,
    }
    write_json(registry_path, registry)
    print(json.dumps({k: registry[k] for k in ("status", "union_validation_count", "per_protocol", "artifact_sha256", "arrow_reads", "test_rows_requested_as_test", "final673_accessed")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
