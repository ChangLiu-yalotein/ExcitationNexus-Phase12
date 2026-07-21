#!/usr/bin/env python3
"""One authorized 17-column Arrow read, immediately split into protocol-local artifacts."""
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import pyarrow.dataset as ds

from gate2e0_common import ROOT, canonical_json_sha, load_config, resolve, sha256, write_json


def main() -> None:
    config = load_config()
    registry_path = resolve("data_registry/gate2e0_auxiliary_extraction_registry.json")
    output_dir = resolve(config["local_output_directory"])
    if registry_path.exists() or output_dir.exists():
        raise RuntimeError("Gate 2-E0 extraction already attempted/frozen; refusing a second source read")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != config["expected_head"]:
        raise RuntimeError("Git HEAD boundary mismatch")
    if sha256(config["source_table"]) != config["source_table_sha256"]:
        raise RuntimeError("source table hash mismatch")

    manifests = {}
    union_ids: set[str] = set()
    for protocol, spec in config["protocols"].items():
        if sha256(spec["manifest"]) != spec["sha256"]:
            raise RuntimeError(f"manifest hash mismatch: {protocol}")
        manifest = pd.read_csv(resolve(spec["manifest"]))
        if len(manifest) != 15016 or not manifest.molecule_id.is_unique:
            raise RuntimeError(f"manifest identity integrity failed: {protocol}")
        for partition in ("train", "val"):
            expected = spec[partition]
            if int(manifest.partition.eq(partition).sum()) != expected:
                raise RuntimeError(f"partition count mismatch: {protocol}/{partition}")
        manifests[protocol] = manifest
        union_ids.update(manifest.loc[manifest.partition.isin(["train", "val"]), "molecule_id"])
    if len(union_ids) != config["expected_train_validation_union"]:
        raise RuntimeError("train/validation union count mismatch")

    columns = ["molecule_id", *config["secondary"], *config["masked"]]
    table = ds.dataset(config["source_table"], format="parquet").to_table(
        columns=columns, filter=ds.field("molecule_id").isin(sorted(union_ids))
    ).to_pandas()
    if len(table) != len(union_ids) or not table.molecule_id.is_unique or set(table.molecule_id) != union_ids:
        raise RuntimeError("Arrow extraction identity integrity failed")
    if any(not np.isfinite(table[col].dropna().to_numpy(float)).all() for col in columns[1:]):
        raise RuntimeError("non-finite auxiliary label")

    output_dir.mkdir(parents=True)
    evidence = {
        "status": "AUTHORIZED_MINIMAL_AUXILIARY_LABEL_EXTRACTION_DONE",
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "arrow_reads": 1,
        "source_table_sha256": config["source_table_sha256"],
        "source_columns": columns,
        "primary_column_read": False,
        "union_count": len(union_ids),
        "generic_union_file_written": False,
        "protocols": {},
        "test_artifact_accessed": False,
        "buffer_rows_written": 0,
        "quarantine_rows_written": 0,
        "final673_accessed": False,
    }
    for protocol, manifest in manifests.items():
        evidence["protocols"][protocol] = {}
        forbidden = set(manifest.loc[~manifest.partition.isin(["train", "val"]), "molecule_id"])
        for partition in ("train", "val"):
            ids = set(manifest.loc[manifest.partition.eq(partition), "molecule_id"])
            block = table.loc[table.molecule_id.isin(ids), columns].sort_values("molecule_id", kind="mergesort").reset_index(drop=True)
            if len(block) != len(ids) or not block.molecule_id.is_unique or set(block.molecule_id) & forbidden:
                raise RuntimeError(f"protocol-local leakage/integrity failure: {protocol}/{partition}")
            path = output_dir / f"{protocol}_{partition}_aux_labels.parquet"
            block.to_parquet(path, index=False)
            check = block.sample(frac=1, random_state=config["statistics"]["row_order_seed"]).sort_values("molecule_id", kind="mergesort").reset_index(drop=True)
            if canonical_json_sha(check.to_dict(orient="list")) != canonical_json_sha(block.to_dict(orient="list")):
                raise RuntimeError("row-order invariance failed")
            evidence["protocols"][protocol][partition] = {
                "count": len(block),
                "artifact_path": str(path.relative_to(ROOT)),
                "artifact_sha256": sha256(path),
                "missing_by_column": {col: int(block[col].isna().sum()) for col in columns[1:]},
                "finite_nonmissing_by_column": {col: int(block[col].notna().sum()) for col in columns[1:]},
                "duplicates": 0,
                "forbidden_partition_overlap": 0,
            }
    write_json(registry_path, evidence)
    write_json("logs/gate2e0_extraction_evidence.json", evidence)
    print(json.dumps({"status": evidence["status"], "arrow_reads": 1, "union_count": len(union_ids), "protocol_counts": {p: {k: v["count"] for k, v in q.items()} for p, q in evidence["protocols"].items()}}, indent=2))


if __name__ == "__main__":
    main()
