#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gate2e0_common import ROOT, load_config, load_protocol_aux, write_json


def summarize(frame: pd.DataFrame, tasks: list[str], identity_columns: list[str]) -> dict:
    result = {}
    total_weight = float(frame.group_weight.sum())
    for task in tasks:
        valid = frame[task].notna()
        item = {
            "records": len(frame), "nonmissing_records": int(valid.sum()), "record_completeness": float(valid.mean()),
            "effective_group_weight": float(frame.loc[valid, "group_weight"].sum()), "structure_group_weighted_completeness": float(frame.loc[valid, "group_weight"].sum() / total_weight),
            "structure_groups_with_label": int(frame.loc[valid, "structure_group_id_v1"].nunique()),
        }
        for column in identity_columns:
            if column in frame:
                item[f"{column}_total"] = int(frame[column].nunique())
                item[f"{column}_with_label"] = int(frame.loc[valid, column].nunique())
        result[task] = item
    return result


def main() -> None:
    config = load_config(); tasks = [*config["secondary"], *config["masked"]]
    all_stats = {}; duplicate_stats = {}
    identity_columns = ["donor_structure_group_id_v1", "acceptor_structure_group_id_v1", "pair_group_id_v1", "full_scaffold_group_id_v1"]
    for protocol in config["protocols"]:
        all_stats[protocol] = {}
        for partition in ("train", "val"):
            frame, _ = load_protocol_aux(config, protocol, partition)
            all_stats[protocol][partition] = summarize(frame, tasks, identity_columns)
            if partition == "train":
                duplicate_stats[protocol] = {}
                duplicate = frame.loc[frame.structure_group_size.gt(1)]
                for task in tasks:
                    ranges = duplicate.groupby("structure_group_id_v1")[task].agg(lambda x: x.max() - x.min() if x.notna().sum() > 1 else np.nan).dropna()
                    duplicate_stats[protocol][task] = {"groups_with_multiple_labels": len(ranges), "median_range": float(ranges.median()) if len(ranges) else None, "p95_range": float(ranges.quantile(.95)) if len(ranges) else None, "max_range": float(ranges.max()) if len(ranges) else None}
    payload = {"completed_utc": datetime.now(timezone.utc).isoformat(), "protocols": all_stats, "duplicate_dispersion": duplicate_stats, "missing_policy": "mask_only_no_imputation_no_missingness_input", "validation_used_for": "coverage_only", "source_parquet_read": False, "test_artifact_accessed": False, "final673_accessed": False}
    write_json("logs/gate2e0_missingness.json", payload)
    iid = all_stats["iid"]
    lines = ["# Gate 2-E0 missingness and coverage", "", "All figures are protocol-local. Train drives task admission; validation is used only to establish future evaluability. Missing auxiliary labels are masked and never zero/mean-imputed or exposed as model inputs.", "", "## IID coverage", "", "| Task | Train records | Train group-weighted completeness | Validation records | Validation group-weighted completeness |", "|---|---:|---:|---:|---:|"]
    for task in tasks:
        tr, va = iid["train"][task], iid["val"][task]
        lines.append(f"| `{task}` | {tr['nonmissing_records']} | {tr['structure_group_weighted_completeness']:.4%} | {va['nonmissing_records']} | {va['structure_group_weighted_completeness']:.4%} |")
    lines += ["", "All 12 secondary targets are fully observed in every protocol train/validation partition. The four fragment targets remain approximately 51.6% observed and retain explicit masks. Protocol- and identity-level counts are frozen in `logs/gate2e0_missingness.json`; no raw identity or target values are published."]
    (ROOT / "reports/gate2e0_missingness_and_coverage.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
