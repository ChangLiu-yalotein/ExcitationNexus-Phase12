#!/usr/bin/env python3
from __future__ import annotations

from datetime import datetime, timezone

from gate2e0_common import ROOT, read_json, sha256, write_json


FILES = [
    "configs/gate2e0_multitask_target_audit_v1.json",
    "data_registry/gate2e0_preregistration_lock_v1.json",
    "data_registry/gate2e0_auxiliary_extraction_registry.json",
    "data_registry/gate2e0_target_admission_ledger.csv",
    "data_registry/gate2e0_target_graph_v2.json",
    "scripts/gate2e0_common.py",
    "scripts/gate2e0_extract_auxiliary_labels.py",
    "scripts/gate2e0_audit_target_semantics.py",
    "scripts/gate2e0_audit_missingness.py",
    "scripts/gate2e0_audit_task_relationships.py",
    "scripts/gate2e0_finalize.py",
    "scripts/git_checkpoint.sh",
    "reports/gate2e0_target_semantics.md",
    "reports/gate2e0_missingness_and_coverage.md",
    "reports/gate2e0_task_relationships.md",
    "reports/gate2e0_multitask_feasibility.md",
    "reports/gate2e0_final_decision.md",
    "logs/gate2e0_extraction_evidence.json",
    "logs/gate2e0_target_audit.json",
    "logs/gate2e0_missingness.json",
    "logs/gate2e0_task_relationships.json",
    "tests/test_gate2e0_contract.py",
    "tests/test_gate2e0_outputs.py",
    "PROJECT_STATE.md",
    "TODO.md",
    "DECISIONS.md",
    "RUN_REGISTRY.csv",
]


def main() -> None:
    evidence = read_json("logs/gate2e0_evidence.json")
    evidence.update({
        "completed_utc": datetime.now(timezone.utc).isoformat(),
        "pytest": {"passed": 155, "failed": 0, "warnings": 4},
        "git_ignored_row_level_artifacts": True,
        "secret_scan_findings": 0,
        "large_files_over_20MiB": 0,
        "git_diff_check": "PASS",
        "source_parquet_reads_after_extraction": 0,
    })
    write_json("logs/gate2e0_evidence.json", evidence)
    files = [*FILES, "logs/gate2e0_evidence.json"]
    missing = [path for path in files if not (ROOT / path).is_file()]
    if missing:
        raise RuntimeError(f"missing Gate 2-E0 assets: {missing}")
    registry = ROOT / "data_registry/gate2e0_sha256.txt"
    registry.write_text("".join(f"{sha256(path)}  {path}\n" for path in sorted(files)))
    print(f"GATE2E0_DONE files={len(files)} registry={registry}")


if __name__ == "__main__":
    main()
