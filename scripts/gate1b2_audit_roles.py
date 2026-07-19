#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from excitationnexus_phase12.role_resolution import (
    ALLOWED_ROLES, file_sha256, load_sidecar_roles, load_structure_fields, resolve_empty_donor,
)

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")
RAW = Path("/home/changliu/ExcitationNexus_Data_v2/raw_compact")
TABLE = Path("/home/changliu/ExcitationNexus_Data_v2/tables/molecule_values_v3.parquet")
MANIFEST = ROOT / "manifests/split_iid_group_seed42_v1.csv"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def main() -> None:
    started = time.perf_counter()
    table = pd.read_parquet(TABLE, columns=[
        "molecule_id", "donor_smiles", "acceptor_smiles", "sidecar_conflict_flag",
    ])
    manifest = pd.read_csv(MANIFEST)
    frame = manifest.merge(table, on="molecule_id", validate="one_to_one")
    rows = []
    for row in frame.itertuples(index=False):
        mid = str(row.molecule_id)
        paths = {
            "pm6_json": RAW / "pm6/results" / mid / f"{mid}_pm6.json",
            "pm6_sidecar": RAW / "pm6/results" / mid / f"{mid}_sidecar.json",
            "dft_json": RAW / "dft/results" / mid / f"{mid}_dft.json",
            "dft_sidecar": RAW / "dft/results" / mid / f"{mid}_sidecar.json",
            "dft_pdb": RAW / "dft/results" / mid / f"{mid}_dft.pdb",
        }
        if not all(path.is_file() and path.stat().st_size > 0 for path in paths.values()):
            raise RuntimeError(f"missing role evidence for {mid}")
        pm6 = load_structure_fields(paths["pm6_json"]); dft = load_structure_fields(paths["dft_json"])
        pm6_roles = [atom["role"] for atom in pm6["atoms"]]; dft_roles = [atom["role"] for atom in dft["atoms"]]
        if any(role not in ALLOWED_ROLES for role in pm6_roles + dft_roles):
            raise RuntimeError(f"invalid role token for {mid}")
        pm6_sidecar = load_sidecar_roles(paths["pm6_sidecar"]); dft_sidecar = load_sidecar_roles(paths["dft_sidecar"])
        exact_sources = pm6_roles == dft_roles == pm6_sidecar == dft_sidecar
        counts = Counter(dft_roles)
        if not exact_sources:
            resolution = {"status": "UNRESOLVED_INCONSISTENT", "mapping_multiplicity": 0,
                          "distinct_role_sets": 0, "resolved_donor_indices": [],
                          "reason": "PM6/DFT/sidecar role vectors differ"}
        elif counts["donor"] == 0 and counts["unknown"] > 0:
            resolution = resolve_empty_donor(dft, str(row.donor_smiles))
        else:
            resolution = {"status": "NOT_APPLICABLE_ORIGINAL_EXPLICIT", "mapping_multiplicity": 1,
                          "distinct_role_sets": 1, "resolved_donor_indices": [],
                          "reason": "original role annotation already contains donor"}
        resolved_donor = len(resolution["resolved_donor_indices"]) if resolution["resolved_donor_indices"] else counts["donor"]
        resolved_unknown = counts["unknown"] - len(resolution["resolved_donor_indices"])
        rows.append({
            "molecule_id": mid, "partition": row.partition,
            "original_donor_count": counts["donor"], "original_acceptor_count": counts["acceptor"],
            "original_unknown_count": counts["unknown"], "resolved_donor_count": resolved_donor,
            "resolved_acceptor_count": counts["acceptor"], "resolved_unknown_count": resolved_unknown,
            "resolution_status": resolution["status"], "mapping_method": "component-formula-constrained full-graph VF2",
            "mapping_multiplicity": resolution["mapping_multiplicity"],
            "distinct_role_sets": resolution["distinct_role_sets"],
            "resolved_donor_atom_indices": ";".join(map(str, resolution["resolved_donor_indices"])),
            "resolution_reason": resolution["reason"],
            "evidence_sources": "PM6_JSON;DFT_JSON;PM6_SIDECAR;DFT_SIDECAR;DONOR_COMPONENT_SMILES;DFT_PDB",
            "pm6_dft_sidecar_roles_exact": exact_sources,
            "sidecar_conflict_flag": bool(row.sidecar_conflict_flag),
            "pm6_json_sha256": file_sha256(paths["pm6_json"]),
            "pm6_sidecar_sha256": file_sha256(paths["pm6_sidecar"]),
            "dft_json_sha256": file_sha256(paths["dft_json"]),
            "dft_sidecar_sha256": file_sha256(paths["dft_sidecar"]),
            "dft_pdb_sha256": file_sha256(paths["dft_pdb"]),
        })
    result = pd.DataFrame(rows).sort_values("molecule_id", kind="mergesort")
    output = ROOT / "manifests/role_resolution_v1.csv"; result.to_csv(output, index=False)
    classes = {
        "pure_DA": int(((result.original_donor_count > 0) & (result.original_unknown_count == 0)).sum()),
        "D_A_unknown": int(((result.original_donor_count > 0) & (result.original_unknown_count > 0)).sum()),
        "empty_donor_unknown": int(((result.original_donor_count == 0) & (result.original_unknown_count > 0)).sum()),
    }
    if classes != {"pure_DA": 14263, "D_A_unknown": 366, "empty_donor_unknown": 387}:
        raise RuntimeError(f"frozen role counts differ: {classes}")
    status = result[result.original_donor_count.eq(0)].resolution_status.value_counts().to_dict()
    spec = {
        "version": "v1", "created_utc": datetime.now(timezone.utc).isoformat(),
        "original_roles_are_primary": True, "resolved_roles_sensitivity_only": True,
        "unknown_never_inferred": True,
        "mapping": "donor non-placeholder neutral element formula constrains original-unknown induced element/bond graph; complete VF2 mappings into full heavy graph",
        "resolution_rule": "RESOLVED only if all symmetry mappings yield one atom-index set; second distinct set is ambiguous",
        "formal_charge_boundary": "component formal charge is checked; charged components are insufficient because DFT atom JSON lacks per-atom formal charge",
        "hydrogen_rule": "inherit donor only when bonded exclusively to resolved donor heavy atoms",
        "classes": classes, "empty_donor_status": status,
        "manifest_sha256": file_sha256(MANIFEST), "table_sha256": file_sha256(TABLE),
        "final673_accessed": False, "target_fields_read": False,
    }
    spec_path = ROOT / "data_registry/role_resolution_spec_v1.json"; write_json(spec_path, spec)
    by_partition = result.groupby(["partition", "resolution_status"]).size().unstack(fill_value=0).to_dict("index")
    report = ROOT / "reports/gate1b2_role_resolution_audit.md"
    report.write_text(
        "# Gate 1-B2 role resolution audit\n\n"
        f"All 15,016 records were audited without target access. Original classes reproduce 14,263 pure D/A, 366 D+A+unknown, and 387 empty-donor+unknown records.\n\n"
        f"Empty-donor resolution status: `{json.dumps(status, sort_keys=True)}`. Partition/status counts: `{json.dumps(by_partition, sort_keys=True)}`.\n\n"
        "Original explicit roles remain the primary analysis. A resolved role set, when uniquely supported, is sensitivity-only. Ambiguous/inconsistent/insufficient records remain explicit unknown and are never deleted or folded into donor. D81_A28 retains its conflict flag.\n"
    )
    print(json.dumps({"status": "ROLE_GOVERNANCE_COMPLETE", "classes": classes,
                      "empty_donor_status": status, "wall_seconds": time.perf_counter() - started}, indent=2))


if __name__ == "__main__":
    main()
