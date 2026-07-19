#!/usr/bin/env python3
"""Read-only Gate 0-A dataset and historical-identity audit.

The script reads final-blind only as an ID column and never emits blind IDs,
SMILES, labels, rows, or per-sample blind membership. Public membership output
uses a redacted boolean for that split while aggregate overlap counts are kept.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, rdBase


PROJECT = Path("/home/changliu/ExcitationNexus")
DATA = Path("/home/changliu/ExcitationNexus_Data_v2")
OUT = PROJECT / "12_Phase4_Multitask_OOD_Training"
NEW_TABLE = DATA / "tables/molecule_values_v3.parquet"
OLD_TABLE = PROJECT / "05_Phase2_Baseline_Protocol/tables/teacher_table_7316_all.csv"
EXTERNAL_TABLE = PROJECT / "07_Phase2C_Smoothed_Memory/tables/external_dev_benchmark_2697.csv"
BLIND_TABLE = PROJECT / "07_Phase2C_Smoothed_Memory/tables/final_blind_test_674.csv"
LEGACY_DIR = PROJECT / "06_Phase2_External_Holdout/raw_3371/excitations"
REGISTRY = PROJECT / "DA_data/structure_60k_sorted.jsonl"
UPLOAD_MANIFEST = DATA / "manifests/closed_loop_compact_upload_manifest_v3.csv"


def normalized_id(value: str) -> str:
    match = re.fullmatch(r"D-?(\d+)_A-?(\d+)", str(value).strip())
    if not match:
        raise ValueError(f"Unrecognized molecule ID format: {value!r}")
    return f"D{int(match.group(1))}_A{int(match.group(2))}"


def canonical_hash(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def id_set_from_csv(path: Path) -> set[str]:
    # Intentionally load only molecule_id; this prevents blind-label access.
    values = pd.read_csv(path, usecols=["molecule_id"])["molecule_id"]
    return {normalized_id(value) for value in values}


def legacy_ids() -> set[str]:
    suffix = "_excitation.json"
    return {
        normalized_id(path.name[: -len(suffix)])
        for path in LEGACY_DIR.glob(f"*{suffix}")
    }


def registry_smiles(target_ids: set[str]) -> tuple[dict[str, str], int, int]:
    found: dict[str, str] = {}
    parsed = 0
    malformed = 0
    with REGISTRY.open("r", encoding="utf-8") as handle:
        for line in handle:
            parsed += 1
            try:
                row = json.loads(line)
                mid = normalized_id(row["id"])
            except (ValueError, KeyError, json.JSONDecodeError):
                malformed += 1
                continue
            if mid in target_ids and row.get("smiles"):
                found[mid] = row["smiles"]
    return found, parsed, malformed


def validate_upload_manifest() -> dict[str, object]:
    ids_by_fidelity: dict[str, set[str]] = {name: set() for name in ("pm6", "dft", "tddft")}
    missing = []
    empty = []
    size_mismatch = []
    unknown = 0
    rows = 0
    paths = set()
    duplicate_paths = 0
    with UPLOAD_MANIFEST.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows += 1
            fidelity = row["fidelity"]
            if fidelity not in ids_by_fidelity:
                unknown += 1
                continue
            ids_by_fidelity[fidelity].add(normalized_id(row["molecule_id"]))
            rel = row["proposed_relative_path"]
            if rel in paths:
                duplicate_paths += 1
            paths.add(rel)
            path = DATA / rel
            if not path.exists():
                missing.append(rel)
                continue
            size = path.stat().st_size
            expected = int(row["file_size"])
            if size == 0:
                empty.append(rel)
            if size != expected:
                size_mismatch.append(rel)
    directory_ids = {
        fidelity: {
            normalized_id(path.name)
            for path in (DATA / f"raw_compact/{fidelity}/results").iterdir()
            if path.is_dir()
        }
        for fidelity in ids_by_fidelity
    }
    common = ids_by_fidelity["pm6"] == ids_by_fidelity["dft"] == ids_by_fidelity["tddft"]
    directory_match = all(directory_ids[key] == ids_by_fidelity[key] for key in ids_by_fidelity)
    return {
        "manifest_rows": rows,
        "manifest_id_counts": {key: len(value) for key, value in ids_by_fidelity.items()},
        "directory_id_counts": {key: len(value) for key, value in directory_ids.items()},
        "fidelity_id_sets_equal": common,
        "directory_sets_match_manifest": directory_match,
        "unknown_fidelity_rows": unknown,
        "duplicate_target_paths": duplicate_paths,
        "missing_files": len(missing),
        "empty_files": len(empty),
        "size_mismatches": len(size_mismatch),
        "first_missing_paths": missing[:10],
        "first_empty_paths": empty[:10],
        "first_size_mismatch_paths": size_mismatch[:10],
    }


def main() -> None:
    new = pd.read_parquet(NEW_TABLE)
    new["normalized_id"] = new["molecule_id"].map(normalized_id)
    new_ids = set(new["normalized_id"])
    old_ids = id_set_from_csv(OLD_TABLE)
    external_ids = id_set_from_csv(EXTERNAL_TABLE)
    blind_ids = id_set_from_csv(BLIND_TABLE)
    legacy = legacy_ids()

    historical_union = old_ids | external_ids | blind_ids | legacy
    smiles_by_id, registry_rows, registry_malformed = registry_smiles(historical_union)
    historical_hashes: dict[str, set[str]] = {}
    hash_failures = 0
    for mid, smiles in smiles_by_id.items():
        digest = canonical_hash(smiles)
        if digest is None:
            hash_failures += 1
        else:
            historical_hashes.setdefault(digest, set()).add(mid)

    new_standard_hashes = new["canonical_smiles"].map(canonical_hash)
    new_hash_counts = new_standard_hashes.value_counts()
    new_hash_set = {value for value in new_standard_hashes if value is not None}
    old_hash_set = {digest for digest, ids in historical_hashes.items() if ids & old_ids}
    external_hash_set = {digest for digest, ids in historical_hashes.items() if ids & external_ids}
    blind_hash_set = {digest for digest, ids in historical_hashes.items() if ids & blind_ids}
    legacy_hash_set = {digest for digest, ids in historical_hashes.items() if ids & legacy}

    upload = validate_upload_manifest()
    parquet_sha = hashlib.sha256(NEW_TABLE.read_bytes()).hexdigest()
    csv_path = DATA / "tables/molecule_values_v3.csv"
    csv_rows = len(pd.read_csv(csv_path, usecols=["molecule_id"])) if csv_path.exists() else None
    conflict = new.loc[new["sidecar_conflict_flag"].fillna(False), "normalized_id"].tolist()

    completeness = {
        column: {
            "non_null": int(new[column].notna().sum()),
            "total": len(new),
            "fraction": float(new[column].notna().mean()),
        }
        for column in new.columns
        if column != "normalized_id"
    }
    integrity = {
        "parquet_rows": len(new),
        "csv_rows": csv_rows,
        "molecule_id_unique": bool(new["molecule_id"].is_unique),
        "normalized_id_unique": bool(new["normalized_id"].is_unique),
        "canonical_smiles_sha256_unique": bool(new["canonical_smiles_sha256"].is_unique),
        "canonical_smiles_rdkit_hash_unique": bool(new_standard_hashes.is_unique),
        "canonical_smiles_stored_hash_unique_count": int(new["canonical_smiles_sha256"].nunique()),
        "canonical_smiles_rdkit_hash_unique_count": int(new_standard_hashes.nunique()),
        "canonical_smiles_rdkit_duplicate_groups": int((new_hash_counts > 1).sum()),
        "canonical_smiles_rdkit_rows_in_duplicate_groups": int(new_hash_counts[new_hash_counts > 1].sum()),
        "canonical_smiles_rdkit_extra_rows": int((new_hash_counts - 1).clip(lower=0).sum()),
        "rdkit_canonicalization_failures": int(new_standard_hashes.isna().sum()),
        "sidecar_conflict_ids": conflict,
        "primary_non_null": int(new["tddft_coulomb_attraction_eV_eps3p5_proxy"].notna().sum()),
        "all_pm6_semantics_unresolved": bool(new["pm6_energy_semantics_unresolved"].all()),
        "upload_manifest": upload,
        "completeness": completeness,
    }

    historical_hash_by_id = {mid: digest for digest, ids in historical_hashes.items() for mid in ids}
    def directional_overlap_count(left, right, right_hashes):
        return sum(mid in right or historical_hash_by_id.get(mid) in right_hashes for mid in left)
    row_level_identity_overlap = {
        "new_old": int(sum(mid in old_ids or digest in old_hash_set for mid, digest in zip(new["normalized_id"], new_standard_hashes))),
        "new_external": int(sum(mid in external_ids or digest in external_hash_set for mid, digest in zip(new["normalized_id"], new_standard_hashes))),
        "new_final": int(sum(mid in blind_ids or digest in blind_hash_set for mid, digest in zip(new["normalized_id"], new_standard_hashes))),
        "new_legacy": int(sum(mid in legacy or digest in legacy_hash_set for mid, digest in zip(new["normalized_id"], new_standard_hashes))),
        "old_rows_matching_external": directional_overlap_count(old_ids, external_ids, external_hash_set),
        "external_rows_matching_old": directional_overlap_count(external_ids, old_ids, old_hash_set),
        "external_rows_matching_final": directional_overlap_count(external_ids, blind_ids, blind_hash_set),
        "final_rows_matching_external": directional_overlap_count(blind_ids, external_ids, external_hash_set),
        "legacy_rows_matching_external": directional_overlap_count(legacy, external_ids, external_hash_set),
        "external_rows_matching_legacy": directional_overlap_count(external_ids, legacy, legacy_hash_set),
        "legacy_rows_matching_final": directional_overlap_count(legacy, blind_ids, blind_hash_set),
        "final_rows_matching_legacy": directional_overlap_count(blind_ids, legacy, legacy_hash_set),
    }

    overlap = {
        "set_sizes": {
            "new15016": len(new_ids),
            "old7316": len(old_ids),
            "external2698": len(external_ids),
            "final673": len(blind_ids),
            "legacy3371": len(legacy),
        },
        "normalized_id_intersections": {
            "new_old": len(new_ids & old_ids),
            "new_external": len(new_ids & external_ids),
            "new_final": len(new_ids & blind_ids),
            "new_legacy": len(new_ids & legacy),
            "old_external": len(old_ids & external_ids),
            "external_final": len(external_ids & blind_ids),
            "legacy_external": len(legacy & external_ids),
            "legacy_final": len(legacy & blind_ids),
        },
        "rdkit_canonical_smiles_intersections": {
            "new_old": len(new_hash_set & old_hash_set),
            "new_external": len(new_hash_set & external_hash_set),
            "new_final": len(new_hash_set & blind_hash_set),
            "new_legacy": len(new_hash_set & legacy_hash_set),
            "old_external": len(old_hash_set & external_hash_set),
            "external_final": len(external_hash_set & blind_hash_set),
            "legacy_external": len(legacy_hash_set & external_hash_set),
            "legacy_final": len(legacy_hash_set & blind_hash_set),
        },
        "row_level_identity_overlap": row_level_identity_overlap,
        "identity_method": [
            "normalized molecule_id: D-<n>_A-<n> and D<n>_A<n> mapped to D<n>_A<n>",
            "RDKit canonical SMILES after Chem.RemoveHs, isomericSmiles=True, SHA-256",
        ],
        "rdkit_version": rdBase.rdkitVersion,
        "registry_rows_parsed": registry_rows,
        "registry_malformed_rows": registry_malformed,
        "historical_ids_with_registry_smiles": len(smiles_by_id),
        "historical_ids_without_registry_smiles": len(historical_union - set(smiles_by_id)),
        "historical_smiles_canonicalization_failures": hash_failures,
    }

    membership = pd.DataFrame(
        {
            "molecule_id": new["molecule_id"],
            "canonical_smiles_sha256": new["canonical_smiles_sha256"],
            "in_old7316": [mid in old_ids or digest in old_hash_set for mid, digest in zip(new["normalized_id"], new_standard_hashes)],
            "in_external2698": [mid in external_ids or digest in external_hash_set for mid, digest in zip(new["normalized_id"], new_standard_hashes)],
            # Redacted by design: aggregate final overlap exists in summary JSON.
            "in_final673": "REDACTED_SEALED_SET",
            "in_legacy3371": [mid in legacy or digest in legacy_hash_set for mid, digest in zip(new["normalized_id"], new_standard_hashes)],
        }
    )
    forbidden = []
    allowed = []
    for mid, digest in zip(new["normalized_id"], new_standard_hashes):
        reasons = []
        if mid in external_ids or digest in external_hash_set:
            reasons.append("external2698_model_selection")
        if mid in legacy or digest in legacy_hash_set:
            reasons.append("legacy3371_not_independent_training_asset")
        forbidden.append(";".join(reasons))
        # Never encode sample-level membership in the sealed final set. Rows
        # otherwise eligible still require an in-memory sealed-set exclusion
        # during split construction, so they are not marked True here.
        allowed.append(False if reasons else "SEALED_CHECK_REQUIRED")
    membership["allowed_for_new_training"] = allowed
    membership["forbidden_reason"] = forbidden

    OUT.joinpath("data_registry").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("manifests").mkdir(parents=True, exist_ok=True)
    OUT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    (OUT / "logs/gate0a_integrity.json").write_text(json.dumps(integrity, indent=2), encoding="utf-8")
    (OUT / "logs/gate0a_overlap.json").write_text(json.dumps(overlap, indent=2), encoding="utf-8")
    membership.to_csv(OUT / "manifests/new15016_historical_membership.csv", index=False)
    (OUT / "data_registry/final15016_id_list.txt").write_text(
        "\n".join(sorted(new_ids)) + "\n", encoding="utf-8"
    )
    registry_record = {
        "dataset_name": "ExcitationNexus final15016",
        "data_root": str(DATA),
        "table": str(NEW_TABLE),
        "parquet_sha256": parquet_sha,
        "rows": len(new),
        "identity_column": "molecule_id",
        "canonical_identity_column": "canonical_smiles_sha256",
        "generated_by": str(Path(__file__).resolve()),
        "rdkit_version": rdBase.rdkitVersion,
    }
    (OUT / "data_registry/final15016_dataset_registry.json").write_text(
        json.dumps(registry_record, indent=2), encoding="utf-8"
    )
    (OUT / "data_registry/final15016_sha256.txt").write_text(
        f"{parquet_sha}  {NEW_TABLE}\n", encoding="utf-8"
    )
    print(json.dumps({"integrity": integrity, "overlap": overlap}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"GATE0A_AUDIT_FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise
