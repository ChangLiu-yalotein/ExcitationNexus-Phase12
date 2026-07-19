#!/usr/bin/env python3
"""Audit, build target-free features, and preregister Gate 1-B1."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")
TABLE = Path("/home/changliu/ExcitationNexus_Data_v2/tables/molecule_values_v3.parquet")
MANIFEST = ROOT / "manifests/split_iid_group_seed42_v1.csv"
TASK_GRAPH = ROOT / "data_registry/TARGET_TASK_GRAPH_V1.json"
FEATURE_CACHE = ROOT / "runs/gate1b1_new_iid_cheap_baselines/features_v1.parquet"
TABLE_SHA = "e7587b1546039f099a4dbd0d352e98885bb2ebdbdcfa18884dd4355eed815a83"
MANIFEST_SHA = "f4572f2c1896d4228dd9eff67220adb7d0a02ad79b70c66766e6da876541c3f2"

DESC_NAMES = [
    "MolWt", "MolLogP", "MolMR", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "NumAromaticHeterocycles", "NumAliphaticHeterocycles", "NumSaturatedRings",
    "NumHeteroatoms", "HeavyAtomCount", "NumValenceElectrons", "NHOHCount",
    "NOCount", "FractionCSP3", "RingCount", "HallKierAlpha",
]
C0_COLUMNS = [f"pair_{name}" for name in DESC_NAMES] + [f"pair_morgan_{i}" for i in range(512)]
SAFE_PM6_COLUMNS = ["pm6_homo_hartree", "pm6_lumo_hartree", "pm6_gap_ev"]
C15_COLUMNS = C0_COLUMNS + SAFE_PM6_COLUMNS


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def assert_firewall(columns: list[str]) -> None:
    forbidden_exact = {"pm6_energy_raw", "partition", "donor_id", "acceptor_id"}
    for column in columns:
        low = column.lower()
        if column in forbidden_exact or low.startswith(("tddft_", "multiwfn_", "target_")):
            raise ValueError(f"forbidden input field: {column}")
        if any(token in low for token in ("coulomb", "wavelength", "dipole", "final673", "split_")):
            raise ValueError(f"forbidden input field: {column}")


def build_features(table: pd.DataFrame) -> pd.DataFrame:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512, includeChirality=False)
    matrix = np.empty((len(table), len(C15_COLUMNS)), dtype=np.float32)
    for row_index, smiles in enumerate(table["canonical_smiles"].astype(str)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"RDKit failed for row {row_index}")
        matrix[row_index, :len(DESC_NAMES)] = [float(getattr(Descriptors, name)(mol)) for name in DESC_NAMES]
        fingerprint = generator.GetFingerprint(mol)
        bits = np.zeros((512,), dtype=np.int8)
        DataStructs.ConvertToNumpyArray(fingerprint, bits)
        matrix[row_index, len(DESC_NAMES):len(C0_COLUMNS)] = bits
    matrix[:, len(C0_COLUMNS):] = table[SAFE_PM6_COLUMNS].to_numpy(dtype=np.float32)
    result = pd.DataFrame(matrix, columns=C15_COLUMNS)
    result.insert(0, "molecule_id", table["molecule_id"].astype(str).to_numpy())
    return result


def historical_mapping() -> pd.DataFrame:
    old = json.loads((ROOT / "data_registry/gate1a1_feature_columns_v1.json").read_text())["columns"]
    rows = []
    for column in old:
        target, admitted, tier, reason, unit = "", False, "BLOCKED", "not mapped", "dimensionless"
        if column in C0_COLUMNS:
            target, admitted, tier, reason = column, True, "M1_C0_OPEN", "deterministically regenerated"
        elif column == "pm6_homo_hartree":
            target, admitted, tier, reason, unit = column, True, "M2_C1P5_SAFE", "resolved orbital", "hartree"
        elif column == "pm6_lumo_hartree":
            target, admitted, tier, reason, unit = column, True, "M2_C1P5_SAFE", "resolved orbital", "hartree"
        elif column == "pm6_homo_lumo_gap_ev":
            target, admitted, tier, reason, unit = "pm6_gap_ev", True, "M2_C1P5_SAFE", "resolved gap", "eV"
        elif column == "pm6_homo_lumo_gap_hartree":
            target, tier, reason, unit = "pm6_gap_hartree", "BLOCKED_REDUNDANT", "deterministic duplicate of gap_eV", "hartree"
        elif column == "pm6_pm6_energy_hartree":
            target, tier, reason, unit = "pm6_energy_raw", "BLOCKED_SEMANTICS", "PM6 energy semantics unresolved", "unresolved"
        elif column == "pm6_num_atoms":
            target, tier, reason, unit = "pm6_num_atoms_total", "BLOCKED_SCOPE", "M2 frozen to orbital information only", "count"
        elif column in {"pm6_normal_termination", "pm6_n_warnings", "pm6_missing_flag"}:
            target, tier, reason = "", "BLOCKED_CONTROL", "control/provenance or unnecessary complete-data flag"
        rows.append({
            "historical_column": column, "new_column": target, "admitted": admitted,
            "feature_tier": tier, "unit": unit, "reason": reason,
        })
    return pd.DataFrame(rows)


def main() -> None:
    started = time.perf_counter()
    if sha256(TABLE) != TABLE_SHA or sha256(MANIFEST) != MANIFEST_SHA:
        raise RuntimeError("frozen table or IID manifest hash mismatch")
    task_graph = json.loads(TASK_GRAPH.read_text())
    primary = task_graph["primary"]
    if len(primary) != 1:
        raise RuntimeError("primary target graph is not singular")
    primary = primary[0]
    assert_firewall(C15_COLUMNS)

    manifest = pd.read_csv(MANIFEST)
    expected = {"train": 10387, "val": 2309, "test": 2319, "historical_quarantine": 1}
    if manifest["partition"].value_counts().to_dict() != expected:
        raise RuntimeError("IID counts differ from frozen contract")
    if manifest["molecule_id"].nunique() != 15016:
        raise RuntimeError("manifest molecule identity failure")
    if manifest.groupby("structure_group_id_v1")["partition"].nunique().max() != 1:
        raise RuntimeError("structure group crosses partitions")
    historical_train = manifest[manifest["historical_status"].eq("HISTORICAL_TRAIN_OVERLAP")]
    if len(historical_train) != 17 or not historical_train["partition"].eq("train").all():
        raise RuntimeError("historical train-overlap boundary failure")
    if len(manifest[manifest["partition"].eq("historical_quarantine")]) != 1:
        raise RuntimeError("quarantine boundary failure")
    effective = manifest.groupby("partition")["group_weight"].sum().to_dict()
    expected_effective = {"historical_quarantine": 1.0, "test": 2195.0, "train": 10248.0, "val": 2195.0}
    if set(effective) != set(expected_effective) or any(
        not np.isclose(effective[key], expected_effective[key], rtol=0.0, atol=1e-9) for key in expected_effective
    ):
        raise RuntimeError(f"effective group counts differ: {effective}")

    columns = ["molecule_id", "canonical_smiles", *SAFE_PM6_COLUMNS,
               "pm6_num_atoms_total", "pm6_num_donor_atoms", "pm6_num_acceptor_atoms", "pm6_energy_raw"]
    table = pd.read_parquet(TABLE, columns=columns)
    joined = manifest[["molecule_id"]].merge(table, on="molecule_id", how="left", validate="one_to_one")
    if len(joined) != 15016 or joined["canonical_smiles"].isna().any():
        raise RuntimeError("table/manifest join failure")
    features = build_features(joined)
    FEATURE_CACHE.parent.mkdir(parents=True, exist_ok=True)
    features.to_parquet(FEATURE_CACHE, index=False)
    feature_sha = sha256(FEATURE_CACHE)
    # Input order must not affect the ID-bound feature representation.
    shuffled = joined.sample(frac=1, random_state=20260719).reset_index(drop=True)
    probe_ids = shuffled["molecule_id"].head(32).tolist()
    probe_a = features.set_index("molecule_id").loc[probe_ids]
    probe_b = build_features(shuffled.head(32)).set_index("molecule_id").loc[probe_ids]
    if not np.array_equal(probe_a.to_numpy(), probe_b.to_numpy(), equal_nan=True):
        raise RuntimeError("shuffled-row feature determinism failure")
    if not np.isfinite(features[C15_COLUMNS].to_numpy()).all():
        raise RuntimeError("safe feature cache contains missing/non-finite values")

    mapping = historical_mapping()
    mapping_path = ROOT / "data_registry/gate1b1_feature_mapping_v1.csv"
    mapping.to_csv(mapping_path, index=False)
    old_columns = json.loads((ROOT / "data_registry/gate1a1_feature_columns_v1.json").read_text())["columns"]
    claim_report = ROOT / "reports/gate1_historical_claim_reconciliation.md"
    claim_report.write_text(
        "# Gate 1 historical claim reconciliation\n\n"
        "Status: **SUPERSEDED_PENDING_RECOMPUTATION** for the legacy B2 percentage/statistical claims.\n\n"
        "- The rounded `0.0750 ± 0.0025 eV` summary is superseded by the asset-backed historical `0.0794090 ± 0.0025025 eV`.\n"
        "- The fixed Gate 1 reproductions give `0.0781230 ± 0.0013170 eV`.\n"
        "- Legacy `13.5% improvement`, p-values, Cohen's d, and `58% improved` claims require recomputation from explicitly identified prediction vectors.\n"
        "- The B2-2a comparison value `0.078015` is a different historical prediction convention and cannot be mixed with the seed42 original run MAE.\n"
        "- Historical checkpoints remain valid inference assets; new engineering baselines use the Gate 1-A2/A3 reproduction vectors.\n"
        "- The cheap-versus-B2-1 ensemble confidence interval crosses zero; no significant-superiority claim is authorized.\n"
    )
    audit_report = ROOT / "reports/gate1b1_feature_admission_audit.md"
    audit_report.write_text(
        "# Gate 1-B1 feature admission audit\n\n"
        f"All 15,016 records joined one-to-one. The target-free cache contains {len(C0_COLUMNS)} C0 features and "
        f"{len(C15_COLUMNS)} C1.5-safe features. Cache SHA-256: `{feature_sha}`.\n\n"
        "The old 541-column no-dipole model cannot be migrated verbatim: its PM6 energy semantics are unresolved, "
        "and atom count, termination, warning, missingness-control, and duplicate-unit gap fields are outside the new safe contract. "
        "Only HOMO, LUMO, and gap_eV are admitted. PM6 dipole and all DFT/TDDFT fields are excluded.\n\n"
        f"RDKit `{rdkit.__version__}`; Morgan radius 2, 512 bits, includeChirality=False. No descriptor parse or missing-value failure occurred.\n"
    )

    config = {
        "gate": "1-B1", "version": "v1", "status": "PREREGISTERED_BEFORE_TRAINING",
        "table": str(TABLE), "table_sha256": TABLE_SHA,
        "manifest": str(MANIFEST), "manifest_sha256": MANIFEST_SHA,
        "feature_cache": str(FEATURE_CACHE), "feature_cache_sha256": feature_sha,
        "primary_target": primary, "primary_report_name": task_graph["primary_report_name"],
        "split_counts": expected, "effective_group_counts": effective,
        "features": {
            "M1_C0_open": C0_COLUMNS, "M2_C1p5_safe_no_dipole": C15_COLUMNS,
            "morgan": {"radius": 2, "nBits": 512, "includeChirality": False},
            "pm6_safe": SAFE_PM6_COLUMNS, "old_feature_count": len(old_columns),
        },
        "models": ["weighted_median", "ridge_c0", "xgb_c0", "xgb_c1p5_safe"],
        "xgboost": {"n_estimators": 500, "max_depth": 6, "learning_rate": 0.05,
                     "tree_method": "hist", "device": "cuda", "verbosity": 0},
        "ridge": {"alpha": 1.0}, "seeds": [42, 123, 456],
        "preprocessing": "train-only group-weighted median imputation and weighted mean/std scaling",
        "sample_weight": "group_weight", "primary_validation_metric": "structure_group_macro_mae",
        "reported_metrics": ["record_mae", "group_macro_mae", "record_rmse", "group_macro_rmse", "record_r2", "group_macro_r2"],
        "test_unlock": "all 8 models complete; validation metrics/model hashes frozen; no test-guided change",
        "role_strata": ["pure_donor_acceptor", "donor_acceptor_unknown", "empty_donor_unknown"],
        "frequency_bins": "train-only donor/acceptor record frequencies: 1, 2-5, 6-20, >20",
        "target_bins": "train-only primary quartile thresholds, applied after test unlock",
        "forbidden_inputs": ["primary/equivalent Coulomb", "tddft_*", "multiwfn_*", "wavelength",
                             "dipole", "pm6_energy_raw", "DFT", "partition/split IDs", "final673"],
        "test_target_access_before_unlock": False, "final673_access": False,
        "versions": {"rdkit": rdkit.__version__, "xgboost": xgboost.__version__, "sklearn": sklearn.__version__},
    }
    config_path = ROOT / "configs/gate1b1_new_iid_cheap_baselines_v1.json"
    write_json(config_path, config)
    prereg_report = ROOT / "reports/gate1b1_preregistration.md"
    prereg_report.write_text(
        "# Gate 1-B1 preregistration\n\n"
        "Status: **FROZEN BEFORE TRAINING AND TEST-TARGET ACCESS**. The frozen IID manifest is bound by SHA-256, "
        "and its historical filename seed is not used as scientific metadata. All preprocessing is train-only and group-weighted. "
        "Validation primary metric is structure-group-macro MAE. Six fixed XGBoost runs, one Ridge-C0, and one weighted median "
        "must finish and freeze validation/model hashes before the test target is unlocked once. No hyperparameter search or test-guided rerun is allowed.\n"
    )
    locked = {str(path.relative_to(ROOT)): sha256(path) for path in (config_path, mapping_path, audit_report, prereg_report, claim_report)}
    aggregate = hashlib.sha256("".join(f"{name}:{value}\n" for name, value in sorted(locked.items())).encode()).hexdigest()
    write_json(ROOT / "data_registry/gate1b1_preregistration_lock_v1.json", {
        "gate": "1-B1", "version": "v1", "status": "FROZEN_BEFORE_TRAINING",
        "locked_utc": datetime.now(timezone.utc).isoformat(), "files": locked,
        "aggregate_sha256": aggregate, "post_lock_policy": "Corrections require explicit v2; never overwrite v1.",
    })
    print(json.dumps({
        "status": "FEATURE_CONTRACT_PASS", "records": len(features), "c0_features": len(C0_COLUMNS),
        "c1p5_features": len(C15_COLUMNS), "feature_cache_sha256": feature_sha,
        "preregistration_aggregate_sha256": aggregate, "wall_seconds": time.perf_counter() - started,
    }, indent=2))


if __name__ == "__main__":
    main()
