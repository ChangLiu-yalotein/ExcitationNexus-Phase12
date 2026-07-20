#!/usr/bin/env python3
"""Train five frozen OOD cheap baselines without reading OOD test targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

try:
    from scripts.gate1b1_train_cheap_baselines import fit_preprocessor, record_group_metrics, save_preprocessor, sha256, transform, weighted_median
except ModuleNotFoundError:  # direct execution from scripts/
    from gate1b1_train_cheap_baselines import fit_preprocessor, record_group_metrics, save_preprocessor, sha256, transform, weighted_median

ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def load_contract(config_path: Path) -> tuple[dict, dict, pd.DataFrame]:
    config = read_json(config_path)
    if sha256(resolve(config["table"])) != config["table_sha256"]:
        raise RuntimeError("frozen table hash mismatch")
    if sha256(resolve(config["feature_cache"])) != config["feature_cache_sha256"]:
        raise RuntimeError("frozen feature cache hash mismatch")
    source_path = resolve(config["feature_source_config"])
    source = read_json(source_path)
    features = pd.read_parquet(resolve(config["feature_cache"]))
    for label, source_key in config["training_features"]["source_keys"].items():
        columns = source["features"][source_key]
        if len(columns) != config["training_features"]["counts"][label]:
            raise RuntimeError(f"{label} feature count mismatch")
        if not set(columns).issubset(features.columns):
            raise RuntimeError(f"{label} feature cache columns missing")
    if source["features"]["morgan"] != config["training_features"]["morgan"]:
        raise RuntimeError("512-bit training fingerprint contract mismatch")
    return config, source, features


def validate_manifest(name: str, spec: dict) -> pd.DataFrame:
    path = resolve(spec["manifest"])
    if sha256(path) != spec["sha256"]:
        raise RuntimeError(f"{name} manifest hash mismatch")
    frame = pd.read_csv(path)
    if len(frame) != 15016 or frame["molecule_id"].nunique() != 15016:
        raise RuntimeError(f"{name} coverage mismatch")
    if frame["partition"].value_counts().to_dict() != spec["records"]:
        raise RuntimeError(f"{name} record counts mismatch")
    if frame.groupby("structure_group_id_v1")["partition"].nunique().max() != 1:
        raise RuntimeError(f"{name} structure leakage")
    for partition, expected in spec["groups"].items():
        actual = float(frame.loc[frame.partition.eq(partition), "group_weight"].sum())
        if not np.isclose(actual, expected, atol=1e-8):
            raise RuntimeError(f"{name} {partition} effective-group mismatch")
    quarantine = frame[frame.historical_status.eq("HISTORICAL_MODEL_SELECTION_QUARANTINE")]
    overlap = frame[frame.historical_status.eq("HISTORICAL_TRAIN_OVERLAP")]
    if len(quarantine) != 1 or not quarantine.partition.eq("historical_quarantine").all():
        raise RuntimeError(f"{name} quarantine failure")
    if len(overlap) != 17 or not overlap.partition.eq("train").all():
        raise RuntimeError(f"{name} historical train overlap failure")
    model = frame[frame.partition.isin(["train", "val", "test"])]
    identity = {
        "donor_cold": "donor_structure_group_id_v1",
        "acceptor_cold": "acceptor_structure_group_id_v1",
        "pair_cold": "pair_group_id_v1",
        "full_scaffold_cold": "full_scaffold_group_id_v1",
    }.get(name)
    if identity:
        sets = {p: set(model.loc[model.partition.eq(p), identity]) for p in ("train", "val", "test")}
        if any(sets[a] & sets[b] for a, b in (("train", "val"), ("train", "test"), ("val", "test"))):
            raise RuntimeError(f"{name} identity leakage")
    if name == "both_cold":
        core = model[model.partition.isin(["train", "val"])]
        test = model[model.partition.eq("test")]
        for column in ("donor_structure_group_id_v1", "acceptor_structure_group_id_v1"):
            if set(core[column]) & set(test[column]):
                raise RuntimeError(f"both_cold {column} leakage")
        if not (frame.partition.eq("buffer").sum() == 3291):
            raise RuntimeError("both_cold buffer boundary mismatch")
    return frame


def read_train_val_targets(config: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    # Firewall scope is protocol-specific: labels are joined only to this protocol's train/val rows.
    allowed = sorted(manifest.loc[manifest.partition.isin(["train", "val"]), "molecule_id"].astype(str))
    dataset = ds.dataset(str(resolve(config["table"])), format="parquet")
    table = dataset.to_table(columns=["molecule_id", config["primary_target"]], filter=ds.field("molecule_id").isin(allowed)).to_pandas()
    if len(table) != len(allowed) or table[config["primary_target"]].isna().any():
        raise RuntimeError("protocol-scoped train/val target read mismatch")
    return table


def diagnostic_mol(smiles: str):
    """Gate 0-C frozen parser for non-kekulizable attachment fragments."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    return mol if Chem.SanitizeMol(mol, sanitizeOps=ops, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE else None


def nearest_similarity(test_smiles: dict[str, str], train_smiles: dict[str, str], generator) -> dict[str, float]:
    train_fps = []
    for smiles in sorted(set(train_smiles.values())):
        mol = diagnostic_mol(smiles)
        if mol is None:
            raise RuntimeError(f"fingerprint parse failed: {smiles[:80]}")
        train_fps.append(generator.GetFingerprint(mol))
    output = {}
    cache = {}
    for key, smiles in sorted(test_smiles.items()):
        if smiles not in cache:
            mol = diagnostic_mol(smiles)
            if mol is None:
                raise RuntimeError(f"fingerprint parse failed: {smiles[:80]}")
            fp = generator.GetFingerprint(mol)
            cache[smiles] = float(max(DataStructs.BulkTanimotoSimilarity(fp, train_fps)))
        output[key] = cache[smiles]
    return output


def build_similarity(config: dict, manifests: dict[str, pd.DataFrame], output: Path) -> dict:
    if output.exists():
        raise RuntimeError("OOD similarity asset already exists; refusing overwrite")
    RDLogger.DisableLog("rdApp.*")
    structures = pd.read_parquet(ROOT / "manifests/new15016_structure_groups_v1.parquet")
    components = pd.read_csv(ROOT / "manifests/component_identity_v1.csv")
    identity = structures[["molecule_id", "canonical_structure_smiles_v1"]].merge(
        components[["molecule_id", "donor_canonical_structure_smiles_v1", "acceptor_canonical_structure_smiles_v1"]],
        on="molecule_id", validate="one_to_one")
    by_id = identity.set_index("molecule_id")
    fp = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048, includeChirality=True)
    rows = []
    for split_name, manifest in manifests.items():
        train_ids = manifest.loc[manifest.partition.eq("train"), "molecule_id"].astype(str).tolist()
        test_ids = manifest.loc[manifest.partition.eq("test"), "molecule_id"].astype(str).tolist()
        views = {
            "full": "canonical_structure_smiles_v1",
            "donor": "donor_canonical_structure_smiles_v1",
            "acceptor": "acceptor_canonical_structure_smiles_v1",
        }
        scores = {}
        for view, column in views.items():
            scores[view] = nearest_similarity(by_id.loc[test_ids, column].to_dict(), by_id.loc[train_ids, column].to_dict(), fp)
        for molecule_id in test_ids:
            rows.append({"split_name": split_name, "molecule_id": molecule_id,
                         "nearest_train_full_morgan2048_chiral": scores["full"][molecule_id],
                         "nearest_train_donor_morgan2048_chiral": scores["donor"][molecule_id],
                         "nearest_train_acceptor_morgan2048_chiral": scores["acceptor"][molecule_id]})
    result = pd.DataFrame(rows).sort_values(["split_name", "molecule_id"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    return {"path": str(output), "sha256": sha256(output), "rows": len(result), "rdkit": rdBase.rdkitVersion,
            "fingerprint": config["ood_diagnostic_fingerprint"], "target_columns": []}


def train_all(config_path: Path, run_root: Path) -> dict:
    if run_root.exists():
        raise RuntimeError(f"run root exists: {run_root}")
    config, source, features = load_contract(config_path)
    manifests = {name: validate_manifest(name, spec) for name, spec in config["splits"].items()}
    run_root.mkdir(parents=True)
    similarity_path = ROOT / "data_registry/gate2a_ood_similarity_v1.parquet"
    if similarity_path.exists():
        similarity_frame = pd.read_parquet(similarity_path)
        expected_rows = sum(spec["records"]["test"] for spec in config["splits"].values())
        if len(similarity_frame) != expected_rows or similarity_frame.isna().any().any():
            raise RuntimeError("frozen target-free OOD similarity asset is incomplete")
        similarity_meta = {"path": str(similarity_path), "sha256": sha256(similarity_path), "rows": len(similarity_frame),
                           "rdkit": rdBase.rdkitVersion, "fingerprint": config["ood_diagnostic_fingerprint"], "target_columns": []}
    else:
        similarity_meta = build_similarity(config, manifests, similarity_path)
    target = config["primary_target"]
    c0 = source["features"][config["training_features"]["source_keys"]["C0"]]
    c15 = source["features"][config["training_features"]["source_keys"]["C1p5_safe"]]
    registry = {"gate": "2-A", "status": "VALIDATION_AND_20_BASELINE_ARTIFACTS_FROZEN", "models": {},
                "preprocessors": {}, "validation_predictions": {}, "similarity": similarity_meta,
                "config_sha256": sha256(config_path), "table_sha256": config["table_sha256"],
                "feature_cache_sha256": config["feature_cache_sha256"], "test_target_accessed": False,
                "created_utc": datetime.now(timezone.utc).isoformat()}
    for split_name, manifest in manifests.items():
        split_dir = run_root / split_name
        split_dir.mkdir()
        targets = read_train_val_targets(config, manifest)
        frame = manifest.merge(features, on="molecule_id", validate="one_to_one").merge(targets, on="molecule_id", how="left", validate="one_to_one")
        if frame.loc[frame.partition.eq("test"), target].notna().any() or frame.loc[frame.partition.isin(["buffer", "historical_quarantine"]), target].notna().any():
            raise RuntimeError("test/buffer/quarantine target entered training frame")
        train, val = frame[frame.partition.eq("train")], frame[frame.partition.eq("val")]
        weights = train.group_weight.to_numpy(np.float64)
        y_train, y_val = train[target].to_numpy(np.float64), val[target].to_numpy(np.float64)
        matrices = {}
        for label, columns in (("c0", c0), ("c1p5_safe", c15)):
            prep = fit_preprocessor(train[columns].to_numpy(np.float64), weights)
            prep_path = split_dir / f"preprocessor_{label}.npz"
            save_preprocessor(prep_path, prep, columns)
            registry["preprocessors"][f"{split_name}/{label}"] = {"path": str(prep_path), "sha256": sha256(prep_path), "fit_partition": "train", "columns": len(columns)}
            matrices[label] = {"train": transform(train[columns].to_numpy(np.float64), prep), "val": transform(val[columns].to_numpy(np.float64), prep)}
        median = weighted_median(y_train, weights)
        median_path = split_dir / "weighted_median.json"
        write_json(median_path, {"value": median, "validation": record_group_metrics(y_val, np.full(len(val), median), val.structure_group_id_v1.to_numpy()), "test_target_accessed": False})
        registry["models"][f"{split_name}/weighted_median"] = {"path": str(median_path), "sha256": sha256(median_path), "kind": "weighted_median"}
        ridge = Ridge(alpha=1.0).fit(matrices["c0"]["train"], y_train, sample_weight=weights)
        ridge_path = split_dir / "ridge_c0.npz"
        np.savez_compressed(ridge_path, coef=ridge.coef_, intercept=np.asarray([ridge.intercept_]))
        ridge_pred = ridge.predict(matrices["c0"]["val"])
        registry["models"][f"{split_name}/ridge_c0"] = {"path": str(ridge_path), "sha256": sha256(ridge_path), "kind": "ridge_c0", "validation": record_group_metrics(y_val, ridge_pred, val.structure_group_id_v1.to_numpy())}
        pd.DataFrame({"molecule_id": val.molecule_id, "prediction": ridge_pred}).to_csv(split_dir / "ridge_c0_val_predictions.csv", index=False)
        for label, matrix_key in (("xgb_c0", "c0"), ("xgb_c1p5_safe", "c1p5_safe")):
            model = XGBRegressor(**config["xgboost"])
            start = time.perf_counter(); model.fit(matrices[matrix_key]["train"], y_train, sample_weight=weights); seconds = time.perf_counter() - start
            pred = model.predict(matrices[matrix_key]["val"])
            model_path = split_dir / f"{label}.json"; model.save_model(model_path)
            pred_path = split_dir / f"{label}_val_predictions.csv"
            pd.DataFrame({"molecule_id": val.molecule_id, "prediction": pred}).to_csv(pred_path, index=False)
            registry["models"][f"{split_name}/{label}"] = {"path": str(model_path), "sha256": sha256(model_path), "kind": label,
                "validation": record_group_metrics(y_val, pred, val.structure_group_id_v1.to_numpy()), "training_wall_seconds": seconds, "seed": 42}
            registry["validation_predictions"][f"{split_name}/{label}"] = {"path": str(pred_path), "sha256": sha256(pred_path)}
    if len(registry["models"]) != 20 or len(registry["preprocessors"]) != 10:
        raise RuntimeError("expected 20 models and 10 preprocessors")
    registry_path = ROOT / "data_registry/gate2a_model_registry.json"
    write_json(registry_path, registry)
    evidence = {"status": registry["status"], "model_registry_sha256": sha256(registry_path), "models": 20, "preprocessors": 10,
                "xgboost_runs": 10, "test_target_accessed": False, "buffer_dataset_created": False, "quarantine_dataset_created": False,
                "final673_accessed": False, "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "")}
    write_json(ROOT / "logs/gate2a_training_evidence.json", evidence)
    return evidence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gate2a_ood_baselines_v1.json")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/gate2a_ood_baselines")
    args = parser.parse_args()
    print(json.dumps(train_all(args.config, args.run_root), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
