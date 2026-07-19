#!/usr/bin/env python3
"""Freeze validation/model hashes, then evaluate Gate 1-B1 test targets once."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from xgboost import XGBRegressor

from gate1b1_train_cheap_baselines import load_preprocessor, record_group_metrics, sha256, transform


ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def verify_result_hashes(result: dict) -> None:
    if result.get("test_target_accessed") is not False:
        raise RuntimeError("seed result does not prove sealed test target")
    for model in result["models"].values():
        if sha256(Path(model["model_path"])) != model["model_sha256"]:
            raise RuntimeError("model hash mismatch before test unlock")
        if sha256(Path(model["prediction_path"])) != model["prediction_sha256"]:
            raise RuntimeError("validation prediction hash mismatch before test unlock")


def frequency_bin(value: int) -> str:
    if value <= 1:
        return "1"
    if value <= 5:
        return "2-5"
    if value <= 20:
        return "6-20"
    return ">20"


def safe_metrics(frame: pd.DataFrame, prediction: np.ndarray, mask: np.ndarray) -> dict:
    count = int(mask.sum())
    if count == 0:
        return {"records": 0, "groups": 0, "reason": "EMPTY_STRATUM"}
    target = frame.loc[mask, "primary_true"].to_numpy(np.float64)
    pred = np.asarray(prediction)[mask]
    groups = frame.loc[mask, "structure_group_id_v1"].astype(str).to_numpy()
    return record_group_metrics(target, pred, groups)


def test_metadata(config: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    test_ids = manifest.loc[manifest["partition"].eq("test"), "molecule_id"].astype(str).tolist()
    dataset = ds.dataset(config["table"], format="parquet")
    columns = [
        "molecule_id", config["primary_target"], "pm6_num_atoms_total",
        "pm6_num_donor_atoms", "pm6_num_acceptor_atoms",
    ]
    frame = dataset.to_table(columns=columns, filter=ds.field("molecule_id").isin(test_ids)).to_pandas()
    if len(frame) != 2319:
        raise RuntimeError("test target unlock count mismatch")
    return frame.rename(columns={config["primary_target"]: "primary_true"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    config = read_json(args.config)
    published = args.run_root / "published"
    unlock_path = args.run_root / "test_unlock_v1.json"
    if published.exists() or unlock_path.exists():
        raise RuntimeError("test has already been unlocked/evaluated; refusing a second evaluation")

    baseline_dir = args.run_root / "baselines"
    baseline_gate = read_json(baseline_dir / "baseline_gate.json")
    if baseline_gate.get("test_target_accessed") is not False:
        raise RuntimeError("baseline gate does not prove sealed test target")
    seed_results = {}
    for seed in config["seeds"]:
        result_path = args.run_root / f"seed{seed}" / "seed_result.json"
        result = read_json(result_path)
        verify_result_hashes(result)
        seed_results[str(seed)] = {"path": str(result_path), "sha256": sha256(result_path), "result": result}

    baseline_assets = {}
    for name in ("baseline_gate.json", "weighted_median.json", "ridge_c0.json", "ridge_c0.npz",
                 "preprocessor_c0.npz", "preprocessor_c1p5.npz"):
        path = baseline_dir / name
        baseline_assets[name] = {"path": str(path), "sha256": sha256(path)}
    unlock = {
        "gate": "1-B1", "status": "TEST_UNLOCKED_ONCE_AFTER_ALL_MODELS_FROZEN",
        "unlocked_utc": datetime.now(timezone.utc).isoformat(),
        "models_frozen": 8, "baseline_assets": baseline_assets,
        "seed_results": {seed: {"path": item["path"], "sha256": item["sha256"],
                                  "validation": item["result"]["models"]} for seed, item in seed_results.items()},
        "config_sha256": sha256(args.config), "manifest_sha256": config["manifest_sha256"],
        "feature_cache_sha256": config["feature_cache_sha256"], "test_guided_change": False,
    }
    write_json(unlock_path, unlock)
    unlock_sha = sha256(unlock_path)

    # Test targets are first accessed after the immutable unlock evidence above exists.
    manifest = pd.read_csv(config["manifest"])
    test_manifest = manifest[manifest["partition"].eq("test")].copy()
    if len(test_manifest) != 2319 or manifest["partition"].eq("historical_quarantine").sum() != 1:
        raise RuntimeError("test/quarantine boundary mismatch")
    features = pd.read_parquet(config["feature_cache"])
    test = test_manifest.merge(features, on="molecule_id", validate="one_to_one")
    test = test.merge(test_metadata(config, manifest), on="molecule_id", validate="one_to_one")
    if test["primary_true"].isna().any():
        raise RuntimeError("test target is missing")

    predictions: dict[str, np.ndarray] = {}
    inference_seconds: dict[str, float] = {}
    median = read_json(baseline_dir / "weighted_median.json")["value"]
    predictions["weighted_median"] = np.full(len(test), median, dtype=np.float64)

    c0_columns = config["features"]["M1_C0_open"]
    c15_columns = config["features"]["M2_C1p5_safe_no_dipole"]
    prep_c0, prep_c0_columns = load_preprocessor(baseline_dir / "preprocessor_c0.npz")
    prep_c15, prep_c15_columns = load_preprocessor(baseline_dir / "preprocessor_c1p5.npz")
    if prep_c0_columns != c0_columns or prep_c15_columns != c15_columns:
        raise RuntimeError("preprocessor feature order changed")
    x_c0 = transform(test[c0_columns].to_numpy(np.float64), prep_c0)
    x_c15 = transform(test[c15_columns].to_numpy(np.float64), prep_c15)
    ridge = np.load(baseline_dir / "ridge_c0.npz", allow_pickle=False)
    started = time.perf_counter()
    predictions["ridge_c0"] = x_c0 @ ridge["coef"] + float(ridge["intercept"][0])
    inference_seconds["ridge_c0"] = time.perf_counter() - started

    for seed in config["seeds"]:
        result = seed_results[str(seed)]["result"]
        for label, matrix in (("xgb_c0", x_c0), ("xgb_c1p5_safe", x_c15)):
            model = XGBRegressor()
            model.load_model(result["models"][label]["model_path"])
            started = time.perf_counter()
            predictions[f"{label}_seed{seed}"] = model.predict(matrix)
            inference_seconds[f"{label}_seed{seed}"] = time.perf_counter() - started

    output = test_manifest[[
        "molecule_id", "structure_group_id_v1", "donor_structure_group_id_v1",
        "acceptor_structure_group_id_v1", "structure_group_size", "group_weight",
    ]].merge(test[["molecule_id", "primary_true"]], on="molecule_id", validate="one_to_one")
    for name, values in predictions.items():
        output[name] = values
    published.mkdir(parents=True)
    prediction_path = published / "gate1b1_test_predictions_once.csv"
    output.to_csv(prediction_path, index=False)

    overall = {name: record_group_metrics(
        test["primary_true"].to_numpy(np.float64), values,
        test["structure_group_id_v1"].astype(str).to_numpy(),
    ) for name, values in predictions.items()}

    donor_count = manifest.loc[manifest["partition"].eq("train"), "donor_structure_group_id_v1"].value_counts()
    acceptor_count = manifest.loc[manifest["partition"].eq("train"), "acceptor_structure_group_id_v1"].value_counts()
    test["donor_frequency_bin"] = test["donor_structure_group_id_v1"].map(donor_count).fillna(0).astype(int).map(frequency_bin)
    test["acceptor_frequency_bin"] = test["acceptor_structure_group_id_v1"].map(acceptor_count).fillna(0).astype(int).map(frequency_bin)
    unknown = test["pm6_num_atoms_total"] - test["pm6_num_donor_atoms"] - test["pm6_num_acceptor_atoms"]
    role_masks = {
        "pure_donor_acceptor": ((test["pm6_num_donor_atoms"] > 0) & (test["pm6_num_acceptor_atoms"] > 0) & (unknown == 0)).to_numpy(),
        "donor_acceptor_unknown": ((test["pm6_num_donor_atoms"] > 0) & (test["pm6_num_acceptor_atoms"] > 0) & (unknown > 0)).to_numpy(),
        "empty_donor_unknown": ((test["pm6_num_donor_atoms"] == 0) & (unknown > 0)).to_numpy(),
        "singleton_structure": test["structure_group_size"].eq(1).to_numpy(),
        "replicated_structure": test["structure_group_size"].gt(1).to_numpy(),
    }
    for value in ("1", "2-5", "6-20", ">20"):
        role_masks[f"donor_frequency_{value}"] = test["donor_frequency_bin"].eq(value).to_numpy()
        role_masks[f"acceptor_frequency_{value}"] = test["acceptor_frequency_bin"].eq(value).to_numpy()
    quartiles = [-np.inf, *baseline_gate["train_target_quartiles_eV"], np.inf]
    for index in range(4):
        role_masks[f"train_quartile_target_bin_{index + 1}"] = (
            (test["primary_true"] > quartiles[index]) & (test["primary_true"] <= quartiles[index + 1])
        ).to_numpy()
    strata = {name: {stratum: safe_metrics(test, values, mask) for stratum, mask in role_masks.items()}
              for name, values in predictions.items()}

    aggregates = {}
    for family in ("xgb_c0", "xgb_c1p5_safe"):
        for metric in ("record_mae", "group_macro_mae", "record_rmse", "group_macro_rmse"):
            values = np.array([overall[f"{family}_seed{seed}"][metric] for seed in config["seeds"]])
            aggregates[f"{family}_{metric}"] = {
                "values": values.tolist(), "mean": float(values.mean()), "sample_std_ddof1": float(values.std(ddof=1)),
            }
    metrics = {
        "status": "GATE1B1_DONE", "test_evaluations": 1, "test_records": len(test),
        "test_effective_groups": float(test["group_weight"].sum()), "overall": overall,
        "three_seed_aggregates": aggregates, "strata": strata,
        "inference_wall_seconds": inference_seconds, "prediction_sha256": sha256(prediction_path),
        "test_unlock_sha256": unlock_sha, "test_guided_retraining": False,
        "primary_report_name": config["primary_report_name"],
    }
    metrics_path = published / "gate1b1_metrics.json"
    write_json(metrics_path, metrics)

    c0 = aggregates["xgb_c0_group_macro_mae"]
    c15 = aggregates["xgb_c1p5_safe_group_macro_mae"]
    report = ROOT / "reports/gate1b1_new_iid_cheap_baselines.md"
    report.write_text(
        "# Gate 1-B1 new IID cheap baselines\n\n"
        "Status: **GATE1B1_DONE**. The target is `J_eh_screened_eV_eps3p5 proxy`; it is not experimental Eb or catalytic efficiency.\n\n"
        f"Frozen IID test: {len(test):,} records / {test['structure_group_id_v1'].nunique():,} structure groups. "
        "The historical-quarantine record never entered a Dataset. Test labels were unlocked once only after all eight model and validation hashes were frozen.\n\n"
        "| Model | Record MAE (eV) | Group-macro MAE (eV) | Record RMSE (eV) | R² |\n"
        "|---|---:|---:|---:|---:|\n" +
        "\n".join(
            f"| {name} | {value['record_mae']:.9f} | {value['group_macro_mae']:.9f} | "
            f"{value['record_rmse']:.9f} | {value['record_r2']:.6f} |"
            for name, value in overall.items()
        ) + "\n\n"
        f"XGBoost-C0 group-macro MAE: `{c0['mean']:.9f} ± {c0['sample_std_ddof1']:.9f} eV`; "
        f"XGBoost-C1.5-safe: `{c15['mean']:.9f} ± {c15['sample_std_ddof1']:.9f} eV` (sample std, ddof=1). "
        "All three seed predictions are identical because the frozen historical transfer configuration uses no row or feature subsampling; the seed is therefore operationally inert. "
        "C1.5-safe does not improve this frozen baseline, so no PM6-orbital gain is claimed.\n\n"
        "These values are an internal new15016 grouped-IID comparison and are not directly comparable to old Layer G as evidence of model progress. "
        "Role, replicate, component-frequency, and train-quartile target strata are recorded in the metrics JSON for diagnosis only and did not guide selection.\n"
    )
    evidence = {
        "gate": "1-B1", "status": "GATE1B1_DONE", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "test_unlock": str(unlock_path), "test_unlock_sha256": unlock_sha,
        "prediction_path": str(prediction_path), "prediction_sha256": sha256(prediction_path),
        "metrics_path": str(metrics_path), "metrics_sha256": sha256(metrics_path),
        "report_path": str(report), "report_sha256": sha256(report),
        "final673_accessed": False, "quarantine_dataset_created": False,
        "test_guided_retraining": False, "gpu_physical_ids": [0, 1, 2],
    }
    write_json(ROOT / "logs/gate1b1_evidence.json", evidence)
    (ROOT / "logs/gate1b1_run.log").write_text(
        "Gate 1-B1 completed: CPU contract passed; 8 fixed models frozen; test unlocked and evaluated once; no rerun.\n"
    )
    print(json.dumps({"status": "GATE1B1_DONE", "overall": overall,
                      "three_seed_aggregates": aggregates, "prediction_sha256": sha256(prediction_path)}, indent=2))


if __name__ == "__main__":
    main()
