#!/usr/bin/env python3
"""One-time union target unlock and evaluation for Gate 2-A."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from scipy.stats import spearmanr
from xgboost import XGBRegressor

try:
    from scripts.gate1b1_train_cheap_baselines import load_preprocessor, record_group_metrics, sha256, transform
    from scripts.gate2a_train_ood_baselines import load_contract, read_json, resolve, validate_manifest, write_json
except ModuleNotFoundError:  # direct execution from scripts/
    from gate1b1_train_cheap_baselines import load_preprocessor, record_group_metrics, sha256, transform
    from gate2a_train_ood_baselines import load_contract, read_json, resolve, validate_manifest, write_json

ROOT = Path(__file__).resolve().parents[1]


def group_abs_errors(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> np.ndarray:
    frame = pd.DataFrame({"group": groups.astype(str), "error": np.abs(p - y)})
    return frame.groupby("group", sort=True).error.mean().to_numpy(np.float64)


def bootstrap_ci(values: np.ndarray, replicates: int, seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    n = len(values)
    draws = np.empty(replicates)
    for index in range(replicates):
        draws[index] = values[rng.integers(0, n, n)].mean()
    return np.quantile(draws, [0.025, 0.975]).astype(float).tolist()


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray, groups: np.ndarray, replicates: int, seed: int) -> list[float]:
    frame = pd.DataFrame({"group": groups.astype(str), "delta": np.abs(a) - np.abs(b)})
    values = frame.groupby("group", sort=True).delta.mean().to_numpy(np.float64)
    return bootstrap_ci(values, replicates, seed)


def independent_degradation_ci(ood: np.ndarray, iid: np.ndarray, replicates: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    delta = np.empty(replicates); ratio = np.empty(replicates)
    for index in range(replicates):
        o = ood[rng.integers(0, len(ood), len(ood))].mean()
        i = iid[rng.integers(0, len(iid), len(iid))].mean()
        delta[index], ratio[index] = o - i, o / i
    return {"absolute_difference_ci95": np.quantile(delta, [0.025, 0.975]).tolist(),
            "ratio_ci95": np.quantile(ratio, [0.025, 0.975]).tolist(), "method": "independent_structure_group_bootstrap"}


def safe_spearman(x, y):
    value = float(spearmanr(x, y).statistic)
    return value if np.isfinite(value) else None


def metric_bundle(frame: pd.DataFrame, prediction: np.ndarray, replicates: int, seed: int) -> dict:
    y = frame.primary_true.to_numpy(np.float64); groups = frame.structure_group_id_v1.astype(str).to_numpy()
    result = record_group_metrics(y, prediction, groups)
    result["group_macro_mae_ci95"] = bootstrap_ci(group_abs_errors(y, prediction, groups), replicates, seed)
    return result


def subgroup_bundle(frame: pd.DataFrame, prediction: np.ndarray, mask: np.ndarray, replicates: int, seed: int) -> dict:
    subset = frame.loc[mask]
    if subset.structure_group_id_v1.nunique() < 30:
        return {"records": int(len(subset)), "groups": int(subset.structure_group_id_v1.nunique()), "status": "INSUFFICIENT_GROUPS_LT30"}
    return metric_bundle(subset, prediction[np.asarray(mask)], replicates, seed)


def verify_registry(registry: dict) -> None:
    if registry.get("status") != "VALIDATION_AND_20_BASELINE_ARTIFACTS_FROZEN" or registry.get("test_target_accessed") is not False:
        raise RuntimeError("model registry is not a sealed pre-test registry")
    if len(registry["models"]) != 20 or len(registry["preprocessors"]) != 10:
        raise RuntimeError("20-model/10-preprocessor freeze mismatch")
    for section in ("models", "preprocessors", "validation_predictions"):
        for item in registry[section].values():
            if sha256(Path(item["path"])) != item["sha256"]:
                raise RuntimeError(f"frozen {section} hash mismatch")
    similarity = registry["similarity"]
    if sha256(Path(similarity["path"])) != similarity["sha256"] or similarity["target_columns"]:
        raise RuntimeError("target-free similarity asset mismatch")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=ROOT / "configs/gate2a_ood_baselines_v1.json")
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/gate2a_ood_baselines")
    args = parser.parse_args()
    config, source, features = load_contract(args.config)
    unlock_path = ROOT / "data_registry/gate2a_test_unlock_v1.json"
    published = args.run_root / "published"
    if unlock_path.exists() or published.exists():
        raise RuntimeError("Gate 2-A test was already unlocked/evaluated; second call is fail-closed")
    registry_path = ROOT / "data_registry/gate2a_model_registry.json"
    registry = read_json(registry_path); verify_registry(registry)
    manifests = {name: validate_manifest(name, spec) for name, spec in config["splits"].items()}
    union_ids = sorted(set().union(*[set(frame.loc[frame.partition.eq("test"), "molecule_id"].astype(str)) for frame in manifests.values()]))
    unlock = {"gate": "2-A", "status": "UNIFIED_TEST_UNLOCK_CREATED_BEFORE_SINGLE_ARROW_READ",
              "created_utc": datetime.now(timezone.utc).isoformat(), "union_unique_molecule_count": len(union_ids),
              "per_split_test_counts": {name: int(frame.partition.eq("test").sum()) for name, frame in manifests.items()},
              "arrow_target_reads_allowed": 1, "arrow_target_reads_completed": 0,
              "model_registry_sha256": sha256(registry_path), "frozen_models": 20, "frozen_preprocessors": 10,
              "config_sha256": sha256(args.config), "test_guided_change": False}
    write_json(unlock_path, unlock)

    # The sole Arrow read containing any OOD test target occurs here, after the unlock is durable.
    dataset = ds.dataset(str(resolve(config["table"])), format="parquet")
    columns = ["molecule_id", config["primary_target"], "pm6_num_atoms_total", "pm6_num_donor_atoms", "pm6_num_acceptor_atoms"]
    test_metadata = dataset.to_table(columns=columns, filter=ds.field("molecule_id").isin(union_ids)).to_pandas()
    if len(test_metadata) != len(union_ids) or test_metadata[config["primary_target"]].isna().any():
        raise RuntimeError("single union Arrow target read mismatch")
    unlock["arrow_target_reads_completed"] = 1
    unlock["target_read_completed_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(unlock_path, unlock)

    similarity = pd.read_parquet(registry["similarity"]["path"])
    c0_columns = source["features"][config["training_features"]["source_keys"]["C0"]]
    c15_columns = source["features"][config["training_features"]["source_keys"]["C1p5_safe"]]
    replicates, boot_seed = config["bootstrap"]["replicates"], config["bootstrap"]["seed"]
    iid_predictions = pd.read_csv(ROOT / "runs/gate1b1_new_iid_cheap_baselines/published/gate1b1_test_predictions_once.csv")
    iid_errors = group_abs_errors(iid_predictions.primary_true.to_numpy(), iid_predictions.xgb_c0_seed42.to_numpy(), iid_predictions.structure_group_id_v1.to_numpy())
    all_metrics = {}; all_comparisons = {}; degradation = {}; prediction_frames = []
    for split_index, (split_name, manifest) in enumerate(manifests.items()):
        test_manifest = manifest[manifest.partition.eq("test")].copy()
        frame = test_manifest.merge(features, on="molecule_id", validate="one_to_one").merge(
            test_metadata.rename(columns={config["primary_target"]: "primary_true"}), on="molecule_id", validate="one_to_one").merge(
            similarity[similarity.split_name.eq(split_name)].drop(columns="split_name"), on="molecule_id", validate="one_to_one")
        split_dir = args.run_root / split_name
        prep_c0, cols0 = load_preprocessor(split_dir / "preprocessor_c0.npz")
        prep_c15, cols15 = load_preprocessor(split_dir / "preprocessor_c1p5_safe.npz")
        if cols0 != c0_columns or cols15 != c15_columns:
            raise RuntimeError("frozen feature order mismatch")
        x0 = transform(frame[c0_columns].to_numpy(np.float64), prep_c0)
        x15 = transform(frame[c15_columns].to_numpy(np.float64), prep_c15)
        median = read_json(split_dir / "weighted_median.json")["value"]
        ridge = np.load(split_dir / "ridge_c0.npz", allow_pickle=False)
        predictions = {"weighted_median": np.full(len(frame), median),
                       "ridge_c0": x0 @ ridge["coef"] + float(ridge["intercept"][0])}
        for label, matrix in (("xgb_c0", x0), ("xgb_c1p5_safe", x15)):
            model = XGBRegressor(); model.load_model(split_dir / f"{label}.json"); predictions[label] = model.predict(matrix)
        out = frame[["molecule_id", "structure_group_id_v1", "group_weight", "primary_true"]].copy()
        for label, prediction in predictions.items(): out[label] = prediction
        out.insert(0, "split_name", split_name); prediction_frames.append(out)
        metrics = {label: metric_bundle(frame, prediction, replicates, boot_seed + split_index * 100 + offset)
                   for offset, (label, prediction) in enumerate(predictions.items())}
        target_iqr = float(np.subtract(*np.quantile(frame.primary_true, [0.75, 0.25])))
        for value in metrics.values(): value["normalized_mae_by_target_iqr"] = value["group_macro_mae"] / target_iqr
        unknown = frame.pm6_num_atoms_total - frame.pm6_num_donor_atoms - frame.pm6_num_acceptor_atoms
        masks = {"singleton": frame.structure_group_size.eq(1).to_numpy(), "replicate": frame.structure_group_size.gt(1).to_numpy(),
                 "pure_donor_acceptor": ((frame.pm6_num_donor_atoms.gt(0)) & (frame.pm6_num_acceptor_atoms.gt(0)) & unknown.eq(0)).to_numpy(),
                 "donor_acceptor_unknown": ((frame.pm6_num_donor_atoms.gt(0)) & (frame.pm6_num_acceptor_atoms.gt(0)) & unknown.gt(0)).to_numpy(),
                 "empty_donor_unknown": ((frame.pm6_num_donor_atoms.eq(0)) & unknown.gt(0)).to_numpy()}
        donor_freq = manifest.loc[manifest.partition.eq("train"), "donor_structure_group_id_v1"].value_counts()
        acceptor_freq = manifest.loc[manifest.partition.eq("train"), "acceptor_structure_group_id_v1"].value_counts()
        for component, frequency in (("donor", donor_freq), ("acceptor", acceptor_freq)):
            values = frame[f"{component}_structure_group_id_v1"].map(frequency).fillna(0)
            masks.update({f"{component}_freq_unseen": values.eq(0).to_numpy(), f"{component}_freq_1_5": values.between(1, 5).to_numpy(),
                          f"{component}_freq_6_20": values.between(6, 20).to_numpy(), f"{component}_freq_gt20": values.gt(20).to_numpy()})
        for view in ("full", "donor", "acceptor"):
            scores = frame[f"nearest_train_{view}_morgan2048_chiral"]
            for lo, hi in zip(config["similarity_bins"][:-1], config["similarity_bins"][1:]):
                masks[f"{view}_similarity_[{lo},{min(hi,1.0)})"] = ((scores >= lo) & (scores < hi)).to_numpy()
        metrics["xgb_c0"]["strata"] = {name: subgroup_bundle(frame, predictions["xgb_c0"], mask, replicates, boot_seed + 500 + i)
                                           for i, (name, mask) in enumerate(masks.items())}
        metrics["xgb_c0"]["similarity_error_spearman"] = {view: safe_spearman(frame[f"nearest_train_{view}_morgan2048_chiral"], np.abs(predictions["xgb_c0"] - frame.primary_true))
                                                              for view in ("full", "donor", "acceptor")}
        all_metrics[split_name] = {"target": {"mean": float(frame.primary_true.mean()), "sd": float(frame.primary_true.std(ddof=1)), "iqr": target_iqr}, "models": metrics}
        y = frame.primary_true.to_numpy(); groups = frame.structure_group_id_v1.to_numpy()
        comparisons = {}
        for challenger in ("ridge_c0", "weighted_median", "xgb_c1p5_safe"):
            delta = group_abs_errors(y, predictions[challenger], groups).mean() - group_abs_errors(y, predictions["xgb_c0"], groups).mean()
            ci = paired_bootstrap_ci(predictions[challenger] - y, predictions["xgb_c0"] - y, groups, replicates, boot_seed + 900 + split_index)
            comparisons[f"{challenger}_minus_xgb_c0"] = {"group_macro_mae_difference": float(delta), "paired_group_bootstrap_ci95": ci,
                                                          "pm6_orbital_gain_claim_allowed": bool(challenger == "xgb_c1p5_safe" and ci[1] < 0)}
        all_comparisons[split_name] = comparisons
        ood_errors = group_abs_errors(y, predictions["xgb_c0"], groups)
        degradation[split_name] = {"iid_reference_group_macro_mae": config["iid_reference_group_macro_mae_eV"],
            "ood_group_macro_mae": metrics["xgb_c0"]["group_macro_mae"],
            "absolute_difference": metrics["xgb_c0"]["group_macro_mae"] - config["iid_reference_group_macro_mae_eV"],
            "degradation_ratio": metrics["xgb_c0"]["group_macro_mae"] / config["iid_reference_group_macro_mae_eV"],
            **independent_degradation_ci(ood_errors, iid_errors, replicates, boot_seed + 1200 + split_index)}
    published.mkdir(parents=True)
    predictions_path = published / "gate2a_test_predictions_once.csv"
    pd.concat(prediction_frames, ignore_index=True).to_csv(predictions_path, index=False)
    metrics_payload = {"status": "GATE2A_DONE_OOD_BASELINES", "test_evaluations": 1, "union_arrow_target_reads": 1,
                       "union_unique_molecule_count": len(union_ids), "metrics": all_metrics, "comparisons": all_comparisons,
                       "protocol_degradation": degradation, "prediction_sha256": sha256(predictions_path),
                       "cross_protocol_bootstrap": "independent_not_paired", "final673_accessed": False}
    metrics_path = published / "gate2a_metrics.json"; write_json(metrics_path, metrics_payload)

    lines = ["# Gate 2-A frozen OOD cheap baselines", "", "Status: **GATE2A_DONE_OOD_BASELINES**.", "",
             "Primary model: XGBoost-C0 (532 frozen C0 columns). C1.5-safe is a secondary PM6-orbital control. The target is `J_eh_screened_eV_eps3p5 proxy`, not experimental Eb.", "",
             "| Protocol | Test records | Test groups | Median | Ridge-C0 | XGB-C0 | XGB-C1.5-safe |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name in config["splits"]:
        m = all_metrics[name]["models"]
        lines.append(f"| {name} | {m['xgb_c0']['records']} | {m['xgb_c0']['groups']} | {m['weighted_median']['group_macro_mae']:.9f} | {m['ridge_c0']['group_macro_mae']:.9f} | {m['xgb_c0']['group_macro_mae']:.9f} | {m['xgb_c1p5_safe']['group_macro_mae']:.9f} |")
    lines += ["", "C0 training fingerprints are Morgan radius 2 / 512 bits / no chirality. OOD diagnostics are separate Morgan radius 2 / 2048 bits / chirality-enabled assets and never enter a model.",
              "", "Cross-protocol degradation is descriptive and uses independent structure-group bootstrap because IID and OOD test sets differ. No paired cross-protocol claim is made.",
              "", "Both-cold contains 587 test records/groups; 3,291 records / 3,234 groups remain buffer and received no predictions. Full-scaffold-cold is not described as deep scaffold extrapolation because 75.60% of full scaffolds are singletons."]
    (ROOT / "reports/gate2a_ood_baselines.md").write_text("\n".join(lines) + "\n")
    (ROOT / "reports/gate2a_protocol_degradation.md").write_text("# Gate 2-A protocol-specific degradation\n\n" + "\n".join(
        f"- `{name}`: MAE {value['ood_group_macro_mae']:.9f} eV; difference from IID {value['absolute_difference']:+.9f} eV; ratio {value['degradation_ratio']:.4f}; independent-bootstrap difference CI {value['absolute_difference_ci95']}."
        for name, value in degradation.items()) + "\n\nThese are protocol-specific descriptive comparisons, not paired or same-distribution causal estimates.\n")
    (ROOT / "reports/gate2a_pm6_orbital_control.md").write_text("# Gate 2-A PM6 orbital secondary control\n\n" + "\n".join(
        f"- `{name}`: C1.5-safe minus C0 {value['xgb_c1p5_safe_minus_xgb_c0']['group_macro_mae_difference']:+.9f} eV; paired group-bootstrap CI {value['xgb_c1p5_safe_minus_xgb_c0']['paired_group_bootstrap_ci95']}; gain claim allowed: {value['xgb_c1p5_safe_minus_xgb_c0']['pm6_orbital_gain_claim_allowed']}."
        for name, value in all_comparisons.items()) + "\n\nC1.5-safe remains secondary and cannot replace the global primary C0 baseline.\n")
    evidence = {"status": "GATE2A_DONE_OOD_BASELINES", "completed_utc": datetime.now(timezone.utc).isoformat(),
                "model_registry_sha256": sha256(registry_path), "test_unlock_sha256": sha256(unlock_path),
                "metrics_sha256": sha256(metrics_path), "predictions_sha256": sha256(predictions_path),
                "models": 20, "union_arrow_target_reads": 1, "test_evaluations": 1, "final673_accessed": False,
                "buffer_predictions": 0, "quarantine_predictions": 0}
    write_json(ROOT / "logs/gate2a_evidence.json", evidence)
    (ROOT / "logs/gate2a_run.log").write_text("Gate 2-A: 20 frozen assets; one union Arrow target read; five test protocols evaluated once; no retraining.\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
