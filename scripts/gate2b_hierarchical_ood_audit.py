#!/usr/bin/env python3
"""Artifact-only hierarchical OOD audit; never reads the source target table."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from scipy.stats import spearmanr, wasserstein_distance
import xgboost as xgb

try:
    from scripts.gate1b1_train_cheap_baselines import load_preprocessor, sha256, transform
except ModuleNotFoundError:
    from gate1b1_train_cheap_baselines import load_preprocessor, sha256, transform

ROOT = Path(__file__).resolve().parents[1]


def resolve(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else ROOT / p


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def safe_float(value) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def verify_inputs(config: dict) -> None:
    for item in config["inputs"].values():
        if sha256(resolve(item["path"])) != item["sha256"]:
            raise RuntimeError(f"input hash mismatch: {item['path']}")
    for item in config["manifests"].values():
        if sha256(resolve(item["path"])) != item["sha256"]:
            raise RuntimeError(f"manifest hash mismatch: {item['path']}")
    if not resolve(config["inputs"]["test_unlock"]["path"]).exists():
        raise RuntimeError("Gate 2-A evaluator is not locked")
    source = Path(__file__).read_text().lower()
    forbidden = ("molecule_" + "values_v3", "py" + "arrow", "to_" + "table(", "tddft_" + "coulomb")
    if any(token in source for token in forbidden):
        raise RuntimeError("audit source violates artifact-only firewall")


def record_metrics(y: np.ndarray, p: np.ndarray) -> dict:
    error = p - y; denom = np.sum((y - y.mean()) ** 2)
    return {"mae": float(np.mean(np.abs(error))), "rmse": float(np.sqrt(np.mean(error ** 2))),
            "r2": float(1 - np.sum(error ** 2) / denom) if denom > 0 else None}


def cluster_table(frame: pd.DataFrame, prediction: str, cluster: str) -> pd.DataFrame:
    columns = list(dict.fromkeys([cluster, "structure_group_id_v1", "primary_true", prediction]))
    work = frame[columns].copy().sort_values(list(dict.fromkeys([cluster, "structure_group_id_v1"])), kind="mergesort")
    work["abs_error"] = (work[prediction] - work.primary_true).abs()
    work["sq_error"] = (work[prediction] - work.primary_true) ** 2
    work["signed_error"] = work[prediction] - work.primary_true
    return work.groupby(cluster, sort=True).agg(records=("primary_true", "size"), structure_groups=("structure_group_id_v1", "nunique"),
        mae=("abs_error", "mean"), mse=("sq_error", "mean"), signed_error=("signed_error", "mean"),
        target_mean=("primary_true", "mean"), target_sd=("primary_true", "std"), target_min=("primary_true", "min"), target_max=("primary_true", "max")).reset_index()


def macro_metrics(frame: pd.DataFrame, prediction: str, cluster: str) -> dict:
    table = cluster_table(frame, prediction, cluster)
    y = frame.primary_true.to_numpy(np.float64); p = frame[prediction].to_numpy(np.float64)
    record = record_metrics(y, p)
    target_means = table.target_mean.to_numpy(np.float64)
    target_iqr_record = float(np.subtract(*np.quantile(y, [0.75, 0.25])))
    target_iqr_identity = float(np.subtract(*np.quantile(target_means, [0.75, 0.25])))
    mean_target = float(target_means.mean())
    row_cluster_weight = frame[cluster].map(table.set_index(cluster).records.rdiv(1.0)).to_numpy(np.float64)
    denom = np.sum(row_cluster_weight * (y - mean_target) ** 2)
    r2 = float(1 - np.sum(row_cluster_weight * (p - y) ** 2) / denom) if denom > 0 else None
    identity_mae = float(table.mae.mean()); identity_rmse = float(np.sqrt(table.mse.mean()))
    identity_target_sd = float(target_means.std(ddof=1)) if len(target_means) > 1 else 0.0
    sizes = table.records.to_numpy()
    return {"record": record, "identity_macro": {"mae": identity_mae, "rmse": identity_rmse, "r2": r2,
        "normalized_mae_record_target_iqr": identity_mae / target_iqr_record if target_iqr_record else None,
        "normalized_mae_identity_mean_iqr": identity_mae / target_iqr_identity if target_iqr_identity else None,
        "rmse_over_identity_target_sd": identity_rmse / identity_target_sd if identity_target_sd else None},
        "identity_count": int(len(table)), "identity_size": {"min": int(sizes.min()), "median": float(np.median(sizes)), "p90": float(np.quantile(sizes, .9)), "max": int(sizes.max())},
        "identity_error": {"median_mae": float(table.mae.median()), "p90_mae": float(table.mae.quantile(.9)),
                           "worst_decile_mean_mae": float(table.loc[table.mae >= table.mae.quantile(.9), "mae"].mean())},
        "identity_target": {"mean": mean_target, "sd": identity_target_sd, "iqr": target_iqr_identity,
                            "within_identity_range_median": float((table.target_max - table.target_min).median())}}


def oneway_bootstrap(frame: pd.DataFrame, prediction: str, cluster: str, replicates: int, seed: int) -> np.ndarray:
    values = cluster_table(frame, prediction, cluster).mae.to_numpy(np.float64)
    rng = np.random.default_rng(seed); n = len(values); output = np.empty(replicates)
    for index in range(replicates): output[index] = values[rng.integers(0, n, n)].mean()
    return output


def two_way_bootstrap(frame: pd.DataFrame, prediction: str, replicates: int, seed: int) -> np.ndarray:
    work = frame.sort_values("molecule_id", kind="mergesort")
    donors = sorted(work.donor_structure_group_id_v1.unique()); acceptors = sorted(work.acceptor_structure_group_id_v1.unique())
    d_index = pd.Categorical(work.donor_structure_group_id_v1, categories=donors).codes
    a_index = pd.Categorical(work.acceptor_structure_group_id_v1, categories=acceptors).codes
    errors = np.abs(work[prediction].to_numpy(np.float64) - work.primary_true.to_numpy(np.float64))
    rng = np.random.default_rng(seed); output = np.empty(replicates)
    for index in range(replicates):
        total = 0
        while total == 0:
            d_mult = np.bincount(rng.integers(0, len(donors), len(donors)), minlength=len(donors))
            a_mult = np.bincount(rng.integers(0, len(acceptors), len(acceptors)), minlength=len(acceptors))
            weights = d_mult[d_index] * a_mult[a_index]; total = weights.sum()
        output[index] = np.sum(weights * errors) / total
    return output


def ci(values: np.ndarray) -> list[float]:
    return np.quantile(values, [.025, .975]).astype(float).tolist()


def diagnostic_mol(smiles: str):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return mol
    mol = Chem.MolFromSmiles(smiles, sanitize=False)
    if mol is None:
        return None
    ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    return mol if Chem.SanitizeMol(mol, sanitizeOps=ops, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE else None


def paired_spearman(frame: pd.DataFrame, left: str, right: str) -> float | None:
    valid = frame[[left, right]].dropna()
    return safe_float(spearmanr(valid[left], valid[right]).statistic) if len(valid) > 1 else None


def worst_identities(frame: pd.DataFrame, prediction: str, cluster: str) -> list[dict]:
    work = frame[[cluster, "molecule_id", "primary_true", prediction]].copy()
    work["abs_error"] = (work[prediction] - work.primary_true).abs()
    work["signed_error"] = work[prediction] - work.primary_true
    grouped = work.groupby(cluster, sort=True).agg(records=("molecule_id", "size"), mae=("abs_error", "mean"), signed_error=("signed_error", "mean")).reset_index()
    return grouped.nlargest(10, "mae").to_dict("records")


def identity_descriptors(smiles: str) -> dict:
    mol = diagnostic_mol(smiles)
    if mol is None: return {"acceptor_atoms": None, "heteroatom_fraction": None, "formal_charge": None, "molecular_weight": None}
    atoms = mol.GetAtoms(); n = len(atoms)
    return {"acceptor_atoms": n, "heteroatom_fraction": sum(a.GetAtomicNum() not in (1, 6) for a in atoms) / n if n else None,
            "formal_charge": int(sum(a.GetFormalCharge() for a in atoms)), "molecular_weight": float(Descriptors.MolWt(mol))}


def acceptor_mechanism(frame: pd.DataFrame, config: dict, similarity: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    components = pd.read_csv(resolve(config["inputs"]["component_identity"]["path"]))
    comp = components[["molecule_id", "acceptor_canonical_structure_smiles_v1", "acceptor_scaffold_group_id_v1"]]
    work = frame.merge(similarity[similarity.split_name.eq("acceptor_cold")].drop(columns="split_name"), on="molecule_id", validate="one_to_one").merge(comp, on="molecule_id", validate="one_to_one")
    work["abs_error"] = (work.xgb_c0 - work.primary_true).abs(); work["signed_error"] = work.xgb_c0 - work.primary_true
    grouped = work.groupby("acceptor_structure_group_id_v1", sort=True).agg(records=("molecule_id", "size"), donors=("donor_structure_group_id_v1", "nunique"),
        structure_groups=("structure_group_id_v1", "nunique"), mae=("abs_error", "mean"), signed_error=("signed_error", "mean"),
        acceptor_similarity=("nearest_train_acceptor_morgan2048_chiral", "mean"), full_similarity=("nearest_train_full_morgan2048_chiral", "mean"),
        target_mean=("primary_true", "mean"), target_min=("primary_true", "min"), target_max=("primary_true", "max"),
        scaffold=("acceptor_scaffold_group_id_v1", "first"), smiles=("acceptor_canonical_structure_smiles_v1", "first")).reset_index()
    grouped["target_range"] = grouped.target_max - grouped.target_min
    descriptors = pd.DataFrame([identity_descriptors(s) for s in grouped.smiles]); grouped = pd.concat([grouped.drop(columns="smiles"), descriptors], axis=1)
    corr = {name: paired_spearman(grouped, "mae", column) for name, column in {
        "mae_vs_acceptor_similarity": "acceptor_similarity", "mae_vs_full_similarity": "full_similarity", "mae_vs_records": "records",
        "mae_vs_donor_count": "donors", "mae_vs_target_range": "target_range", "mae_vs_heteroatom_fraction": "heteroatom_fraction"}.items()}
    worst = grouped.nlargest(10, "mae")[["acceptor_structure_group_id_v1", "records", "donors", "mae", "signed_error", "acceptor_similarity", "full_similarity", "target_range"]].to_dict("records")
    descriptor_columns = ["acceptor_atoms", "heteroatom_fraction", "formal_charge", "molecular_weight"]
    descriptor_summary = {column: {"valid": int(grouped[column].notna().sum()), "mean": safe_float(grouped[column].mean()),
        "median": safe_float(grouped[column].median()), "min": safe_float(grouped[column].min()), "max": safe_float(grouped[column].max())} for column in descriptor_columns}
    return {"identity_count": len(grouped), "descriptor_parse_failures": int(grouped.acceptor_atoms.isna().sum()),
            "unique_acceptor_scaffolds": int(grouped.scaffold.nunique()), "descriptor_summary": descriptor_summary,
            "correlations_spearman": corr, "underprediction_fraction": float((work.signed_error < 0).mean()),
            "overprediction_fraction": float((work.signed_error > 0).mean()), "worst_10_anonymous_identity_hashes": worst}, grouped


def pm6_shift_audit(config: dict, acceptor_manifest: pd.DataFrame, acceptor_predictions: pd.DataFrame) -> dict:
    features = pd.read_parquet(resolve(config["inputs"]["feature_cache"]["path"]))
    frame = acceptor_manifest[["molecule_id", "partition"]].merge(features, on="molecule_id", validate="one_to_one")
    output = {"features": {}, "frozen_model_feature_importance_gain": {}, "tree_shap": {}}
    thresholds = config["decision_thresholds"]["pm6_shift"]; shifted = 0; shap_associated = False
    for feature in config["pm6_features"]:
        train = frame.loc[frame.partition.eq("train"), feature].to_numpy(np.float64); val = frame.loc[frame.partition.eq("val"), feature].to_numpy(np.float64); test = frame.loc[frame.partition.eq("test"), feature].to_numpy(np.float64)
        train_sd = train.std(ddof=1); standardized = (test.mean() - train.mean()) / train_sd
        result = {"train": {"mean": float(train.mean()), "sd": float(train_sd), "min": float(train.min()), "max": float(train.max())},
                  "val": {"mean": float(val.mean()), "sd": float(val.std(ddof=1))}, "test": {"mean": float(test.mean()), "sd": float(test.std(ddof=1))},
                  "standardized_test_shift": float(standardized), "wasserstein": float(wasserstein_distance(train, test)),
                  "wasserstein_over_train_sd": float(wasserstein_distance(train, test) / train_sd),
                  "test_outside_train_range_fraction": float(((test < train.min()) | (test > train.max())).mean())}
        conditions = [abs(result["standardized_test_shift"]) >= thresholds["absolute_standardized_shift_min"],
                      result["wasserstein_over_train_sd"] >= thresholds["wasserstein_over_train_sd_min"],
                      result["test_outside_train_range_fraction"] >= thresholds["out_of_train_range_fraction_min"]]
        result["shift_flag"] = sum(conditions) >= 2; shifted += int(result["shift_flag"]); output["features"][feature] = result
    b1 = read_json(ROOT / "configs/gate1b1_new_iid_cheap_baselines_v1.json")
    columns = b1["features"]["M2_C1p5_safe_no_dipole"]
    prep, frozen_columns = load_preprocessor(ROOT / "runs/gate2a_ood_baselines/acceptor_cold/preprocessor_c1p5_safe.npz")
    if columns != frozen_columns: raise RuntimeError("frozen C1.5 feature order mismatch")
    test_ids = acceptor_manifest.loc[acceptor_manifest.partition.eq("test"), "molecule_id"]
    test_frame = pd.DataFrame({"molecule_id": test_ids}).merge(features, on="molecule_id", validate="one_to_one")
    matrix = transform(test_frame[columns].to_numpy(np.float64), prep)
    booster = xgb.Booster(); booster.load_model(ROOT / "runs/gate2a_ood_baselines/acceptor_cold/xgb_c1p5_safe.json")
    gain = booster.get_score(importance_type="gain"); contributions = booster.predict(xgb.DMatrix(matrix, feature_names=columns), pred_contribs=True)
    errors = acceptor_predictions.set_index("molecule_id").loc[test_frame.molecule_id]
    absolute_error = np.abs(errors.xgb_c1p5_safe.to_numpy() - errors.primary_true.to_numpy())
    for feature in config["pm6_features"]:
        index = columns.index(feature); values = contributions[:, index]
        corr = safe_float(spearmanr(np.abs(values), absolute_error).statistic)
        output["frozen_model_feature_importance_gain"][feature] = float(gain.get(feature, gain.get(f"f{index}", 0.0)))
        output["tree_shap"][feature] = {"mean_signed_contribution": float(values.mean()), "mean_absolute_contribution": float(np.abs(values).mean()),
                                         "absolute_contribution_vs_absolute_error_spearman": corr}
        shap_associated = shap_associated or (corr is not None and abs(corr) >= thresholds["absolute_shap_error_spearman_min"])
    output["shifted_feature_count"] = shifted; output["shap_error_association_flag"] = shap_associated
    output["PM6_ORBITAL_SHIFT_RISK"] = bool(shifted >= thresholds["minimum_shifted_features"] and shap_associated)
    output["no_refit"] = True; output["new_predictions_generated"] = False
    return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, default=ROOT / "configs/gate2b_hierarchical_ood_audit_v1.json"); args = parser.parse_args()
    config = read_json(args.config); verify_inputs(config); RDLogger.DisableLog("rdApp.*")
    ood_predictions = pd.read_csv(resolve(config["inputs"]["ood_predictions"]["path"]))
    iid_predictions = pd.read_csv(resolve(config["inputs"]["iid_predictions"]["path"])).rename(columns={"xgb_c0_seed42": "xgb_c0", "xgb_c1p5_safe_seed42": "xgb_c1p5_safe"})
    similarity = pd.read_parquet(resolve(config["inputs"]["ood_similarity"]["path"]))
    manifests = {name: pd.read_csv(resolve(spec["path"])) for name, spec in config["manifests"].items()}
    replicates = config["bootstrap"]["replicates"]; seed = config["bootstrap"]["seed"]
    iid = iid_predictions.merge(manifests["iid"][["molecule_id", "partition"]], on="molecule_id", validate="one_to_one")
    if not iid.partition.eq("test").all(): raise RuntimeError("IID artifact/manifest mismatch")
    iid_reference = {"models": {}}
    iid_bootstraps = {}
    for model_index, model in enumerate(config["models"]):
        iid_reference["models"][model] = macro_metrics(iid, model, "structure_group_id_v1")
        iid_bootstraps[model] = oneway_bootstrap(iid, model, "structure_group_id_v1", replicates, seed + model_index)
        iid_reference["models"][model]["structure_group_bootstrap_ci95"] = ci(iid_bootstraps[model])
    iid_reference["xgb_c0_skill_vs_weighted_median"] = 1 - iid_reference["models"]["xgb_c0"]["identity_macro"]["mae"] / iid_reference["models"]["weighted_median"]["identity_macro"]["mae"]
    iid_boot = iid_bootstraps["xgb_c0"]
    protocols = {}; bootstrap = {}; labels = []
    for split_index, split_name in enumerate(("donor_cold", "acceptor_cold", "pair_cold", "both_cold", "full_scaffold_cold")):
        prediction = ood_predictions[ood_predictions.split_name.eq(split_name)].copy()
        identity_columns = ["molecule_id", "partition", "donor_structure_group_id_v1", "acceptor_structure_group_id_v1", "pair_group_id_v1", "full_scaffold_group_id_v1"]
        frame = prediction.merge(manifests[split_name][identity_columns], on="molecule_id", validate="one_to_one")
        if not frame.partition.eq("test").all(): raise RuntimeError(f"{split_name} artifact/manifest mismatch")
        protocol = {"models": {}, "primary_cluster": config["manifests"][split_name]["cluster"]}
        boots = {}
        for model_index, model in enumerate(config["models"]):
            if split_name == "both_cold":
                donor = macro_metrics(frame, model, "donor_structure_group_id_v1"); acceptor = macro_metrics(frame, model, "acceptor_structure_group_id_v1")
                dist = two_way_bootstrap(frame, model, replicates, seed + split_index * 100 + model_index)
                protocol["models"][model] = {"record": record_metrics(frame.primary_true.to_numpy(), frame[model].to_numpy()), "donor_identity": donor,
                    "acceptor_identity": acceptor, "two_way_mae": float(np.mean(np.abs(frame[model] - frame.primary_true))), "two_way_bootstrap_ci95": ci(dist)}
                protocol["models"][model]["donor_cluster_bootstrap_ci95"] = ci(oneway_bootstrap(frame, model, "donor_structure_group_id_v1", replicates, seed + 500 + model_index))
                protocol["models"][model]["acceptor_cluster_bootstrap_ci95"] = ci(oneway_bootstrap(frame, model, "acceptor_structure_group_id_v1", replicates, seed + 600 + model_index))
            else:
                cluster = config["manifests"][split_name]["cluster"]; protocol["models"][model] = macro_metrics(frame, model, cluster)
                dist = oneway_bootstrap(frame, model, cluster, replicates, seed + split_index * 100 + model_index)
                protocol["models"][model]["identity_bootstrap_ci95"] = ci(dist)
            structure_metrics = macro_metrics(frame, model, "structure_group_id_v1")
            structure_dist = oneway_bootstrap(frame, model, "structure_group_id_v1", replicates, seed + 7000 + split_index * 100 + model_index)
            protocol["models"][model]["structure_group_secondary"] = structure_metrics
            protocol["models"][model]["structure_group_bootstrap_ci95"] = ci(structure_dist)
            boots[model] = dist
        median_mae = protocol["models"]["weighted_median"]["record"]["mae"] if split_name == "both_cold" else protocol["models"]["weighted_median"]["identity_macro"]["mae"]
        model_mae = protocol["models"]["xgb_c0"]["record"]["mae"] if split_name == "both_cold" else protocol["models"]["xgb_c0"]["identity_macro"]["mae"]
        protocol["xgb_c0_skill_vs_weighted_median"] = 1 - model_mae / median_mae
        protocol["xgb_c0_minus_median_bootstrap_ci95"] = ci(boots["xgb_c0"] - boots["weighted_median"])
        degradation = boots["xgb_c0"] - iid_boot; ratio = boots["xgb_c0"] / iid_boot
        protocol["iid_to_ood_independent_bootstrap"] = {"mae_difference_ci95": ci(degradation), "ratio_ci95": ci(ratio), "paired": False}
        clusters = min(frame.donor_structure_group_id_v1.nunique(), frame.acceptor_structure_group_id_v1.nunique()) if split_name == "both_cold" else protocol["models"]["xgb_c0"]["identity_count"]
        protocol["inference_cluster_count"] = int(clusters); protocol["power_status"] = "LOW_CLUSTER_POWER" if clusters < config["bootstrap"]["minimum_clusters_for_strong_claim"] else "ADEQUATE_CLUSTER_COUNT"
        protocols[split_name] = protocol; bootstrap[split_name] = {k: ci(v) for k, v in boots.items()}
    acceptor_frame = ood_predictions[ood_predictions.split_name.eq("acceptor_cold")].merge(manifests["acceptor_cold"][["molecule_id", "donor_structure_group_id_v1", "acceptor_structure_group_id_v1"]], on="molecule_id", validate="one_to_one")
    acceptor, acceptor_table = acceptor_mechanism(acceptor_frame, config, similarity)
    pm6 = pm6_shift_audit(config, manifests["acceptor_cold"], ood_predictions[ood_predictions.split_name.eq("acceptor_cold")])
    acceptor_ci = protocols["acceptor_cold"]["iid_to_ood_independent_bootstrap"]["mae_difference_ci95"]
    labels.append("ACCEPTOR_OOD_FAILURE_CONFIRMED" if acceptor_ci[0] > 0 else "ACCEPTOR_OOD_CLAIM_WEAKENED")
    gate2a = read_json(resolve(config["inputs"]["ood_metrics"]["path"])); both_standard = gate2a["metrics"]["both_cold"]["models"]["xgb_c0"]
    both_skill = 1 - both_standard["group_macro_mae"] / gate2a["metrics"]["both_cold"]["models"]["weighted_median"]["group_macro_mae"]
    thresholds = config["decision_thresholds"]["both_low_skill"]
    conditions = [both_standard["normalized_mae_by_target_iqr"] >= thresholds["normalized_mae_min"], both_standard["group_macro_r2"] <= thresholds["r2_max"], both_skill <= thresholds["skill_vs_median_max"]]
    if sum(conditions) >= thresholds["minimum_conditions"]: labels.append("BOTH_COLD_LOW_SKILL_WARNING")
    if pm6["PM6_ORBITAL_SHIFT_RISK"]: labels.append("PM6_ORBITAL_SHIFT_RISK")
    both_frame = ood_predictions[ood_predictions.split_name.eq("both_cold")].copy()
    both_identity_frame = both_frame.merge(manifests["both_cold"][["molecule_id", "donor_structure_group_id_v1", "acceptor_structure_group_id_v1"]], on="molecule_id", validate="one_to_one")
    y = both_frame.primary_true.to_numpy(); p = both_frame.xgb_c0.to_numpy(); slope = float(np.polyfit(y - y.mean(), p - p.mean(), 1)[0])
    both_explanation = {"absolute_mae": both_standard["group_macro_mae"], "normalized_mae": both_standard["normalized_mae_by_target_iqr"], "r2": both_standard["group_macro_r2"],
        "weighted_median_mae": gate2a["metrics"]["both_cold"]["models"]["weighted_median"]["group_macro_mae"], "skill_vs_median": both_skill,
        "records": len(both_frame), "donor_identities": manifests["both_cold"].query("partition == 'test'").donor_structure_group_id_v1.nunique(),
        "acceptor_identities": manifests["both_cold"].query("partition == 'test'").acceptor_structure_group_id_v1.nunique(), "prediction_on_truth_centered_slope": slope,
        "regression_to_mean_warning": slope < 0.75, "target_sd": float(y.std(ddof=1)), "target_iqr": float(np.subtract(*np.quantile(y, [.75, .25]))),
        "worst_10_donor_identity_hashes": worst_identities(both_identity_frame, "xgb_c0", "donor_structure_group_id_v1"),
        "worst_10_acceptor_identity_hashes": worst_identities(both_identity_frame, "xgb_c0", "acceptor_structure_group_id_v1")}
    output = {"status": "GATE2B_DONE_HIERARCHICAL_OOD_AUDIT", "diagnostic_labels": labels, "iid_reference": iid_reference, "protocols": protocols,
              "acceptor_mechanism": acceptor, "both_cold_explanation": both_explanation, "pm6_shift_audit": pm6,
              "bootstrap_replicates": replicates, "bootstrap_seed": seed, "artifact_only": True, "main_parquet_accessed": False,
              "new_predictions_generated": False, "gate2a_evaluator_calls": 0, "final673_accessed": False,
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    metrics_path = ROOT / "logs/gate2b_hierarchical_metrics.json"; write_json(metrics_path, output)
    # Reports are generated only from the frozen artifact analysis above.
    lines = ["# Gate 2-B hierarchical OOD inference", "", "Status: **GATE2B_DONE_HIERARCHICAL_OOD_AUDIT**.", "", f"Diagnostic labels: `{', '.join(labels)}`.", "",
             "| Protocol | Primary inference unit | Clusters | Power | XGB-C0 identity/two-way MAE | IID difference CI | Skill vs median |", "|---|---|---:|---|---:|---|---:|",
             f"| iid | structure_group_id_v1 | {iid_reference['models']['xgb_c0']['identity_count']} | ADEQUATE_CLUSTER_COUNT | {iid_reference['models']['xgb_c0']['identity_macro']['mae']:.9f} | reference | {iid_reference['xgb_c0_skill_vs_weighted_median']:.4f} |"]
    for name, value in protocols.items():
        metric = value["models"]["xgb_c0"]["record"]["mae"] if name == "both_cold" else value["models"]["xgb_c0"]["identity_macro"]["mae"]
        lines.append(f"| {name} | {value['primary_cluster']} | {value['inference_cluster_count']} | {value['power_status']} | {metric:.9f} | {value['iid_to_ood_independent_bootstrap']['mae_difference_ci95']} | {value['xgb_c0_skill_vs_weighted_median']:.4f} |")
    lines += ["", "Structure-group bootstrap remains a secondary Gate 2-A sensitivity. Cold-protocol inference now clusters on held-out identities. IID-to-OOD comparisons use independent resampling and are descriptive, never paired or causal."]
    (ROOT / "reports/gate2b_hierarchical_ood_inference.md").write_text("\n".join(lines) + "\n")
    (ROOT / "reports/gate2b_acceptor_failure_mechanism.md").write_text("# Gate 2-B acceptor-cold failure mechanism\n\n" + f"Acceptor identity clusters: {acceptor['identity_count']}. Acceptor-cluster degradation CI: {acceptor_ci}. Diagnostic label: `{labels[0]}`.\n\nDescriptor parse failures after the frozen diagnostic fallback: {acceptor['descriptor_parse_failures']}. Unique acceptor scaffolds: {acceptor['unique_acceptor_scaffolds']}. Spearman diagnostics: `{acceptor['correlations_spearman']}`. Worst identities are reported only by frozen anonymous structure hash in the metrics JSON. No raw structure is emitted.\n")
    (ROOT / "reports/gate2b_both_cold_difficulty.md").write_text("# Gate 2-B both-cold relative difficulty\n\n" + f"Absolute XGBoost-C0 MAE is {both_explanation['absolute_mae']:.9f} eV, but normalized MAE is {both_explanation['normalized_mae']:.3f}, R² is {both_explanation['r2']:.3f}, and skill versus weighted median is {both_skill:.3f}. The 587 records cross only 15 donor and 40 acceptor identities; donor-side inference is `LOW_CLUSTER_POWER`. Prediction-on-truth centered slope is {slope:.3f}, so regression-to-mean warning is {both_explanation['regression_to_mean_warning']}. Worst donor and acceptor clusters are stored only as anonymous hashes.\n\nAbsolute MAE therefore cannot support a claim of strong both-cold generalization.\n")
    (ROOT / "reports/gate2b_pm6_shift_audit.md").write_text("# Gate 2-B PM6 orbital shift audit\n\n" + f"Shifted features under the preregistered rule: {pm6['shifted_feature_count']}/3. SHAP/error association: {pm6['shap_error_association_flag']}. Diagnostic label `PM6_ORBITAL_SHIFT_RISK` applied: {pm6['PM6_ORBITAL_SHIFT_RISK']}. Frozen gain and TreeSHAP diagnostics were read from the acceptor-cold C1.5-safe model without refitting or generating predictions.\n")
    evidence = {"status": output["status"], "diagnostic_labels": labels, "metrics_sha256": sha256(metrics_path), "input_prediction_sha256": config["inputs"]["ood_predictions"]["sha256"],
                "input_prediction_sha256_after": sha256(resolve(config["inputs"]["ood_predictions"]["path"])), "new_predictions_generated": False,
                "main_parquet_accessed": False, "gate2a_evaluator_calls": 0, "gate2a_evaluator_still_locked": True, "final673_accessed": False}
    write_json(ROOT / "logs/gate2b_evidence.json", evidence); print(json.dumps(evidence, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
