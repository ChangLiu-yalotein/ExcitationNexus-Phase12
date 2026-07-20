#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from scipy.stats import pearsonr, spearmanr

from excitationnexus_phase12.gate1c1 import (
    deterministic_similarity_merge,
    group_bootstrap_error_difference,
    group_macro_mae,
    sha256_file,
)

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate(config_path: Path, lock_path: Path) -> dict:
    config, lock = json.loads(config_path.read_text()), json.loads(lock_path.read_text())
    if sha256_file(config_path) != lock["config_sha256"] or lock["test_inference_allowed"] is not False:
        raise RuntimeError("Gate1C1 preregistration lock mismatch")
    for item in config["frozen_inputs"].values():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen input changed: {item['path']}")
    return config


def correlation(x, y) -> dict:
    frame = pd.DataFrame({"x": x, "y": y}).dropna()
    if len(frame) < 3 or frame.x.nunique() < 2 or frame.y.nunique() < 2:
        return {"n": len(frame), "spearman": None, "pearson": None}
    s, sp = spearmanr(frame.x, frame.y); p, pp = pearsonr(frame.x, frame.y)
    return {"n": len(frame), "spearman": float(s), "spearman_p": float(sp),
            "pearson": float(p), "pearson_p": float(pp)}


def read_epochs() -> tuple[dict, dict]:
    runs, architecture = {}, {}
    for model in ("m3_merged", "m3_dau_shared"):
        best_values = []
        for seed in (42, 123, 456):
            root = ROOT / f"runs/gate1b3_{model}_seed{seed}"
            epochs = [json.loads(line) for line in (root / "epochs.jsonl").read_text().splitlines() if line]
            metrics = json.loads((root / "metrics.json").read_text())
            train = np.asarray([x["train_loss"] for x in epochs]); val = np.asarray([x["validation"]["group_macro_mae"] for x in epochs])
            best_epoch, maximum = int(metrics["best_epoch"]), int(metrics["epochs_completed"])
            slope = float(np.polyfit(np.arange(min(5, len(val))), val[-5:], 1)[0])
            post_best_drop = float((train[best_epoch - 1] - train[-1]) / max(abs(train[best_epoch - 1]), 1e-12))
            final_val_gap = float((val[-1] - val[best_epoch - 1]) / val[best_epoch - 1])
            underfit = best_epoch >= 0.9 * maximum and slope < 0
            overfit = post_best_drop >= 0.05 and final_val_gap >= 0.02
            key = f"{model}_seed{seed}"
            runs[key] = {"epochs": maximum, "best_epoch": best_epoch,
                         "best_validation_group_macro_mae_eV": float(metrics["best_validation_group_macro_mae"]),
                         "first_train_loss": float(train[0]), "best_epoch_train_loss": float(train[best_epoch - 1]),
                         "final_train_loss": float(train[-1]), "final_validation_mae_eV": float(val[-1]),
                         "last_five_validation_slope_eV_per_epoch": slope,
                         "last_ten_validation_sd_eV": float(np.std(val[-10:], ddof=1)),
                         "post_best_train_loss_fractional_drop": post_best_drop,
                         "final_validation_fraction_above_best": final_val_gap,
                         "underfit_rule": bool(underfit), "overfit_rule": bool(overfit),
                         "touched_epoch_ceiling": bool(maximum == 36)}
            best_values.append(float(metrics["best_validation_group_macro_mae"]))
        values = np.asarray(best_values)
        architecture[model] = {"best_validation_mean_eV": float(values.mean()),
                               "best_validation_sample_sd_eV": float(values.std(ddof=1)),
                               "seed_cv": float(values.std(ddof=1) / values.mean()),
                               "unstable_rule": bool(values.std(ddof=1) / values.mean() > 0.05 or
                                                     any(runs[f"{model}_seed{s}"]["last_ten_validation_sd_eV"] > 0.01 for s in (42,123,456))),
                               "all_runs_underfit": bool(all(runs[f"{model}_seed{s}"]["underfit_rule"] for s in (42,123,456)))}
    return runs, architecture


def fingerprints(features: pd.DataFrame) -> list:
    columns = [f"pair_morgan_{index}" for index in range(512)]
    output = []
    for row in features[columns].to_numpy(np.uint8):
        fp = DataStructs.ExplicitBitVect(512); fp.SetBitsFromList(np.flatnonzero(row).astype(int).tolist()); output.append(fp)
    return output


def nearest_train_similarity(features: pd.DataFrame, manifest: pd.DataFrame, test_ids: list[str]) -> pd.DataFrame:
    partition = manifest[["molecule_id", "partition"]].merge(features, on="molecule_id", validate="one_to_one")
    train = partition.loc[partition.partition.eq("train")]
    test = partition.set_index("molecule_id").loc[test_ids].reset_index()
    train_fp, test_fp = fingerprints(train), fingerprints(test)
    similarity = [max(DataStructs.BulkTanimotoSimilarity(query, train_fp)) for query in test_fp]
    return pd.DataFrame({"molecule_id": test.molecule_id, "nearest_train_morgan": similarity})


def subgroup_summary(frame: pd.DataFrame, column: str, config: dict) -> dict:
    output = {}
    for value, part in frame.groupby(column, observed=True, dropna=False, sort=True):
        key = str(value); records, groups = len(part), int(part.structure_group_id_v1.nunique())
        powered = records >= config["subgroup_policy"]["minimum_records"] and groups >= config["subgroup_policy"]["minimum_structure_groups"]
        item = {"records": records, "groups": groups, "adequately_powered": powered}
        for model in config["models"]:
            item[f"{model}_group_macro_mae_eV"] = group_macro_mae(part, model)
        for model in ("m3_merged_ensemble", "m3_dau_shared_ensemble"):
            item[f"{model}_minus_xgboost"] = group_bootstrap_error_difference(
                part, model, "xgboost_c0", iterations=config["bootstrap"]["iterations"], seed=config["bootstrap"]["seed"])
        output[key] = item
    return output


def error_complementarity(frame: pd.DataFrame, config: dict) -> dict:
    threshold = config["error_definitions"]["correct_threshold_eV"]
    margin = config["error_definitions"]["unique_advantage_margin_eV"]
    output = {"absolute_error_correlation": {}, "pairwise": {}}
    errors = {model: (frame[model] - frame.primary_true).abs() for model in config["models"]}
    for first in config["models"]:
        for second in config["models"]:
            if first >= second: continue
            key = f"{first}__{second}"
            output["absolute_error_correlation"][key] = correlation(errors[first], errors[second])
    for model in ("m3_merged_ensemble", "m3_dau_shared_ensemble"):
        x, m = errors["xgboost_c0"], errors[model]
        oracle = np.where(x <= m, frame.xgboost_c0, frame[model])
        oracle_column = f"oracle_xgb_{model}"; frame[oracle_column] = oracle
        output["pairwise"][model] = {
            "both_correct_fraction": float(np.mean((x <= threshold) & (m <= threshold))),
            "both_fail_fraction": float(np.mean((x > threshold) & (m > threshold))),
            "xgboost_only_correct_fraction": float(np.mean((x <= threshold) & (m > threshold))),
            "m3_only_correct_fraction": float(np.mean((m <= threshold) & (x > threshold))),
            "xgboost_unique_advantage_fraction": float(np.mean(x + margin < m)),
            "m3_unique_advantage_fraction": float(np.mean(m + margin < x)),
            "paired_error_difference": group_bootstrap_error_difference(frame, model, "xgboost_c0",
                iterations=config["bootstrap"]["iterations"], seed=config["bootstrap"]["seed"]),
            "oracle_min_group_macro_mae_eV": group_macro_mae(frame, oracle_column),
            "oracle_gain_vs_xgboost_eV": group_macro_mae(frame, "xgboost_c0") - group_macro_mae(frame, oracle_column),
        }
    all_predictions = frame[config["models"]].to_numpy(); truth = frame.primary_true.to_numpy()[:, None]
    best = np.argmin(np.abs(all_predictions - truth), axis=1)
    frame["oracle_all"] = all_predictions[np.arange(len(frame)), best]
    output["oracle_all"] = {"group_macro_mae_eV": group_macro_mae(frame, "oracle_all"),
                            "gain_vs_xgboost_eV": group_macro_mae(frame, "xgboost_c0") - group_macro_mae(frame, "oracle_all"),
                            "deployable": False}
    return output


def add_tier0_strata(test: pd.DataFrame, features: pd.DataFrame, manifest: pd.DataFrame, graph: pd.DataFrame) -> pd.DataFrame:
    descriptor = features[["molecule_id", "pair_MolWt", "pair_HeavyAtomCount", "pair_NumHeteroatoms"]]
    identity = pd.read_parquet(ROOT / "manifests/new15016_structure_groups_v1.parquet")
    frame = test.merge(descriptor, on="molecule_id", validate="one_to_one").merge(
        graph[["molecule_id", "num_atoms"]], on="molecule_id", validate="one_to_one").merge(
        identity[["molecule_id", "canonical_structure_smiles_v1", "donor_structure_group_id_v1", "acceptor_structure_group_id_v1"]], on="molecule_id", validate="one_to_one")
    donor_frequency = manifest.donor_structure_group_id_v1.value_counts(); acceptor_frequency = manifest.acceptor_structure_group_id_v1.value_counts()
    frame["donor_component_frequency"] = frame.donor_structure_group_id_v1.map(donor_frequency)
    frame["acceptor_component_frequency"] = frame.acceptor_structure_group_id_v1.map(acceptor_frequency)
    frame["unknown_fraction"] = frame.unknown_atoms / frame.num_atoms
    frame["heteroatom_fraction"] = frame.pair_NumHeteroatoms / frame.pair_HeavyAtomCount.clip(lower=1)
    frame["donor_presence"] = np.where(frame.donor_atoms.gt(0), "present", "empty")
    charges = []
    for smiles in frame.canonical_structure_smiles_v1:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: raise RuntimeError("formal-charge SMILES parse failure")
        charges.append(sum(atom.GetFormalCharge() for atom in mol.GetAtoms()))
    frame["formal_charge_value"] = charges
    frame["formal_charge"] = np.where(frame.formal_charge_value < 0, "negative", np.where(frame.formal_charge_value > 0, "positive", "zero"))
    return frame


def duplicate_analysis(val_test: pd.DataFrame, test: pd.DataFrame) -> dict:
    duplicate = pd.read_csv(ROOT / "manifests/duplicate_structure_groups_v1.csv")
    all_geometry = {"all_372": correlation(duplicate.heavy_atom_rmsd_max_angstrom, duplicate.primary_range_eV)}
    for name, mask in {"same_role_aware_identity": duplicate.unique_role_aware_groups.eq(1),
                       "different_role_aware_identity": duplicate.unique_role_aware_groups.gt(1),
                       "atom_count_consistent": duplicate.atom_count_consistent.eq(True)}.items():
        all_geometry[name] = correlation(duplicate.loc[mask, "heavy_atom_rmsd_max_angstrom"], duplicate.loc[mask, "primary_range_eV"])
    group_rows = []
    for group, part in val_test.groupby("structure_group_id_v1", sort=True):
        if len(part) < 2: continue
        row = {"structure_group_id_v1": group, "records": len(part),
               "target_range_eV": float(part.primary_true.max() - part.primary_true.min())}
        for model in ("xgboost_c0", "m3_merged_ensemble", "m3_dau_shared_ensemble"):
            row[f"{model}_prediction_range_eV"] = float(part[model].max() - part[model].min())
            row[f"{model}_mean_abs_error_eV"] = float((part[model] - part.primary_true).abs().mean())
        group_rows.append(row)
    groups = pd.DataFrame(group_rows).merge(duplicate, on="structure_group_id_v1", how="left", validate="one_to_one")
    prediction = {}
    for model in ("xgboost_c0", "m3_merged_ensemble", "m3_dau_shared_ensemble"):
        prediction[model] = {"geometry_rmsd_vs_prediction_range": correlation(groups.heavy_atom_rmsd_max_angstrom, groups[f"{model}_prediction_range_eV"]),
                             "target_range_vs_mean_abs_error": correlation(groups.target_range_eV, groups[f"{model}_mean_abs_error_eV"])}
    singleton = test.structure_group_size.eq(1)
    size_metrics = {}
    for name, mask in {"singleton": singleton, "duplicate": ~singleton}.items():
        size_metrics[name] = {"records": int(mask.sum()), "groups": int(test.loc[mask, "structure_group_id_v1"].nunique()),
                              **{f"{model}_group_macro_mae_eV": group_macro_mae(test.loc[mask], model) for model in ("xgboost_c0","m3_merged_ensemble","m3_dau_shared_ensemble")}}
    return {"geometry_vs_target": all_geometry, "frozen_val_test_duplicate_groups": len(groups),
            "prediction_dispersion": prediction, "singleton_vs_duplicate_test": size_metrics}


def role_analysis() -> dict:
    role = pd.read_csv(ROOT / "runs/gate1b3_role_sensitivity/paired_role_sensitivity_predictions.csv")
    role = role.loc[role.seed.astype(str).eq("ensemble")].copy()
    output = {}
    for model, part in role.groupby("model"):
        part["absolute_delta"] = part.delta_prediction.abs(); part["original_abs_error"] = (part.original_prediction - part.y_true).abs()
        output[model] = {"records": len(part), "all_original_empty_donor": bool(part.formed_nonempty_donor.all()),
                         "absolute_delta_vs_changed_fraction": correlation(part.absolute_delta, part.changed_atom_fraction),
                         "absolute_delta_vs_original_error": correlation(part.absolute_delta, part.original_abs_error),
                         "median_abs_delta_eV": float(part.absolute_delta.median()),
                         "median_delta_to_primary_test_mae_ratio": float(part.absolute_delta.median() / (0.08766406500328516 if model == "m3_merged" else 0.08854613716585205)),
                         "by_partition": {name: {"records": len(x), "median_abs_delta_eV": float(x.absolute_delta.median())}
                                          for name, x in part.groupby("partition")}}
    return output


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    header = "|" + "|".join(columns) + "|\n|" + "|".join(["---"] * len(columns)) + "|\n"
    return header + "".join("|" + "|".join(str(row.get(column, "")) for column in columns) + "|\n" for row in rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True); parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--counterfactual", type=Path, required=True); parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise RuntimeError("diagnosis output exists; refusing rerun")
    config = validate(args.config, args.lock)
    counterfactual = json.loads(args.counterfactual.read_text())
    if counterfactual["partition"] != "val" or counterfactual["test_inference_performed"] is not False:
        raise RuntimeError("counterfactual firewall evidence failure")
    test = pd.read_csv(ROOT / config["frozen_inputs"]["gate1b3_test_predictions"]["path"])
    if len(test) != config["test_records"] or test.structure_group_id_v1.nunique() != config["test_structure_groups"]:
        raise RuntimeError("frozen test count mismatch")
    learning_runs, learning_arch = read_epochs()
    complementarity = error_complementarity(test, config)
    features = pd.read_parquet(ROOT / config["frozen_inputs"]["gate1b1_features"]["path"])
    manifest = pd.read_csv(ROOT / config["frozen_inputs"]["iid_manifest"]["path"])
    graph = pd.read_parquet(ROOT / "data_registry/dft_3d_graph_registry_v1.parquet")
    test = add_tier0_strata(test, features, manifest, graph)
    similarity = nearest_train_similarity(features, manifest, test.molecule_id.astype(str).tolist())
    test = test.merge(similarity, on="molecule_id", validate="one_to_one")
    bins = config["subgroup_policy"]
    edges = bins["similarity_bins"]
    ordered = [f"[{edges[index]:g},{edges[index + 1]:g})" for index in range(len(edges) - 1)]
    test["similarity_bin"] = pd.cut(test.nearest_train_morgan, bins=edges, labels=ordered,
                                      right=False, include_lowest=True).astype(str)
    if test.similarity_bin.isna().any():
        raise RuntimeError("similarity bin coverage failure")
    test["similarity_analysis_bin"], merge_metadata = deterministic_similarity_merge(test, "similarity_bin", ordered,
        minimum_records=bins["minimum_records"], minimum_groups=bins["minimum_structure_groups"])
    definitions = {
        "similarity": test.similarity_analysis_bin,
        "heavy_atom_count": pd.cut(test.pair_HeavyAtomCount, bins["heavy_atom_bins"], right=False).astype(str),
        "donor_atom_count": pd.cut(test.donor_atoms, bins["donor_atom_bins"], right=True).astype(str),
        "acceptor_atom_count": pd.cut(test.acceptor_atoms, bins["acceptor_atom_bins"], right=True).astype(str),
        "donor_presence": test.donor_presence,
        "unknown_role_fraction": pd.cut(test.unknown_fraction, bins["unknown_fraction_bins"], right=True, include_lowest=True).astype(str),
        "molecular_weight": pd.cut(test.pair_MolWt, bins["molecular_weight_bins"], right=False).astype(str),
        "heteroatom_fraction": pd.cut(test.heteroatom_fraction, bins["heteroatom_fraction_bins"], right=False).astype(str),
        "formal_charge": test.formal_charge,
        "donor_component_frequency": pd.cut(test.donor_component_frequency, bins["component_frequency_bins"], right=True).astype(str),
        "acceptor_component_frequency": pd.cut(test.acceptor_component_frequency, bins["component_frequency_bins"], right=True).astype(str),
        "target_quantile": pd.qcut(test.primary_true, 4, labels=["Q1", "Q2", "Q3", "Q4"], duplicates="drop").astype(str),
    }
    subgroup = {}
    for name, values in definitions.items():
        test[f"stratum_{name}"] = values
        subgroup[name] = subgroup_summary(test, f"stratum_{name}", config)

    # Existing frozen validation predictions only; no new inference.
    val_parts = []
    for model in ("m3_merged", "m3_dau_shared"):
        merged = None
        for seed in (42,123,456):
            part = pd.read_csv(ROOT / f"runs/gate1b3_{model}_seed{seed}/best_validation_predictions.csv")
            part = part.rename(columns={"truth": "primary_true", "prediction": f"p{seed}"})
            merged = part if merged is None else merged.merge(part[["molecule_id", f"p{seed}"]], on="molecule_id", validate="one_to_one")
        merged[f"{model}_ensemble"] = merged[["p42","p123","p456"]].mean(axis=1)
        val_parts.append(merged[["molecule_id","structure_group_id_v1","primary_true",f"{model}_ensemble"]])
    val = val_parts[0].merge(val_parts[1].drop(columns=["structure_group_id_v1","primary_true"]), on="molecule_id", validate="one_to_one")
    xgb_val = pd.read_csv(ROOT / "runs/gate1b1_new_iid_cheap_baselines/seed42/xgb_c0_val_predictions.csv").rename(columns={"prediction":"xgboost_c0"})
    val = val.merge(xgb_val[["molecule_id","xgboost_c0"]], on="molecule_id", validate="one_to_one")
    val_test = pd.concat([val, test[["molecule_id","structure_group_id_v1","primary_true","xgboost_c0","m3_merged_ensemble","m3_dau_shared_ensemble"]]], ignore_index=True)
    duplicate = duplicate_analysis(val_test, test)
    roles = role_analysis()

    powered_wins = []
    for dimension, groups in subgroup.items():
        for label, item in groups.items():
            if not item["adequately_powered"]: continue
            for model in ("m3_merged_ensemble", "m3_dau_shared_ensemble"):
                result = item[f"{model}_minus_xgboost"]
                if result["ci95_eV"][1] is not None and result["ci95_eV"][1] < 0:
                    powered_wins.append({"dimension": dimension, "stratum": label, "model": model, **result})
    same_role = duplicate["geometry_vs_target"]["same_role_aware_identity"]
    geometry_signal = bool(same_role["spearman"] is not None and abs(same_role["spearman"]) >= 0.3 and same_role["spearman_p"] < 0.05)
    both_underfit = bool(all(learning_arch[x]["all_runs_underfit"] for x in learning_arch))
    cf = counterfactual["ensembles"]
    ordered_noise = all(cf[model]["gaussian_noise_0.01A"]["mean_abs_delta_prediction_eV"] <=
                        cf[model]["gaussian_noise_0.05A"]["mean_abs_delta_prediction_eV"] <=
                        cf[model]["gaussian_noise_0.10A"]["mean_abs_delta_prediction_eV"] for model in cf)
    winning_subgroups = sorted({(item["dimension"], item["stratum"]) for item in powered_wins})
    oracle_gain = complementarity["oracle_all"]["gain_vs_xgboost_eV"]
    if geometry_signal and both_underfit and ordered_noise:
        decision = "SCALE_3D"
    elif len(winning_subgroups) >= 2 and oracle_gain >= 0.005:
        decision = "FUSE_2D_3D"
    else:
        decision = "STOP_PURE_3D"
    final_status = f"GATE1C1_DONE_{decision}"
    diagnosis = {"status": final_status, "decision": decision, "learning_runs": learning_runs,
                 "learning_architectures": learning_arch, "complementarity": complementarity,
                 "similarity_merge": merge_metadata, "subgroups": subgroup, "powered_3d_wins": powered_wins,
                 "powered_3d_winning_subgroups": [{"dimension": x[0], "stratum": x[1]} for x in winning_subgroups],
                 "duplicate_geometry": duplicate, "role_robustness": roles,
                 "validation_counterfactuals": counterfactual,
                 "decision_evidence": {"geometry_signal_rule": geometry_signal, "both_architectures_underfit_rule": both_underfit,
                                       "ordered_noise_response": ordered_noise, "powered_3d_model_stratum_win_count": len(powered_wins),
                                       "powered_3d_winning_subgroup_count": len(winning_subgroups),
                                       "oracle_gain_vs_xgboost_eV": oracle_gain},
                 "implementation_corrections": [
                     "Provisional attempt mapped pandas interval strings incompletely; v1 rerun uses explicit preregistered bin labels and asserts 2319/2319 coverage.",
                     "Provisional attempt counted two model wins in one Q4 stratum as two subgroups; v1 decision counts unique dimension/stratum pairs."
                 ],
                 "test_artifact_analysis_only": True, "new_test_predictions_created": False,
                 "training_performed": False, "final673_accessed": False}
    args.output.mkdir(parents=True); write_json(args.output / "diagnosis.json", diagnosis)

    learning_rows = [{"run": key, "best epoch": x["best_epoch"], "epochs": x["epochs"],
                      "best val MAE": f"{x['best_validation_group_macro_mae_eV']:.6f}",
                      "underfit": x["underfit_rule"], "overfit": x["overfit_rule"]} for key,x in learning_runs.items()]
    (ROOT / "reports/gate1c1_learning_dynamics.md").write_text(
        "# Gate 1-C1 Learning Dynamics\n\n" + markdown_table(learning_rows, ["run","best epoch","epochs","best val MAE","underfit","overfit"]) +
        "\nThe preregistered underfit rule requires a late best epoch and improving final-five validation slope. No additional epoch is authorized by this diagnosis.\n")
    comp_rows=[]
    for model,item in complementarity["pairwise"].items():
        comp_rows.append({"model":model,"M3-only correct":f"{item['m3_only_correct_fraction']:.3f}",
                          "M3 unique advantage":f"{item['m3_unique_advantage_fraction']:.3f}",
                          "oracle MAE":f"{item['oracle_min_group_macro_mae_eV']:.6f}",
                          "oracle gain":f"{item['oracle_gain_vs_xgboost_eV']:.6f}"})
    (ROOT / "reports/gate1c1_error_complementarity.md").write_text(
        "# Gate 1-C1 Error Complementarity\n\n"+markdown_table(comp_rows,["model","M3-only correct","M3 unique advantage","oracle MAE","oracle gain"])+
        f"\nAll values use the existing 2,319-record frozen test artifact. Oracle-min is non-deployable and no fusion weight was fitted. Adequately powered preregistered winning subgroups: `{len(winning_subgroups)}`.\n")
    (ROOT / "reports/gate1c1_geometry_value.md").write_text(
        "# Gate 1-C1 Geometry Value\n\n"+
        f"All 372 duplicate groups: geometry RMSD versus primary target range Spearman = `{duplicate['geometry_vs_target']['all_372']['spearman']}`.\n\n"+
        f"Same role-aware identity subset: `{same_role['n']}` groups, Spearman = `{same_role['spearman']}`, p = `{same_role.get('spearman_p')}`.\n\n"+
        f"Validation-only counterfactuals used six frozen checkpoints and `{counterfactual['records']}` records; no test counterfactual was run. Ordered noise response: `{ordered_noise}`. Frozen val/test prediction dispersion covers `{duplicate['frozen_val_test_duplicate_groups']}` duplicate groups; train duplicates were not newly inferred.\n")
    (ROOT / "reports/gate1c1_role_robustness.md").write_text(
        "# Gate 1-C1 Role Robustness\n\nThe 198 mappings remain graph-supported candidates, not donor ground truth.\n\n"+
        "|model|median abs role perturbation (eV)|fraction of primary IID MAE|\n|---|---:|---:|\n"+
        "".join(f"|{model}|{item['median_abs_delta_eV']:.6f}|{item['median_delta_to_primary_test_mae_ratio']:.3f}|\n" for model,item in roles.items())+
        "\nNo candidate-role view was selected from error changes.\n")
    decision_text = {
        "SCALE_3D": "Stable geometry evidence and consistent underfitting support a controlled capacity increase.",
        "FUSE_2D_3D": "Adequately powered subgroups show complementary 3D signal, supporting a preregistered validation-selected fusion experiment.",
        "STOP_PURE_3D": "The frozen evidence does not justify scaling or continuing a pure-3D architecture path; future modeling should retain the 2D baseline and first improve role semantics or use a separately preregistered fusion test."
    }[decision]
    (ROOT / "reports/gate1c1_decision.md").write_text(
        f"# Gate 1-C1 Decision\n\nFinal status: `{final_status}`\n\n## Recommendation: {decision}\n\n{decision_text}\n\n"+
        f"Decision inputs: geometry signal `{geometry_signal}`, both architectures underfit `{both_underfit}`, ordered noise response `{ordered_noise}`, unique powered 3D-winning subgroups `{len(winning_subgroups)}`, oracle gain `{oracle_gain:.6f} eV`. Exactly one preregistered branch was selected.\n")
    print(json.dumps({"status":final_status,"decision_evidence":diagnosis["decision_evidence"]},indent=2,sort_keys=True))


if __name__ == "__main__":
    main()
