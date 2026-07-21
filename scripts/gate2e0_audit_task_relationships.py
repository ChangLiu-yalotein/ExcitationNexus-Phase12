#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from gate2e0_common import ROOT, load_config, load_primary_labels, load_protocol_aux, standardized_linear_residual_rmse, weighted_corr, weighted_spearman, write_json


def main() -> None:
    config = load_config(); primary = config["primary"]
    frame, _ = load_protocol_aux(config, "iid", "train")
    frame = frame.merge(load_primary_labels(config), on="molecule_id", validate="one_to_one")
    tasks = [primary, *config["secondary"], *config["masked"]]
    pairs = []; pearson = np.full((len(tasks), len(tasks)), np.nan); spearman = pearson.copy(); residual = pearson.copy(); counts = np.zeros_like(pearson, dtype=int)
    for i, left in enumerate(tasks):
        for j, right in enumerate(tasks):
            if i == j:
                valid = frame[[left, "group_weight"]].dropna()
                counts[i, j] = len(valid)
                if len(valid) > 1 and valid[left].nunique() > 1:
                    pearson[i, j] = 1.0; spearman[i, j] = 1.0; residual[i, j] = 0.0
                continue
            valid = frame[[left, right, "group_weight"]].dropna()
            x, y, w = valid[left].to_numpy(float), valid[right].to_numpy(float), valid.group_weight.to_numpy(float)
            counts[i, j] = len(valid)
            if len(valid) > 1:
                pearson[i, j] = weighted_corr(x, y, w); spearman[i, j] = weighted_spearman(x, y, w); residual[i, j] = standardized_linear_residual_rmse(x, y, w)
            if i < j:
                pairs.append({"left": left, "right": right, "pair_count": len(valid), "weighted_pearson": pearson[i, j], "weighted_spearman": spearman[i, j], "standardized_linear_residual_rmse": residual[i, j]})

    complete = frame[[*config["secondary"], "group_weight"]].dropna()
    x = complete[config["secondary"]].to_numpy(float); w = complete.group_weight.to_numpy(float)
    mean = np.sum(x * w[:, None], axis=0) / np.sum(w); std = np.sqrt(np.sum(w[:, None] * (x - mean) ** 2, axis=0) / np.sum(w))
    standardized = (x - mean) / std
    covariance = (standardized * w[:, None]).T @ standardized / np.sum(w)
    eig = np.linalg.eigvalsh(covariance); positive = eig[eig > 1e-12]
    effective_tasks = float((eig.sum() ** 2) / np.sum(eig**2)); condition = float(positive.max() / positive.min())

    pair_df = pd.DataFrame(pairs)
    gate = config["redundancy_gate"]
    numeric_candidates = pair_df.loc[(pair_df.weighted_spearman.abs() >= gate["absolute_spearman_min"]) & (pair_df.standardized_linear_residual_rmse <= gate["standardized_linear_residual_rmse_max"])]
    ledger = pd.read_csv(ROOT / "data_registry/gate2e0_target_admission_ledger.csv")
    admitted_secondary = ledger.loc[ledger.allowed_role.eq("OPTIMIZATION_ALLOWED"), "canonical_name"].tolist()
    admitted_masked = ledger.loc[ledger.allowed_role.eq("MASKED_ONLY"), "canonical_name"].tolist()
    graph = {
        "version": "gate2e0_target_graph_v2", "primary": {"column": primary, "report_name": config["primary_report_name"], "weight": 1.0},
        "secondary_optimization": admitted_secondary, "secondary_total_weight": 0.5, "secondary_per_task_weight": 0.5 / len(admitted_secondary),
        "masked_auxiliary": admitted_masked, "masked_total_weight": 0.25, "masked_per_task_weight": 0.25 / len(admitted_masked),
        "report_only_deterministic": config["report_only_deterministic"], "report_only_redundant": ["tddft_t_index_angstrom"], "disabled": config["disabled"],
        "normalization": "protocol_train_only_group_weighted_observed_label_mean_std", "missing_policy": "mask_only", "dynamic_weighting": False,
    }
    write_json("data_registry/gate2e0_target_graph_v2.json", graph)
    relationship = {"completed_utc": datetime.now(timezone.utc).isoformat(), "partition": "iid_train_only", "tasks": tasks, "pairwise": pairs, "numeric_redundancy_gate_candidates": numeric_candidates.to_dict(orient="records"), "known_multivariate_relation": {"tddft_t_index_angstrom": "t = D - H_CT within frozen source rounding"}, "secondary_covariance_rank": int(np.linalg.matrix_rank(covariance)), "secondary_covariance_condition_number": condition, "effective_number_of_secondary_tasks": effective_tasks, "complete_secondary_records": len(complete), "validation_used": False, "model_fitted": False, "primary_residual_generated": False}
    write_json("logs/gate2e0_task_relationships.json", relationship)

    missing = json.loads((ROOT / "logs/gate2e0_missingness.json").read_text())
    iid_train = missing["protocols"]["iid"]["train"]
    acceptor_val = missing["protocols"]["acceptor_cold"]["val"]
    full_secondary = sum(iid_train[t]["structure_group_weighted_completeness"] >= .95 for t in admitted_secondary)
    sufficient_masked = sum(iid_train[t]["structure_group_weighted_completeness"] >= .45 for t in admitted_masked)
    acceptor_identity_counts = {t: acceptor_val[t].get("acceptor_structure_group_id_v1_with_label", 0) for t in admitted_secondary + admitted_masked}
    decision = "MULTITASK_TARGET_GRAPH_ADMITTED" if full_secondary >= 8 and sufficient_masked >= 2 and min(acceptor_identity_counts.values()) >= 30 else "MULTITASK_TARGET_GRAPH_REDUCED"
    evidence = {"status": "GATE2E0_DONE", "scientific_decision": decision, "admitted_secondary": len(admitted_secondary), "admitted_masked": len(admitted_masked), "secondary_ge95pct": full_secondary, "masked_ge45pct": sufficient_masked, "acceptor_cold_validation_identity_counts": acceptor_identity_counts, "primary_equivalent_auxiliary_in_loss": False, "source_arrow_reads": 1, "model_training": False, "prediction_generation": False, "gpu_used": False, "test_artifact_accessed": False, "final673_accessed": False}
    write_json("logs/gate2e0_evidence.json", evidence)

    (ROOT / "reports/gate2e0_task_relationships.md").write_text("\n".join(["# Gate 2-E0 train-only task relationships", "", f"All correlations use IID train with group weights. The 12-secondary standardized covariance rank is {relationship['secondary_covariance_rank']}; condition number is {condition:.4g}; effective task count is {effective_tasks:.3f}.", "", f"The fixed pairwise numeric redundancy gate produced {len(numeric_candidates)} candidate pair(s). Physical/schema support remains mandatory. Separately, the known multivariate descriptor identity `t = D - H_CT` makes `tddft_t_index_angstrom` report-only redundant.", "", "No model, primary residual, validation correlation, or prediction was produced."]) + "\n")
    (ROOT / "reports/gate2e0_multitask_feasibility.md").write_text("\n".join(["# Gate 2-E0 multitask feasibility", "", f"Decision: `{decision}`.", "", f"Admitted optimization graph: one primary, {len(admitted_secondary)} non-redundant secondary targets, and {len(admitted_masked)} masked fragment targets. Of these, {full_secondary} secondaries exceed 95% IID-train structure-group completeness and {sufficient_masked} masked targets exceed 45%.", "", "Gate 2-E1 frozen initial weights: primary 1.0; secondary total 0.5 (equal per admitted task); masked total 0.25 (equal per admitted task). All normalization is protocol-train-only, group-weighted, and observed-label-only. Dynamic loss weighting is prohibited in the first admission experiment.", "", "Validation was used only for coverage/evaluability; no validation relationship or performance selected tasks."]) + "\n")
    (ROOT / "reports/gate2e0_final_decision.md").write_text("\n".join(["# Gate 2-E0 final decision", "", f"## `{decision}`", "", "The primary Coulomb proxy remains the sole optimized member of its unit/dielectric algebra family. Eleven non-redundant TDDFT secondary targets are admitted; `t_index` is retained only for reporting because it is derived from D and H_CT within source rounding. All four fragment fractions remain masked-only because donor/acceptor sums expose nonzero unknown/unassigned contributions in a subset of records.", "", "The target graph is suitable for a fixed-weight, validation-only Gate 2-E1 admission experiment using frozen C0-512 inputs. This decision does not establish prediction improvement and authorizes no test access."]) + "\n")


if __name__ == "__main__":
    main()
