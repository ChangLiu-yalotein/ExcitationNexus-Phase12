from __future__ import annotations

import hashlib
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from .collate import collate_phase12
from .dft_graph_dataset import DFTGraphCache
from .edge_cache import ShardedEdgeCache, sha256_file
from .gate1b3_evaluation import build_model
from .gate1b3_pipeline import read_targets

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")
RESOLVED = "RESOLVED_UNIQUE_OR_SYMMETRY_EQUIVALENT"


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def parsed_indices(value: str) -> np.ndarray:
    return np.asarray([int(x) - 1 for x in str(value).split(";") if x], dtype=np.int64)


def candidate_role_tensor(original: torch.Tensor, one_based_indices: str) -> torch.Tensor:
    candidate = original.clone(); indices = torch.from_numpy(parsed_indices(one_based_indices)).long()
    if not len(indices) or int(indices.min()) < 0 or int(indices.max()) >= len(original):
        raise ValueError("candidate atom index outside frozen graph")
    if not original[indices].eq(2).all():
        raise ValueError("candidate donor atom was not original unknown")
    candidate[indices] = 0
    changed = candidate.ne(original)
    if int(changed.sum()) != len(indices) or not torch.equal(torch.where(changed)[0], indices.sort().values):
        raise ValueError("only-role-changed assertion failed")
    return candidate


def reconcile(original: np.ndarray, frozen: np.ndarray, tolerance: float) -> float:
    if original.shape != frozen.shape:
        raise RuntimeError("BLOCKED_SENSITIVITY_MISMATCH: prediction shape")
    maximum = float(np.max(np.abs(original - frozen))) if len(original) else 0.0
    if maximum > tolerance:
        raise RuntimeError(f"BLOCKED_SENSITIVITY_MISMATCH: {maximum} > {tolerance}")
    return maximum


class RoleViewDataset(Dataset):
    def __init__(self, graph: DFTGraphCache, edges: ShardedEdgeCache, frame: pd.DataFrame, *, candidate: bool):
        self.graph, self.edges = graph, edges
        self.frame = frame.sort_values(["partition", "molecule_id"], kind="mergesort").reset_index(drop=True)
        self.candidate = candidate

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Data:
        row = self.frame.iloc[int(index)]; source = self.graph.registry.iloc[int(row.cache_index)]
        a, b = int(source.atom_offset_start), int(source.atom_offset_end)
        role = torch.from_numpy(self.graph.role[a:b].copy()).long()
        if self.candidate:
            role = candidate_role_tensor(role, row.resolved_donor_atom_indices)
        return Data(z=torch.from_numpy(self.graph.z[a:b].copy()).long(),
                    pos=torch.from_numpy(self.graph.pos[a:b].copy()).float(), role=role,
                    edge_index=self.edges.edge_index(str(row.molecule_id)), num_nodes=b-a,
                    molecule_id=str(row.molecule_id), partition=str(row.partition))


def prediction_summary(delta: np.ndarray, thresholds: list[float]) -> dict:
    absolute = np.abs(np.asarray(delta, dtype=np.float64))
    return {"records": len(absolute), "mean_signed_delta_eV": float(np.mean(delta)),
            "median_signed_delta_eV": float(np.median(delta)),
            "median_abs_delta_eV": float(np.median(absolute)),
            "p90_abs_delta_eV": float(np.quantile(absolute, .90)),
            "p95_abs_delta_eV": float(np.quantile(absolute, .95)),
            "max_abs_delta_eV": float(np.max(absolute)),
            "threshold_fractions": {str(value): float(np.mean(absolute > value)) for value in thresholds}}


def group_bootstrap(values: pd.DataFrame, column: str, *, seed: int, iterations: int) -> dict:
    group_values = values.groupby("structure_group_id_v1", sort=True)[column].mean().to_numpy(np.float64)
    rng = np.random.default_rng(seed); samples = np.empty(iterations)
    for index in range(iterations):
        samples[index] = group_values[rng.integers(0, len(group_values), len(group_values))].mean()
    low, high = np.quantile(samples, [.025, .975])
    return {"groups": len(group_values), "point_mean_eV": float(group_values.mean()),
            "ci95_percentile_eV": [float(low), float(high)], "iterations": iterations, "seed": seed,
            "ci_excludes_zero": bool(low > 0 or high < 0)}


def error_summary(frame: pd.DataFrame, unchanged_tolerance: float, bootstrap: dict) -> dict:
    delta = frame.delta_absolute_error.to_numpy(np.float64)
    return {"records": len(frame), "mean_delta_absolute_error_eV": float(delta.mean()),
            "median_delta_absolute_error_eV": float(np.median(delta)),
            "improved_fraction": float(np.mean(delta < -unchanged_tolerance)),
            "worsened_fraction": float(np.mean(delta > unchanged_tolerance)),
            "unchanged_fraction": float(np.mean(np.abs(delta) <= unchanged_tolerance)),
            "group_bootstrap": group_bootstrap(frame, "delta_absolute_error", **bootstrap)}


def infer(model, loader: DataLoader, normalization: dict, device: torch.device) -> tuple[list[str], np.ndarray]:
    ids, values = [], []; model.eval()
    with torch.inference_mode():
        for batch in loader:
            ids.extend(list(batch.molecule_id)); batch = batch.to(device)
            values.extend((model(batch) * normalization["std"] + normalization["mean"]).cpu().tolist())
    return ids, np.asarray(values, dtype=np.float64)


def run(config_path: Path, output: Path, physical_gpu: int) -> dict:
    if output.exists():
        raise RuntimeError("role sensitivity output exists; refusing rerun")
    config = json.loads(config_path.read_text())
    frozen = {ROOT / config["role_manifest"]: config["role_manifest_sha256"],
              ROOT / "data_registry/role_resolution_spec_v1.json": config["role_spec_sha256"],
              ROOT / "data_registry/dft_3d_graph_registry_v1.parquet": config["graph_registry_sha256"],
              ROOT / "runs/gate1b2_3d_admission/dft_graph_cache_v1.npz": config["graph_cache_sha256"],
              ROOT / "data_registry/gate1b3_model_registry.json": config["model_registry_sha256"],
              ROOT / "data_registry/gate1b3_test_unlock_v1.json": config["test_unlock_sha256"],
              ROOT / config["frozen_test_predictions"]: config["frozen_test_predictions_sha256"],
              ROOT / "runs/gate1b3_test_once/metrics.json": config["frozen_test_metrics_sha256"]}
    for path, expected in frozen.items():
        if sha256_file(path) != expected: raise RuntimeError(f"frozen sensitivity input changed: {path}")
    role = pd.read_csv(ROOT / config["role_manifest"])
    if int(role.resolution_status.eq("UNRESOLVED_AMBIGUOUS").sum()) != config["excluded_records"]:
        raise RuntimeError("unresolved exclusion count changed")
    selected = role.loc[role.resolution_status.eq(config["included_status"])].copy()
    if len(selected) != 198 or selected.molecule_id.nunique() != 198:
        raise RuntimeError("resolved candidate identity failure")
    counts = selected.partition.value_counts().to_dict()
    if counts != {k: v for k, v in config["expected_partition_counts"].items() if v}:
        raise RuntimeError(f"candidate partition counts changed: {counts}")
    if selected.partition.eq("historical_quarantine").any():
        raise RuntimeError("quarantine candidate cannot enter sensitivity Dataset")

    manifest = pd.read_csv(ROOT / "manifests/split_iid_group_seed42_v1.csv")
    graph = DFTGraphCache(ROOT / "runs/gate1b2_3d_admission/dft_graph_cache_v1.npz",
                          ROOT / "data_registry/dft_3d_graph_registry_v1.parquet")
    edges = ShardedEdgeCache(ROOT / "data_registry/gate1b3_edge_cache_registry.json", verify_hashes=True)
    frame = selected.merge(manifest[["molecule_id", "structure_group_id_v1", "structure_group_size",
                                     "donor_structure_group_id_v1"]], on="molecule_id", validate="one_to_one")
    frame = frame.merge(graph.registry[["molecule_id", "cache_index", "num_atoms", "graph_content_sha256"]],
                        on="molecule_id", validate="one_to_one")
    changed_counts, candidate_hashes = [], []
    for row in frame.itertuples(index=False):
        source = graph.registry.iloc[int(row.cache_index)]; a, b = int(source.atom_offset_start), int(source.atom_offset_end)
        original = torch.from_numpy(graph.role[a:b].copy()).long(); candidate = candidate_role_tensor(original, row.resolved_donor_atom_indices)
        changed_counts.append(int(candidate.ne(original).sum()))
        digest = hashlib.sha256(); digest.update(candidate.numpy().tobytes()); candidate_hashes.append(digest.hexdigest())
        if b-a != int(row.num_atoms) or int(candidate.eq(0).sum()) != int(row.resolved_donor_count):
            raise RuntimeError("candidate atom count/role count mismatch")
    frame["changed_atom_count"] = changed_counts; frame["changed_atom_fraction"] = frame.changed_atom_count / frame.num_atoms
    frame["candidate_role_sha256"] = candidate_hashes; frame["formed_nonempty_donor"] = frame.resolved_donor_count.gt(0)
    group_consistency = frame.groupby("structure_group_id_v1").agg(
        records=("molecule_id", "size"), candidate_role_hashes=("candidate_role_sha256", "nunique"),
        atom_counts=("num_atoms", "nunique"))
    inconsistent_groups = group_consistency.loc[(group_consistency.records > 1) &
                                                 ((group_consistency.candidate_role_hashes > 1) |
                                                  (group_consistency.atom_counts > 1))]

    original_ds = RoleViewDataset(graph, edges, frame, candidate=False)
    candidate_ds = RoleViewDataset(graph, edges, frame, candidate=True)
    original_loader = DataLoader(original_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_phase12)
    candidate_loader = DataLoader(candidate_ds, batch_size=16, shuffle=False, num_workers=0, collate_fn=collate_phase12)
    ordered = original_ds.frame; expected_ids = ordered.molecule_id.astype(str).tolist()
    frozen_test = pd.read_csv(ROOT / config["frozen_test_predictions"])
    truth = pd.DataFrame({"molecule_id": ordered.molecule_id, "partition": ordered.partition})
    train_ids = truth.loc[truth.partition.eq("train"), "molecule_id"].astype(str).tolist()
    train_target = read_targets("/home/changliu/ExcitationNexus_Data_v2/tables/molecule_values_v3.parquet",
                                "tddft_coulomb_attraction_eV_eps3p5_proxy", train_ids,
                                requested_partition="train", allow_test=False)
    truth = truth.merge(train_target.rename(columns={"target": "train_truth"}), on="molecule_id", how="left", validate="one_to_one")
    val_source = pd.read_csv(ROOT / "runs/gate1b3_m3_merged_seed42/best_validation_predictions.csv")[["molecule_id", "truth"]]
    truth = truth.merge(val_source.rename(columns={"truth": "val_truth"}), on="molecule_id", how="left", validate="one_to_one")
    truth = truth.merge(frozen_test[["molecule_id", "primary_true"]].rename(columns={"primary_true": "test_truth"}),
                        on="molecule_id", how="left", validate="one_to_one")
    truth["y_true"] = truth.train_truth.fillna(truth.val_truth).fillna(truth.test_truth)
    if truth.y_true.isna().any(): raise RuntimeError("sensitivity truth-source failure")

    device = torch.device("cuda:0"); rows, reconciliation = [], {}
    started = time.perf_counter()
    for model_name in config["models"]:
        for seed in config["seeds"]:
            label = f"{model_name}_seed{seed}"; run_root = ROOT / f"runs/gate1b3_{model_name}_seed{seed}"
            checkpoint = torch.load(run_root / "best_checkpoint.pt", map_location="cpu", weights_only=False)
            normalization = json.loads((run_root / "normalization.json").read_text())["statistics"]
            model = build_model(json.loads((ROOT / "configs/gate1b3_formal_training_v1.json").read_text()), model_name)
            model.load_state_dict(checkpoint["model"]); model.to(device)
            original_ids, original_prediction = infer(model, original_loader, normalization, device)
            candidate_ids, candidate_prediction = infer(model, candidate_loader, normalization, device)
            if original_ids != expected_ids or candidate_ids != expected_ids:
                raise RuntimeError("paired batch order changed")
            reference = np.full(len(ordered), np.nan)
            val_artifact = pd.read_csv(run_root / "best_validation_predictions.csv").set_index("molecule_id")
            test_artifact = frozen_test.set_index("molecule_id")
            val_mask = ordered.partition.eq("val").to_numpy(); test_mask = ordered.partition.eq("test").to_numpy()
            reference[val_mask] = val_artifact.loc[ordered.loc[val_mask, "molecule_id"], "prediction"].to_numpy()
            reference[test_mask] = test_artifact.loc[ordered.loc[test_mask, "molecule_id"], label].to_numpy()
            check_mask = val_mask | test_mask
            reconciliation[label] = reconcile(original_prediction[check_mask], reference[check_mask],
                                               config["original_prediction_reconciliation_tolerance_eV"])
            for index, row in ordered.iterrows():
                delta = candidate_prediction[index] - original_prediction[index]
                y = float(truth.iloc[index].y_true)
                rows.append({"molecule_id": row.molecule_id, "partition": row.partition,
                             "structure_group_id_v1": row.structure_group_id_v1,
                             "structure_group_size": row.structure_group_size,
                             "donor_structure_group_id_v1": row.donor_structure_group_id_v1,
                             "changed_atom_count": row.changed_atom_count,
                             "changed_atom_fraction": row.changed_atom_fraction,
                             "formed_nonempty_donor": row.formed_nonempty_donor,
                             "model": model_name, "seed": seed, "label": label, "y_true": y,
                             "original_prediction": original_prediction[index],
                             "candidate_prediction": candidate_prediction[index], "delta_prediction": delta,
                             "delta_absolute_error": abs(candidate_prediction[index]-y)-abs(original_prediction[index]-y)})
    result_frame = pd.DataFrame(rows)
    ensemble_rows = []
    for model_name in config["models"]:
        subset = result_frame[result_frame.model.eq(model_name)]
        for molecule_id, group in subset.groupby("molecule_id", sort=False):
            base = group.iloc[0]; original = group.original_prediction.mean(); candidate = group.candidate_prediction.mean(); y=base.y_true
            ensemble_rows.append({**{key: base[key] for key in ["molecule_id","partition","structure_group_id_v1",
                                  "structure_group_size","donor_structure_group_id_v1","changed_atom_count",
                                  "changed_atom_fraction","formed_nonempty_donor"]},
                                  "model": model_name, "seed": "ensemble", "label": f"{model_name}_ensemble",
                                  "y_true": y, "original_prediction": original, "candidate_prediction": candidate,
                                  "delta_prediction": candidate-original,
                                  "delta_absolute_error": abs(candidate-y)-abs(original-y)})
    full = pd.concat([result_frame, pd.DataFrame(ensemble_rows)], ignore_index=True)
    summaries = {}; bootstrap_args={"seed":config["bootstrap"]["seed"],"iterations":config["bootstrap"]["iterations"]}
    for label, values in full.groupby("label", sort=True):
        summaries[label] = {}
        for partition, part in values.groupby("partition", sort=True):
            summaries[label][partition] = {"prediction": prediction_summary(part.delta_prediction.to_numpy(), config["absolute_prediction_thresholds_eV"]),
                                           "error": error_summary(part, config["error_unchanged_tolerance_eV"], bootstrap_args)}
        summaries[label]["all"] = {"prediction": prediction_summary(values.delta_prediction.to_numpy(), config["absolute_prediction_thresholds_eV"]),
                                    "error": error_summary(values, config["error_unchanged_tolerance_eV"], bootstrap_args)}

    seed_consistency = {}
    for model_name in config["models"]:
        pivot = result_frame[result_frame.model.eq(model_name)].pivot(index="molecule_id", columns="seed", values="delta_prediction")
        seed_consistency[model_name] = {}
        for left, right in combinations(config["seeds"], 2):
            seed_consistency[model_name][f"{left}_vs_{right}"] = {
                "spearman": float(spearmanr(pivot[left], pivot[right]).statistic),
                "sign_agreement": float(np.mean(np.sign(pivot[left]) == np.sign(pivot[right])))}
    ensemble = full[full.seed.eq("ensemble")].pivot(index="molecule_id", columns="model", values="delta_prediction")
    architecture_sensitivity = {"spearman": float(spearmanr(ensemble["m3_merged"], ensemble["m3_dau_shared"]).statistic),
                                "sign_agreement": float(np.mean(np.sign(ensemble["m3_merged"]) == np.sign(ensemble["m3_dau_shared"]))),
                                "mean_abs_delta_difference_merged_minus_dau_eV": float(np.mean(np.abs(ensemble["m3_merged"]))-np.mean(np.abs(ensemble["m3_dau_shared"]))) }
    stratification = {}
    ensemble_full = full[full.seed.eq("ensemble")]
    for model_name, values in ensemble_full.groupby("model"):
        stratification[model_name] = {
            "by_partition": values.groupby("partition").delta_prediction.agg(["count","mean","median"]).to_dict("index"),
            "by_structure_group_size": values.groupby("structure_group_size").delta_prediction.agg(["count","mean","median"]).to_dict("index"),
            "by_donor_component_identity": values.groupby("donor_structure_group_id_v1").delta_prediction.agg(["count","mean","median"]).to_dict("index")}
    output.mkdir(parents=True); full.to_csv(output / "paired_role_sensitivity_predictions.csv", index=False)
    result = {"status":"ROLE_SENSITIVITY_COMPLETE_NO_RETRAINING","candidate_records":198,
              "partition_counts":counts,"unresolved_excluded":189,"quarantine_inference_records":0,
              "all_candidates_formed_nonempty_donor":bool(frame.formed_nonempty_donor.all()),
              "changed_atom_count":{"min":int(frame.changed_atom_count.min()),"max":int(frame.changed_atom_count.max()),"mean":float(frame.changed_atom_count.mean())},
              "same_structure_candidate_inconsistent_groups":int(len(inconsistent_groups)),
              "original_prediction_max_abs_reconciliation_eV":reconciliation,
              "summaries":summaries,"seed_consistency":seed_consistency,
              "architecture_sensitivity":architecture_sensitivity,"stratification":stratification,
              "wall_seconds":time.perf_counter()-started,
              "prediction_sha256":sha256_file(output / "paired_role_sensitivity_predictions.csv"),
              "test_y_true_source":"frozen Gate1B3 test-once artifact","test_parquet_target_read":False,
              "training_performed":False,"parameters_updated":False,"final673_accessed":False}
    write_json(output / "role_sensitivity.json", result); return result
