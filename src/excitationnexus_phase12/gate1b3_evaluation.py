from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from torch_geometric.data import Data

from .collate import collate_phase12
from .dft_graph_dataset import DFTGraphCache
from .edge_cache import ShardedEdgeCache, sha256_file
from .gate1b3_pipeline import read_targets
from .metrics import regression_metrics
from .models import M3DAUSharedModel, M3MergedModel

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def build_model(config: dict, model_name: str):
    common = config["model_common"]
    kwargs = dict(hidden_dim=common["hidden_dim"], num_rbf=common["num_rbf"],
                  layers=common["layers"], cutoff=common["cutoff_angstrom"],
                  head_width=config["models"][model_name]["head_width"])
    model = M3MergedModel(**kwargs) if model_name == "m3_merged" else M3DAUSharedModel(**kwargs)
    if sum(p.numel() for p in model.parameters()) != int(config["models"][model_name]["parameters"]):
        raise RuntimeError("frozen parameter count mismatch")
    return model


class Gate1B3TestDataset(Dataset):
    def __init__(self, graph: DFTGraphCache, edges: ShardedEdgeCache,
                 manifest: pd.DataFrame, targets: pd.DataFrame, unlock: dict):
        if unlock.get("status") != "TEST_UNLOCKED_ONCE_AFTER_SIX_MODELS_FROZEN":
            raise PermissionError("TEST_TARGET_FIREWALL_LOCKED")
        subset = manifest.loc[manifest.partition.eq("test")].copy()
        if len(subset) != 2319 or manifest.partition.eq("historical_quarantine").sum() != 1:
            raise RuntimeError("test/quarantine boundary mismatch")
        self.frame = subset.merge(graph.registry[["molecule_id", "cache_index"]], on="molecule_id",
                                  validate="one_to_one").merge(targets, on="molecule_id", validate="one_to_one")
        self.frame = self.frame.sort_values("molecule_id", kind="mergesort").reset_index(drop=True)
        if len(self.frame) != 2319 or self.frame.target.isna().any():
            raise RuntimeError("test graph/target join failure")
        self.graph, self.edges = graph, edges

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> Data:
        row = self.frame.iloc[int(index)]; source = self.graph.registry.iloc[int(row.cache_index)]
        a, b = int(source.atom_offset_start), int(source.atom_offset_end)
        role = torch.from_numpy(self.graph.role[a:b].copy()).long()
        return Data(z=torch.from_numpy(self.graph.z[a:b].copy()).long(),
                    pos=torch.from_numpy(self.graph.pos[a:b].copy()).float(), role=role,
                    edge_index=self.edges.edge_index(str(row.molecule_id)), num_nodes=b-a,
                    molecule_id=str(row.molecule_id), structure_group_id_v1=str(row.structure_group_id_v1),
                    target=torch.tensor(float(row.target), dtype=torch.float32),
                    group_weight=torch.tensor(float(row.group_weight), dtype=torch.float32))


def group_bootstrap_difference(frame: pd.DataFrame, first: str, second: str, *,
                               seed: int = 20260720, iterations: int = 10000) -> dict:
    values = frame.assign(first_error=(frame[first] - frame.primary_true).abs(),
                          second_error=(frame[second] - frame.primary_true).abs()).groupby(
                              "structure_group_id_v1", sort=True)[["first_error", "second_error"]].mean()
    difference = (values.first_error - values.second_error).to_numpy(np.float64)
    rng = np.random.default_rng(seed); bootstrap = np.empty(iterations)
    for index in range(iterations):
        bootstrap[index] = difference[rng.integers(0, len(difference), size=len(difference))].mean()
    low, high = np.quantile(bootstrap, [.025, .975])
    return {"estimand": f"group-macro MAE({first}) - group-macro MAE({second})",
            "groups": len(difference), "iterations": iterations, "seed": seed,
            "point_difference_eV": float(difference.mean()),
            "ci95_percentile_eV": [float(low), float(high)],
            "ci_excludes_zero": bool(low > 0 or high < 0), "negative_favors_first": True}


def metric(frame: pd.DataFrame, prediction: str) -> dict:
    return regression_metrics(frame.primary_true, frame[prediction], frame.structure_group_id_v1)


def validate_unlock(unlock: dict, config: dict, config_path: Path) -> None:
    if unlock.get("status") != "TEST_UNLOCKED_ONCE_AFTER_SIX_MODELS_FROZEN":
        raise PermissionError("TEST_TARGET_FIREWALL_LOCKED")
    if sha256_file(config_path) != unlock["formal_config_sha256"]:
        raise RuntimeError("formal config changed after unlock")
    if sha256_file(ROOT / "data_registry/gate1b3_model_registry.json") != unlock["model_registry_sha256"]:
        raise RuntimeError("model registry changed after unlock")
    for model in ("m3_merged", "m3_dau_shared"):
        for seed in config["seeds"]:
            key = f"{model}_seed{seed}"; item = unlock["models"][key]
            root = ROOT / f"runs/gate1b3_{model}_seed{seed}"
            if sha256_file(root / "best_checkpoint.pt") != item["checkpoint_sha256"]:
                raise RuntimeError(f"checkpoint hash mismatch: {key}")
            if sha256_file(root / "best_validation_predictions.csv") != item["validation_predictions_sha256"]:
                raise RuntimeError(f"validation prediction hash mismatch: {key}")


def evaluate(config_path: Path, unlock_path: Path, output: Path, physical_gpu: int) -> dict:
    if output.exists():
        raise RuntimeError("one-time test output already exists; refusing second evaluation")
    config = json.loads(config_path.read_text()); unlock = json.loads(unlock_path.read_text())
    validate_unlock(unlock, config, config_path); output.mkdir(parents=True)
    write_json(output / "TEST_EVALUATION_STARTED.json", {
        "status": "STARTED_AFTER_EXPLICIT_UNLOCK", "unlock_sha256": sha256_file(unlock_path),
        "physical_gpu": physical_gpu, "test_guided_retraining": False})

    manifest = pd.read_csv(ROOT / config["manifest"])
    test_ids = manifest.loc[manifest.partition.eq("test"), "molecule_id"].astype(str).tolist()
    target = read_targets(config["table"], config["primary_target"], test_ids,
                          requested_partition="test", allow_test=True)
    graph = DFTGraphCache(ROOT / config["graph_cache"], ROOT / config["graph_registry"])
    edges = ShardedEdgeCache(ROOT / config["edge_cache_registry"], verify_hashes=True)
    dataset = Gate1B3TestDataset(graph, edges, manifest, target, unlock)
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=False,
                        num_workers=0, collate_fn=collate_phase12)
    device = torch.device("cuda:0")
    combined = dataset.frame[["molecule_id", "structure_group_id_v1", "structure_group_size",
                              "group_weight"]].copy()
    combined["primary_true"] = dataset.frame.target.to_numpy(np.float64)
    all_metrics, inference_seconds, prediction_hashes = {}, {}, {}
    for model_name in ("m3_merged", "m3_dau_shared"):
        for seed in config["seeds"]:
            label = f"{model_name}_seed{seed}"; run = ROOT / f"runs/gate1b3_{model_name}_seed{seed}"
            checkpoint = torch.load(run / "best_checkpoint.pt", map_location="cpu", weights_only=False)
            normalization = json.loads((run / "normalization.json").read_text())["statistics"]
            model = build_model(config, model_name); model.load_state_dict(checkpoint["model"])
            model.to(device).eval(); ids, predictions = [], []; started = time.perf_counter()
            with torch.inference_mode():
                for batch in loader:
                    ids.extend(list(batch.molecule_id)); batch = batch.to(device)
                    predictions.extend((model(batch) * normalization["std"] + normalization["mean"]).cpu().tolist())
            inference_seconds[label] = time.perf_counter() - started
            if ids != combined.molecule_id.astype(str).tolist():
                raise RuntimeError(f"test prediction ID/order mismatch: {label}")
            combined[label] = np.asarray(predictions, dtype=np.float64)
            path = output / f"{label}_test_predictions.csv"
            combined[["molecule_id", "structure_group_id_v1", "primary_true", label]].to_csv(path, index=False)
            prediction_hashes[label] = sha256_file(path); all_metrics[label] = metric(combined, label)

    xgb_path = ROOT / "runs/gate1b1_new_iid_cheap_baselines/published/gate1b1_test_predictions_once.csv"
    if sha256_file(xgb_path) != unlock["xgboost_c0"]["prediction_file_sha256"]:
        raise RuntimeError("XGBoost-C0 frozen prediction hash mismatch")
    xgb = pd.read_csv(xgb_path)[["molecule_id", "primary_true", "xgb_c0_seed42"]]
    combined = combined.merge(xgb, on="molecule_id", suffixes=("", "_xgb"), validate="one_to_one")
    if not np.allclose(combined.primary_true, combined.primary_true_xgb, rtol=0, atol=1e-7):
        raise RuntimeError("Gate1B1/Gate1B3 test truth mismatch")
    combined = combined.drop(columns="primary_true_xgb").rename(columns={"xgb_c0_seed42": "xgboost_c0"})
    all_metrics["xgboost_c0"] = metric(combined, "xgboost_c0")

    summaries, ensemble_labels = {}, []
    for model_name in ("m3_merged", "m3_dau_shared"):
        labels = [f"{model_name}_seed{seed}" for seed in config["seeds"]]
        ensemble = f"{model_name}_ensemble"; combined[ensemble] = combined[labels].mean(axis=1)
        ensemble_labels.append(ensemble); all_metrics[ensemble] = metric(combined, ensemble)
        values = np.asarray([all_metrics[label]["group_macro_mae"] for label in labels])
        summaries[model_name] = {"seed_group_macro_mae_eV": values.tolist(),
                                 "mean_eV": float(values.mean()), "sample_std_ddof1_eV": float(values.std(ddof=1)),
                                 "ensemble": all_metrics[ensemble]}

    comparisons = {f"{label}_vs_xgboost_c0": group_bootstrap_difference(combined, label, "xgboost_c0")
                   for label in all_metrics if label.startswith("m3_")}
    comparisons["merged_ensemble_vs_dau_ensemble"] = group_bootstrap_difference(
        combined, "m3_merged_ensemble", "m3_dau_shared_ensemble")

    role = graph.registry[["molecule_id", "donor_atoms", "acceptor_atoms", "unknown_atoms"]]
    combined = combined.merge(role, on="molecule_id", validate="one_to_one")
    masks = {"pure_DA": combined.donor_atoms.gt(0) & combined.acceptor_atoms.gt(0) & combined.unknown_atoms.eq(0),
             "DA_unknown": combined.donor_atoms.gt(0) & combined.acceptor_atoms.gt(0) & combined.unknown_atoms.gt(0),
             "empty_donor_unknown": combined.donor_atoms.eq(0) & combined.unknown_atoms.gt(0),
             "singleton_structure": combined.structure_group_size.eq(1),
             "replicated_structure": combined.structure_group_size.gt(1)}
    strata = {label: {name: ({"records": len(combined.loc[mask]), **metric(combined.loc[mask], label)}
                             if mask.any() else {"records": 0}) for name, mask in masks.items()}
              for label in [*ensemble_labels, "xgboost_c0"]}

    prediction_path = output / "gate1b3_test_predictions_once.csv"; combined.to_csv(prediction_path, index=False)
    result = {"status": "TEST_EVALUATED_EXACTLY_ONCE", "test_records": len(combined),
              "test_structure_groups": int(combined.structure_group_id_v1.nunique()),
              "individual_metrics": all_metrics, "architecture_summary": summaries,
              "paired_group_bootstrap": comparisons, "strata": strata,
              "inference_wall_seconds": inference_seconds, "individual_prediction_sha256": prediction_hashes,
              "combined_prediction_sha256": sha256_file(prediction_path),
              "unlock_sha256": sha256_file(unlock_path), "test_guided_retraining": False,
              "final673_accessed": False}
    write_json(output / "metrics.json", result)
    write_json(output / "TEST_EVALUATION_COMPLETE.json", {
        "status": "COMPLETE_NO_RERUN", "metrics_sha256": sha256_file(output / "metrics.json"),
        "predictions_sha256": sha256_file(prediction_path), "test_guided_retraining": False})
    return result
