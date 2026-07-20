#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from excitationnexus_phase12.collate import collate_phase12
from excitationnexus_phase12.dft_graph_dataset import DFTGraphCache
from excitationnexus_phase12.edge_cache import ShardedEdgeCache
from excitationnexus_phase12.gate1b3_evaluation import build_model
from excitationnexus_phase12.gate1b3_pipeline import Gate1B3GraphDataset, read_targets
from excitationnexus_phase12.gate1c1 import group_macro_mae, perturb_positions, sha256_file

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def validate_preregistration(config_path: Path, lock_path: Path) -> dict:
    config, lock = json.loads(config_path.read_text()), json.loads(lock_path.read_text())
    if sha256_file(config_path) != lock["config_sha256"] or lock["status"] != "LOCKED_BEFORE_DIAGNOSIS":
        raise RuntimeError("Gate1C1 preregistration lock mismatch")
    if config["validation_counterfactuals"]["partition"] != "val only" or config["validation_counterfactuals"]["test_counterfactuals_forbidden"] is not True:
        raise PermissionError("counterfactual partition firewall failure")
    for item in config["frozen_inputs"].values():
        if sha256_file(ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen input changed: {item['path']}")
    return config


def infer(model, loader, condition: str, normalization: dict, device: torch.device, seed: int) -> tuple[list[str], np.ndarray, int]:
    ids, predictions, skipped = [], [], 0
    model.eval()
    with torch.inference_mode():
        for batch in loader:
            ids.extend(list(batch.molecule_id)); batch = batch.to(device)
            batch.pos, count = perturb_positions(batch, condition, seed); skipped += count
            output = model(batch) * normalization["std"] + normalization["mean"]
            predictions.extend(output.cpu().numpy().tolist())
    return ids, np.asarray(predictions, dtype=np.float64), skipped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise RuntimeError("counterfactual output exists; refusing rerun")
    config = validate_preregistration(args.config, args.lock)
    formal = json.loads((ROOT / config["frozen_inputs"]["formal_config"]["path"]).read_text())
    registry = json.loads((ROOT / "data_registry/gate1b3_model_registry.json").read_text())
    manifest = pd.read_csv(ROOT / formal["manifest"])
    if set(manifest.loc[manifest.partition.eq("val"), "partition"]) != {"val"}:
        raise RuntimeError("validation-only manifest failure")
    val_ids = manifest.loc[manifest.partition.eq("val"), "molecule_id"].astype(str).tolist()
    targets = read_targets(formal["table"], formal["primary_target"], val_ids,
                           requested_partition="val", allow_test=False)
    graph = DFTGraphCache(ROOT / formal["graph_cache"], ROOT / formal["graph_registry"])
    edges = ShardedEdgeCache(ROOT / formal["edge_cache_registry"], verify_hashes=True)
    dataset = Gate1B3GraphDataset(graph, edges, manifest, partition="val", targets=targets)
    loader = DataLoader(dataset, batch_size=formal["batch_size"], shuffle=False, num_workers=0,
                        collate_fn=collate_phase12)
    base = dataset.frame[["molecule_id", "structure_group_id_v1", "target"]].rename(columns={"target": "truth"})
    expected_ids = base.molecule_id.astype(str).tolist()
    conditions = config["validation_counterfactuals"]["conditions"]
    device = torch.device("cuda:0")
    rows, reconciliations, started = [], {}, time.perf_counter()
    for model_name in config["validation_counterfactuals"]["models"]:
        wave = "wave1" if model_name == "m3_merged" else "wave2"
        for seed in config["validation_counterfactuals"]["seeds"]:
            run = ROOT / f"runs/gate1b3_{model_name}_seed{seed}"
            frozen = registry[wave]["runs"][str(seed)]
            if sha256_file(run / "best_checkpoint.pt") != frozen["checkpoint_sha256"]:
                raise RuntimeError("frozen checkpoint changed")
            checkpoint = torch.load(run / "best_checkpoint.pt", map_location="cpu", weights_only=False)
            normalization = json.loads((run / "normalization.json").read_text())["statistics"]
            model = build_model(formal, model_name); model.load_state_dict(checkpoint["model"]); model.to(device)
            by_condition = {}
            for condition in conditions:
                ids, prediction, skipped = infer(model, loader, condition, normalization, device,
                                                  config["validation_counterfactuals"]["random_seed"])
                if ids != expected_ids:
                    raise RuntimeError("counterfactual ID/order changed")
                by_condition[condition] = (prediction, skipped)
            frozen_val = pd.read_csv(run / "best_validation_predictions.csv").set_index("molecule_id")
            original = by_condition["original"][0]
            reference = frozen_val.loc[expected_ids, "prediction"].to_numpy(np.float64)
            maximum = float(np.max(np.abs(original - reference)))
            if maximum > 2e-6:
                raise RuntimeError(f"original validation prediction mismatch: {maximum}")
            label = f"{model_name}_seed{seed}"; reconciliations[label] = maximum
            for condition, (prediction, skipped) in by_condition.items():
                for index, molecule_id in enumerate(expected_ids):
                    rows.append({"molecule_id": molecule_id,
                                 "structure_group_id_v1": base.iloc[index].structure_group_id_v1,
                                 "truth": float(base.iloc[index].truth), "model": model_name,
                                 "seed": seed, "label": label, "condition": condition,
                                 "prediction": float(prediction[index]),
                                 "delta_prediction": float(prediction[index] - original[index]),
                                 "skipped_graphs_in_condition": skipped})
    frame = pd.DataFrame(rows)
    summaries = {}
    for (label, condition), part in frame.groupby(["label", "condition"], sort=True):
        summaries.setdefault(label, {})[condition] = {
            "records": len(part),
            "group_macro_mae_eV": group_macro_mae(part.rename(columns={"truth": "primary_true"}), "prediction"),
            "mean_abs_delta_prediction_eV": float(part.delta_prediction.abs().mean()),
            "median_abs_delta_prediction_eV": float(part.delta_prediction.abs().median()),
            "p95_abs_delta_prediction_eV": float(part.delta_prediction.abs().quantile(0.95)),
            "max_abs_delta_prediction_eV": float(part.delta_prediction.abs().max()),
            "skipped_graphs": int(part.skipped_graphs_in_condition.max()),
        }
    ensemble_rows = []
    for (model, condition, molecule_id), part in frame.groupby(["model", "condition", "molecule_id"], sort=False):
        base_row = part.iloc[0]
        ensemble_rows.append({"molecule_id": molecule_id, "structure_group_id_v1": base_row.structure_group_id_v1,
                              "truth": base_row.truth, "model": model, "condition": condition,
                              "prediction": float(part.prediction.mean()),
                              "delta_prediction": float(part.delta_prediction.mean())})
    ensembles = pd.DataFrame(ensemble_rows)
    ensemble_summaries = {}
    for (model, condition), part in ensembles.groupby(["model", "condition"], sort=True):
        ensemble_summaries.setdefault(model, {})[condition] = {
            "records": len(part),
            "group_macro_mae_eV": group_macro_mae(part.rename(columns={"truth": "primary_true"}), "prediction"),
            "mean_abs_delta_prediction_eV": float(part.delta_prediction.abs().mean()),
            "median_abs_delta_prediction_eV": float(part.delta_prediction.abs().median()),
            "p95_abs_delta_prediction_eV": float(part.delta_prediction.abs().quantile(0.95)),
            "max_abs_delta_prediction_eV": float(part.delta_prediction.abs().max()),
        }
    args.output.mkdir(parents=True)
    frame.to_parquet(args.output / "counterfactual_predictions.parquet", index=False)
    result = {"status": "VALIDATION_COUNTERFACTUALS_COMPLETE_NO_TRAINING",
              "partition": "val", "records": len(dataset), "models": 6, "conditions": conditions,
              "original_reconciliation_max_eV": reconciliations, "per_checkpoint": summaries,
              "ensembles": ensemble_summaries, "prediction_artifact_sha256": sha256_file(args.output / "counterfactual_predictions.parquet"),
              "wall_seconds": time.perf_counter() - started, "physical_gpu": args.physical_gpu,
              "training_performed": False, "parameters_updated": False, "test_inference_performed": False,
              "test_target_accessed": False, "final673_accessed": False}
    write_json(args.output / "counterfactuals.json", result)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
