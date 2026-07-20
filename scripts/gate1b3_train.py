#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import resource
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from excitationnexus_phase12.collate import collate_phase12
from excitationnexus_phase12.dft_graph_dataset import DFTGraphCache
from excitationnexus_phase12.edge_cache import ShardedEdgeCache, sha256_file
from excitationnexus_phase12.gate1b3_pipeline import Gate1B3GraphDataset, read_targets, target_blind_subset
from excitationnexus_phase12.losses import weighted_masked_multitask_loss
from excitationnexus_phase12.metrics import regression_metrics
from excitationnexus_phase12.models import M3DAUSharedModel, M3MergedModel
from excitationnexus_phase12.normalization import weighted_stats

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def seed_all(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def build_model(config: dict, model_name: str):
    common = config["model_common"]
    kwargs = dict(hidden_dim=common["hidden_dim"], num_rbf=common["num_rbf"],
                  layers=common["layers"], cutoff=common["cutoff_angstrom"],
                  head_width=config["models"][model_name]["head_width"])
    model = M3MergedModel(**kwargs) if model_name == "m3_merged" else M3DAUSharedModel(**kwargs)
    expected = int(config["models"][model_name]["parameters"])
    if sum(p.numel() for p in model.parameters()) != expected:
        raise RuntimeError("frozen architecture parameter mismatch")
    return model


def validate_inputs(config: dict) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    manifest_path = ROOT / config["manifest"]
    if sha256_file(manifest_path) != config["manifest_sha256"] or sha256_file(config["table"]) != config["table_sha256"]:
        raise RuntimeError("frozen table/manifest hash mismatch")
    manifest = pd.read_csv(manifest_path)
    counts = manifest.partition.value_counts().to_dict()
    if counts != {"train": 10387, "test": 2319, "val": 2309, "historical_quarantine": 1}:
        raise RuntimeError(f"frozen IID counts mismatch: {counts}")
    if manifest.groupby("structure_group_id_v1").partition.nunique().max() != 1:
        raise RuntimeError("structure group crosses partition")
    ids = manifest.loc[manifest.partition.isin(["train", "val"]), "molecule_id"].astype(str).tolist()
    targets = read_targets(config["table"], config["primary_target"], ids,
                           requested_partition="train", allow_test=False)
    if len(targets) != 12696 or targets.target.isna().any():
        raise RuntimeError("train/val-only target read failure")
    train = manifest[manifest.partition.eq("train")].merge(targets, on="molecule_id", validate="one_to_one")
    stats = weighted_stats(train.target, train.group_weight)
    return manifest, targets, stats


def datasets(config: dict, subset: dict | None = None):
    manifest, targets, stats = validate_inputs(config)
    graph = DFTGraphCache(ROOT / config["graph_cache"], ROOT / config["graph_registry"])
    edges = ShardedEdgeCache(ROOT / config["edge_cache_registry"], verify_hashes=True)
    train = Gate1B3GraphDataset(graph, edges, manifest, partition="train", targets=targets)
    val = Gate1B3GraphDataset(graph, edges, manifest, partition="val", targets=targets)
    if subset:
        train.frame = target_blind_subset(train.frame, subset["train_records"]).reset_index(drop=True)
        val.frame = target_blind_subset(val.frame, subset["val_records"]).reset_index(drop=True)
    return train, val, stats


def one_epoch(model, loader, optimizer, stats, device, clip: float) -> tuple[float, float]:
    model.train(); losses = []; loader_seconds = 0.0; previous = time.perf_counter()
    for batch in loader:
        now = time.perf_counter(); loader_seconds += now - previous
        batch = batch.to(device); normalized = (batch.target - stats["mean"]) / stats["std"]
        prediction = model(batch)
        loss, _ = weighted_masked_multitask_loss(
            {"primary": prediction}, normalized[:, None],
            torch.ones((len(prediction), 1), dtype=torch.bool, device=device),
            batch.group_weight, ["primary"], {"primary": 1.0}, base_loss="smooth_l1")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError("non-finite train loss/gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip); optimizer.step()
        losses.append(float(loss)); previous = time.perf_counter()
    return float(np.mean(losses)), loader_seconds


def evaluate_val(model, loader, stats, device) -> tuple[dict, pd.DataFrame]:
    model.eval(); molecule_ids = []; truth = []; prediction = []; groups = []
    with torch.no_grad():
        for batch in loader:
            ids = list(batch.molecule_id); group_ids = list(batch.structure_group_id_v1)
            batch = batch.to(device); pred = model(batch) * stats["std"] + stats["mean"]
            molecule_ids.extend(ids); groups.extend(group_ids)
            truth.extend(batch.target.detach().cpu().numpy().tolist()); prediction.extend(pred.cpu().numpy().tolist())
    frame = pd.DataFrame({"molecule_id": molecule_ids, "structure_group_id_v1": groups,
                          "truth": truth, "prediction": prediction})
    return regression_metrics(frame.truth, frame.prediction, frame.structure_group_id_v1), frame


def gpu_state(physical_gpu: int) -> dict:
    try:
        text = subprocess.check_output([
            "nvidia-smi", f"--id={physical_gpu}",
            "--query-gpu=temperature.gpu,memory.used,utilization.gpu",
            "--format=csv,noheader,nounits"], text=True).strip()
        temp, memory, utilization = [int(x.strip()) for x in text.split(",")]
        return {"temperature_c": temp, "memory_used_mib": memory, "utilization_percent": utilization}
    except Exception as exc:
        return {"query_error": type(exc).__name__}


def run(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text()); seed_all(args.seed)
    if args.output.exists():
        raise RuntimeError("run output already exists; refusing silent restart")
    calibration = args.mode == "calibration"
    subset = config["calibration"] if calibration else None
    train, val, stats = datasets(config, subset)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True, generator=generator,
                              num_workers=0, collate_fn=collate_phase12)
    val_loader = DataLoader(val, batch_size=config["batch_size"], shuffle=False, num_workers=0,
                            collate_fn=collate_phase12)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = build_model(config, args.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimizer"]["learning_rate"],
                                  weight_decay=config["optimizer"]["weight_decay"])
    if calibration:
        budget = config["calibration"]
    elif args.mode == "dry_run":
        budget = {"max_epochs": 1, "minimum_epochs": 1, "patience": 1, "min_delta_eV": 0.0}
        train.frame = train.frame.head(32); val.frame = val.frame.head(32)
    else:
        budget = config["formal_budget"]
    args.output.mkdir(parents=True); write_json(args.output / "normalization.json", {
        "scope": "full partition=train only", "weighting": "group_weight", "statistics": stats,
        "manifest_sha256": config["manifest_sha256"], "table_sha256": config["table_sha256"],
        "test_target_accessed": False,
    })
    log = (args.output / "epochs.jsonl").open("w")
    best = float("inf"); best_epoch = 0; stale = 0; epochs = []
    torch.cuda.reset_peak_memory_stats() if device.type == "cuda" else None
    started = time.perf_counter(); total_loader = 0.0
    for epoch in range(1, int(budget["max_epochs"]) + 1):
        epoch_started = time.perf_counter()
        train_loss, loader_seconds = one_epoch(model, train_loader, optimizer, stats, device, config["gradient_clip_norm"])
        val_metrics, val_predictions = evaluate_val(model, val_loader, stats, device)
        metric = float(val_metrics["group_macro_mae"]); improved = metric < best - float(budget["min_delta_eV"])
        if improved:
            best, best_epoch, stale = metric, epoch, 0
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "epoch": epoch,
                        "seed": args.seed, "config_sha256": sha256_file(args.config),
                        "normalization": stats, "validation": val_metrics}, args.output / "best_checkpoint.pt")
            val_predictions.to_csv(args.output / "best_validation_predictions.csv", index=False)
        else:
            stale += 1
        state = {"epoch": epoch, "train_loss": train_loss, "validation": val_metrics,
                 "improved": improved, "best_epoch": best_epoch, "best_group_macro_mae": best,
                 "epoch_wall_seconds": time.perf_counter() - epoch_started,
                 "loader_wait_seconds": loader_seconds, "gpu": gpu_state(args.physical_gpu)}
        epochs.append(state); log.write(json.dumps(state, sort_keys=True) + "\n"); log.flush()
        print(json.dumps(state, sort_keys=True), flush=True)
        if state["gpu"].get("temperature_c", 0) >= 90:
            raise RuntimeError("BLOCKED_HARDWARE_TEMPERATURE")
        total_loader += loader_seconds
        if epoch >= int(budget["minimum_epochs"]) and stale >= int(budget["patience"]):
            break
    log.close(); wall = time.perf_counter() - started
    best_path = args.output / "best_checkpoint.pt"
    result = {
        "status": "TRAIN_VAL_COMPLETE_TEST_LOCKED", "mode": args.mode, "model": args.model,
        "seed": args.seed, "physical_gpu": args.physical_gpu, "parameters": sum(p.numel() for p in model.parameters()),
        "train_records": len(train), "val_records": len(val), "epochs_completed": len(epochs),
        "best_epoch": best_epoch, "best_validation_group_macro_mae": best,
        "best_checkpoint_sha256": sha256_file(best_path), "wall_seconds": wall,
        "graphs_per_second_train_approx": len(train) * len(epochs) / wall,
        "cpu_loader_wait_seconds": total_loader,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "test_target_accessed": False, "final673_accessed": False,
    }
    write_json(args.output / "metrics.json", result); print(json.dumps(result, indent=2)); return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", choices=["m3_merged", "m3_dau_shared"], required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--physical-gpu", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["calibration", "dry_run", "formal"], required=True)
    args = parser.parse_args(); run(args)


if __name__ == "__main__":
    main()
