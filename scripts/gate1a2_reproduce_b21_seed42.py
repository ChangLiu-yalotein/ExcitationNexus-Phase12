#!/usr/bin/env python3
"""Checkpoint inference and one fixed seed42 B2-1 reproduction adapter."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.db import connect
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader

HISTORICAL_ROOT = Path("/home/changliu/ExcitationNexus/equiformer_v3_model")
FAIRCHEM_SRC = HISTORICAL_ROOT / "equiformer_v3/src"
sys.path.insert(0, str(FAIRCHEM_SRC))
sys.path.insert(0, str(HISTORICAL_ROOT))

from equiformer_v3.datasets.dual_tower import DualTowerPyGDataset, dual_tower_collate_fn  # noqa: E402
from equiformer_v3.models.naive_dual_tower import NaiveDualTowerEquiformerV3Batched  # noqa: E402,F401
from fairchem.core.trainers.b2_1_trainer import B2_1Trainer, move_batch_to_device  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def metric_dict(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
        "n_records": int(len(truth)),
    }


def load_config(path: Path) -> tuple[dict, object]:
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(path)
    return OmegaConf.to_container(cfg, resolve=True), cfg


def make_trainer(config: dict, checkpoint: Path, run_dir: Path, cpu: bool) -> B2_1Trainer:
    run_dir.mkdir(parents=True, exist_ok=True)
    trainer = B2_1Trainer(
        task=config.get("task", {}), model=config["model"], outputs=config.get("outputs", {}),
        dataset=config["dataset"], optimizer=config["optim"], loss_functions=config["loss_functions"],
        evaluation_metrics=config.get("evaluation_metrics", {}), identifier="gate1a2_inference",
        local_rank=0, run_dir=str(run_dir), is_debug=True, print_every=1000, seed=42,
        logger="tensorboard", amp=False, cpu=cpu,
    )
    payload = torch.load(checkpoint, map_location=trainer.device, weights_only=False)
    trainer.model.load_state_dict(payload["state_dict"])
    if cpu:
        # The frozen historical forward unconditionally calls tower.cuda().
        # Mask only that device-forcing method for the required CPU smoke.
        trainer.model.tower.cuda = lambda device=None: trainer.model.tower
    return trainer


def database_sids(path: Path) -> list[str]:
    return [str(getattr(row, "name", getattr(row, "sid", row.id))) for row in connect(path).select()]


def infer(
    config_path: Path,
    checkpoint: Path,
    test_db: Path,
    historical_predictions: Path,
    output_dir: Path,
    cpu: bool,
) -> dict:
    started = time.perf_counter()
    config, cfg = load_config(config_path)
    trainer = make_trainer(config, checkpoint, output_dir / "trainer_runtime", cpu)
    dataset = DualTowerPyGDataset(
        db_path=str(test_db), cutoff=cfg.model.max_radius, max_neighbors=cfg.model.max_neighbors,
    )
    sids = database_sids(test_db)
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0, collate_fn=dual_tower_collate_fn)
    trainer.model.eval()
    predictions: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            batch = move_batch_to_device(batch, trainer.device)
            predictions.append(trainer.model(batch)["energy"].detach().cpu().numpy())
            targets.append(batch["energy"].detach().cpu().numpy())
    pred = np.concatenate(predictions)
    truth = np.concatenate(targets)
    if len(pred) != len(sids) or not np.isfinite(pred).all():
        raise RuntimeError("inference length/finite check failed")
    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame({
        "sid": sids, "target_energy": truth, "pred_energy": pred,
        "absolute_error": np.abs(pred - truth), "squared_error": (pred - truth) ** 2,
    })
    prediction_path = output_dir / "gate1a2_checkpoint_test_predictions.csv"
    frame.to_csv(prediction_path, index=False)
    old_path = historical_predictions
    old = pd.read_csv(old_path)
    if old["sid"].astype(str).tolist() != sids:
        raise RuntimeError("old prediction SID order mismatch")
    result = {
        "metrics": metric_dict(truth, pred),
        "historical_metrics": metric_dict(old["target_energy"].to_numpy(), old["pred_energy"].to_numpy()),
        "max_abs_truth_delta": float(np.max(np.abs(truth - old["target_energy"].to_numpy()))),
        "max_abs_prediction_delta": float(np.max(np.abs(pred - old["pred_energy"].to_numpy()))),
        "prediction_sha256": sha256(prediction_path),
        "historical_prediction_sha256": sha256(old_path),
        "checkpoint_sha256": sha256(checkpoint),
        "parameter_count": sum(parameter.numel() for parameter in trainer.model.parameters()),
        "device": str(trainer.device),
        "inference_wall_seconds": time.perf_counter() - started,
        "throughput_records_per_second": len(pred) / (time.perf_counter() - started),
        "final673_accessed": False,
        "new15016_accessed": False,
    }
    result["status"] = "REPRODUCED_STRICT" if (
        result["max_abs_prediction_delta"] <= 1e-7
        and abs(result["metrics"]["mae"] - result["historical_metrics"]["mae"]) <= 1e-12
    ) else ("REPRODUCED_NUMERIC" if abs(result["metrics"]["mae"] - result["historical_metrics"]["mae"]) <= 0.001 else "FAILED_REPRODUCTION")
    (output_dir / "gate1a2_checkpoint_inference.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def cpu_smoke(config_path: Path, test_db: Path, checkpoint: Path, output_dir: Path) -> dict:
    config, cfg = load_config(config_path)
    trainer = make_trainer(config, checkpoint, output_dir / "cpu_trainer_runtime", True)
    dataset = DualTowerPyGDataset(str(test_db), cutoff=cfg.model.max_radius, max_neighbors=cfg.model.max_neighbors)
    loader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0, collate_fn=dual_tower_collate_fn)
    batch = next(iter(loader))
    with torch.no_grad():
        output = trainer.model(move_batch_to_device(batch, trainer.device))["energy"]
    result = {
        "status": "CPU_SMOKE_PASS", "shape": list(output.shape),
        "finite": bool(torch.isfinite(output).all()), "parameter_count": sum(p.numel() for p in trainer.model.parameters()),
        "test_metric_computed": False, "final673_accessed": False, "new15016_accessed": False,
    }
    if result["shape"] != [2] or not result["finite"]:
        raise RuntimeError(f"CPU smoke failed: {result}")
    return result


def run_training(config_path: Path, output_dir: Path) -> dict:
    if output_dir.exists():
        raise RuntimeError(f"formal training output already exists: {output_dir}")
    output_dir.mkdir(parents=True)
    command = [
        sys.executable, str(HISTORICAL_ROOT / "scripts/run_b2_1_training_wrapper.py"),
        "--config-yml", str(config_path), "--mode", "train", "--num-gpus", "1",
        "--seed", "42", "--identifier", "gate1a2_b21_seed42",
        "--timestamp-id", "gate1a2-b21-seed42-20260719", "--run-dir", str(output_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{FAIRCHEM_SRC}:{HISTORICAL_ROOT}:{env.get('PYTHONPATH', '')}"
    env["WANDB_MODE"] = "disabled"
    started = time.perf_counter()
    log_path = output_dir / "training.log"
    with log_path.open("w") as log:
        completed = subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    result = {
        "returncode": completed.returncode, "wall_seconds": time.perf_counter() - started,
        "command": command, "cuda_visible_devices": env.get("CUDA_VISIBLE_DEVICES", ""),
        "python": sys.version.split()[0], "platform": platform.platform(),
    }
    if completed.returncode != 0:
        result["status"] = "FAILED_TRAINING_PROCESS"
        (output_dir / "training_process.json").write_text(json.dumps(result, indent=2) + "\n")
        raise RuntimeError(f"training failed; see {log_path}")
    checkpoints = sorted(output_dir.glob("checkpoints/**/best_checkpoint.pt"))
    if len(checkpoints) != 1:
        raise RuntimeError(f"expected one best checkpoint, found {checkpoints}")
    result.update({"status": "TRAINING_COMPLETE", "best_checkpoint": str(checkpoints[0]), "best_checkpoint_sha256": sha256(checkpoints[0])})
    (output_dir / "training_process.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path, help="Explicit checkpoint override for post-training inference")
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--inference", action="store_true")
    parser.add_argument("--train", action="store_true")
    args = parser.parse_args()
    gate = json.loads(args.config.read_text())
    config_path = Path(gate["historical_assets"]["config"])
    checkpoint = args.checkpoint or Path(gate["historical_assets"]["best_checkpoint"])
    test_db = Path(gate["historical_assets"]["test_db"])
    historical_predictions = Path(gate["historical_assets"]["historical_predictions"])
    if sum((args.cpu_smoke, args.inference, args.train)) != 1:
        parser.error("choose exactly one of --cpu-smoke, --inference, --train")
    if args.cpu_smoke:
        result = cpu_smoke(config_path, test_db, checkpoint, args.output_dir)
    elif args.inference:
        result = infer(config_path, checkpoint, test_db, historical_predictions, args.output_dir, cpu=False)
    else:
        result = run_training(config_path, args.output_dir)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
