#!/usr/bin/env python3
"""Gate 1-A3 CPU/inference gates and isolated dual-GPU launcher."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")
HIST = Path("/home/changliu/ExcitationNexus/equiformer_v3_model")
PYTHON = Path("/home/changliu/miniconda3/envs/ML/bin/python")


def load_gate1a2_adapter():
    path = ROOT / "scripts/gate1a2_reproduce_b21_seed42.py"
    spec = importlib.util.spec_from_file_location("gate1a2_adapter", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    if config["gate"] != "1-A3" or config["version"] != "v1":
        raise RuntimeError("unexpected Gate 1-A3 config")
    return config


def seed_assets(config: dict, seed: int) -> dict[str, Path]:
    return {name: Path(value) for name, value in config["seed_assets"][str(seed)].items()}


def cpu_smoke(config_path: Path, output_root: Path) -> dict:
    config = load_config(config_path)
    adapter = load_gate1a2_adapter()
    results = {}
    for seed in config["seeds"]:
        assets = seed_assets(config, seed)
        result = adapter.cpu_smoke(
            Path(config["common_assets"]["config"]), Path(config["common_assets"]["test_db"]),
            assets["checkpoint"], output_root / f"seed{seed}",
        )
        result["seed"] = seed
        results[str(seed)] = result
    payload = {"status": "CPU_SMOKE_PASS_ALL", "seeds": results, "final673_accessed": False, "new15016_accessed": False}
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "gate1a3_cpu_smoke.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def checkpoint_inference(config_path: Path, seed: int, output_dir: Path, checkpoint_override: Path | None = None) -> dict:
    config = load_config(config_path)
    adapter = load_gate1a2_adapter()
    assets = seed_assets(config, seed)
    checkpoint = checkpoint_override or assets["checkpoint"]
    result = adapter.infer(
        Path(config["common_assets"]["config"]), checkpoint,
        Path(config["common_assets"]["test_db"]), assets["test_predictions"], output_dir, cpu=False,
    )
    old_csv = output_dir / "gate1a2_checkpoint_test_predictions.csv"
    new_csv = output_dir / f"gate1a3_seed{seed}_historical_checkpoint_test_predictions.csv"
    old_csv.rename(new_csv)
    old_json = output_dir / "gate1a2_checkpoint_inference.json"
    if old_json.exists():
        old_json.unlink()
    result.update({"seed": seed, "prediction_path": str(new_csv)})
    (output_dir / f"gate1a3_seed{seed}_checkpoint_inference.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    return result


def gpu_inventory() -> list[dict]:
    query = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,memory.used,utilization.gpu,temperature.gpu",
        "--format=csv,noheader,nounits",
    ], text=True)
    compute = subprocess.check_output([
        "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"
    ], text=True).strip()
    busy_uuids = {line.split(",")[0].strip() for line in compute.splitlines() if line.strip()}
    inventory = []
    for line in query.splitlines():
        idx, name, uuid, memory, util, temperature = [part.strip() for part in line.split(",")]
        inventory.append({
            "index": int(idx), "name": name, "uuid": uuid, "memory_used_mib": int(memory),
            "utilization_percent": int(util), "temperature_c": int(temperature),
            "compute_process": uuid in busy_uuids,
            "free": uuid not in busy_uuids and int(memory) < 1024 and int(util) <= 5,
        })
    return inventory


def worker(config_path: Path, seed: int, physical_gpu: int, output_dir: Path) -> int:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=False)
    log_path = output_dir / "training.log"
    command = [
        str(PYTHON), str(HIST / "scripts/run_b2_1_training_wrapper.py"),
        "--config-yml", config["common_assets"]["config"], "--mode", "train", "--num-gpus", "1",
        "--seed", str(seed), "--identifier", f"gate1a3_b21_seed{seed}",
        "--timestamp-id", f"gate1a3-b21-seed{seed}-20260719", "--run-dir", str(output_dir),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(physical_gpu)
    env["PYTHONPATH"] = f"{HIST / 'equiformer_v3/src'}:{HIST}:{env.get('PYTHONPATH', '')}"
    env["WANDB_MODE"] = "disabled"
    started = time.monotonic()
    high_temp_count = 0
    peak_memory = 0
    max_temperature = 0
    with log_path.open("w") as log:
        process = subprocess.Popen(command, env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
        (output_dir / "trainer.pid").write_text(f"{process.pid}\n")
        while process.poll() is None:
            time.sleep(30)
            try:
                row = subprocess.check_output([
                    "nvidia-smi", f"--id={physical_gpu}",
                    "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits",
                ], text=True).strip().split(",")
                memory, temperature = int(row[0].strip()), int(row[1].strip())
                peak_memory = max(peak_memory, memory)
                max_temperature = max(max_temperature, temperature)
                high_temp_count = high_temp_count + 1 if temperature >= 90 else 0
                if high_temp_count >= 3:
                    process.send_signal(signal.SIGTERM)
                    process.wait(timeout=60)
                    status = "BLOCKED_HARDWARE_TEMPERATURE"
                    break
            except Exception:
                pass
        else:
            status = "TRAINING_COMPLETE" if process.returncode == 0 else "FAILED_TRAINING_PROCESS"
    result = {
        "seed": seed, "physical_gpu": physical_gpu, "pid": process.pid, "returncode": process.returncode,
        "status": status, "wall_seconds": time.monotonic() - started, "peak_memory_mib_observed": peak_memory,
        "max_temperature_c_observed": max_temperature, "command": command,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    checkpoints = sorted(output_dir.glob("checkpoints/**/best_checkpoint.pt"))
    if status == "TRAINING_COMPLETE" and len(checkpoints) != 1:
        result["status"] = "FAILED_CHECKPOINT_CARDINALITY"
    result["best_checkpoints"] = [str(path) for path in checkpoints]
    (output_dir / "training_process.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0 if result["status"] == "TRAINING_COMPLETE" else 1


def launch(config_path: Path, launch_log: Path) -> dict:
    config = load_config(config_path)
    inventory = gpu_inventory()
    by_id = {gpu["index"]: gpu for gpu in inventory}
    mapping = {int(seed): int(gpu) for seed, gpu in config["seed_to_physical_gpu"].items()}
    if any(not by_id[gpu]["free"] for gpu in mapping.values()):
        raise RuntimeError(f"preregistered GPU is not free: {inventory}")
    processes = {}
    for seed, gpu in mapping.items():
        output_dir = Path(config["output_roots"][str(seed)]) / "formal_training"
        if output_dir.exists():
            raise RuntimeError(f"formal output already exists: {output_dir}")
        top_log = ROOT / f"logs/gate1a3_seed{seed}_worker.log"
        handle = top_log.open("w")
        command = [
            str(PYTHON), str(Path(__file__).resolve()), "--config", str(config_path), "--worker",
            "--seed", str(seed), "--physical-gpu", str(gpu), "--output-dir", str(output_dir),
        ]
        proc = subprocess.Popen(command, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True, close_fds=True)
        handle.close()
        processes[str(seed)] = {
            "launcher_pid": proc.pid, "physical_gpu": gpu, "output_dir": str(output_dir),
            "training_log": str(output_dir / "training.log"), "worker_log": str(top_log), "command": command,
        }
    payload = {
        "status": "LAUNCHED", "launched_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_inventory_before_launch": inventory, "seeds": processes,
        "final673_accessed": False, "new15016_accessed": False,
    }
    launch_log.parent.mkdir(parents=True, exist_ok=True)
    launch_log.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, choices=(123, 456))
    parser.add_argument("--physical-gpu", type=int)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--checkpoint-inference", action="store_true")
    parser.add_argument("--launch", action="store_true")
    parser.add_argument("--worker", action="store_true")
    args = parser.parse_args()
    if sum((args.cpu_smoke, args.checkpoint_inference, args.launch, args.worker)) != 1:
        parser.error("choose exactly one operation")
    if args.cpu_smoke:
        result = cpu_smoke(args.config, args.output_dir or ROOT / "runs/gate1a3_cpu_smoke")
    elif args.checkpoint_inference:
        if args.seed is None or args.output_dir is None:
            parser.error("checkpoint inference requires --seed and --output-dir")
        result = checkpoint_inference(args.config, args.seed, args.output_dir, args.checkpoint)
    elif args.launch:
        result = launch(args.config, ROOT / "logs/gate1a3_launch.json")
    else:
        if args.seed is None or args.physical_gpu is None or args.output_dir is None:
            parser.error("worker requires --seed, --physical-gpu and --output-dir")
        raise SystemExit(worker(args.config, args.seed, args.physical_gpu, args.output_dir))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
