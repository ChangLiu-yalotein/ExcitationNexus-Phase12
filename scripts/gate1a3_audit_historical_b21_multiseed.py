#!/usr/bin/env python3
"""Audit and preregister the frozen historical B2-1 seed123/456 assets."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from ase.db import connect

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")
HIST = Path("/home/changliu/ExcitationNexus/equiformer_v3_model")
SHARED = HIST / "checkpoints/2026-04-25-09-33-52"
LOGS = {
    123: HIST / "logs/b2_1_seed123_gpu1_isolated_20260425_093310",
    456: HIST / "logs/b2_1_seed456_gpu2_isolated_20260425_093315",
}
CHECKPOINTS = {123: SHARED / "checkpoint.pt", 456: SHARED / "best_checkpoint.pt"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def ids(path: Path) -> list[str]:
    return [str(getattr(row, "name", getattr(row, "sid", row.id))) for row in connect(path).select()]


def log_summary(path: Path) -> dict:
    text = path.read_text(errors="replace")
    values = [(int(float(epoch)), float(mae)) for mae, epoch in re.findall(
        r"energy_mae: ([0-9.eE+-]+).*epoch: ([0-9]+\.0000)$", text, re.MULTILINE
    )]
    wall = re.findall(r"Total time taken: ([0-9.]+)", text)
    return {
        "validation_epoch_count": len(values),
        "best_epoch_from_rounded_log": min(values, key=lambda item: (item[1], item[0]))[0],
        "best_val_batch_macro_mae_rounded_log": min(value for _, value in values),
        "final_val_batch_macro_mae_rounded_log": values[-1][1],
        "wall_seconds": float(wall[-1]),
        "completed_80_epochs": values[-1][0] == 80,
    }


def main() -> None:
    common = {
        "config": HIST / "configs/dual_tower/b2_1_naive_dual_tower.yml",
        "train_db": HIST / "data/dual_tower_7316/dft_dual_tower_train.db",
        "val_db": HIST / "data/dual_tower_7316/dft_dual_tower_val.db",
        "test_db": HIST / "data/dual_tower_7316/dft_dual_tower_test.db",
        "model_source": HIST / "equiformer_v3/models/naive_dual_tower.py",
        "dataset_source": HIST / "equiformer_v3/datasets/dual_tower.py",
        "trainer_source": HIST / "equiformer_v3/src/fairchem/core/trainers/b2_1_trainer.py",
        "wrapper": HIST / "scripts/run_b2_1_training_wrapper.py",
        "normalizers": SHARED / "normalizers.pt",
    }
    split_ids = {part: ids(common[f"{part}_db"]) for part in ("train", "val", "test")}
    assert {part: len(value) for part, value in split_ids.items()} == {"train": 5120, "val": 1098, "test": 1098}
    assert len(set().union(*(set(value) for value in split_ids.values()))) == 7316

    seeds = {}
    for seed in (123, 456):
        log_dir = LOGS[seed]
        paths = {
            "checkpoint": CHECKPOINTS[seed],
            "training_log": log_dir / "training.log",
            "evaluation_results": log_dir / "eval_final/evaluation_results.json",
            "test_predictions": log_dir / "eval_final/test_predictions_with_sid.csv",
            "test_predictions_without_sid": log_dir / "eval_final/test_predictions.csv",
            "val_predictions": log_dir / "eval_final/val_predictions.csv",
        }
        evaluation = json.loads(paths["evaluation_results"].read_text())
        prediction = pd.read_csv(paths["test_predictions"])
        assert prediction["sid"].astype(str).tolist() == split_ids["test"]
        payload = torch.load(paths["checkpoint"], map_location="cpu", weights_only=False)
        checkpoint_seed = int(payload["config"]["cmd"]["seed"])
        assert checkpoint_seed == seed
        state_elements = sum(t.numel() for t in payload["state_dict"].values())
        assert state_elements == 1_075_318
        seeds[str(seed)] = {
            "paths": {name: str(path) for name, path in paths.items()},
            "sha256": {name: sha256(path) for name, path in paths.items()},
            "historical_evaluation": evaluation,
            "checkpoint_seed": checkpoint_seed,
            "checkpoint_epoch": payload.get("epoch"),
            "checkpoint_step": payload.get("step"),
            "checkpoint_val_metrics": payload.get("val_metrics"),
            "checkpoint_kind": "final_epoch80" if seed == 123 else "best_validation",
            "state_tensor_elements": state_elements,
            "log_summary": log_summary(paths["training_log"]),
        }

    historical_maes = np.array([
        seeds[str(seed)]["historical_evaluation"]["test_metrics"]["mae"] for seed in (123, 456)
    ])
    seed42_mae = 0.07659060508012772
    all_maes = np.array([seed42_mae, *historical_maes])
    registry = {
        "gate": "1-A3",
        "status": "ASSET_AUDIT_PASS",
        "common_assets": {name: {"path": str(path), "sha256": sha256(path)} for name, path in common.items()},
        "split_counts": {part: len(value) for part, value in split_ids.items()},
        "split_order_sha256": {
            part: hashlib.sha256(("\n".join(value) + "\n").encode()).hexdigest() for part, value in split_ids.items()
        },
        "seeds": seeds,
        "historical_three_seed": {
            "maes_eV": [float(value) for value in all_maes],
            "mean_mae_eV": float(all_maes.mean()),
            "sample_std_mae_eV_ddof1": float(all_maes.std(ddof=1)),
        },
        "shared_checkpoint_collision": {
            "present": True,
            "directory": str(SHARED),
            "seed123_policy": "Use final epoch-80 checkpoint.pt; historical best was overwritten/unavailable.",
            "seed456_policy": "Use best_checkpoint.pt.",
            "causality": "Both jobs used the same timestamp-id and wrote the same directory.",
        },
        "historical_protocol_limitations": [
            "Validation selection is batch-macro and overweights the final two-record batch.",
            "B2-1 uses 7,316 records while cheap Layer G uses 7,313.",
            "Cheap paired comparisons are restricted to the common 1,097 test SIDs.",
        ],
        "final673_accessed": False,
        "new15016_accessed": False,
    }
    registry_path = ROOT / "data_registry/gate1a3_historical_asset_registry.json"
    write_json(registry_path, registry)

    config = {
        "gate": "1-A3", "version": "v1", "status": "PREREGISTERED_BEFORE_INFERENCE_OR_TRAINING",
        "objective": "Exactly one B2-1 reproduction each for seeds 123 and 456 on separate GPUs.",
        "seeds": [123, 456], "seed_to_physical_gpu": {"123": 0, "456": 1},
        "common_assets": {name: str(path) for name, path in common.items()},
        "seed_assets": {seed: data["paths"] for seed, data in seeds.items()},
        "expected_sha256": {
            "common": {name: sha256(path) for name, path in common.items()},
            "seeds": {seed: data["sha256"] for seed, data in seeds.items()},
        },
        "historical_results": {
            seed: {
                "test": data["historical_evaluation"]["test_metrics"],
                "validation_independent": data["historical_evaluation"]["val_metrics"],
                "checkpoint_kind": data["checkpoint_kind"],
                "checkpoint_epoch": data["checkpoint_epoch"],
                "checkpoint_val_metrics": data["checkpoint_val_metrics"],
                "training_wall_seconds": data["log_summary"]["wall_seconds"],
            } for seed, data in seeds.items()
        },
        "training_contract": {
            "runs_per_seed": 1, "batch_size": 8, "epochs": 80, "optimizer": "AdamW",
            "checkpoint_selection": "historical batch-macro validation MAE", "early_stop": False,
            "test_after_training_only": True, "hyperparameter_search": False, "restart_on_bad_metric": False,
            "independent_output_directories": True,
        },
        "model_contract": {
            "architecture": "shared EquiformerV3 graph-scalar towers plus 1->64 projection and late fusion",
            "parameter_count": 1_065_570, "checkpoint_state_tensor_elements": 1_075_318,
            "max_radius_angstrom": 8.0, "max_neighbors": 20,
            "normalization_mean_eV": 0.800596, "normalization_stdev_eV": 0.194270,
        },
        "reproduction_tolerance_mae_eV_per_seed": 0.001,
        "reproduction_tolerance_mae_eV_aggregate_mean": 0.001,
        "aggregate_std": "sample standard deviation, ddof=1",
        "forbidden": ["final673", "new15016", "B2-0", "B2-2a", "seed42 rerun", "target-derived inputs"],
        "failure_policy": "No restart, seed replacement, tuning, or post-test selection.",
        "output_roots": {
            "123": str(ROOT / "runs/gate1a3_b21_seed123"),
            "456": str(ROOT / "runs/gate1a3_b21_seed456"),
        },
        "final673_access": False, "new15016_access": False,
    }
    config_path = ROOT / "configs/gate1a3_b21_seeds123_456_reproduction_v1.json"
    write_json(config_path, config)
    report_path = ROOT / "reports/gate1a3_reproduction_preregistration.md"
    report_path.write_text(
        "# Gate 1-A3 preregistration\n\n"
        "Status: **FROZEN BEFORE CHECKPOINT INFERENCE OR TRAINING**.\n\n"
        "Exactly one fixed 80-epoch B2-1 run will be executed for each of seeds 123 and 456 on separate physical GPUs. "
        "The historical shared-directory collision is retained as provenance: seed123 uses its epoch-80 final checkpoint; "
        "seed456 uses the surviving best-validation checkpoint. No result-driven restart or tuning is allowed.\n\n"
        f"Historical seed123 test MAE: `{historical_maes[0]:.17g}` eV.  "
        f"Historical seed456 test MAE: `{historical_maes[1]:.17g}` eV.  "
        f"Historical exact three-seed mean/sample std: `{all_maes.mean():.17g} ± {all_maes.std(ddof=1):.17g}` eV.\n\n"
        "The 7,316-record B2-1 protocol, batch-macro validation limitation, and 1,097-SID cheap comparison boundary remain frozen. "
        "Final673 and new15016 are outside scope.\n"
    )
    locked = {str(path.relative_to(ROOT)): sha256(path) for path in (config_path, registry_path, report_path)}
    aggregate = hashlib.sha256("".join(f"{name}:{value}\n" for name, value in sorted(locked.items())).encode()).hexdigest()
    write_json(ROOT / "data_registry/gate1a3_preregistration_lock_v1.json", {
        "gate": "1-A3", "version": "v1", "status": "FROZEN_BEFORE_INFERENCE_AND_TRAINING",
        "locked_utc": datetime.now(timezone.utc).isoformat(), "files": locked,
        "aggregate_sha256": aggregate, "post_lock_policy": "Corrections require explicit v2; never overwrite v1.",
    })
    print(json.dumps({"status": registry["status"], "aggregate_sha256": aggregate, "historical_three_seed": registry["historical_three_seed"]}, indent=2))


if __name__ == "__main__":
    main()
