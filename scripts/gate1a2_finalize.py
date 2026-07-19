#!/usr/bin/env python3
"""Finalize Gate 1-A2 metrics and the frozen 1,097-SID paired audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
        "n_records": int(len(truth)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--formal-inference", required=True, type=Path)
    parser.add_argument("--checkpoint-inference", required=True, type=Path)
    parser.add_argument("--cheap-predictions", required=True, type=Path)
    parser.add_argument("--training-process", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = json.loads(args.config.read_text())
    formal_json = json.loads((args.formal_inference.parent / "gate1a2_checkpoint_inference.json").read_text())
    checkpoint_json = json.loads((args.checkpoint_inference.parent / "gate1a2_checkpoint_inference.json").read_text())
    training = json.loads(args.training_process.read_text())
    formal = pd.read_csv(args.formal_inference)
    checkpoint = pd.read_csv(args.checkpoint_inference)
    cheap = pd.read_csv(args.cheap_predictions)

    if formal["sid"].duplicated().any() or cheap["molecule_id"].duplicated().any():
        raise RuntimeError("duplicate SID in paired inputs")
    common = formal.merge(cheap, left_on="sid", right_on="molecule_id", how="inner", validate="one_to_one")
    if len(common) != 1097:
        raise RuntimeError(f"expected 1097 paired SIDs, observed {len(common)}")
    truth_delta = np.abs(common["target_energy"].to_numpy() - common["true_eb_eV"].to_numpy())
    if float(truth_delta.max()) > 1e-6:
        raise RuntimeError("paired truth mismatch")

    truth = common["true_eb_eV"].to_numpy(dtype=np.float64)
    b21_pred = common["pred_energy"].to_numpy(dtype=np.float64)
    cheap_pred = common["pred_eb_eV"].to_numpy(dtype=np.float64)
    cheap_abs = np.abs(cheap_pred - truth)
    b21_abs = np.abs(b21_pred - truth)
    statistic, nominal_p = wilcoxon(cheap_abs, b21_abs, alternative="two-sided")
    comparison = {
        "paired_records": 1097,
        "excluded_b21_only_test_sid": sorted(set(formal["sid"]) - set(common["sid"])),
        "max_abs_truth_delta_eV": float(truth_delta.max()),
        "cheap": metrics(truth, cheap_pred),
        "b21_new_training": metrics(truth, b21_pred),
        "delta_mae_cheap_minus_b21_eV": float(cheap_abs.mean() - b21_abs.mean()),
        "nominal_same_pair_wilcoxon_statistic": float(statistic),
        "nominal_same_pair_wilcoxon_p": float(nominal_p),
        "interpretation": "Audit-only paired calculation on an already inspected test set; no new superiority claim.",
        "locked_project_comparison_p": 0.145,
    }
    paired = pd.DataFrame({
        "molecule_id": common["sid"], "true_eb_eV": truth,
        "cheap_pred_eb_eV": cheap_pred, "b21_pred_eb_eV": b21_pred,
        "cheap_abs_error_eV": cheap_abs, "b21_abs_error_eV": b21_abs,
        "abs_error_delta_cheap_minus_b21_eV": cheap_abs - b21_abs,
    })
    paired_path = args.output_dir / "gate1a2_cheap_vs_b21_paired_1097.csv"
    paired.to_csv(paired_path, index=False)
    formal_copy = args.output_dir / "gate1a2_formal_test_predictions_1098.csv"
    checkpoint_copy = args.output_dir / "gate1a2_historical_checkpoint_predictions_1098.csv"
    shutil.copyfile(args.formal_inference, formal_copy)
    shutil.copyfile(args.checkpoint_inference, checkpoint_copy)

    original_mae = config["historical_result"]["test_mae"]
    formal_mae = formal_json["metrics"]["mae"]
    result = {
        "status": "REPRODUCED_NUMERIC" if abs(formal_mae - original_mae) <= 0.001 else "FAILED_REPRODUCTION",
        "historical_checkpoint_inference": checkpoint_json,
        "formal_seed42_training": {
            "metrics": formal_json["metrics"],
            "historical_original_test_mae": original_mae,
            "historical_sid_vector_test_mae": config["historical_result"]["sid_vector_recomputed_test_mae"],
            "absolute_mae_delta_vs_original": abs(formal_mae - original_mae),
            "absolute_mae_delta_vs_sid_vector": abs(formal_mae - config["historical_result"]["sid_vector_recomputed_test_mae"]),
            "best_epoch": 80,
            "best_val_batch_macro_mae": 0.07653653435409069,
            "parameter_count": 1065570,
            "checkpoint_sha256": formal_json["checkpoint_sha256"],
            "prediction_sha256": sha256(formal_copy),
            "wall_seconds": training["wall_seconds"],
            "peak_gpu_memory_mib": {"value": 14767, "qualifier": "maximum observed by external polling; not continuous profiler peak"},
            "physical_gpu_id": 0,
        },
        "paired_comparison": comparison,
        "environment": {
            "python": sys.version.split()[0], "platform": platform.platform(),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "numpy": np.__version__, "pandas": pd.__version__,
            "scipy": scipy.__version__, "scikit_learn": sklearn.__version__,
        },
        "published_artifacts": {
            "formal_predictions": {"path": str(formal_copy), "sha256": sha256(formal_copy)},
            "checkpoint_predictions": {"path": str(checkpoint_copy), "sha256": sha256(checkpoint_copy)},
            "paired_predictions": {"path": str(paired_path), "sha256": sha256(paired_path)},
        },
        "scope": {
            "formal_training_runs": 1, "other_seeds": False, "new15016": False,
            "final673": False, "b2_0": False, "b2_2a": False,
        },
    }
    (args.output_dir / "gate1a2_metrics.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
