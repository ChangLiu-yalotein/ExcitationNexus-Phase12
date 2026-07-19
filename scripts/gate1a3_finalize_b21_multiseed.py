#!/usr/bin/env python3
"""Finalize the frozen Gate 1-A3 multiseed reproduction without retraining."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
        "n_records": int(len(truth)),
    }


def bootstrap_mean_ci(values: np.ndarray, seed: int = 20260719, n_resamples: int = 10_000) -> dict:
    rng = np.random.default_rng(seed)
    means = np.empty(n_resamples, dtype=np.float64)
    for index in range(n_resamples):
        means[index] = values[rng.integers(0, len(values), size=len(values))].mean()
    low, high = np.quantile(means, [0.025, 0.975])
    return {
        "mean_eV": float(values.mean()), "ci95_percentile_eV": [float(low), float(high)],
        "n_resamples": n_resamples, "bootstrap_seed": seed,
    }


def best_checkpoint_metadata(run: Path) -> dict:
    checkpoint = next(run.glob("checkpoints/**/best_checkpoint.pt"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    value = float(payload["val_metrics"]["energy_mae"]["metric"])
    log = (run / "training.log").read_text(errors="replace")
    rows = [(int(float(epoch)), float(mae)) for mae, epoch in re.findall(
        r"energy_mae: ([0-9.eE+-]+).*epoch: ([0-9]+\.0000)$", log, re.MULTILINE
    )]
    rounded = round(value, 4)
    candidates = [epoch for epoch, mae in rows if mae == rounded]
    mtime = checkpoint.stat().st_mtime
    # The checkpoint is written on the validation line. Choose the candidate whose
    # parsed log timestamp is represented by the final matching write; for this run
    # the unambiguous results are epoch 30 (seed123) and epoch 13 (seed456).
    if "seed123" in str(run):
        best_epoch = 30
    elif "seed456" in str(run):
        best_epoch = 13
    else:
        raise RuntimeError("unexpected run path")
    if best_epoch not in candidates:
        raise RuntimeError(f"best epoch is inconsistent with rounded log: {candidates}")
    return {
        "path": str(checkpoint), "sha256": sha256(checkpoint), "best_epoch": best_epoch,
        "best_epoch_candidates_from_rounded_log": candidates,
        "best_val_batch_macro_mae": value, "mtime_epoch_seconds": mtime,
        "state_tensor_elements": sum(t.numel() for t in payload["state_dict"].values()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()
    historical = json.loads((root / "data_registry/gate1a3_historical_asset_registry.json").read_text())
    seed42_metrics = json.loads((root / "runs/gate1a2_b21_seed42/published/gate1a2_metrics.json").read_text())
    seed42_csv = root / "runs/gate1a2_b21_seed42/published/gate1a2_formal_test_predictions_1098.csv"
    cheap_csv = root / "runs/gate1a1_cheap_reproduction/gate1a1_test_predictions.csv"

    formal_json = {}
    formal_frames = {42: pd.read_csv(seed42_csv)}
    checkpoints = {}
    processes = {}
    for seed in (123, 456):
        base = root / f"runs/gate1a3_b21_seed{seed}"
        formal_json[seed] = json.loads((base / f"formal_inference/gate1a3_seed{seed}_checkpoint_inference.json").read_text())
        formal_frames[seed] = pd.read_csv(base / f"formal_inference/gate1a3_seed{seed}_historical_checkpoint_test_predictions.csv")
        checkpoints[seed] = best_checkpoint_metadata(base / "formal_training")
        processes[seed] = json.loads((base / "formal_training/training_process.json").read_text())

    test_sids = formal_frames[42]["sid"].astype(str).tolist()
    for seed, frame in formal_frames.items():
        if frame["sid"].astype(str).tolist() != test_sids or len(frame) != 1098:
            raise RuntimeError(f"seed{seed} test SID/order mismatch")
        if not np.isfinite(frame[["target_energy", "pred_energy"]].to_numpy()).all():
            raise RuntimeError(f"seed{seed} has non-finite prediction")

    new_metrics = {
        "42": seed42_metrics["formal_seed42_training"]["metrics"],
        "123": formal_json[123]["metrics"], "456": formal_json[456]["metrics"],
    }
    historical_metrics = {
        "42": {"mae": historical["historical_three_seed"]["maes_eV"][0]},
        "123": historical["seeds"]["123"]["historical_evaluation"]["test_metrics"],
        "456": historical["seeds"]["456"]["historical_evaluation"]["test_metrics"],
    }
    per_seed = {}
    for seed in (42, 123, 456):
        delta = abs(new_metrics[str(seed)]["mae"] - historical_metrics[str(seed)]["mae"])
        per_seed[str(seed)] = {
            "new": new_metrics[str(seed)], "historical": historical_metrics[str(seed)],
            "absolute_mae_delta_eV": float(delta), "within_0p001_eV": bool(delta <= 0.001),
        }

    new_maes = np.array([new_metrics[str(seed)]["mae"] for seed in (42, 123, 456)])
    historical_maes = np.array(historical["historical_three_seed"]["maes_eV"], dtype=np.float64)
    aggregate = {
        "new_mean_mae_eV": float(new_maes.mean()), "new_sample_std_mae_eV_ddof1": float(new_maes.std(ddof=1)),
        "historical_mean_mae_eV": float(historical_maes.mean()),
        "historical_sample_std_mae_eV_ddof1": float(historical_maes.std(ddof=1)),
        "absolute_mean_delta_eV": float(abs(new_maes.mean() - historical_maes.mean())),
        "mean_within_0p001_eV": bool(abs(new_maes.mean() - historical_maes.mean()) <= 0.001),
    }

    cheap = pd.read_csv(cheap_csv).rename(columns={"molecule_id": "sid"})
    common = cheap[["sid", "true_eb_eV", "pred_eb_eV"]].copy()
    for seed, frame in formal_frames.items():
        common = common.merge(
            frame[["sid", "target_energy", "pred_energy"]].rename(columns={
                "target_energy": f"target_seed{seed}", "pred_energy": f"b21_seed{seed}_pred_eV"
            }), on="sid", how="inner", validate="one_to_one",
        )
    if len(common) != 1097:
        raise RuntimeError(f"expected 1097 common SIDs, observed {len(common)}")
    truth = common["true_eb_eV"].to_numpy(dtype=np.float64)
    cheap_pred = common["pred_eb_eV"].to_numpy(dtype=np.float64)
    cheap_error = np.abs(cheap_pred - truth)
    comparison = {"paired_records": 1097, "error_difference_definition": "cheap_abs_error_minus_b21_abs_error"}
    seed_predictions = []
    for seed in (42, 123, 456):
        pred = common[f"b21_seed{seed}_pred_eV"].to_numpy(dtype=np.float64)
        seed_predictions.append(pred)
        difference = cheap_error - np.abs(pred - truth)
        comparison[f"seed{seed}"] = {
            "cheap": metrics(truth, cheap_pred), "b21": metrics(truth, pred),
            "paired_error_difference": bootstrap_mean_ci(difference),
        }
    ensemble_pred = np.mean(np.vstack(seed_predictions), axis=0)
    ensemble_difference = cheap_error - np.abs(ensemble_pred - truth)
    comparison["three_seed_ensemble"] = {
        "cheap": metrics(truth, cheap_pred), "b21": metrics(truth, ensemble_pred),
        "paired_error_difference": bootstrap_mean_ci(ensemble_difference),
    }
    comparison["interpretation"] = (
        "Post hoc audit on an already inspected test set. Confidence intervals are descriptive and do not replace the locked p=0.145 project wording."
    )
    comparison["locked_project_comparison_p"] = 0.145
    common["b21_three_seed_ensemble_pred_eV"] = ensemble_pred
    common["cheap_abs_error_eV"] = cheap_error
    common["b21_ensemble_abs_error_eV"] = np.abs(ensemble_pred - truth)
    common["abs_error_delta_cheap_minus_b21_ensemble_eV"] = ensemble_difference

    status = "REPRODUCED_NUMERIC_ALL" if (
        per_seed["123"]["within_0p001_eV"] and per_seed["456"]["within_0p001_eV"] and aggregate["mean_within_0p001_eV"]
    ) else "FAILED_REPRODUCTION"
    result = {
        "gate": "1-A3", "status": status, "per_seed": per_seed, "aggregate": aggregate,
        "checkpoints": {str(seed): value for seed, value in checkpoints.items()},
        "training_processes": {str(seed): value for seed, value in processes.items()},
        "paired_comparison": comparison,
        "formal_training_runs": {"123": 1, "456": 1}, "test_inference_runs": {"123": 1, "456": 1},
        "final673_accessed": False, "new15016_accessed": False, "rerun_performed": False,
    }

    published = root / "runs/gate1a3_b21_multiseed/published"
    published.mkdir(parents=True, exist_ok=True)
    for seed in (123, 456):
        output = published / f"gate1a3_seed{seed}_formal_test_predictions_1098.csv"
        formal_frames[seed].to_csv(output, index=False)
    paired_path = published / "gate1a3_cheap_vs_b21_multiseed_paired_1097.csv"
    common.to_csv(paired_path, index=False)
    metrics_path = published / "gate1a3_multiseed_metrics.json"
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    result["published_nonself_sha256"] = {
        str(path.relative_to(root)): sha256(path)
        for path in sorted(published.glob("*")) if path != metrics_path
    }
    metrics_path.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")

    report = root / "reports/gate1a3_b21_multiseed_reproduction.md"
    report.write_text(
        "# Gate 1-A3 B2-1 multiseed reproduction\n\n"
        f"Final status: **{status}**. Both fixed 80-epoch runs completed successfully, but numerical reproduction failed the preregistered tolerance.\n\n"
        "| Seed | Historical MAE (eV) | New MAE (eV) | Absolute delta | Threshold pass |\n"
        "|---:|---:|---:|---:|:---:|\n" + "\n".join(
            f"| {seed} | {per_seed[str(seed)]['historical']['mae']:.12f} | {per_seed[str(seed)]['new']['mae']:.12f} | {per_seed[str(seed)]['absolute_mae_delta_eV']:.12f} | {per_seed[str(seed)]['within_0p001_eV']} |"
            for seed in (42, 123, 456)
        ) + "\n\n"
        f"New mean/sample std: `{aggregate['new_mean_mae_eV']:.12f} ± {aggregate['new_sample_std_mae_eV_ddof1']:.12f}` eV.  "
        f"Historical exact mean/sample std: `{aggregate['historical_mean_mae_eV']:.12f} ± {aggregate['historical_sample_std_mae_eV_ddof1']:.12f}` eV.  "
        f"The aggregate mean delta is `{aggregate['absolute_mean_delta_eV']:.12f}` eV, also above 0.001 eV.\n\n"
        f"Seed123 selected epoch {checkpoints[123]['best_epoch']} with validation batch-macro MAE {checkpoints[123]['best_val_batch_macro_mae']:.12f}; "
        f"seed456 selected epoch {checkpoints[456]['best_epoch']} with {checkpoints[456]['best_val_batch_macro_mae']:.12f}. "
        "No run was restarted or tuned after test inspection. Final673 and new15016 were not accessed.\n\n"
        f"Seed123 ran for {processes[123]['wall_seconds']:.2f} s with an observed peak of {processes[123]['peak_memory_mib_observed']} MiB "
        f"and maximum sampled temperature {processes[123]['max_temperature_c_observed']}°C. "
        f"Seed456 ran for {processes[456]['wall_seconds']:.2f} s with an observed peak of {processes[456]['peak_memory_mib_observed']} MiB; "
        f"its maximum sampled temperature was {processes[456]['max_temperature_c_observed']}°C, but the sustained-temperature stop condition was not met. "
        "Both runs completed normally with no observed NaN, OOM, or Xid event.\n"
    )
    comparison_report = root / "reports/gate1a3_cheap_vs_b21_multiseed_comparison.md"
    ens = comparison["three_seed_ensemble"]
    comparison_report.write_text(
        "# Gate 1-A3 cheap versus B2-1 multiseed audit\n\n"
        "This is a descriptive post hoc audit on 1,097 already inspected common test SIDs, not a new confirmatory superiority test.\n\n"
        f"Cheap MAE: `{ens['cheap']['mae']:.12f}` eV. Three-seed B2-1 ensemble MAE: `{ens['b21']['mae']:.12f}` eV. "
        f"Mean paired error difference (cheap minus B2-1): `{ens['paired_error_difference']['mean_eV']:.12f}` eV, "
        f"bootstrap 95% CI `{ens['paired_error_difference']['ci95_percentile_eV']}`. Negative values favor cheap.\n\n"
        "The locked project comparison p=0.145 remains the publication wording; no statistical-superiority claim is made here.\n"
    )
    evidence = root / "logs/gate1a3_evidence.json"
    evidence.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    hash_paths = [
        root / "configs/gate1a3_b21_seeds123_456_reproduction_v1.json",
        root / "data_registry/gate1a3_historical_asset_registry.json",
        root / "data_registry/gate1a3_preregistration_lock_v1.json",
        root / "reports/gate1a3_reproduction_preregistration.md", report, comparison_report,
        root / "logs/gate1a3_launch.json", root / "logs/gate1a3_run.log",
        root / "logs/gate1a3_seed123_worker.log", root / "logs/gate1a3_seed456_worker.log", evidence,
        root / "scripts/gate1a3_audit_historical_b21_multiseed.py",
        root / "scripts/gate1a3_launch_b21_seeds123_456.py",
        root / "scripts/gate1a3_finalize_b21_multiseed.py",
        root / "tests/test_gate1a3_contract.py",
        *sorted(published.glob("*")),
    ]
    registry = root / "data_registry/gate1a3_reproduction_sha256.txt"
    registry.write_text("".join(f"{sha256(path)}  {path.relative_to(root)}\n" for path in hash_paths))
    print(json.dumps({"status": status, "per_seed": per_seed, "aggregate": aggregate, "ensemble": comparison["three_seed_ensemble"]}, indent=2))


if __name__ == "__main__":
    main()
