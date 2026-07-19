#!/usr/bin/env python3
"""Train Gate 1-B1 baselines without reading test targets."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import resource
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return 0.0
    order = np.argsort(values[valid], kind="mergesort")
    x, w = values[valid][order], weights[valid][order]
    return float(x[np.searchsorted(np.cumsum(w), 0.5 * w.sum(), side="left")])


def fit_preprocessor(matrix: np.ndarray, weights: np.ndarray) -> dict[str, np.ndarray]:
    medians = np.array([weighted_median(matrix[:, j], weights) for j in range(matrix.shape[1])])
    imputed = np.where(np.isfinite(matrix), matrix, medians)
    total = weights.sum()
    means = np.sum(imputed * weights[:, None], axis=0) / total
    variances = np.sum(((imputed - means) ** 2) * weights[:, None], axis=0) / total
    scales = np.sqrt(np.maximum(variances, 0.0))
    scales[~np.isfinite(scales) | (scales < 1e-12)] = 1.0
    return {"medians": medians, "means": means, "scales": scales}


def transform(matrix: np.ndarray, prep: dict[str, np.ndarray]) -> np.ndarray:
    imputed = np.where(np.isfinite(matrix), matrix, prep["medians"])
    return ((imputed - prep["means"]) / prep["scales"]).astype(np.float32)


def record_group_metrics(y: np.ndarray, p: np.ndarray, groups: np.ndarray) -> dict:
    error = np.asarray(p) - np.asarray(y)
    unique = np.unique(groups)
    group_abs = np.array([np.mean(np.abs(error[groups == group])) for group in unique])
    group_sq = np.array([np.mean(error[groups == group] ** 2) for group in unique])
    record_denom = np.sum((y - np.mean(y)) ** 2)
    group_weights = np.array([1.0 / np.sum(groups == group) for group in groups])
    weighted_mean = np.sum(group_weights * y) / np.sum(group_weights)
    group_denom = np.sum(group_weights * (y - weighted_mean) ** 2)
    return {
        "records": int(len(y)), "groups": int(len(unique)),
        "record_mae": float(np.mean(np.abs(error))), "group_macro_mae": float(group_abs.mean()),
        "record_rmse": float(np.sqrt(np.mean(error ** 2))), "group_macro_rmse": float(np.sqrt(group_sq.mean())),
        "record_r2": float(1 - np.sum(error ** 2) / record_denom) if record_denom > 0 else None,
        "group_macro_r2": float(1 - np.sum(group_weights * error ** 2) / group_denom) if group_denom > 0 else None,
    }


def load_inputs(config_path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame]:
    config = json.loads(config_path.read_text())
    if sha256(Path(config["table"])) != config["table_sha256"]:
        raise RuntimeError("table hash mismatch")
    if sha256(Path(config["manifest"])) != config["manifest_sha256"]:
        raise RuntimeError("manifest hash mismatch")
    if sha256(Path(config["feature_cache"])) != config["feature_cache_sha256"]:
        raise RuntimeError("feature cache hash mismatch")
    manifest = pd.read_csv(config["manifest"])
    features = pd.read_parquet(config["feature_cache"])
    frame = manifest.merge(features, on="molecule_id", how="left", validate="one_to_one")
    if len(frame) != 15016 or frame[config["features"]["M2_C1p5_safe_no_dipole"]].isna().any().any():
        raise RuntimeError("feature join or finite check failed")
    return config, manifest, frame


def read_train_val_targets(config: dict, manifest: pd.DataFrame) -> pd.DataFrame:
    allowed = manifest[manifest["partition"].isin(["train", "val"])]["molecule_id"].astype(str).tolist()
    dataset = ds.dataset(config["table"], format="parquet")
    table = dataset.to_table(
        columns=["molecule_id", config["primary_target"]],
        filter=ds.field("molecule_id").isin(allowed),
    ).to_pandas()
    if len(table) != config["split_counts"]["train"] + config["split_counts"]["val"]:
        raise RuntimeError("train/val-only target read count mismatch")
    return table


def save_preprocessor(path: Path, prep: dict[str, np.ndarray], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, columns=np.asarray(columns), **prep)


def load_preprocessor(path: Path) -> tuple[dict[str, np.ndarray], list[str]]:
    payload = np.load(path, allow_pickle=False)
    prep = {name: payload[name] for name in ("medians", "means", "scales")}
    return prep, payload["columns"].astype(str).tolist()


def baseline_run(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError(f"baseline output exists: {output}")
    output.mkdir(parents=True)
    config, manifest, frame = load_inputs(config_path)
    targets = read_train_val_targets(config, manifest)
    frame = frame.merge(targets, on="molecule_id", how="left", validate="one_to_one")
    if frame.loc[frame["partition"].eq("test"), config["primary_target"]].notna().any():
        raise RuntimeError("test target entered training frame")
    train = frame[frame["partition"].eq("train")].copy()
    val = frame[frame["partition"].eq("val")].copy()
    target = config["primary_target"]
    weights = train["group_weight"].to_numpy(np.float64)
    if not np.isclose(weights.sum(), 10248.0, atol=1e-9):
        raise RuntimeError("train group-weight sum mismatch")

    preprocessors = {}
    transformed = {}
    for label, key in (("c0", "M1_C0_open"), ("c1p5", "M2_C1p5_safe_no_dipole")):
        columns = config["features"][key]
        prep = fit_preprocessor(train[columns].to_numpy(np.float64), weights)
        path = output / f"preprocessor_{label}.npz"
        save_preprocessor(path, prep, columns)
        preprocessors[label] = {"path": str(path), "sha256": sha256(path), "columns": columns}
        transformed[label] = {
            "train": transform(train[columns].to_numpy(np.float64), prep),
            "val": transform(val[columns].to_numpy(np.float64), prep),
        }

    y_train = train[target].to_numpy(np.float64)
    y_val = val[target].to_numpy(np.float64)
    median = weighted_median(y_train, weights)
    median_pred = np.full(len(val), median)
    median_metrics = record_group_metrics(y_val, median_pred, val["structure_group_id_v1"].to_numpy())
    write_json(output / "weighted_median.json", {
        "model": "weighted_median", "value": median, "validation": median_metrics,
        "fit_scope": "train-only", "sample_weight_sum": float(weights.sum()), "test_target_accessed": False,
    })

    ridge = Ridge(alpha=float(config["ridge"]["alpha"]))
    started = time.perf_counter()
    ridge.fit(transformed["c0"]["train"], y_train, sample_weight=weights)
    ridge_seconds = time.perf_counter() - started
    ridge_pred = ridge.predict(transformed["c0"]["val"])
    ridge_path = output / "ridge_c0.npz"
    np.savez_compressed(ridge_path, coef=ridge.coef_, intercept=np.asarray([ridge.intercept_]))
    ridge_metrics = record_group_metrics(y_val, ridge_pred, val["structure_group_id_v1"].to_numpy())
    write_json(output / "ridge_c0.json", {
        "model": "ridge_c0", "model_path": str(ridge_path), "model_sha256": sha256(ridge_path),
        "validation": ridge_metrics, "training_wall_seconds": ridge_seconds,
        "fit_scope": "train-only", "sample_weight_sum": float(weights.sum()), "test_target_accessed": False,
    })
    pd.DataFrame({"molecule_id": val["molecule_id"], "prediction": ridge_pred}).to_csv(output / "ridge_c0_val_predictions.csv", index=False)
    quantiles = np.quantile(y_train, [0.25, 0.5, 0.75]).tolist()
    gate = {
        "status": "BASELINES_AND_PREPROCESSING_FROZEN", "preprocessors": preprocessors,
        "weighted_median_sha256": sha256(output / "weighted_median.json"),
        "ridge_model_sha256": sha256(ridge_path), "ridge_validation": ridge_metrics,
        "train_target_quartiles_eV": quantiles, "test_target_accessed": False,
    }
    write_json(output / "baseline_gate.json", gate)
    return gate


class GPUMonitor:
    def __init__(self, physical_gpu: int | None):
        self.physical_gpu = physical_gpu
        self.peak_mib = 0
        self.max_temperature_c = 0
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        if self.physical_gpu is None:
            return
        while not self.stop.wait(0.2):
            try:
                out = subprocess.check_output([
                    "nvidia-smi", f"--id={self.physical_gpu}",
                    "--query-gpu=memory.used,temperature.gpu", "--format=csv,noheader,nounits",
                ], text=True).strip().split(",")
                self.peak_mib = max(self.peak_mib, int(out[0].strip()))
                self.max_temperature_c = max(self.max_temperature_c, int(out[1].strip()))
            except Exception:
                pass

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop.set()
        self.thread.join()


def seed_run(config_path: Path, baseline_dir: Path, output: Path, seed: int, device: str, physical_gpu: int | None) -> dict:
    if output.exists():
        raise RuntimeError(f"seed output exists: {output}")
    if not (baseline_dir / "baseline_gate.json").exists():
        raise RuntimeError("baseline/preprocessing gate is absent")
    output.mkdir(parents=True)
    config, manifest, frame = load_inputs(config_path)
    targets = read_train_val_targets(config, manifest)
    frame = frame.merge(targets, on="molecule_id", how="left", validate="one_to_one")
    train, val = frame[frame["partition"].eq("train")], frame[frame["partition"].eq("val")]
    y_train, y_val = train[config["primary_target"]].to_numpy(), val[config["primary_target"]].to_numpy()
    weights = train["group_weight"].to_numpy(np.float64)
    results = {"seed": seed, "device": device, "physical_gpu": physical_gpu, "models": {}, "test_target_accessed": False}
    for label, feature_key, prep_name in (
        ("xgb_c0", "M1_C0_open", "c0"), ("xgb_c1p5_safe", "M2_C1p5_safe_no_dipole", "c1p5")
    ):
        prep_path = baseline_dir / f"preprocessor_{prep_name}.npz"
        prep, columns = load_preprocessor(prep_path)
        if columns != config["features"][feature_key]:
            raise RuntimeError("preprocessor feature order mismatch")
        x_train = transform(train[columns].to_numpy(np.float64), prep)
        x_val = transform(val[columns].to_numpy(np.float64), prep)
        params = dict(config["xgboost"])
        params.update({"random_state": seed, "device": device})
        model = XGBRegressor(**params)
        started = time.perf_counter()
        with GPUMonitor(physical_gpu if device == "cuda" else None) as monitor:
            model.fit(x_train, y_train, sample_weight=weights)
            training_seconds = time.perf_counter() - started
            inference_started = time.perf_counter()
            pred = model.predict(x_val)
            inference_seconds = time.perf_counter() - inference_started
        model_path = output / f"{label}.json"
        model.save_model(model_path)
        prediction_path = output / f"{label}_val_predictions.csv"
        pd.DataFrame({"molecule_id": val["molecule_id"], "prediction": pred}).to_csv(prediction_path, index=False)
        results["models"][label] = {
            "model_path": str(model_path), "model_sha256": sha256(model_path),
            "prediction_path": str(prediction_path), "prediction_sha256": sha256(prediction_path),
            "validation": record_group_metrics(y_val, pred, val["structure_group_id_v1"].to_numpy()),
            "training_wall_seconds": training_seconds, "validation_inference_wall_seconds": inference_seconds,
            "peak_gpu_memory_mib_observed": monitor.peak_mib, "max_temperature_c_observed": monitor.max_temperature_c,
            "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
            "preprocessor_sha256": sha256(prep_path), "sample_weight_sum": float(weights.sum()),
        }
    write_json(output / "seed_result.json", results)
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--baselines", action="store_true")
    parser.add_argument("--seed", type=int, choices=(42, 123, 456))
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    if args.baselines == (args.seed is not None):
        parser.error("choose exactly one of --baselines or --seed")
    if args.baselines:
        result = baseline_run(args.config, args.baseline_dir)
    else:
        if args.output is None:
            parser.error("seed run requires --output")
        result = seed_run(args.config, args.baseline_dir, args.output, args.seed, args.device, args.physical_gpu)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
