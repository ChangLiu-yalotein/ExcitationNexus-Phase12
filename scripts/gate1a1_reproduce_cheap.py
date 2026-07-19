#!/usr/bin/env python3
"""Fixed-protocol reproduction of the historical Stage 2C no-dipole model."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import sklearn
import xgboost
from rdkit import Chem
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


DESCRIPTORS = [
    "MolWt", "MolLogP", "MolMR", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "NumAromaticHeterocycles", "NumAliphaticHeterocycles", "NumSaturatedRings",
    "NumHeteroatoms", "HeavyAtomCount", "NumValenceElectrons", "NHOHCount",
    "NOCount", "FractionCSP3", "RingCount", "HallKierAlpha",
]
PM6_RAW_FIELDS = [
    "pm6_energy_hartree", "homo_hartree", "lumo_hartree",
    "homo_lumo_gap_hartree", "homo_lumo_gap_ev", "dipole_debye",
    "dipole_x", "dipole_y", "dipole_z", "num_atoms",
]
PM6_ORBITAL = [
    "pm6_homo_hartree", "pm6_lumo_hartree",
    "pm6_homo_lumo_gap_hartree", "pm6_homo_lumo_gap_ev",
]
PM6_NO_DIPOLE = PM6_ORBITAL + [
    "pm6_pm6_energy_hartree", "pm6_num_atoms", "pm6_normal_termination",
    "pm6_n_warnings", "pm6_missing_flag",
]
FORBIDDEN_SUBSTRINGS = (
    "coulomb_attraction", "tddft_", "multiwfn_", "wavelength", "dipole",
    "final673", "final_blind", "target_", "_label",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compact_sid(sid: str) -> str:
    donor, acceptor = sid.split("_")
    return donor.replace("D-", "D") + "_" + acceptor.replace("A-", "A")


def load_inputs(config: dict) -> tuple[list[str], dict[str, list[str]], np.ndarray, pd.DataFrame, list[str]]:
    assets = {key: Path(value) for key, value in config["historical_assets"].items()}
    for name, expected in config["expected_sha256"].items():
        path_key = config["hash_key_to_asset"][name]
        observed = sha256(assets[path_key])
        if observed != expected:
            raise RuntimeError(f"{name} hash mismatch: {observed} != {expected}")

    split = json.loads(assets["split_manifest"].read_text())
    if {key: len(split[key]) for key in ("train", "val", "test")} != {
        "train": 5118, "val": 1098, "test": 1097,
    }:
        raise RuntimeError("Layer G split count mismatch")
    all_sids = split["train"] + split["val"] + split["test"]
    if len(all_sids) != len(set(all_sids)):
        raise RuntimeError("Layer G split overlap or duplicate")

    with assets["teacher_table"].open(newline="") as handle:
        teacher = {row["molecule_id"]: float(row["eb_eV"]) for row in csv.DictReader(handle)}
    y = np.asarray([teacher[sid] for sid in all_sids], dtype=np.float64)

    wanted = {compact_sid(sid) for sid in all_sids}
    smiles: dict[str, str] = {}
    with assets["structure_registry"].open() as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("id") in wanted:
                smiles[row["id"]] = row["smiles"]
                if len(smiles) == len(wanted):
                    break
    if len(smiles) != len(wanted):
        raise RuntimeError("structure registry coverage mismatch")

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=config["features"]["morgan_radius"],
        fpSize=config["features"]["morgan_nbits"],
        includeChirality=config["features"]["morgan_include_chirality"],
    )
    rows: list[dict[str, float]] = []
    pm6_dir = assets["pm6_energy_dir"]
    for sid in all_sids:
        mol = Chem.MolFromSmiles(smiles[compact_sid(sid)])
        if mol is None:
            raise RuntimeError(f"SMILES parse failure: {sid}")
        features: dict[str, float] = {}
        for name in DESCRIPTORS:
            try:
                features[f"pair_{name}"] = float(getattr(Descriptors, name)(mol))
            except Exception:
                features[f"pair_{name}"] = np.nan
        fp = generator.GetFingerprint(mol)
        for index, value in enumerate(np.asarray(fp)):
            features[f"pair_morgan_{index}"] = float(value)

        metadata_path = pm6_dir / f"{sid}_metadata.json"
        if metadata_path.exists():
            try:
                metadata = json.loads(metadata_path.read_text())
                for field in PM6_RAW_FIELDS:
                    features[f"pm6_{field}"] = metadata.get(field, np.nan)
                features["pm6_normal_termination"] = 1.0 if metadata.get("normal_termination", False) else 0.0
                warnings = metadata.get("warnings", [])
                features["pm6_n_warnings"] = float(len(warnings)) if isinstance(warnings, list) else 0.0
                features["pm6_missing_flag"] = 0.0
            except (json.JSONDecodeError, ValueError):
                for column in PM6_NO_DIPOLE:
                    features[column] = np.nan
                features["pm6_missing_flag"] = 1.0
        else:
            for column in PM6_NO_DIPOLE:
                features[column] = np.nan
            features["pm6_missing_flag"] = 1.0
        rows.append(features)

    c0_columns = [f"pair_{name}" for name in DESCRIPTORS] + [
        f"pair_morgan_{index}" for index in range(config["features"]["morgan_nbits"])
    ]
    columns = c0_columns + PM6_NO_DIPOLE
    if len(columns) != 541 or len(columns) != len(set(columns)):
        raise RuntimeError(f"unexpected feature contract: {len(columns)}")
    forbidden = [column for column in columns if any(token.lower() in column.lower() for token in FORBIDDEN_SUBSTRINGS)]
    if forbidden:
        raise RuntimeError(f"target/dipole firewall failure: {forbidden[:10]}")
    frame = pd.DataFrame(rows)
    return all_sids, split, y, frame.loc[:, columns], columns


def indices(split: dict[str, list[str]], all_sids: list[str]) -> dict[str, np.ndarray]:
    lookup = {sid: index for index, sid in enumerate(all_sids)}
    return {name: np.asarray([lookup[sid] for sid in split[name]], dtype=np.int64) for name in ("train", "val", "test")}


def preprocess(frame: pd.DataFrame, idx: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], SimpleImputer, StandardScaler]:
    values = frame.to_numpy(dtype=np.float64)
    imputer = SimpleImputer(strategy="median")
    scaler = StandardScaler()
    train = scaler.fit_transform(imputer.fit_transform(values[idx["train"]]))
    transformed = {
        "train": train,
        "val": scaler.transform(imputer.transform(values[idx["val"]])),
        "test": scaler.transform(imputer.transform(values[idx["test"]])),
    }
    return transformed, imputer, scaler


def environment() -> dict[str, str]:
    return {
        "python": sys.version.split()[0], "platform": platform.platform(),
        "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__,
        "xgboost": xgboost.__version__, "rdkit": rdkit.__version__,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "physical_gpu_id": os.environ.get("EXCITATIONNEXUS_PHYSICAL_GPU_ID", ""),
    }


def monitor_gpu(stop: threading.Event, samples: list[int]) -> None:
    pid = str(os.getpid())
    while not stop.wait(0.1):
        try:
            output = subprocess.check_output(
                ["nvidia-smi", "--query-compute-apps=pid,used_gpu_memory", "--format=csv,noheader,nounits"],
                text=True, stderr=subprocess.DEVNULL,
            )
            for line in output.splitlines():
                fields = [field.strip() for field in line.split(",")]
                if len(fields) == 2 and fields[0] == pid:
                    samples.append(int(fields[1]))
        except (OSError, subprocess.SubprocessError, ValueError):
            pass


def metric_dict(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float | int]:
    return {
        "mae": float(mean_absolute_error(truth, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(truth, prediction))),
        "r2": float(r2_score(truth, prediction)),
        "n_records": int(len(truth)),
    }


def cpu_smoke(config: dict) -> None:
    all_sids, split, y, frame, columns = load_inputs(config)
    idx = indices(split, all_sids)
    transformed, _, _ = preprocess(frame, idx)
    smoke = XGBRegressor(
        n_estimators=2, max_depth=2, learning_rate=0.05, tree_method="hist",
        device="cpu", random_state=42, verbosity=0,
    )
    smoke.fit(transformed["train"][:64], y[idx["train"]][:64])
    prediction = smoke.predict(transformed["val"][:8])
    if prediction.shape != (8,) or not np.isfinite(prediction).all():
        raise RuntimeError("CPU prediction smoke failed")
    print(json.dumps({
        "status": "CPU_SMOKE_PASS", "records": len(all_sids), "features": len(columns),
        "prediction_count": len(prediction), "test_used": False, "final673_accessed": False,
    }, sort_keys=True))


def reproduce(config: dict, output_dir: Path) -> None:
    started = time.perf_counter()
    all_sids, split, y, frame, columns = load_inputs(config)
    idx = indices(split, all_sids)
    transformed, imputer, scaler = preprocess(frame, idx)
    params = dict(config["model"])

    samples: list[int] = []
    stop = threading.Event()
    monitor = threading.Thread(target=monitor_gpu, args=(stop, samples), daemon=True)
    monitor.start()
    try:
        model = XGBRegressor(**params)
        model.fit(transformed["train"], y[idx["train"]])
        predictions = {name: model.predict(transformed[name]) for name in ("train", "val", "test")}
    finally:
        stop.set()
        monitor.join(timeout=2)

    if any(not np.isfinite(prediction).all() for prediction in predictions.values()):
        raise RuntimeError("non-finite reproduction prediction")
    output_dir.mkdir(parents=True, exist_ok=False)
    test_ids = [all_sids[index] for index in idx["test"]]
    prediction_frame = pd.DataFrame({
        "molecule_id": test_ids,
        "true_eb_eV": y[idx["test"]],
        "pred_eb_eV": predictions["test"],
    })
    prediction_path = output_dir / "gate1a1_test_predictions.csv"
    prediction_frame.to_csv(prediction_path, index=False)

    historical = pd.read_csv(config["historical_assets"]["historical_predictions"])
    aligned = historical.set_index("molecule_id").loc[test_ids]
    historical_prediction = aligned["pred_eb_eV"].to_numpy(dtype=np.float64)
    max_abs_prediction_delta = float(np.max(np.abs(predictions["test"] - historical_prediction)))
    serialized = pd.read_csv(prediction_path)
    exact_vector_match = bool(
        serialized["molecule_id"].tolist() == historical["molecule_id"].tolist()
        and np.array_equal(serialized["true_eb_eV"].to_numpy(), historical["true_eb_eV"].to_numpy())
        and np.array_equal(serialized["pred_eb_eV"].to_numpy(), historical["pred_eb_eV"].to_numpy())
        and sha256(prediction_path) == sha256(Path(config["historical_assets"]["historical_predictions"]))
    )

    metrics = {
        name: metric_dict(y[idx[name]], predictions[name]) for name in ("train", "val", "test")
    }
    metrics.update({
        "historical_test_mae": config["historical_result"]["test_mae"],
        "absolute_mae_delta": abs(metrics["test"]["mae"] - config["historical_result"]["test_mae"]),
        "old_new_max_abs_prediction_delta": max_abs_prediction_delta,
        "old_new_exact_prediction_vector_match": exact_vector_match,
        "old_prediction_sha256": sha256(Path(config["historical_assets"]["historical_predictions"])),
        "new_prediction_sha256": sha256(prediction_path),
        "feature_count": len(columns),
        "feature_columns_sha256": hashlib.sha256(("\n".join(columns) + "\n").encode()).hexdigest(),
        "imputer_fit_scope": "train_only_5118",
        "scaler_fit_scope": "train_only_5118",
        "test_used_for_fit_or_early_stopping": False,
        "historical_test_selection_limitation": True,
        "peak_process_gpu_memory_mib_observed": max(samples) if samples else None,
        "wall_seconds": time.perf_counter() - started,
        "checkpoint_saved": False,
        "final673_accessed": False,
    })
    status = "REPRODUCED_STRICT" if exact_vector_match else (
        "REPRODUCED_NUMERIC" if metrics["absolute_mae_delta"] <= 0.001 else "FAILED_REPRODUCTION"
    )
    metrics["status"] = status
    (output_dir / "gate1a1_metrics.json").write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    (output_dir / "gate1a1_environment.json").write_text(json.dumps(environment(), indent=2, sort_keys=True) + "\n")
    (output_dir / "gate1a1_feature_columns.json").write_text(json.dumps(columns, indent=2) + "\n")
    evidence = {
        "status": status,
        "metrics": metrics,
        "model_parameters": params,
        "split_counts": {name: len(split[name]) for name in ("train", "val", "test")},
        "environment": environment(),
        "no_model_selection_or_hyperparameter_search": True,
        "formal_models_trained": 1,
    }
    (output_dir / "gate1a1_run_evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(json.dumps(evidence, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--cpu-smoke", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if args.cpu_smoke:
        cpu_smoke(config)
        return
    if args.output_dir is None:
        parser.error("--output-dir is required for reproduction")
    reproduce(config, args.output_dir)


if __name__ == "__main__":
    main()
