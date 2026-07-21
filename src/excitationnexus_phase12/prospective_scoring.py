from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator

DESC_NAMES = [
    "MolWt", "MolLogP", "MolMR", "TPSA", "NumHDonors", "NumHAcceptors",
    "NumRotatableBonds", "NumAromaticRings", "NumAliphaticRings",
    "NumAromaticHeterocycles", "NumAliphaticHeterocycles", "NumSaturatedRings",
    "NumHeteroatoms", "HeavyAtomCount", "NumValenceElectrons", "NHOHCount",
    "NOCount", "FractionCSP3", "RingCount", "HallKierAlpha",
]
C0_COLUMNS = [f"pair_{name}" for name in DESC_NAMES] + [f"pair_morgan_{i}" for i in range(512)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ordered_hash(values: list[str]) -> str:
    return hashlib.sha256(("\n".join(values) + "\n").encode()).hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def assert_c0_contract(columns: list[str]) -> None:
    if columns != C0_COLUMNS or len(columns) != 532:
        raise RuntimeError("C0 feature order/count mismatch")
    forbidden = ("tddft", "multiwfn", "coulomb", "wavelength", "dipole", "partition", "split", "final673")
    if any(any(token in column.lower() for token in forbidden) for column in columns):
        raise RuntimeError("forbidden field entered C0")


def c0_matrix(smiles: list[str]) -> np.ndarray:
    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=512, includeChirality=False)
    matrix = np.empty((len(smiles), len(C0_COLUMNS)), dtype=np.float32)
    for i, text in enumerate(smiles):
        mol = Chem.MolFromSmiles(str(text))
        if mol is None:
            raise RuntimeError(f"candidate sanitize failure at row {i}")
        matrix[i, :20] = [float(getattr(Descriptors, name)(mol)) for name in DESC_NAMES]
        bits = np.zeros(512, dtype=np.int8)
        DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), bits)
        matrix[i, 20:] = bits
    if matrix.shape[1] != 532 or not np.isfinite(matrix).all():
        raise RuntimeError("candidate feature integrity failure")
    return matrix


def weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        raise RuntimeError("no valid value for weighted median")
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


def load_config(root: Path) -> dict:
    return json.loads((root / "configs/gate3a1_prospective_scoring_v1.json").read_text())


def verify_inputs(root: Path, config: dict) -> None:
    for item in config["inputs"].values():
        path = root / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"frozen input hash mismatch: {item['path']}")
    for item in config["label_sources"]:
        path = root / item["path"]
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"label artifact hash mismatch: {item['path']}")


def load_deployment_labels(root: Path, config: dict, manifest: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    target = config["primary_target"]
    blocks = []
    for source_index, item in enumerate(config["label_sources"]):
        block = pd.read_parquet(root / item["path"], columns=["molecule_id", item["target_column"]])
        block = block.rename(columns={item["target_column"]: target})
        block["molecule_id"] = block["molecule_id"].astype(str)
        block["_source"] = source_index
        blocks.append(block)
    all_labels = pd.concat(blocks, ignore_index=True)
    if not np.isfinite(all_labels[target].to_numpy(np.float64)).all():
        raise RuntimeError("non-finite frozen label")
    spread = all_labels.groupby("molecule_id")[target].agg(lambda x: float(x.max() - x.min()))
    if float(spread.max()) > 1e-12:
        raise RuntimeError("inconsistent frozen primary labels")
    labels = all_labels.sort_values(["molecule_id", "_source"], kind="mergesort").drop_duplicates("molecule_id")
    labels = labels[["molecule_id", target]]
    eligible = manifest.loc[manifest["partition"].ne("historical_quarantine"), ["molecule_id"]].copy()
    eligible["molecule_id"] = eligible["molecule_id"].astype(str)
    joined = eligible.merge(labels, on="molecule_id", how="left", validate="one_to_one")
    missing = int(joined[target].isna().sum())
    extra = int(len(set(labels.molecule_id) - set(eligible.molecule_id)))
    if len(joined) != 15015 or missing or extra:
        raise RuntimeError(f"deployment label coverage failure: missing={missing}, extra={extra}")
    evidence = {
        "eligible": 15015, "covered": len(joined), "missing": missing, "extra": extra,
        "source_count": len(blocks), "cross_source_max_abs_difference": float(spread.max()),
        "main_parquet_read": False, "test_artifact_read": False,
    }
    return joined, evidence


def save_preprocessor(path: Path, prep: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, columns=np.asarray(C0_COLUMNS), **prep)


def load_preprocessor(path: Path) -> dict[str, np.ndarray]:
    payload = np.load(path, allow_pickle=False)
    if payload["columns"].astype(str).tolist() != C0_COLUMNS:
        raise RuntimeError("preprocessor column mismatch")
    return {key: payload[key] for key in ("medians", "means", "scales")}


def deterministic_rank_percentile(prediction: np.ndarray, hashes: np.ndarray) -> np.ndarray:
    order = np.lexsort((hashes.astype(str), np.asarray(prediction)))
    rank = np.empty(len(order), dtype=np.float64)
    rank[order] = np.arange(len(order), dtype=np.float64)
    return rank / max(len(order) - 1, 1)


def identity_cap_ok(row: pd.Series, donor_counts: dict, acceptor_counts: dict, maximum: int = 2) -> bool:
    return donor_counts.get(row.donor_identity_hash, 0) < maximum and acceptor_counts.get(row.acceptor_identity_hash, 0) < maximum
