from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def group_macro_mae(frame: pd.DataFrame, prediction: str, truth: str = "primary_true") -> float:
    errors = (frame[prediction] - frame[truth]).abs()
    return float(errors.groupby(frame.structure_group_id_v1).mean().mean())


def group_bootstrap_error_difference(
    frame: pd.DataFrame,
    first: str,
    second: str,
    *,
    truth: str = "primary_true",
    iterations: int = 10000,
    seed: int = 20260720,
) -> dict:
    grouped = pd.DataFrame({
        "group": frame.structure_group_id_v1,
        "difference": (frame[first] - frame[truth]).abs() - (frame[second] - frame[truth]).abs(),
    }).groupby("group", sort=True).difference.mean().to_numpy(np.float64)
    if not len(grouped):
        return {"groups": 0, "point_difference_eV": None, "ci95_eV": [None, None]}
    rng = np.random.default_rng(seed)
    draws = np.empty(iterations, dtype=np.float64)
    for start in range(0, iterations, 1000):
        stop = min(start + 1000, iterations)
        indices = rng.integers(0, len(grouped), size=(stop - start, len(grouped)))
        draws[start:stop] = grouped[indices].mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return {
        "groups": int(len(grouped)),
        "records": int(len(frame)),
        "point_difference_eV": float(grouped.mean()),
        "ci95_eV": [float(low), float(high)],
        "ci_excludes_zero": bool(low > 0 or high < 0),
        "negative_favors_first": True,
        "iterations": iterations,
        "seed": seed,
    }


def deterministic_similarity_merge(
    frame: pd.DataFrame,
    label_column: str,
    ordered_labels: list[str],
    *,
    minimum_records: int,
    minimum_groups: int,
) -> tuple[pd.Series, list[dict]]:
    blocks: list[list[str]] = []
    current: list[str] = []
    for label in ordered_labels:
        current.append(label)
        selected = frame[label_column].isin(current)
        if int(selected.sum()) >= minimum_records and int(frame.loc[selected, "structure_group_id_v1"].nunique()) >= minimum_groups:
            blocks.append(current)
            current = []
    if current:
        if blocks:
            blocks[-1].extend(current)
        else:
            blocks.append(current)
    mapping: dict[str, str] = {}
    metadata = []
    for block in blocks:
        merged = "+".join(block)
        for item in block:
            mapping[item] = merged
        selected = frame[label_column].isin(block)
        metadata.append({"label": merged, "source_bins": block, "records": int(selected.sum()),
                         "groups": int(frame.loc[selected, "structure_group_id_v1"].nunique())})
    return frame[label_column].map(mapping), metadata


def fixed_rotation(dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    # Product of fixed 30-degree z and 20-degree y rotations.
    a, b = np.deg2rad(30.0), np.deg2rad(20.0)
    rz = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
    return torch.tensor(rz @ ry, dtype=dtype, device=device)


def perturb_positions(batch, condition: str, seed: int = 20260720) -> tuple[torch.Tensor, int]:
    pos = batch.pos.clone()
    skipped = 0
    if condition == "original":
        return pos, skipped
    if condition == "zero_coordinates":
        return torch.zeros_like(pos), skipped
    if condition == "global_rotation_translation":
        rotation = fixed_rotation(pos.dtype, pos.device)
        translation = pos.new_tensor([1.25, -0.75, 0.50])
        return pos @ rotation.T + translation, skipped
    if condition.startswith("gaussian_noise_"):
        sigma = float(condition.split("_")[-1].removesuffix("A"))
        for graph_index, molecule_id in enumerate(batch.molecule_id):
            mask = batch.batch.eq(graph_index)
            digest = hashlib.sha256(f"{seed}:{condition}:{molecule_id}".encode()).digest()
            generator = torch.Generator(device="cpu").manual_seed(int.from_bytes(digest[:8], "big") % (2**63 - 1))
            noise = torch.randn((int(mask.sum()), 3), generator=generator, dtype=pos.dtype).to(pos.device) * sigma
            pos[mask] += noise
        return pos, skipped
    if condition.startswith("DA_separation_"):
        distance = float(condition.split("_")[-1].removesuffix("A"))
        for graph_index in range(int(batch.num_graphs)):
            graph = batch.batch.eq(graph_index)
            donor, acceptor = graph & batch.role.eq(0), graph & batch.role.eq(1)
            if not donor.any() or not acceptor.any():
                skipped += 1
                continue
            axis = pos[acceptor].mean(0) - pos[donor].mean(0)
            norm = axis.norm()
            if float(norm) < 1e-8:
                skipped += 1
                continue
            unit = axis / norm
            pos[donor] -= 0.5 * distance * unit
            pos[acceptor] += 0.5 * distance * unit
        return pos, skipped
    raise ValueError(f"unknown counterfactual condition: {condition}")
