#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import random
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.dataset as ds
import torch
from torch.utils.data import DataLoader

from excitationnexus_phase12.collate import collate_phase12
from excitationnexus_phase12.dft_graph_dataset import DFTGraphCache, Gate1B2GraphDataset
from excitationnexus_phase12.losses import weighted_masked_multitask_loss
from excitationnexus_phase12.metrics import regression_metrics
from excitationnexus_phase12.models import M3DAUSharedModel, M3MergedModel
from excitationnexus_phase12.normalization import weighted_stats

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True); path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""): digest.update(chunk)
    return digest.hexdigest()


def resolve(path: str) -> Path:
    candidate = Path(path); return candidate if candidate.is_absolute() else ROOT / candidate


def target_blind_key(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def model_kwargs(config: dict) -> dict:
    common = config["model_common"]
    return {"hidden_dim": common["hidden_dim"], "num_rbf": common["num_rbf"],
            "layers": common["layers"], "cutoff": common["cutoff_angstrom"]}


def prepare(config_path: Path, output: Path) -> dict:
    if output.exists():
        raise RuntimeError("prepared output already exists")
    config = json.loads(config_path.read_text()); manifest_path = resolve(config["manifest"])
    if sha256(manifest_path) != config["manifest_sha256"] or sha256(Path(config["table"])) != config["table_sha256"]:
        raise RuntimeError("frozen input hash mismatch")
    task_graph = json.loads((ROOT / "data_registry/TARGET_TASK_GRAPH_V1.json").read_text())
    if task_graph["primary"] != [config["primary_target_from_contract"]]:
        raise RuntimeError("primary target contract mismatch")
    manifest = pd.read_csv(manifest_path)
    allowed = manifest.loc[manifest.partition.isin(["train", "val"]), "molecule_id"].astype(str).tolist()
    dataset = ds.dataset(config["table"], format="parquet")
    target = dataset.to_table(
        columns=["molecule_id", config["primary_target_from_contract"]],
        filter=ds.field("molecule_id").isin(allowed),
    ).to_pandas().rename(columns={config["primary_target_from_contract"]: "target"})
    if len(target) != 12696 or target.target.isna().any():
        raise RuntimeError("train/val-only target read mismatch")
    joined = manifest[manifest.partition.isin(["train", "val"])].merge(target, on="molecule_id", validate="one_to_one")
    train = joined[joined.partition.eq("train")]
    stats = weighted_stats(train.target, train.group_weight)
    train_ids = sorted(train.molecule_id.astype(str), key=target_blind_key)[:config["smoke_train_records"]]
    val_ids = sorted(joined.loc[joined.partition.eq("val"), "molecule_id"].astype(str), key=target_blind_key)[:config["smoke_val_records"]]
    output.mkdir(parents=True)
    target.to_parquet(output / "train_val_targets.parquet", index=False)
    write_json(output / "normalization.json", {
        "scope": "full partition=train only", "weighting": "group_weight", "target": stats,
        "manifest_sha256": config["manifest_sha256"], "table_sha256": config["table_sha256"],
        "test_target_accessed": False,
    })
    write_json(output / "subset_ids.json", {"rule": config["subset_rule"], "train": train_ids, "val": val_ids})
    merged = M3MergedModel(**model_kwargs(config), head_width=config["models"]["m3_merged"]["head_width"])
    dau = M3DAUSharedModel(**model_kwargs(config), head_width=config["models"]["m3_dau_shared"]["head_width"])
    params = {"m3_merged": sum(p.numel() for p in merged.parameters()), "m3_dau_shared": sum(p.numel() for p in dau.parameters())}
    relative = abs(params["m3_merged"] - params["m3_dau_shared"]) / max(params.values())
    if relative > 0.05:
        raise RuntimeError("parameter fairness exceeds 5 percent")
    fairness = {
        "same_backbone_family": True, "same_hidden_cutoff_layers_optimizer_loss": True,
        "parameters": params, "relative_parameter_difference": relative,
        "backbone_invocations": {"m3_merged": 1, "m3_dau_shared": 3},
        "compute_note": "DAU uses three explicit shared-weight subgraph invocations; measured runtime is reported after smoke.",
        "test_target_accessed": False,
    }
    write_json(output / "fairness_gate.json", fairness)
    (ROOT / "reports/gate1b2_model_fairness.md").write_text(
        "# Gate 1-B2 model fairness\n\n"
        f"M3-Merged and M3-DAU-Shared use one identical `{config['model_common']}` backbone family and the same optimizer/loss. "
        f"Parameter counts are {params['m3_merged']:,} and {params['m3_dau_shared']:,}; relative difference is {relative * 100:.3f}%, within 5%. "
        "DAU has one shared parameter set but invokes it on donor, acceptor, and unknown subgraphs separately; empty roles use zero vectors and presence=0. Runtime/throughput, rather than parameter count alone, records its compute overhead.\n"
    )
    return {"status": "PREPARED_TRAIN_VAL_ONLY", "target_stats": stats, "fairness": fairness,
            "train_subset": len(train_ids), "val_subset": len(val_ids)}


def seed_everything(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


def build_model(config: dict, name: str):
    if name == "m3_merged":
        return M3MergedModel(**model_kwargs(config), head_width=config["models"][name]["head_width"])
    return M3DAUSharedModel(**model_kwargs(config), head_width=config["models"][name]["head_width"])


def invariance(model, batch) -> dict:
    model.eval()
    with torch.no_grad():
        base = model(batch)
        shifted = batch.clone(); shifted.pos = batch.pos + torch.tensor([1.2, -0.7, 2.3], device=batch.pos.device)
        angle = torch.tensor(0.731, device=batch.pos.device)
        zero, one = torch.tensor(0.0, device=batch.pos.device), torch.tensor(1.0, device=batch.pos.device)
        rotation = torch.stack([torch.stack([torch.cos(angle), -torch.sin(angle), zero]),
                                torch.stack([torch.sin(angle), torch.cos(angle), zero]),
                                torch.stack([zero, zero, one])])
        rotated = batch.clone(); rotated.pos = batch.pos @ rotation.T
        translation_delta = float((model(shifted) - base).abs().max())
        rotation_delta = float((model(rotated) - base).abs().max())
    if translation_delta > 2e-5 or rotation_delta > 2e-5:
        raise RuntimeError("GPU invariance failed")
    return {"translation_max_abs_delta": translation_delta, "rotation_max_abs_delta": rotation_delta}


def run_model(config_path: Path, prepared: Path, output: Path, model_name: str, physical_gpu: int) -> dict:
    if output.exists(): raise RuntimeError("model smoke output already exists")
    config = json.loads(config_path.read_text()); seed_everything(config["seed"])
    manifest = pd.read_csv(resolve(config["manifest"])); targets = pd.read_parquet(prepared / "train_val_targets.parquet")
    subset = json.loads((prepared / "subset_ids.json").read_text()); stats = json.loads((prepared / "normalization.json").read_text())["target"]
    cache = DFTGraphCache(resolve(config["graph_cache"]), resolve(config["graph_registry"]))
    train = Gate1B2GraphDataset(cache, manifest, partition="train", targets=targets,
                               cutoff=config["model_common"]["cutoff_angstrom"], max_neighbors=config["model_common"]["max_neighbors"])
    val = Gate1B2GraphDataset(cache, manifest, partition="val", targets=targets,
                             cutoff=config["model_common"]["cutoff_angstrom"], max_neighbors=config["model_common"]["max_neighbors"])
    train.frame = train.frame[train.frame.molecule_id.isin(subset["train"])].sort_values("molecule_id").reset_index(drop=True)
    val.frame = val.frame[val.frame.molecule_id.isin(subset["val"])].sort_values("molecule_id").reset_index(drop=True)
    generator = torch.Generator().manual_seed(config["seed"])
    train_loader = DataLoader(train, batch_size=config["batch_size"], shuffle=True, generator=generator,
                              num_workers=0, collate_fn=collate_phase12)
    val_loader = DataLoader(val, batch_size=config["batch_size"], shuffle=False, num_workers=0, collate_fn=collate_phase12)
    device = torch.device("cuda:0"); model = build_model(config, model_name).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["optimizer"]["learning_rate"], weight_decay=config["optimizer"]["weight_decay"])
    torch.cuda.reset_peak_memory_stats(); output.mkdir(parents=True)
    first_batch = next(iter(train_loader)).to(device); invariant = invariance(model, first_batch)
    parameter_before = next(model.parameters()).detach().clone(); two_batch_losses = []
    model.train()
    for step, batch in enumerate(train_loader):
        if step >= 2: break
        batch = batch.to(device); normalized = (batch.target - stats["mean"]) / stats["std"]
        pred = model(batch)
        loss, _ = weighted_masked_multitask_loss({"primary": pred}, normalized[:, None],
            torch.ones((len(pred), 1), dtype=torch.bool, device=device), batch.group_weight,
            ["primary"], {"primary": 1.0}, base_loss="smooth_l1")
        optimizer.zero_grad(set_to_none=True); loss.backward()
        if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
            raise RuntimeError("non-finite loss or gradient")
        torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"]); optimizer.step()
        two_batch_losses.append(float(loss))
    parameter_changed = not torch.equal(parameter_before, next(model.parameters()).detach())
    if not parameter_changed: raise RuntimeError("optimizer did not change parameters")
    epochs, started = [], time.perf_counter()
    for epoch in range(1, config["max_epochs"] + 1):
        model.train(); losses = []
        for batch in train_loader:
            batch = batch.to(device); normalized = (batch.target - stats["mean"]) / stats["std"]
            pred = model(batch); loss, _ = weighted_masked_multitask_loss(
                {"primary": pred}, normalized[:, None], torch.ones((len(pred), 1), dtype=torch.bool, device=device),
                batch.group_weight, ["primary"], {"primary": 1.0}, base_loss="smooth_l1")
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if not torch.isfinite(loss): raise RuntimeError("non-finite train loss")
            torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"]); optimizer.step(); losses.append(float(loss))
        model.eval(); truth, prediction, groups = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                group_values = list(batch.structure_group_id_v1)
                batch = batch.to(device); raw = model(batch) * stats["std"] + stats["mean"]
                truth.extend(batch.target.detach().cpu().numpy()); prediction.extend(raw.detach().cpu().numpy()); groups.extend(group_values)
        metric = regression_metrics(truth, prediction, groups)
        epochs.append({"epoch": epoch, "train_loss": float(np.mean(losses)), "validation": metric})
    wall = time.perf_counter() - started
    checkpoint = output / "plumbing_only_state.pt"; torch.save(model.state_dict(), checkpoint)
    result = {
        "status": "GPU_ADMISSION_SMOKE_COMPLETE", "model": model_name, "physical_gpu": physical_gpu,
        "seed": config["seed"], "parameters": sum(p.numel() for p in model.parameters()),
        "two_batch_losses": two_batch_losses, "parameter_changed": parameter_changed, "invariance": invariant,
        "epochs": epochs, "wall_seconds_3epochs": wall, "train_records": len(train), "val_records": len(val),
        "graphs_per_second_train_approx": config["max_epochs"] * len(train) / wall,
        "peak_gpu_memory_mib": torch.cuda.max_memory_allocated() / 2 ** 20,
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "checkpoint_sha256": sha256(checkpoint), "checkpoint_scientific_status": "PLUMBING_ONLY",
        "test_target_accessed": False, "final673_accessed": False,
    }
    write_json(output / "result.json", result); print(json.dumps(result, indent=2)); return result


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--prepared", type=Path, required=True); parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--model", choices=["m3_merged", "m3_dau_shared"]); parser.add_argument("--output", type=Path)
    parser.add_argument("--physical-gpu", type=int)
    args = parser.parse_args()
    if args.prepare:
        print(json.dumps(prepare(args.config, args.prepared), indent=2)); return
    if args.model is None or args.output is None or args.physical_gpu is None: parser.error("model run requires --model, --output, --physical-gpu")
    run_model(args.config, args.prepared, args.output, args.model, args.physical_gpu)


if __name__ == "__main__": main()
