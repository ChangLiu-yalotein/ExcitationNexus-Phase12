#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from excitationnexus_phase12.dft_graph_dataset import DFTGraphCache
from excitationnexus_phase12.edge_cache import ShardedEdgeCache, sha256_file
from excitationnexus_phase12.graph_builder import _directed_radius_graph

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")


def edge_worker(task: tuple[str, np.ndarray, float, int]) -> tuple[str, np.ndarray]:
    molecule_id, pos, cutoff, max_neighbors = task
    torch.set_num_threads(1)
    edge = _directed_radius_graph(torch.from_numpy(pos.copy()).float(), cutoff, max_neighbors)
    return molecule_id, edge.numpy().astype(np.int32, copy=False)


def stratified_sample(registry: pd.DataFrame, n: int) -> list[str]:
    frame = registry.copy()
    frame["atom_bin"] = pd.qcut(frame.num_atoms, 4, labels=False, duplicates="drop")
    frame["role_class"] = np.select(
        [frame.donor_atoms.eq(0), frame.unknown_atoms.gt(0)],
        ["empty_donor", "has_unknown"], default="pure_da")
    frame["sample_key"] = frame.molecule_id.map(lambda x: hashlib.sha256(str(x).encode()).hexdigest())
    selected: list[str] = []
    groups = list(frame.groupby(["partition", "role_class", "atom_bin"], sort=True))
    quota = max(1, n // max(1, len(groups)))
    for _, group in groups:
        selected.extend(group.sort_values("sample_key").molecule_id.astype(str).head(quota))
    if len(selected) < n:
        remaining = frame.loc[~frame.molecule_id.isin(selected)].sort_values("sample_key")
        selected.extend(remaining.molecule_id.astype(str).head(n - len(selected)))
    return sorted(selected[:n])


def build(args: argparse.Namespace) -> dict:
    config = json.loads(args.config.read_text())
    cache_root = ROOT / config["edge_cache"]["root"]
    registry_out = ROOT / config["edge_cache"]["registry"]
    if cache_root.exists() or registry_out.exists():
        raise RuntimeError("edge-cache output already exists; refusing to overwrite")
    source_cache = DFTGraphCache(ROOT / config["graph_cache"], ROOT / config["graph_registry"])
    registry = source_cache.registry.sort_values("cache_index").reset_index(drop=True)
    if len(registry) != 15016 or not registry.molecule_id.is_unique:
        raise RuntimeError("source graph registry coverage failure")
    cache_root.mkdir(parents=True)
    shard_size = int(config["edge_cache"]["shard_size"])
    workers = int(args.workers)
    records: dict[str, dict] = {}
    shards: list[dict] = []
    started = time.perf_counter()
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers) as pool:
        for shard_number, row_start in enumerate(range(0, len(registry), shard_size)):
            rows = registry.iloc[row_start:row_start + shard_size]
            tasks = []
            for row in rows.itertuples(index=False):
                a, b = int(row.atom_offset_start), int(row.atom_offset_end)
                tasks.append((str(row.molecule_id), source_cache.pos[a:b],
                              float(config["model_common"]["cutoff_angstrom"]),
                              int(config["model_common"]["max_neighbors"])))
            results = pool.map(edge_worker, tasks, chunksize=max(1, len(tasks) // (workers * 4)))
            src_parts, dst_parts, offsets = [], [], [0]
            for (molecule_id, edge), row in zip(results, rows.itertuples(index=False)):
                if molecule_id != str(row.molecule_id):
                    raise RuntimeError("worker result order mismatch")
                src_parts.append(edge[0]); dst_parts.append(edge[1]); offsets.append(offsets[-1] + edge.shape[1])
            name = f"radius_edges_{shard_number:04d}.npz"
            path = cache_root / name
            np.savez_compressed(
                path,
                molecule_ids=rows.molecule_id.astype(str).to_numpy(dtype="U"),
                atom_counts=rows.num_atoms.to_numpy(dtype=np.int32),
                edge_offsets=np.asarray(offsets, dtype=np.int64),
                edge_src=np.concatenate(src_parts).astype(np.int32, copy=False),
                edge_dst=np.concatenate(dst_parts).astype(np.int32, copy=False),
                graph_hashes=rows.graph_content_sha256.astype(str).to_numpy(dtype="U64"),
            )
            shard_hash = sha256_file(path)
            for local_index, row in enumerate(rows.itertuples(index=False)):
                records[str(row.molecule_id)] = {
                    "shard": name, "row": local_index,
                    "edge_start": offsets[local_index], "edge_end": offsets[local_index + 1],
                    "atom_count": int(row.num_atoms), "source_graph_sha256": str(row.graph_content_sha256),
                }
            shards.append({"file": name, "sha256": shard_hash, "records": len(rows),
                           "edges": offsets[-1], "bytes": path.stat().st_size})
            print(json.dumps({"shard": name, "records": len(rows), "edges": offsets[-1]}, sort_keys=True), flush=True)
    aggregate_text = "".join(f"{x['file']}:{x['sha256']}\n" for x in shards)
    aggregate = hashlib.sha256(aggregate_text.encode()).hexdigest()
    payload = {
        "schema_version": "gate1b3_radius_edges_v1", "status": "TARGET_FREE_READ_ONLY",
        "cache_root": str(cache_root.relative_to(ROOT)), "source_graph_cache": config["graph_cache"],
        "source_graph_registry": config["graph_registry"],
        "source_graph_registry_sha256": sha256_file(ROOT / config["graph_registry"]),
        "cutoff_angstrom": config["model_common"]["cutoff_angstrom"],
        "max_neighbors": config["model_common"]["max_neighbors"],
        "atom_order": "Gate1B2 cache_index and frozen DFT JSON atom index",
        "distance_reconstruction": "Euclidean distance from frozen float32 DFT coordinates",
        "record_count": len(records), "missing_ids": 0, "duplicate_ids": 0,
        "aggregate_sha256": aggregate, "shards": shards, "records": records,
        "build_wall_seconds": time.perf_counter() - started,
        "build_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "target_columns_read": [],
    }
    registry_out.parent.mkdir(parents=True, exist_ok=True)
    registry_out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return payload


def verify_and_benchmark(args: argparse.Namespace) -> dict:
    # Freeze the same single-thread cdist path used by every cache worker.
    torch.set_num_threads(1)
    config = json.loads(args.config.read_text())
    source = DFTGraphCache(ROOT / config["graph_cache"], ROOT / config["graph_registry"])
    cached = ShardedEdgeCache(ROOT / config["edge_cache"]["registry"], verify_hashes=True)
    sample_ids = stratified_sample(source.registry, int(config["edge_cache"]["verification_records"]))
    by_id = source.registry.set_index("molecule_id")
    dynamic: dict[str, torch.Tensor] = {}
    started = time.perf_counter()
    for molecule_id in sample_ids:
        row = by_id.loc[molecule_id]
        a, b = int(row.atom_offset_start), int(row.atom_offset_end)
        dynamic[molecule_id] = _directed_radius_graph(
            torch.from_numpy(source.pos[a:b].copy()).float(),
            config["model_common"]["cutoff_angstrom"], config["model_common"]["max_neighbors"])
    dynamic_seconds = time.perf_counter() - started
    started = time.perf_counter(); mismatch = atom_mismatch = 0; total_edges = 0
    for molecule_id in sample_ids:
        edge = cached.edge_index(molecule_id); total_edges += edge.shape[1]
        mismatch += int(not torch.equal(edge, dynamic[molecule_id]))
        atom_mismatch += int(int(edge.max()) >= int(by_id.loc[molecule_id].num_atoms))
    cached_seconds = time.perf_counter() - started
    inventory = cached.registry["shards"]
    result = {
        "verified_records": len(sample_ids), "sampling": "partition/role/atom-count stratified; SHA-256 tie-break",
        "edge_mismatch": mismatch, "atom_order_mismatch": atom_mismatch,
        "dynamic_seconds": dynamic_seconds, "cached_seconds": cached_seconds,
        "dynamic_graphs_per_second": len(sample_ids) / dynamic_seconds,
        "cached_graphs_per_second": len(sample_ids) / cached_seconds,
        "speedup": dynamic_seconds / cached_seconds, "verified_edges": total_edges,
        "cache_disk_bytes": sum(x["bytes"] for x in inventory),
        "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "gpu_waiting_time_seconds": 0.0,
        "torch_cdist_threads": 1,
    }
    if mismatch or atom_mismatch or len(cached.records) != 15016:
        raise RuntimeError(f"edge-cache verification failed: {result}")
    report = ROOT / "reports/gate1b3_edge_cache_benchmark.md"
    report.write_text(
        "# Gate 1-B3 target-free radius edge cache\n\n"
        f"Coverage is **{len(cached.records):,}/15,016** with zero missing/duplicate IDs. "
        f"A deterministic stratified sample of **{len(sample_ids)}** records had **{mismatch}** edge and "
        f"**{atom_mismatch}** atom-order mismatches. Dynamic construction processed "
        f"{result['dynamic_graphs_per_second']:.2f} graphs/s; cached lookup processed "
        f"{result['cached_graphs_per_second']:.2f} graphs/s ({result['speedup']:.2f}x). "
        f"The {len(inventory)} local shards occupy {result['cache_disk_bytes']/2**20:.1f} MiB. "
        "The cache contains only identities, atom counts, radius edges, and source graph hashes; distances "
        "are reconstructed from frozen DFT coordinates. No target scalar was read. GPU waiting time during "
        "this CPU benchmark was 0 s.\n"
    )
    (ROOT / "logs/gate1b3_edge_cache_benchmark.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.build:
        payload = build(args)
        print(json.dumps({k: payload[k] for k in ("record_count", "aggregate_sha256", "build_wall_seconds")}, indent=2))
    if args.verify:
        print(json.dumps(verify_and_benchmark(args), indent=2))


if __name__ == "__main__":
    main()
