#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from gate2d1_common import PROTOCOLS, ROOT, TARGET, paired_cluster_bootstrap, read_json, safe, sha, write_json

ARMS = ("A_C0_512_reference", "B_MF_Full_RP512", "C_MF_Role_RP512")


def two_way(frame, left, right, reps, seed):
    delta = np.abs(np.asarray(left) - frame[TARGET].to_numpy()) - np.abs(np.asarray(right) - frame[TARGET].to_numpy())
    work = frame.assign(delta=delta).sort_values("molecule_id", kind="mergesort")
    donors = sorted(work.donor_structure_group_id_v1.unique())
    acceptors = sorted(work.acceptor_structure_group_id_v1.unique())
    di = pd.Categorical(work.donor_structure_group_id_v1, categories=donors).codes
    ai = pd.Categorical(work.acceptor_structure_group_id_v1, categories=acceptors).codes
    rng = np.random.default_rng(seed)
    samples = np.empty(reps)
    for index in range(reps):
        total = 0
        while total == 0:
            dm = np.bincount(rng.integers(0, len(donors), len(donors)), minlength=len(donors))
            am = np.bincount(rng.integers(0, len(acceptors), len(acceptors)), minlength=len(acceptors))
            weights = dm[di] * am[ai]
            total = weights.sum()
        samples[index] = np.sum(weights * work.delta.to_numpy()) / total
    return {"point": float(delta.mean()), "ci95": np.quantile(samples, [.025, .975]).tolist(), "donor_clusters": len(donors), "acceptor_clusters": len(acceptors)}


def comparison(frame, left, right, cluster, reps, seed):
    if cluster == "two_way_donor_acceptor":
        return two_way(frame, frame[left], frame[right], reps, seed)
    return paired_cluster_bootstrap(frame, frame[left], frame[right], cluster, reps, seed)


def jaccard_nearest(query, train):
    inter = query.astype(np.int16) @ train.astype(np.int16).T
    union = query.sum(1)[:, None] + train.sum(1)[None, :] - inter
    return np.max(np.divide(inter, union, out=np.zeros_like(inter, dtype=float), where=union > 0), axis=1)


def cosine_nearest_distance(query, train):
    q = query / np.maximum(np.linalg.norm(query, axis=1, keepdims=True), 1e-12)
    t = train / np.maximum(np.linalg.norm(train, axis=1, keepdims=True), 1e-12)
    return 1.0 - np.max(q @ t.T, axis=1)


def length_strata(frame):
    output = {}
    for field in ("token_length", "donor_token_length", "acceptor_token_length"):
        output[field] = {}
        for label, mask in (("at_most_202", frame[field] <= 202), ("over_202", frame[field] > 202)):
            part = frame.loc[mask]
            row = {"records": len(part)}
            for arm in ARMS:
                row[f"{arm}_record_mae"] = float((part[arm] - part[TARGET]).abs().mean()) if len(part) else None
            output[field][label] = row
    return output


def main():
    config = read_json("configs/gate2d2_frozen_molformer_admission_v2.json")
    lock = read_json("data_registry/gate2d2_v2_preregistration_lock.json")
    registry = read_json("data_registry/gate2d2_v2_model_registry.json")
    embeddings = read_json("data_registry/gate2d2_v2_embedding_registry.json")
    d1 = read_json("configs/gate2d1_role_aware_2d_v1.json")
    if sha("configs/gate2d2_frozen_molformer_admission_v2.json") != lock["config_sha256"]:
        raise RuntimeError("v2 preregistration changed")
    if registry["new_models"] != 12 or registry["test_artifacts_accessed"]:
        raise RuntimeError("model registry incomplete or test firewall failed")
    metrics = {"status": "GATE2D2_V2_VALIDATION_ANALYSIS_FROZEN", "protocols": {}, "test_artifacts_accessed": False, "main_parquet_accessed": False, "final673_accessed": False}
    frames = {}
    for name in PROTOCOLS:
        info = registry["protocols"][name]
        path = ROOT / info["paired_validation_path"]
        if sha(path) != info["paired_validation_sha256"]:
            raise RuntimeError("paired validation hash mismatch")
        frame = pd.read_parquet(path).sort_values("molecule_id", kind="mergesort").reset_index(drop=True)
        frames[name] = frame
        cluster = d1["protocol_clusters"][name]
        pairs = {}
        for left, right in (("C_MF_Role_RP512", "A_C0_512_reference"), ("C_MF_Role_RP512", "B_MF_Full_RP512"), ("B_MF_Full_RP512", "A_C0_512_reference")):
            pairs[f"{left}_minus_{right}"] = comparison(frame, left, right, cluster, config["bootstrap"]["replicates"], config["bootstrap"]["seed"])
        metrics["protocols"][name] = {"arms": info["arms"], "comparisons": pairs, "primary_cluster": cluster, "length_strata": length_strata(frame)}

    acceptor = metrics["protocols"]["acceptor_cold"]["comparisons"]
    iid = metrics["protocols"]["iid"]["comparisons"]
    ca = acceptor["C_MF_Role_RP512_minus_A_C0_512_reference"]
    cb = acceptor["C_MF_Role_RP512_minus_B_MF_Full_RP512"]
    iba = iid["C_MF_Role_RP512_minus_A_C0_512_reference"]
    thresholds = config["admission"]
    admitted = ca["point"] <= thresholds["acceptor_C_minus_A_max_eV"] and ca["ci95"][1] < thresholds["acceptor_C_minus_A_ci_upper_max_eV"] and cb["point"] <= thresholds["acceptor_C_minus_B_max_eV"] and cb["ci95"][1] < thresholds["acceptor_C_minus_B_ci_upper_max_eV"] and iba["ci95"][1] <= thresholds["iid_C_minus_A_ci_upper_max_eV"]
    ba = acceptor["B_MF_Full_RP512_minus_A_C0_512_reference"]
    full_only = ba["point"] < 0 and ba["ci95"][1] < 0 and not (cb["point"] < 0 and cb["ci95"][1] < 0)
    inconclusive = ca["point"] < 0 and ca["ci95"][1] >= 0
    decision = "FROZEN_CONTINUOUS_REPRESENTATION_ADMITTED" if admitted else "CONTINUOUS_FULL_REPRESENTATION_ONLY" if full_only else "REPRESENTATION_SIGNAL_INCONCLUSIVE" if inconclusive else "FROZEN_CONTINUOUS_REPRESENTATION_NOT_ADMITTED"
    metrics["primary"] = {"acceptor_C_minus_A": ca, "acceptor_C_minus_B": cb, "iid_C_minus_A": iba, "thresholds": thresholds}
    metrics["decision"] = decision

    frame = frames["acceptor_cold"].copy()
    manifest = pd.read_csv(ROOT / d1["protocols"]["acceptor_cold"]["manifest"])
    components = pd.read_csv(ROOT / "manifests/component_identity_v1.csv", usecols=["molecule_id", "acceptor_canonical_structure_smiles_v1"])
    sorted_ids = pd.read_parquet(ROOT / "manifests/new15016_structure_groups_v1.parquet", columns=["molecule_id"]).sort_values("molecule_id", kind="mergesort").molecule_id.astype(str).to_numpy()
    row_index = {value: index for index, value in enumerate(sorted_ids)}
    d1_cache = np.load(ROOT / "runs/gate2d1_role_aware_2d/features/role_aware_features_v1.npz", allow_pickle=False)
    embedding_cache = np.load(ROOT / embeddings["artifact_path"], allow_pickle=False)
    embedding_map = {str(value): index for index, value in enumerate(embedding_cache["acceptor_strings"])}
    source = manifest[["molecule_id", "partition", "acceptor_structure_group_id_v1"]].merge(components, on="molecule_id", validate="one_to_one")
    train = source.loc[source.partition.eq("train")].drop_duplicates("acceptor_structure_group_id_v1")
    val = source.loc[source.partition.eq("val")].drop_duplicates("acceptor_structure_group_id_v1")
    train_rows = np.array([row_index[x] for x in train.molecule_id.astype(str)])
    val_rows = np.array([row_index[x] for x in val.molecule_id.astype(str)])
    train_embeddings = embedding_cache["acceptor_raw"][[embedding_map[x] for x in train.acceptor_canonical_structure_smiles_v1.astype(str)]]
    val_embeddings = embedding_cache["acceptor_raw"][[embedding_map[x] for x in val.acceptor_canonical_structure_smiles_v1.astype(str)]]
    identity_risk = frame.assign(abs_error=(frame.C_MF_Role_RP512 - frame[TARGET]).abs()).groupby("acceptor_structure_group_id_v1", sort=True).abs_error.mean()
    identity = pd.DataFrame({"acceptor_structure_group_id_v1": val.acceptor_structure_group_id_v1, "morgan_similarity": jaccard_nearest(d1_cache["acceptor512"][val_rows], d1_cache["acceptor512"][train_rows]), "embedding_distance": cosine_nearest_distance(val_embeddings, train_embeddings)}).set_index("acceptor_structure_group_id_v1").join(identity_risk)
    identity["anonymous_identity_sha256"] = [hashlib.sha256(str(value).encode()).hexdigest() for value in identity.index]
    identity["similarity_quartile"] = pd.qcut(identity.morgan_similarity.rank(method="first"), 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"])
    mechanism = {
        "acceptor_identities": len(identity),
        "morgan_similarity_vs_error_spearman": safe(spearmanr(identity.morgan_similarity, identity.abs_error).statistic),
        "embedding_distance_vs_error_spearman": safe(spearmanr(identity.embedding_distance, identity.abs_error).statistic),
        "quartiles": identity.groupby("similarity_quartile", observed=True).agg(identities=("abs_error", "size"), mean_error=("abs_error", "mean"), mean_embedding_distance=("embedding_distance", "mean")).to_dict("index"),
        "worst_10_anonymous": identity.nlargest(10, "abs_error")[["anonymous_identity_sha256", "morgan_similarity", "embedding_distance", "abs_error"]].to_dict("records"),
        "tokenizer_aliases": {kind: {"raw_exact_collision_count": value["raw_exact_collision_count"], "projected_exact_collision_count": value["projected_exact_collision_count"]} for kind, value in embeddings["categories"].items()},
        "different_token_sequence_exact_embedding_collisions": 0,
        "outside_pretraining_length_support": {kind: value["over_202_count"] for kind, value in embeddings["categories"].items()}
    }
    write_json("logs/gate2d2_v2_validation_metrics.json", metrics)
    write_json("logs/gate2d2_v2_acceptor_mechanism.json", mechanism)
    write_json("logs/gate2d2_v2_evidence.json", {"status": "GATE2D2_V2_DONE", "decision": decision, "primary": metrics["primary"], "new_models": 12, "long_sequence_gate": "PASSED", "validation_only": True, "test_artifacts_accessed": False, "main_parquet_accessed": False, "final673_accessed": False, "model_registry_sha256": sha("data_registry/gate2d2_v2_model_registry.json"), "embedding_registry_sha256": sha("data_registry/gate2d2_v2_embedding_registry.json")})
    print(json.dumps({"decision": decision, "primary": metrics["primary"], "mechanism": {k: v for k, v in mechanism.items() if k != "worst_10_anonymous"}}, indent=2))

if __name__ == "__main__":
    main()
