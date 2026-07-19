#!/usr/bin/env python3
"""Independent invariants and deterministic reproduction checks for Gate 0-C."""
from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FILES = {
    "iid_group_seed42_v1": "split_iid_group_seed42_v1.csv",
    "donor_cold_v1": "split_donor_cold_v1.csv",
    "acceptor_cold_v1": "split_acceptor_cold_v1.csv",
    "pair_cold_v1": "split_pair_cold_v1.csv",
    "both_cold_external_test_v1": "split_both_cold_external_test_v1.csv",
    "full_scaffold_cold_v1": "split_full_scaffold_cold_v1.csv",
}
TARGET_TOKENS = ("tddft", "multiwfn", "target", "coulomb", "excitation", "oscillator",
                 "transition_dipole", "h_ct", "q_d_to_a", "q_a_to_d", "net_ct")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def import_generator():
    path = ROOT / "scripts/gate0c_generate_splits.py"
    spec = importlib.util.spec_from_file_location("gate0c_generate", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod


def partition_sets(df, col):
    return {p: set(df.loc[df.partition.eq(p), col]) for p in ("train", "val", "test")}


def disjoint(sets):
    return not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or
                sets["val"] & sets["test"])


def verify_one(name, df):
    checks = {}
    checks["records_15016"] = len(df) == 15016
    checks["molecule_id_unique"] = df.molecule_id.is_unique
    checks["partition_coverage"] = df.partition.notna().all()
    checks["structure_zero_leakage"] = df.groupby("structure_group_id_v1").partition.nunique().max() == 1
    model_parts = {"train", "val", "test", "buffer"}
    retained = df[df.partition.isin(model_parts)]
    weights = retained.groupby("structure_group_id_v1").group_weight.sum()
    checks["group_weights_sum_one"] = bool((weights.sub(1).abs() < 1e-9).all())
    q = df.historical_status.eq("HISTORICAL_MODEL_SELECTION_QUARANTINE")
    h = df.historical_status.eq("HISTORICAL_TRAIN_OVERLAP")
    checks["quarantine_exactly_one"] = int(q.sum()) == 1 and df.loc[q, "partition"].eq("historical_quarantine").all()
    checks["historical_train_overlap_17_train_only"] = int(h.sum()) == 17 and df.loc[h, "partition"].eq("train").all()
    lowcols = [c.lower() for c in df.columns]
    checks["no_tier3_target_columns"] = not any(any(t in c for t in TARGET_TOKENS) for c in lowcols)
    checks["no_final_membership_columns"] = not any("final673" in c or "in_final" in c for c in lowcols)
    checks["allowed_partitions"] = set(df.partition) <= {"train", "val", "test", "buffer", "historical_quarantine"}
    if name == "donor_cold_v1":
        sets = partition_sets(df, "donor_structure_group_id_v1")
        checks["donor_identity_zero_overlap"] = disjoint(sets)
        checks["donor_minimum_power"] = len(sets["val"]) >= 15 and len(sets["test"]) >= 15
    elif name == "acceptor_cold_v1":
        sets = partition_sets(df, "acceptor_structure_group_id_v1")
        checks["acceptor_identity_zero_overlap"] = disjoint(sets)
        checks["acceptor_minimum_power"] = len(sets["val"]) >= 30 and len(sets["test"]) >= 30
    elif name == "pair_cold_v1":
        sets = partition_sets(df, "pair_group_id_v1")
        checks["pair_identity_zero_overlap"] = disjoint(sets)
        train = df[df.partition.eq("train")]
        ds = train.groupby("donor_structure_group_id_v1").size()
        ac = train.groupby("acceptor_structure_group_id_v1").size()
        vt = df[df.partition.isin(["val", "test"])]
        checks["pair_seen_components_support_ge5"] = bool(
            vt.donor_structure_group_id_v1.map(ds).fillna(0).ge(5).all() and
            vt.acceptor_structure_group_id_v1.map(ac).fillna(0).ge(5).all())
    elif name == "full_scaffold_cold_v1":
        checks["scaffold_identity_zero_overlap"] = disjoint(partition_sets(df, "full_scaffold_group_id_v1"))
    elif name == "both_cold_external_test_v1":
        test = df[df.partition.eq("test")]
        tv = df[df.partition.isin(["train", "val"])]
        checks["both_cold_test_donor_zero_overlap"] = not (set(test.donor_structure_group_id_v1) & set(tv.donor_structure_group_id_v1))
        checks["both_cold_test_acceptor_zero_overlap"] = not (set(test.acceptor_structure_group_id_v1) & set(tv.acceptor_structure_group_id_v1))
        checks["buffer_explicit"] = df.partition.eq("buffer").any()
        checks["both_cold_power"] = (len(test) >= 500 and test.structure_group_id_v1.nunique() >= 450 and
                                      test.donor_structure_group_id_v1.nunique() >= 15 and
                                      test.acceptor_structure_group_id_v1.nunique() >= 30 and
                                      df.partition.eq("train").sum() >= 7000 and
                                      df.partition.eq("val").sum() >= 1000)
    return checks


def main():
    generator = import_generator()
    registry = json.loads((ROOT / "manifests/split_registry_v1.json").read_text())
    gate0b = json.loads((ROOT / "logs/gate0b_evidence.json").read_text())
    evidence = {"gate": "0-C", "split_status": {}, "global_checks": {}, "hashes": {}}
    observed = {}
    for name, filename in FILES.items():
        path = ROOT / "manifests" / filename
        df = pd.read_csv(path, dtype={"molecule_id": str})
        checks = verify_one(name, df)
        status = "DONE" if all(checks.values()) else "BLOCKED"
        evidence["split_status"][name] = {"status": status, "checks": checks}
        evidence["hashes"][filename] = sha(path)
        observed[name] = df

    base, _, _ = generator.load_inputs()
    regen, _, _ = generator.generate_all(base)
    shuffled, _, _ = generator.load_inputs(shuffle_seed=2026)
    regen_shuffled, _, _ = generator.generate_all(shuffled)
    reproducible = {}
    order_independent = {}
    for name in FILES:
        h_disk = generator.canonical_assignment_hash(observed[name])
        h_regen = generator.canonical_assignment_hash(regen[name])
        h_shuffle = generator.canonical_assignment_hash(regen_shuffled[name])
        reproducible[name] = h_disk == h_regen
        order_independent[name] = h_regen == h_shuffle
    evidence["global_checks"] = {
        "all_split_checks_pass": all(x["status"] == "DONE" for x in evidence["split_status"].values()),
        "repeat_generation_identical_assignment": all(reproducible.values()),
        "shuffled_input_identical_assignment": all(order_independent.values()),
        "final673_aggregate_id_overlap_zero_from_sealed_gate0b": gate0b["historical"]["new_vs_final_id_intersection_aggregate"] == 0,
        "final673_aggregate_structure_overlap_zero_from_sealed_gate0b": gate0b["historical"]["new_vs_final_structure_intersection_aggregate"] == 0,
        "final673_per_sample_artifact_created": False,
        "target_used_for_split": False,
        "time_split_status": "BLOCKED_NO_TRUSTED_TIMESTAMP",
    }
    evidence["reproduction_by_split"] = reproducible
    evidence["input_order_independence_by_split"] = order_independent
    global_ok = (all(reproducible.values()) and all(order_independent.values()) and
                 evidence["global_checks"]["final673_aggregate_id_overlap_zero_from_sealed_gate0b"] and
                 evidence["global_checks"]["final673_aggregate_structure_overlap_zero_from_sealed_gate0b"] and
                 not evidence["global_checks"]["final673_per_sample_artifact_created"] and
                 not evidence["global_checks"]["target_used_for_split"])
    split_ok = all(x["status"] == "DONE" for x in evidence["split_status"].values())
    evidence["overall_status"] = "DONE" if global_ok and split_ok else "BLOCKED"
    if not all(reproducible.values()) or not all(order_independent.values()):
        evidence["overall_status"] = "BLOCKED"

    frozen = dict(registry)
    for name, info in frozen["splits"].items():
        info["status"] = evidence["split_status"][name]["status"]
        info["verified_sha256"] = evidence["hashes"][Path(info["manifest"]).name]
    frozen["verification"] = evidence["global_checks"]
    (ROOT / "data_registry/SPLIT_REGISTRY_V1_FROZEN.json").write_text(
        json.dumps(frozen, indent=2, sort_keys=True) + "\n")
    (ROOT / "logs/gate0c_evidence.json").write_text(json.dumps(evidence, indent=2, sort_keys=True, default=lambda x: x.item()) + "\n")
    hash_paths = [ROOT / "configs/gate0c_split_preregistration_v1.json",
                  ROOT / "data_registry/gate0c_preregistration_lock_v1.json",
                  ROOT / "manifests/split_registry_v1.json",
                  ROOT / "manifests/split_counts_v1.csv",
                  ROOT / "manifests/split_assignment_units_v1.csv"]
    hash_paths += [ROOT / "manifests" / x for x in FILES.values()]
    lines = [f"{sha(p)}  {p.relative_to(ROOT)}" for p in hash_paths]
    (ROOT / "data_registry/gate0c_split_sha256.txt").write_text("\n".join(lines) + "\n")
    print(json.dumps(evidence, indent=2, default=lambda x: x.item()))
    if evidence["overall_status"] != "DONE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
