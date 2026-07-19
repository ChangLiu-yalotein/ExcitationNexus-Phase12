#!/usr/bin/env python3
"""Gate 0-C target-blind, deterministic grouped split generator."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import scipy
import sklearn

ROOT = Path(__file__).resolve().parents[1]
STRUCTURE_PATH = ROOT / "manifests/new15016_structure_groups_v1.parquet"
COMPONENT_PATH = ROOT / "manifests/component_identity_v1.csv"
HISTORICAL_PATH = ROOT / "manifests/historical_overlap_quarantine_v1.csv"
CONFIG_PATH = ROOT / "configs/gate0c_split_preregistration_v1.json"
LOCK_PATH = ROOT / "data_registry/gate0c_preregistration_lock_v1.json"
INPUT_PATH = Path("/home/changliu/ExcitationNexus_Data_v2/tables/molecule_values_v3.parquet")
EXPECTED_INPUT_SHA256 = "e7587b1546039f099a4dbd0d352e98885bb2ebdbdcfa18884dd4355eed815a83"
SPLIT_VERSION = "v1"
SEEDS = [42, 123, 456, 789, 2026]
PARTITIONS = ("train", "val", "test")
MANIFEST_COLUMNS = [
    "molecule_id", "partition", "structure_group_id_v1",
    "donor_structure_group_id_v1", "acceptor_structure_group_id_v1",
    "pair_group_id_v1", "full_scaffold_group_id_v1", "role_aware_group_id_v1",
    "structure_group_size", "group_weight", "assignment_unit_id",
    "historical_status", "forced_train_reason", "split_name", "split_version",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stable_hash(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def canonical_assignment_hash(df: pd.DataFrame) -> str:
    lines = [f"{r.molecule_id}\t{r.partition}\t{r.assignment_unit_id}" for r in
             df.sort_values("molecule_id", kind="mergesort").itertuples()]
    return stable_hash("\n".join(lines) + "\n")


class DSU:
    def __init__(self, values):
        self.parent = {x: x for x in values}

    def find(self, x):
        p = self.parent[x]
        if p != x:
            self.parent[x] = self.find(p)
        return self.parent[x]

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)


def load_inputs(shuffle_seed: int | None = None) -> tuple[pd.DataFrame, dict, dict]:
    config_bytes = CONFIG_PATH.read_bytes()
    lock = json.loads(LOCK_PATH.read_text())
    assert hashlib.sha256(config_bytes).hexdigest() == lock["preregistration_sha256"]
    assert sha256_file(INPUT_PATH) == EXPECTED_INPUT_SHA256
    config = json.loads(config_bytes)
    assert config["candidate_seeds"] == SEEDS
    s = pd.read_parquet(STRUCTURE_PATH)
    c = pd.read_csv(COMPONENT_PATH, dtype=str)
    keep = ["molecule_id", "donor_canonical_structure_smiles_v1",
            "acceptor_canonical_structure_smiles_v1", "full_scaffold_status"]
    d = s.merge(c[keep], on="molecule_id", how="left", validate="one_to_one")
    q = pd.read_csv(HISTORICAL_PATH, dtype=str)
    status = dict(zip(q.structure_group_id_v1, q.governance_action))
    d["historical_status"] = d.structure_group_id_v1.map(status).fillna("NONE")
    if shuffle_seed is not None:
        d = d.sample(frac=1, random_state=shuffle_seed).reset_index(drop=True)
    assert len(d) == 15016 and d.molecule_id.is_unique
    assert d.full_scaffold_status.eq("OK").all()
    assert d.historical_status.eq("HISTORICAL_MODEL_SELECTION_QUARANTINE").sum() == 1
    assert d.historical_status.eq("HISTORICAL_TRAIN_OVERLAP").sum() == 17
    return d, config, lock


def atomic_units(df: pd.DataFrame, identity_col: str | None, split_name: str):
    model = df[df.historical_status.ne("HISTORICAL_MODEL_SELECTION_QUARANTINE")].copy()
    structures = sorted(model.structure_group_id_v1.unique())
    dsu = DSU(structures)
    if identity_col and identity_col != "structure_group_id_v1":
        for _, g in model.groupby(identity_col, sort=True):
            ids = sorted(g.structure_group_id_v1.unique())
            for x in ids[1:]:
                dsu.union(ids[0], x)
    components = defaultdict(list)
    for x in structures:
        components[dsu.find(x)].append(x)
    structure_to_unit = {}
    unit_rows = []
    for members in components.values():
        members = sorted(members)
        uid = stable_hash(split_name + "|" + "|".join(members))
        mask = model.structure_group_id_v1.isin(members)
        g = model.loc[mask]
        forced = g.historical_status.eq("HISTORICAL_TRAIN_OVERLAP").any()
        for x in members:
            structure_to_unit[x] = uid
        unit_rows.append({
            "assignment_unit_id": uid, "record_count": len(g),
            "effective_weight": float(g.group_weight.sum()),
            "structure_group_count": len(members), "forced_train": bool(forced),
            "identity_count": int(g[identity_col].nunique()) if identity_col else len(members),
        })
    return model, structure_to_unit, pd.DataFrame(unit_rows).sort_values("assignment_unit_id")


def objective(counts: dict[str, float], targets: dict[str, float]) -> float:
    return sum(abs(counts[p] - targets[p]) for p in targets)


def balance_units(units: pd.DataFrame, proportions: dict[str, float], seed: int,
                  partitions=PARTITIONS) -> tuple[dict[str, str], tuple]:
    total_r = float(units.record_count.sum())
    total_w = float(units.effective_weight.sum())
    target_r = {p: total_r * proportions[p] for p in partitions}
    target_w = {p: total_w * proportions[p] for p in partitions}
    assigned, cr, cw = {}, {p: 0.0 for p in partitions}, {p: 0.0 for p in partitions}
    forced = units[units.forced_train].sort_values("assignment_unit_id")
    for r in forced.itertuples():
        assigned[r.assignment_unit_id] = "train"
        cr["train"] += r.record_count; cw["train"] += r.effective_weight
    rest = units[~units.forced_train].copy()
    rest["tie"] = rest.assignment_unit_id.map(lambda x: stable_hash(f"{seed}:{x}"))
    rest = rest.sort_values(["record_count", "effective_weight", "tie"], ascending=[False, False, True])
    for r in rest.itertuples():
        choices = []
        for p in partitions:
            nr, nw = cr.copy(), cw.copy()
            nr[p] += r.record_count; nw[p] += r.effective_weight
            score = objective(nr, target_r) / total_r + objective(nw, target_w) / total_w
            choices.append((score, stable_hash(f"{seed}:{r.assignment_unit_id}:{p}"), p))
        p = min(choices)[2]
        assigned[r.assignment_unit_id] = p
        cr[p] += r.record_count; cw[p] += r.effective_weight
    score = (objective(cr, target_r) / total_r, objective(cw, target_w) / total_w,
             seed, stable_hash(json.dumps(sorted(assigned.items()))))
    return assigned, score


def apply_assignment(df, structure_to_unit, assigned, split_name):
    out = df.copy()
    out["assignment_unit_id"] = out.structure_group_id_v1.map(structure_to_unit)
    out["partition"] = out.assignment_unit_id.map(assigned)
    q = out.historical_status.eq("HISTORICAL_MODEL_SELECTION_QUARANTINE")
    out.loc[q, "partition"] = "historical_quarantine"
    out.loc[q, "assignment_unit_id"] = out.loc[q, "structure_group_id_v1"].map(
        lambda x: stable_hash(split_name + "|quarantine|" + x))
    out["forced_train_reason"] = np.where(
        out.historical_status.eq("HISTORICAL_TRAIN_OVERLAP"),
        "HISTORICAL_TRAIN_OVERLAP", "")
    out["split_name"] = split_name
    out["split_version"] = SPLIT_VERSION
    return out[MANIFEST_COLUMNS].sort_values("molecule_id", kind="mergesort").reset_index(drop=True)


def generic_split(df, split_name, identity_col, proportions=None):
    proportions = proportions or {"train": .70, "val": .15, "test": .15}
    model, s2u, units = atomic_units(df, identity_col, split_name)
    candidates = []
    for seed in SEEDS:
        a, score = balance_units(units, proportions, seed)
        candidates.append((score, seed, a))
    score, seed, assigned = min(candidates, key=lambda x: x[0])
    out = apply_assignment(df, s2u, assigned, split_name)
    return out, units.assign(partition=units.assignment_unit_id.map(assigned), split_name=split_name), {
        "selected_seed": seed, "selected_objective": score[:2],
        "candidate_objectives": [{"seed": s, "record_l1": z[0], "weight_l1": z[1]}
                                 for z, s, _ in candidates],
    }


def pair_split(df):
    name = "pair_cold_v1"
    model, s2u, units = atomic_units(df, "pair_group_id_v1", name)
    row_unit = model.structure_group_id_v1.map(s2u)
    candidates = []
    for seed in SEEDS:
        assigned, base_score = balance_units(units, {"train": .70, "val": .15, "test": .15}, seed)
        for _ in range(10):
            parts = row_unit.map(assigned)
            train = model[parts.eq("train")]
            ds = train.groupby("donor_structure_group_id_v1").size().to_dict()
            ac = train.groupby("acceptor_structure_group_id_v1").size().to_dict()
            bad = model[~parts.eq("train") & (
                model.donor_structure_group_id_v1.map(ds).fillna(0).lt(5) |
                model.acceptor_structure_group_id_v1.map(ac).fillna(0).lt(5))]
            bad_units = sorted(set(bad.structure_group_id_v1.map(s2u)))
            if not bad_units:
                break
            for uid in bad_units:
                assigned[uid] = "train"
        temp = apply_assignment(df, s2u, assigned, name)
        cnt = temp.partition.value_counts()
        err = sum(abs(cnt.get(p, 0) / 15015 - x) for p, x in
                  {"train": .70, "val": .15, "test": .15}.items())
        candidates.append(((err, seed, canonical_assignment_hash(temp)), seed, assigned, temp))
    score, seed, assigned, out = min(candidates, key=lambda x: x[0])
    units = units.assign(partition=units.assignment_unit_id.map(assigned), split_name=name)
    return out, units, {"selected_seed": seed, "selected_objective": score[0],
                        "candidate_objectives": [{"seed": s, "record_ratio_l1": z[0]}
                                                 for z, s, _, _ in candidates]}


def both_candidate(df, seed, kd, ka):
    model = df[df.historical_status.ne("HISTORICAL_MODEL_SELECTION_QUARANTINE")].copy()
    forced = model[model.historical_status.eq("HISTORICAL_TRAIN_OVERLAP")]
    forbidden_d = set(forced.donor_structure_group_id_v1)
    forbidden_a = set(forced.acceptor_structure_group_id_v1)
    donors = sorted(set(model.donor_structure_group_id_v1) - forbidden_d)
    acceptors = sorted(set(model.acceptor_structure_group_id_v1) - forbidden_a)
    ta = set(sorted(acceptors, key=lambda x: stable_hash(f"{seed}:A:{x}"))[:ka])
    td = set()
    for _ in range(4):
        dscore = model[model.acceptor_structure_group_id_v1.isin(ta)].groupby(
            "donor_structure_group_id_v1").size().to_dict()
        td = set(sorted(donors, key=lambda x: (-dscore.get(x, 0), stable_hash(f"{seed}:D:{x}")))[:kd])
        ascore = model[model.donor_structure_group_id_v1.isin(td)].groupby(
            "acceptor_structure_group_id_v1").size().to_dict()
        ta = set(sorted(acceptors, key=lambda x: (-ascore.get(x, 0), stable_hash(f"{seed}:A2:{x}")))[:ka])
    raw = np.where(model.donor_structure_group_id_v1.isin(td) & model.acceptor_structure_group_id_v1.isin(ta),
                   "test", np.where(model.donor_structure_group_id_v1.isin(td) |
                                    model.acceptor_structure_group_id_v1.isin(ta), "buffer", "core"))
    model["raw"] = raw
    mixed = model.groupby("structure_group_id_v1").raw.nunique()
    mixed_ids = set(mixed[mixed.gt(1)].index)
    model.loc[model.structure_group_id_v1.isin(mixed_ids), "raw"] = "buffer"
    test = model[model.raw.eq("test")]
    core = model[model.raw.eq("core")]
    buffer = model[model.raw.eq("buffer")]
    stats = {
        "test_records": len(test), "test_groups": test.structure_group_id_v1.nunique(),
        "test_donors": test.donor_structure_group_id_v1.nunique(),
        "test_acceptors": test.acceptor_structure_group_id_v1.nunique(),
        "core_records": len(core), "buffer_records": len(buffer),
        "test_fraction": len(test) / 15015,
    }
    power = (stats["test_records"] >= 500 and stats["test_groups"] >= 450 and
             stats["test_donors"] >= 15 and stats["test_acceptors"] >= 30 and
             stats["core_records"] >= 8000)
    in_band = .05 <= stats["test_fraction"] <= .10
    key = (0 if power else 1, -stats["core_records"], stats["buffer_records"],
           0 if in_band else 1, abs(stats["test_fraction"] - .075), seed, kd, ka)
    return key, model, stats, td, ta


def both_cold_split(df):
    name = "both_cold_external_test_v1"
    candidates = []
    for seed in SEEDS:
        for kd in (15, 20, 25, 30, 40):
            for ka in (30, 40, 50, 60, 80):
                candidates.append((*both_candidate(df, seed, kd, ka), seed, kd, ka))
    key, model, stats, td, ta, seed, kd, ka = min(candidates, key=lambda x: x[0])
    if key[0] != 0:
        raise RuntimeError("BOTH_COLD_BLOCKED_INSUFFICIENT_POWER")
    core_ids = set(model.loc[model.raw.eq("core"), "molecule_id"])
    core_df = df[df.molecule_id.isin(core_ids)].copy()
    _, core_s2u, core_units = atomic_units(core_df, "structure_group_id_v1", name + "_core")
    core_assign, _ = balance_units(core_units, {"train": .85, "val": .15}, seed,
                                   partitions=("train", "val"))
    full_s2u, assigned = {}, {}
    for sid, uid in core_s2u.items():
        full_s2u[sid] = uid; assigned[uid] = core_assign[uid]
    for rawpart in ("test", "buffer"):
        for sid in sorted(model.loc[model.raw.eq(rawpart), "structure_group_id_v1"].unique()):
            uid = stable_hash(name + "|" + rawpart + "|" + sid)
            full_s2u[sid] = uid; assigned[uid] = rawpart
    out = apply_assignment(df, full_s2u, assigned, name)
    cnt = out.partition.value_counts()
    if cnt.get("train", 0) < 7000 or cnt.get("val", 0) < 1000:
        raise RuntimeError("BOTH_COLD_BLOCKED_INSUFFICIENT_POWER_AFTER_CORE_SPLIT")
    unit_rows = []
    for uid, g in out[out.partition.ne("historical_quarantine")].groupby("assignment_unit_id"):
        unit_rows.append({"assignment_unit_id": uid, "record_count": len(g),
                          "effective_weight": g.group_weight.sum(),
                          "structure_group_count": g.structure_group_id_v1.nunique(),
                          "forced_train": g.forced_train_reason.ne("").any(),
                          "identity_count": 1, "partition": g.partition.iloc[0],
                          "split_name": name})
    candidate_summaries = []
    for x in candidates:
        candidate_summaries.append({"seed": x[-3], "test_donor_request": x[-2],
                                    "test_acceptor_request": x[-1], **x[2], "power": x[0][0] == 0})
    meta = {"selected_seed": seed, "selected_test_donor_request": kd,
            "selected_test_acceptor_request": ka, "selected_stats": stats,
            "candidate_objectives": candidate_summaries}
    return out, pd.DataFrame(unit_rows), meta


def validate_local(out, regime):
    assert len(out) == 15016 and out.molecule_id.is_unique
    assert out.loc[out.historical_status.eq("HISTORICAL_MODEL_SELECTION_QUARANTINE"),
                   "partition"].eq("historical_quarantine").all()
    assert out.loc[out.historical_status.eq("HISTORICAL_TRAIN_OVERLAP"), "partition"].eq("train").all()
    assert out.groupby("structure_group_id_v1").partition.nunique().max() == 1
    model = out[out.partition.isin(["train", "val", "test"])]
    if regime:
        sets = {p: set(model.loc[model.partition.eq(p), regime]) for p in PARTITIONS}
        assert not (sets["train"] & sets["val"] or sets["train"] & sets["test"] or sets["val"] & sets["test"])


def generate_all(df: pd.DataFrame):
    results, units, metadata = {}, [], {}
    specs = [
        ("iid_group_seed42_v1", "structure_group_id_v1", None),
        ("donor_cold_v1", "donor_structure_group_id_v1", "donor_structure_group_id_v1"),
        ("acceptor_cold_v1", "acceptor_structure_group_id_v1", "acceptor_structure_group_id_v1"),
        ("full_scaffold_cold_v1", "full_scaffold_group_id_v1", "full_scaffold_group_id_v1"),
    ]
    for name, identity, check in specs:
        out, u, m = generic_split(df, name, identity)
        validate_local(out, check)
        results[name], metadata[name] = out, m; units.append(u)
    out, u, m = pair_split(df); validate_local(out, "pair_group_id_v1")
    results["pair_cold_v1"], metadata["pair_cold_v1"] = out, m; units.append(u)
    out, u, m = both_cold_split(df); validate_local(out, None)
    results["both_cold_external_test_v1"], metadata["both_cold_external_test_v1"] = out, m; units.append(u)
    return results, pd.concat(units, ignore_index=True), metadata


def counts_table(results):
    rows = []
    for name, df in results.items():
        for p, g in df.groupby("partition", sort=True):
            rows.append({"split_name": name, "partition": p, "records": len(g),
                         "structure_groups": g.structure_group_id_v1.nunique(),
                         "effective_weight": float(g.group_weight.sum()),
                         "fraction_records_all15016": len(g) / 15016})
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--shuffle-seed", type=int)
    parser.add_argument("--output-root", type=Path, default=ROOT)
    args = parser.parse_args()
    df, config, lock = load_inputs(args.shuffle_seed)
    results, units, metadata = generate_all(df)
    outroot = args.output_root
    mdir = outroot / "manifests"; mdir.mkdir(parents=True, exist_ok=True)
    files = {
        "iid_group_seed42_v1": "split_iid_group_seed42_v1.csv",
        "donor_cold_v1": "split_donor_cold_v1.csv",
        "acceptor_cold_v1": "split_acceptor_cold_v1.csv",
        "pair_cold_v1": "split_pair_cold_v1.csv",
        "both_cold_external_test_v1": "split_both_cold_external_test_v1.csv",
        "full_scaffold_cold_v1": "split_full_scaffold_cold_v1.csv",
    }
    registry = {"split_version": SPLIT_VERSION, "preregistration_sha256": lock["preregistration_sha256"],
                "input_sha256": EXPECTED_INPUT_SHA256, "splits": {}}
    for name, filename in files.items():
        path = mdir / filename
        results[name].to_csv(path, index=False, float_format="%.12g", lineterminator="\n")
        registry["splits"][name] = {"status": "GENERATED_PENDING_INDEPENDENT_VERIFICATION",
                                     "manifest": str(path.relative_to(outroot)),
                                     "sha256": sha256_file(path),
                                     "assignment_sha256": canonical_assignment_hash(results[name]),
                                     "solver": metadata[name]}
    counts_table(results).to_csv(mdir / "split_counts_v1.csv", index=False, float_format="%.12g")
    units.sort_values(["split_name", "assignment_unit_id"]).to_csv(
        mdir / "split_assignment_units_v1.csv", index=False, float_format="%.12g")
    registry["versions"] = {"python": platform.python_version(), "pandas": pd.__version__,
                            "numpy": np.__version__, "scipy": scipy.__version__,
                            "sklearn": sklearn.__version__, "rdkit": rdkit.__version__}
    (mdir / "split_registry_v1.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": "GENERATED", "counts": counts_table(results).to_dict("records")}, indent=2))


if __name__ == "__main__":
    main()
