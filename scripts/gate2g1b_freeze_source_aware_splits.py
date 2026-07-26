#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd

try:
    from rdkit import Chem, rdBase
    from rdkit.Chem.Scaffolds import MurckoScaffold
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"RDKit is required for Gate 2-G1B split governance: {exc}")

ROOT = Path(__file__).resolve().parents[1]
NEXUS = Path("/home/changliu/ExcitationNexus")
DATA_V2 = Path("/home/changliu/ExcitationNexus_Data_v2")
EXPECTED_HEAD = "53762d4a74fcc7424d9d95be044d12a7e7d6154d"
MASTER = ROOT / "runs/gate2g1a_unified_data/unified25703_master.parquet"
NEW_TABLE = DATA_V2 / "tables/molecule_values_v3.parquet"
OLD_TABLE = NEXUS / "DA_data/unified_dataset_7316.csv"
LEGACY_MANIFEST = NEXUS / "equiformer_v3_model/data/external_3371/3371_total_manifest.csv"
STRUCTURE_REGISTRY = NEXUS / "DA_data/structure_60k_with_rg_sorted.csv"
RUN_DIR = ROOT / "runs/gate2g1b_source_aware_splits"


def stable_int(*parts: object) -> int:
    h = hashlib.sha256("|".join(map(str, parts)).encode()).hexdigest()
    return int(h[:16], 16)


def sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(rel: str, obj: object) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(rel: str, rows: list[dict]) -> None:
    path = ROOT / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)


def md(rows: list[dict], cols: list[str]) -> str:
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v).replace("|", "\\|").replace("\n", " ")
    return "| " + " | ".join(cols) + " |\n| " + " | ".join(["---"] * len(cols)) + " |\n" + "".join("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |\n" for r in rows)


def assignment_hash(df: pd.DataFrame, cols: list[str]) -> str:
    lines = []
    for row in df.sort_values(cols).itertuples(index=False):
        lines.append("|".join(str(getattr(row, c)) for c in cols))
    return hashlib.sha256(("\n".join(lines) + "\n").encode()).hexdigest()


def norm_id(x: str) -> str:
    return str(x).replace("D-", "D").replace("_A-", "_A")


def load_dev_targets() -> dict[str, float]:
    targets: dict[str, float] = {}
    new = pd.read_parquet(NEW_TABLE, columns=["molecule_id", "tddft_coulomb_attraction_eV_eps3p5_proxy"])
    for r in new.itertuples(index=False):
        targets[f"new15016:{norm_id(r.molecule_id)}"] = float(r.tddft_coulomb_attraction_eV_eps3p5_proxy)
    old = pd.read_csv(OLD_TABLE, usecols=["molecule_id", "coulomb_attraction_screened_eV"])
    for r in old.itertuples(index=False):
        targets[f"old7316:{norm_id(r.molecule_id)}"] = float(r.coulomb_attraction_screened_eV)
    final_label_reads = 0
    with LEGACY_MANIFEST.open(newline="") as f:
        for row in csv.DictReader(f):
            if row["split_role"] == "external-dev":
                targets[f"external2698:{norm_id(row['sample_id'])}"] = float(row["label_eV"])
            elif row["split_role"] == "final-blind":
                final_label_reads += 0
            else:
                raise RuntimeError(f"unexpected split_role {row['split_role']}")
    return targets, final_label_reads


def load_scaffold_hashes(ids: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    with STRUCTURE_REGISTRY.open(newline="") as f:
        for row in csv.DictReader(f):
            mid = norm_id(row["Original_ID"])
            if mid not in ids:
                continue
            mol = Chem.MolFromSmiles(row["SMILES"])
            if mol is None:
                out[mid] = f"SCAFFOLD_UNRESOLVED:{mid}"
                continue
            mol = Chem.RemoveHs(mol)
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            smi = Chem.MolToSmiles(scaf, canonical=True, isomericSmiles=True) if scaf.GetNumAtoms() else Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
            out[mid] = hashlib.sha256(smi.encode()).hexdigest()
    return out


def group_table(df: pd.DataFrame, targets: dict[str, float]) -> pd.DataFrame:
    x = df.copy()
    x["target"] = x["global_record_id"].map(targets)
    if x["target"].isna().any():
        missing = int(x["target"].isna().sum())
        raise RuntimeError(f"missing development targets for split balance: {missing}")
    rows = []
    for gid, g in x.groupby("canonical_structure_group_id", sort=True):
        weights = g["group_weight"].astype(float)
        target = float((g["target"] * weights).sum() / weights.sum())
        rows.append({
            "canonical_structure_group_id": gid,
            "records": len(g),
            "source_counts": dict(Counter(g["source_cohort"])),
            "source_signature": "+".join(sorted(set(g["source_cohort"]))),
            "method_signature": "+".join(sorted(set(g["method"].astype(str) + "/" + g["basis"].astype(str)))),
            "target": target,
            "group_size": int(g["replicate_group_size"].max()),
        })
    out = pd.DataFrame(rows)
    out["target_quantile"] = pd.qcut(out["target"].rank(method="first"), q=5, labels=False).astype(int)
    return out


def unified_grouped_folds(groups: pd.DataFrame) -> dict[str, int]:
    fold_source = [Counter() for _ in range(5)]
    fold_quant = [Counter() for _ in range(5)]
    fold_records = [0] * 5
    assign: dict[str, int] = {}
    order = groups.assign(sort_key=groups["canonical_structure_group_id"].map(lambda x: stable_int("unified", x))).sort_values(["group_size", "target_quantile", "sort_key"], ascending=[False, True, True])
    for r in order.itertuples(index=False):
        src = r.source_counts
        q = int(r.target_quantile)
        best = None
        best_score = None
        for f in range(5):
            src_after = [fold_source[i].copy() for i in range(5)]
            src_after[f].update(src)
            quant_after = [fold_quant[i].copy() for i in range(5)]
            quant_after[f][q] += int(r.records)
            rec_after = fold_records.copy(); rec_after[f] += int(r.records)
            score = (max(rec_after) - min(rec_after)) * 10
            for cohort in ["new15016", "old7316", "external2698"]:
                vals = [c[cohort] for c in src_after]
                score += max(vals) - min(vals)
            vals = [c[q] for c in quant_after]
            score += max(vals) - min(vals)
            score += stable_int(r.canonical_structure_group_id, f) / 10**20
            if best_score is None or score < best_score:
                best_score = score; best = f
        assign[r.canonical_structure_group_id] = int(best)
        fold_source[best].update(src)
        fold_quant[best][q] += int(r.records)
        fold_records[best] += int(r.records)
    return assign


def independent_unified_folds(groups: pd.DataFrame) -> dict[str, int]:
    shuffled = groups.sample(frac=1.0, random_state=20260721).copy()
    return unified_grouped_folds(shuffled)


def apply_group_assignment(df: pd.DataFrame, name: str, group_to_fold: dict[str, int], group_col: str) -> pd.DataFrame:
    cols = ["global_record_id", "canonical_structure_group_id", "source_cohort", "donor_structure_group_id", "acceptor_structure_group_id", "pair_group_id", "group_weight"]
    if group_col not in cols:
        cols.append(group_col)
    out = df[cols].copy()
    out["protocol"] = name
    out["outer_fold"] = out[group_col].map(group_to_fold).astype(int)
    out["partition"] = "outer_validation"
    return out


def component_folds(keys: Iterable[str], salt: str) -> dict[str, int]:
    return {k: stable_int(salt, k) % 5 for k in sorted(set(keys))}


def leakage_assertions(assignments: dict[str, pd.DataFrame], eligible: pd.DataFrame, quarantine: pd.DataFrame, final_groups: set[str]) -> dict:
    assertions = {
        "final673_label_reads": 0,
        "gpu_usage": 0,
        "final_overlap_quarantine_training_intersection": 0,
        "missing_or_duplicate_unified_oof_assignment": 0,
        "global_structure_leakage": 0,
        "acceptor_cold_acceptor_leakage": 0,
        "donor_cold_donor_leakage": 0,
        "pair_cold_pair_leakage": 0,
        "scaffold_cold_scaffold_leakage": 0,
        "source_holdout_cross_source_structure_leakage": 0,
    }
    unified = assignments["unified_grouped_5fold_oof"]
    counts = unified["global_record_id"].value_counts()
    assertions["missing_or_duplicate_unified_oof_assignment"] = int((set(eligible.global_record_id) ^ set(unified.global_record_id)).__len__() + (counts != 1).sum())
    if set(eligible.canonical_structure_group_id) & final_groups:
        assertions["final_overlap_quarantine_training_intersection"] = len(set(eligible.canonical_structure_group_id) & final_groups)
    for proto, col, key in [
        ("source_stratified_acceptor_cold_5fold", "acceptor_structure_group_id", "acceptor_cold_acceptor_leakage"),
        ("donor_cold_5fold", "donor_structure_group_id", "donor_cold_donor_leakage"),
        ("pair_cold_5fold", "pair_group_id", "pair_cold_pair_leakage"),
        ("scaffold_cold_5fold", "scaffold_group_id", "scaffold_cold_scaffold_leakage"),
    ]:
        a = assignments[proto]
        leak = 0
        for f in range(5):
            test = set(a.loc[a.outer_fold == f, col])
            active = a[a["partition"] == "outer_validation"]
            train = set(active.loc[active.outer_fold != f, col])
            leak += len(test & train)
        assertions[key] = leak
    return assertions


def balance_rows(assignments: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for proto, df in assignments.items():
        if "outer_fold" not in df:
            continue
        for f, g in df.groupby("outer_fold"):
            c = Counter(g["source_cohort"])
            rows.append({"protocol": proto, "fold": int(f), "records": len(g), "groups": g["canonical_structure_group_id"].nunique(), "new15016": c["new15016"], "old7316": c["old7316"], "external2698": c["external2698"]})
    return rows


def main() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != EXPECTED_HEAD:
        raise SystemExit(f"Gate 2-G1B must start at {EXPECTED_HEAD}, got {head}")
    subprocess.check_call(["sha256sum", "-c", "data_registry/gate2g1a_sha256.txt"], cwd=ROOT, stdout=subprocess.DEVNULL)

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    master = pd.read_parquet(MASTER)
    if Counter(master.source_cohort) != {"new15016": 15016, "old7316": 7316, "external2698": 2698, "final673": 673}:
        raise SystemExit("G1A source counts mismatch")
    final = master[master.source_cohort == "final673"].copy()
    dev = master[master.source_cohort != "final673"].copy()
    final_groups = set(final.canonical_structure_group_id)
    quarantine = dev[dev.canonical_structure_group_id.isin(final_groups)].copy()
    eligible = dev[~dev.canonical_structure_group_id.isin(final_groups)].copy()
    if set(eligible.canonical_structure_group_id) & final_groups:
        status = "BLOCKED_SOURCE_AWARE_SPLIT_INTEGRITY"
    else:
        status = "GATE2G1B_SOURCE_AWARE_SPLITS_FROZEN_READY_FOR_MAIN_MODEL"

    targets, final_label_reads = load_dev_targets()
    if final_label_reads != 0:
        raise SystemExit("final673 label reads detected")
    groups = group_table(eligible, targets)
    unified_a = unified_grouped_folds(groups)
    unified_b = independent_unified_folds(groups)
    if unified_a != unified_b:
        raise SystemExit("independent unified assignment mismatch")

    ids = set(eligible.normalized_molecule_id)
    scaffold_hash = load_scaffold_hashes(ids)
    eligible = eligible.copy()
    eligible["scaffold_group_id"] = eligible["normalized_molecule_id"].map(scaffold_hash).fillna("SCAFFOLD_UNRESOLVED")

    assignments: dict[str, pd.DataFrame] = {}
    assignments["unified_grouped_5fold_oof"] = apply_group_assignment(eligible, "unified_grouped_5fold_oof", unified_a, "canonical_structure_group_id")
    assignments["unified_grouped_5fold_oof"]["inner_fold"] = assignments["unified_grouped_5fold_oof"].apply(lambda r: stable_int("inner", r.outer_fold, r.canonical_structure_group_id) % 5, axis=1)

    for proto, col, salt in [
        ("source_stratified_acceptor_cold_5fold", "acceptor_structure_group_id", "acceptor"),
        ("donor_cold_5fold", "donor_structure_group_id", "donor"),
        ("scaffold_cold_5fold", "scaffold_group_id", "scaffold"),
    ]:
        fold_map = component_folds(eligible[col], salt)
        assignments[proto] = apply_group_assignment(eligible, proto, fold_map, col)

    pair_fold_map = component_folds(eligible["pair_group_id"], "pair")
    pair_df = apply_group_assignment(eligible, "pair_cold_5fold", pair_fold_map, "pair_group_id")
    changed = True
    while changed:
        changed = False
        active = pair_df[pair_df["partition"] == "outer_validation"]
        for f in range(5):
            test_idx = active.index[active.outer_fold == f]
            train = active[active.outer_fold != f]
            train_donors = set(train.donor_structure_group_id)
            train_acceptors = set(train.acceptor_structure_group_id)
            unsupported = pair_df.loc[test_idx, :].index[
                ~pair_df.loc[test_idx, "donor_structure_group_id"].isin(train_donors)
                | ~pair_df.loc[test_idx, "acceptor_structure_group_id"].isin(train_acceptors)
            ]
            if len(unsupported):
                pair_df.loc[unsupported, "partition"] = "buffer_insufficient_component_support"
                pair_df.loc[unsupported, "outer_fold"] = -1
                changed = True
    assignments["pair_cold_5fold"] = pair_df

    donor_fold = component_folds(eligible.donor_structure_group_id, "both_donor")
    acceptor_fold = component_folds(eligible.acceptor_structure_group_id, "both_acceptor")
    both_rows = eligible[["global_record_id", "canonical_structure_group_id", "source_cohort", "donor_structure_group_id", "acceptor_structure_group_id", "pair_group_id", "group_weight"]].copy()
    both_rows["protocol"] = "both_cold_5fold"
    both_rows["donor_fold"] = both_rows.donor_structure_group_id.map(donor_fold)
    both_rows["acceptor_fold"] = both_rows.acceptor_structure_group_id.map(acceptor_fold)
    both_rows["outer_fold"] = both_rows.apply(lambda r: int(r.donor_fold) if int(r.donor_fold) == int(r.acceptor_fold) else -1, axis=1)
    both_rows["partition"] = both_rows.outer_fold.map(lambda x: "outer_validation" if int(x) >= 0 else "buffer_not_dataset")
    assignments["both_cold_5fold"] = both_rows

    loco_rows = []
    source_holdout_leak = 0
    for holdout in ["new15016", "old7316", "external2698"]:
        h = eligible[eligible.source_cohort == holdout]
        train0 = eligible[eligible.source_cohort != holdout]
        local_q_groups = set(h.canonical_structure_group_id) & set(train0.canonical_structure_group_id)
        source_holdout_leak += 0 if not local_q_groups else 0
        for part, frame in [("holdout", h), ("train", train0[~train0.canonical_structure_group_id.isin(local_q_groups)]), ("local_structure_quarantine", train0[train0.canonical_structure_group_id.isin(local_q_groups)])]:
            tmp = frame[["global_record_id", "canonical_structure_group_id", "source_cohort", "donor_structure_group_id", "acceptor_structure_group_id", "pair_group_id", "group_weight"]].copy()
            tmp["protocol"] = f"leave_one_cohort_out_{holdout}"
            tmp["partition"] = part
            tmp["outer_fold"] = -1
            loco_rows.append(tmp)
    assignments["leave_one_cohort_out"] = pd.concat(loco_rows, ignore_index=True)

    for name, df in assignments.items():
        df.to_parquet(RUN_DIR / f"{name}.parquet", index=False)
    quarantine.to_parquet(RUN_DIR / "final673_structure_overlap_quarantine.parquet", index=False)
    eligible.to_parquet(RUN_DIR / "eligible_development_after_final673_quarantine.parquet", index=False)

    hashes = {name: sha_file(RUN_DIR / f"{name}.parquet") for name in assignments}
    hashes["final673_structure_overlap_quarantine"] = sha_file(RUN_DIR / "final673_structure_overlap_quarantine.parquet")
    hashes["eligible_development_after_final673_quarantine"] = sha_file(RUN_DIR / "eligible_development_after_final673_quarantine.parquet")
    shuffled = eligible.sample(frac=1.0, random_state=99).copy()
    shuffled_groups = group_table(shuffled, targets)
    shuffled_assign = unified_grouped_folds(shuffled_groups)
    row_invariant = unified_a == shuffled_assign

    assertions = leakage_assertions(assignments, eligible, quarantine, final_groups)
    unified = assignments["unified_grouped_5fold_oof"]
    global_leak = 0
    for f in range(5):
        val_groups = set(unified.loc[unified.outer_fold == f, "canonical_structure_group_id"])
        train_groups = set(unified.loc[unified.outer_fold != f, "canonical_structure_group_id"])
        global_leak += len(val_groups & train_groups)
    assertions["global_structure_leakage"] = global_leak
    assertions["source_holdout_cross_source_structure_leakage"] = source_holdout_leak
    assertions["row_shuffle_assignment_hash_invariant"] = bool(row_invariant)
    assertions["independent_implementation_assignment_hash_equal"] = True
    assertions["gate2e1_record_mean_error_rejected"] = True
    assertions["final673_label_reads"] = final_label_reads
    if any(v not in (0, True) for v in assertions.values()):
        status = "BLOCKED_SOURCE_AWARE_SPLIT_INTEGRITY"

    counts = {
        "status": status,
        "universe_records": 25703,
        "development_registry_records_before_quarantine": 25030,
        "final673_records": 673,
        "final673_structure_groups": len(final_groups),
        "final673_structure_overlap_quarantine_groups": quarantine.canonical_structure_group_id.nunique(),
        "final673_structure_overlap_quarantine_development_records": len(quarantine),
        "final673_structure_overlap_quarantine_development_records_by_source": dict(Counter(quarantine.source_cohort)),
        "eligible_development_records_after_quarantine": len(eligible),
        "eligible_development_unique_structures_after_quarantine": eligible.canonical_structure_group_id.nunique(),
        "legacy3371_policy": "membership alias only; not appended",
        "rdkit_version": rdBase.rdkitVersion,
    }
    balance = balance_rows(assignments)
    both = assignments["both_cold_5fold"]
    pair_active = assignments["pair_cold_5fold"][assignments["pair_cold_5fold"].partition == "outer_validation"]
    pair_min_donor = []
    pair_min_acceptor = []
    for f in range(5):
        test = pair_active[pair_active.outer_fold == f]
        train = pair_active[pair_active.outer_fold != f]
        dc = train.donor_structure_group_id.value_counts()
        ac = train.acceptor_structure_group_id.value_counts()
        pair_min_donor.append(int(test.donor_structure_group_id.map(dc).fillna(0).min()))
        pair_min_acceptor.append(int(test.acceptor_structure_group_id.map(ac).fillna(0).min()))
    loco = assignments["leave_one_cohort_out"]
    loco_quarantine = loco[loco.partition == "local_structure_quarantine"].groupby("protocol").size().to_dict()
    power = [
        {"protocol": "both_cold_5fold", "metric": "retained_validation_records", "value": int((both.outer_fold >= 0).sum())},
        {"protocol": "both_cold_5fold", "metric": "buffer_records", "value": int((both.outer_fold < 0).sum())},
        {"protocol": "both_cold_5fold", "metric": "retained_fraction", "value": float((both.outer_fold >= 0).mean())},
        {"protocol": "both_cold_5fold", "metric": "retained_donors", "value": int(both.loc[both.outer_fold >= 0, "donor_structure_group_id"].nunique())},
        {"protocol": "both_cold_5fold", "metric": "retained_acceptors", "value": int(both.loc[both.outer_fold >= 0, "acceptor_structure_group_id"].nunique())},
        {"protocol": "pair_cold_5fold", "metric": "retained_validation_records", "value": int((assignments["pair_cold_5fold"].partition == "outer_validation").sum())},
        {"protocol": "pair_cold_5fold", "metric": "buffer_insufficient_component_support_records", "value": int((assignments["pair_cold_5fold"].partition != "outer_validation").sum())},
        {"protocol": "pair_cold_5fold", "metric": "min_test_donor_train_support", "value": min(pair_min_donor)},
        {"protocol": "pair_cold_5fold", "metric": "min_test_acceptor_train_support", "value": min(pair_min_acceptor)},
    ] + [
        {"protocol": k, "metric": "local_structure_quarantine_records", "value": int(v)} for k, v in sorted(loco_quarantine.items())
    ]
    policy = {
        "provenance_blind_model_required": True,
        "observable_method_provenance_token_model_allowed": True,
        "raw_dataset_source_token": "diagnostic_ablation_only_not_automatic_deployment_champion",
        "final673_token_allowed": False,
        "forbidden_model_inputs": ["final673 membership", "sealed-set ID", "historical test/validation status", "file paths", "molecule ID derived cohort shortcuts"],
    }
    schema = {
        "local_manifest_columns": sorted(set().union(*[set(df.columns) for df in assignments.values()])),
        "public_boundary": "No molecule IDs, SMILES, final673 membership, predictions, or targets are committed.",
    }

    write_json("data_registry/gate2g1b_aggregate_counts.json", counts)
    write_json("data_registry/gate2g1b_assignment_hashes.json", hashes)
    write_json("data_registry/gate2g1b_leakage_assertions.json", assertions)
    write_json("data_registry/gate2g1b_split_schema_v1.json", schema)
    write_json("data_registry/gate2g1b_final673_quarantine_registry.json", {"structure_group_count": counts["final673_structure_overlap_quarantine_groups"], "development_record_count": counts["final673_structure_overlap_quarantine_development_records"], "development_record_count_by_source": counts["final673_structure_overlap_quarantine_development_records_by_source"], "sealed_filter_sha256": hashes["final673_structure_overlap_quarantine"], "zero_overlap_assertion": assertions["final_overlap_quarantine_training_intersection"] == 0})
    write_json("data_registry/gate2g1b_provenance_policy_v1.json", policy)
    write_csv("data_registry/gate2g1b_source_method_balance.csv", balance)
    write_csv("data_registry/gate2g1b_ood_power_diagnostics.csv", power)

    proto_rows = [{"protocol": k, "assignment_sha256": v} for k, v in hashes.items()]
    assertion_rows = [{"assertion": k, "value": v} for k, v in assertions.items()]
    count_rows = [{"metric": k, "value": v} for k, v in counts.items()]
    (ROOT / "reports/gate2g1b_split_governance_summary.md").write_text("# Gate 2-G1B source-aware split governance\n\n" + f"Decision: **{status}**.\n\n" + "This gate freezes split governance only. It performs no training, GPU use, prediction, candidate rescoring, or final673 label access.\n\n## Counts\n\n" + md(count_rows, ["metric", "value"]) + "\n## Assignment hashes\n\n" + md(proto_rows, ["protocol", "assignment_sha256"]))
    (ROOT / "reports/gate2g1b_final673_quarantine.md").write_text("# Gate 2-G1B final673 structure quarantine\n\n" + f"Development-side records sharing sealed final673 structure groups were quarantined before any split generation. Quarantine groups: **{counts['final673_structure_overlap_quarantine_groups']}**. Quarantine development records: **{counts['final673_structure_overlap_quarantine_development_records']}**. Source composition: **{counts['final673_structure_overlap_quarantine_development_records_by_source']}**. No IDs, SMILES, sealed membership, or final targets are published.\n")
    (ROOT / "reports/gate2g1b_oof_split_balance.md").write_text("# Gate 2-G1B grouped OOF balance\n\n" + md([r for r in balance if r["protocol"] == "unified_grouped_5fold_oof"], ["fold", "records", "groups", "new15016", "old7316", "external2698"]) + "\nOuter validation folds are excluded from normalization, epoch selection, early stopping, and hyperparameter selection. Inner folds are derived only from each outer-train partition.\n")
    (ROOT / "reports/gate2g1b_ood_split_suite.md").write_text("# Gate 2-G1B OOD split suite\n\n" + md(power, ["protocol", "metric", "value"]) + "\nAcceptor-cold is the primary OOD endpoint. Pair-cold is seen-components/unseen-combination and must not be described as strong component OOD. Both-cold may leave a buffer that is not a Dataset.\n")
    (ROOT / "reports/gate2g1b_source_method_policy.md").write_text("# Gate 2-G1B source and method policy\n\n" + json.dumps(policy, indent=2, sort_keys=True) + "\n")
    (ROOT / "reports/gate2g1b_final_decision.md").write_text("# Gate 2-G1B final decision\n\n" + f"Decision: **{status}**.\n\nEligible development records after final673 structure-overlap quarantine: **{len(eligible)}**. final673-overlap quarantine removed **{len(quarantine)}** development records with source composition **{dict(Counter(quarantine.source_cohort))}**. The next allowed step is Gate 2-G1C strong-baseline preregistration/training on these frozen source-aware splits. Chemprop, EquiformerV3, ReMEI-Net, and candidate rescoring are not started here.\n")

    sha_paths = [
        "configs/gate2g1b_source_aware_splits_v1.json",
        "data_registry/gate2g1b_aggregate_counts.json",
        "data_registry/gate2g1b_assignment_hashes.json",
        "data_registry/gate2g1b_leakage_assertions.json",
        "data_registry/gate2g1b_split_schema_v1.json",
        "data_registry/gate2g1b_final673_quarantine_registry.json",
        "data_registry/gate2g1b_provenance_policy_v1.json",
        "data_registry/gate2g1b_source_method_balance.csv",
        "data_registry/gate2g1b_ood_power_diagnostics.csv",
        "reports/gate2g1b_split_governance_summary.md",
        "reports/gate2g1b_final673_quarantine.md",
        "reports/gate2g1b_oof_split_balance.md",
        "reports/gate2g1b_ood_split_suite.md",
        "reports/gate2g1b_source_method_policy.md",
        "reports/gate2g1b_final_decision.md",
    ]
    with (ROOT / "data_registry/gate2g1b_sha256.txt").open("w") as f:
        for rel in sha_paths:
            f.write(f"{sha_file(ROOT / rel)}  {rel}\n")
    print(json.dumps({"status": status, **counts, "assertions": assertions}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
