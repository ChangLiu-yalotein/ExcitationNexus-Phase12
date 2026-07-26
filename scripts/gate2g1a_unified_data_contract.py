#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

try:
    from rdkit import Chem, rdBase
except Exception as exc:  # pragma: no cover
    raise SystemExit(f"RDKit is required for Gate 2-G1A identity governance: {exc}")


ROOT = Path(__file__).resolve().parents[1]
DATA_V2 = Path("/home/changliu/ExcitationNexus_Data_v2")
NEXUS = Path("/home/changliu/ExcitationNexus")
EXPECTED_HEAD = "95f70354dff0955882e852d4363b81df59378460"

NEW_TABLE = DATA_V2 / "tables/molecule_values_v3.parquet"
OLD_TABLE = NEXUS / "DA_data/unified_dataset_7316.csv"
LEGACY_MANIFEST = NEXUS / "equiformer_v3_model/data/external_3371/3371_total_manifest.csv"
STRUCTURE_REGISTRY = NEXUS / "DA_data/structure_60k_with_rg_sorted.csv"


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
    fields = sorted({k for row in rows for k in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def md_table(rows: list[dict], cols: list[str]) -> str:
    def fmt(v: object) -> str:
        if isinstance(v, float):
            return f"{v:.6f}"
        return str(v).replace("|", "\\|").replace("\n", " ")

    return (
        "| "
        + " | ".join(cols)
        + " |\n| "
        + " | ".join(["---"] * len(cols))
        + " |\n"
        + "".join("| " + " | ".join(fmt(r.get(c, "")) for c in cols) + " |\n" for r in rows)
    )


def norm_id(value: str) -> str:
    return value.replace("D-", "D").replace("_A-", "_A")


def split_da(value: str) -> tuple[str, str]:
    d, a = norm_id(value).split("_")
    return d, a


def structure_hash(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    mol = Chem.RemoveHs(mol)
    can = Chem.MolToSmiles(mol, isomericSmiles=True, canonical=True)
    return hashlib.sha256(can.encode("utf-8")).hexdigest()


def needed_structure_ids() -> set[str]:
    ids = set(pd.read_parquet(NEW_TABLE, columns=["molecule_id"])["molecule_id"].map(norm_id))
    ids.update(pd.read_csv(OLD_TABLE, usecols=["molecule_id"])["molecule_id"].map(norm_id))
    legacy_ids = pd.read_csv(LEGACY_MANIFEST, usecols=["sample_id"])
    ids.update(legacy_ids["sample_id"].map(norm_id))
    return ids


def load_structure_hashes(needed_ids: set[str]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    with STRUCTURE_REGISTRY.open(newline="") as f:
        for row in csv.DictReader(f):
            mid = norm_id(row["Original_ID"])
            if mid not in needed_ids:
                continue
            h = structure_hash(row["SMILES"])
            if h:
                hashes[mid] = h
    return hashes


def legacy_rows_without_final_labels(struct_hash: dict[str, str]) -> list[dict]:
    rows: list[dict] = []
    usecols = ["sample_id", "split_role", "geometry_source", "label_formula_version", "conversion_status"]
    legacy = pd.read_csv(LEGACY_MANIFEST, usecols=usecols)
    for row in legacy.to_dict(orient="records"):
        mid = norm_id(row["sample_id"])
        d, a = split_da(mid)
        source = "external2698" if row["split_role"] == "external-dev" else "final673"
        sealed = source == "final673"
        rows.append(
                {
                    "global_record_id": f"{source}:{mid}",
                    "source_cohort": source,
                    "original_molecule_id": row["sample_id"],
                    "normalized_molecule_id": mid,
                    "canonical_structure_group_id": struct_hash.get(mid, f"UNRESOLVED:{mid}"),
                    "donor_structure_group_id": d,
                    "acceptor_structure_group_id": a,
                    "pair_group_id": f"{d}_{a}",
                    "original_historical_partition": row["split_role"],
                    "method": "CAM-B3LYP",
                    "basis": "6-31G(d)",
                    "program": "legacy_external_holdout_manifest",
                    "geometry_fidelity": row["geometry_source"],
                    "label_source": "label_eV_redacted" if sealed else "label_eV_manifest",
                    "epsilon": 3.5,
                    "target_semantics_version": row["label_formula_version"],
                    "PM6_available": True,
                    "DFT_available": True,
                    "TDDFT_available": True,
                    "2D_available": mid in struct_hash,
                    "3D_available": row["geometry_source"] in {"both", "dft", "pm6"},
                    "auxiliary_targets_available": True,
                    "sealed_status": "SEALED_FINAL673" if sealed else "OPEN_DEVELOPMENT",
                    "training_eligibility": "FORBIDDEN_FINAL_CONFIRMATION" if sealed else "DEVELOPMENT_ELIGIBLE_SOURCE_AWARE",
                    "exclusion_reason": "FINAL673_LABEL_SEALED" if sealed else "",
                    "cohort_target_available_for_training": not sealed,
                }
            )
    return rows


def new_rows(struct_hash: dict[str, str]) -> list[dict]:
    df = pd.read_parquet(NEW_TABLE)
    rows: list[dict] = []
    for row in df.itertuples(index=False):
        mid = norm_id(row.molecule_id)
        d, a = split_da(mid)
        rows.append(
            {
                "global_record_id": f"new15016:{mid}",
                "source_cohort": "new15016",
                "original_molecule_id": row.molecule_id,
                "normalized_molecule_id": mid,
                "canonical_structure_group_id": structure_hash(row.canonical_smiles) or struct_hash.get(mid, row.canonical_smiles_sha256),
                "donor_structure_group_id": d,
                "acceptor_structure_group_id": a,
                "pair_group_id": f"{d}_{a}",
                "original_historical_partition": "new15016",
                "method": row.tddft_method,
                "basis": row.tddft_basis,
                "program": str(row.dft_program),
                "geometry_fidelity": row.dft_source_geometry,
                "label_source": "tddft_coulomb_attraction_eV_eps3p5_proxy",
                "epsilon": 3.5,
                "target_semantics_version": "J_eh_screened_eV_eps3p5_proxy_v3",
                "PM6_available": True,
                "DFT_available": True,
                "TDDFT_available": True,
                "2D_available": True,
                "3D_available": True,
                "auxiliary_targets_available": True,
                "sealed_status": "OPEN_DEVELOPMENT",
                "training_eligibility": "DEVELOPMENT_ELIGIBLE_SOURCE_AWARE",
                "exclusion_reason": "",
                "cohort_target_available_for_training": True,
            }
        )
    return rows


def old_rows(struct_hash: dict[str, str]) -> list[dict]:
    df = pd.read_csv(OLD_TABLE)
    rows: list[dict] = []
    for row in df.itertuples(index=False):
        mid = norm_id(row.molecule_id)
        d, a = split_da(mid)
        rows.append(
            {
                "global_record_id": f"old7316:{mid}",
                "source_cohort": "old7316",
                "original_molecule_id": row.molecule_id,
                "normalized_molecule_id": mid,
                "canonical_structure_group_id": struct_hash.get(mid, f"UNRESOLVED:{mid}"),
                "donor_structure_group_id": d,
                "acceptor_structure_group_id": a,
                "pair_group_id": f"{d}_{a}",
                "original_historical_partition": "old7316_training",
                "method": "CAM-B3LYP",
                "basis": "legacy_6-31G(d)_assumed_from_external_protocol",
                "program": "legacy_teacher_table",
                "geometry_fidelity": "legacy_dft_pm6",
                "label_source": "coulomb_attraction_screened_eV",
                "epsilon": 3.5,
                "target_semantics_version": "legacy_eb_screened_eV",
                "PM6_available": True,
                "DFT_available": True,
                "TDDFT_available": True,
                "2D_available": mid in struct_hash,
                "3D_available": True,
                "auxiliary_targets_available": True,
                "sealed_status": "OPEN_DEVELOPMENT",
                "training_eligibility": "DEVELOPMENT_ELIGIBLE_SOURCE_AWARE",
                "exclusion_reason": "",
                "cohort_target_available_for_training": True,
            }
        )
    return rows


def add_group_weights(rows: list[dict]) -> None:
    counts = Counter(r["canonical_structure_group_id"] for r in rows)
    for r in rows:
        size = counts[r["canonical_structure_group_id"]]
        r["replicate_group_size"] = size
        r["group_weight"] = 1.0 / size


def summarize_identity(rows: list[dict]) -> dict:
    by_source = defaultdict(set)
    ids_by_source = defaultdict(set)
    for r in rows:
        by_source[r["source_cohort"]].add(r["canonical_structure_group_id"])
        ids_by_source[r["source_cohort"]].add(r["normalized_molecule_id"])

    def inter(a: str, b: str) -> int:
        return len(by_source[a] & by_source[b])

    group_size_counts = Counter(Counter(r["canonical_structure_group_id"] for r in rows).values())
    source_counts = Counter(r["source_cohort"] for r in rows)
    return {
        "record_counts": dict(source_counts),
        "total_records": len(rows),
        "development_records": source_counts["new15016"] + source_counts["old7316"] + source_counts["external2698"],
        "final_confirmation_records": source_counts["final673"],
        "legacy3371_decomposition": {
            "external2698": source_counts["external2698"],
            "final673": source_counts["final673"],
            "sum": source_counts["external2698"] + source_counts["final673"],
        },
        "global_unique_structures": len(set(r["canonical_structure_group_id"] for r in rows)),
        "structure_group_size_distribution": {str(k): v for k, v in sorted(group_size_counts.items())},
        "computed_structure_overlaps": {
            "new_vs_old": inter("new15016", "old7316"),
            "new_vs_external": inter("new15016", "external2698"),
            "new_vs_final": inter("new15016", "final673"),
            "old_vs_external": inter("old7316", "external2698"),
            "external_vs_final": inter("external2698", "final673"),
        },
        "known_frozen_overlap_contract": json.loads((ROOT / "logs/gate0a_overlap.json").read_text()),
        "identity_method": {
            "id_normalization": "D-<n>_A-<n> and D<n>_A<n> mapped to D<n>_A<n>",
            "structure_group": "RDKit RemoveHs canonical isomeric SMILES SHA-256",
            "rdkit_version": rdBase.rdkitVersion,
        },
    }


def method_inventory(rows: list[dict]) -> list[dict]:
    counts = Counter((r["source_cohort"], r["method"], r["basis"], r["program"], r["geometry_fidelity"]) for r in rows)
    return [
        {
            "source_cohort": k[0],
            "method": k[1],
            "basis": k[2],
            "program": k[3],
            "geometry_fidelity": k[4],
            "records": v,
        }
        for k, v in sorted(counts.items())
    ]


def target_ledger(rows: list[dict]) -> list[dict]:
    source_counts = Counter(r["source_cohort"] for r in rows)
    return [
        {
            "source_cohort": "new15016",
            "records": source_counts["new15016"],
            "target_field": "tddft_coulomb_attraction_eV_eps3p5_proxy",
            "raw_quantity": "tddft_coulomb_attraction_eV / epsilon",
            "epsilon": 3.5,
            "unit": "eV",
            "semantics": "J_eh_screened_eV_eps3p5 proxy",
            "status": "COMPATIBLE_PROXY_NEW_PIPELINE",
        },
        {
            "source_cohort": "old7316",
            "records": source_counts["old7316"],
            "target_field": "coulomb_attraction_screened_eV / eb_screened_eV",
            "raw_quantity": "legacy screened Coulomb/EB proxy",
            "epsilon": 3.5,
            "unit": "eV",
            "semantics": "legacy screened proxy; same nominal epsilon but parser/method lineage differs",
            "status": "SOURCE_AWARE_REQUIRED",
        },
        {
            "source_cohort": "external2698",
            "records": source_counts["external2698"],
            "target_field": "label_eV",
            "raw_quantity": "eb_screened_v1",
            "epsilon": 3.5,
            "unit": "eV",
            "semantics": "legacy external screened proxy; historical model-selection use means not blind",
            "status": "SOURCE_AWARE_REQUIRED",
        },
        {
            "source_cohort": "final673",
            "records": source_counts["final673"],
            "target_field": "REDACTED_NOT_READ_FOR_GATE2G1A",
            "raw_quantity": "sealed aggregate only",
            "epsilon": 3.5,
            "unit": "eV",
            "semantics": "final confirmation labels remain sealed",
            "status": "SEALED_CONFIRMATION_ONLY",
        },
    ]


def tier_registry(rows: list[dict]) -> dict:
    open_rows = [r for r in rows if r["source_cohort"] != "final673"]
    final_rows = [r for r in rows if r["source_cohort"] == "final673"]

    def tier(name: str, predicate) -> dict:
        dev = [r for r in open_rows if predicate(r)]
        final = [r for r in final_rows if predicate(r)]
        return {
            "eligible_development_records": len(dev),
            "eligible_development_unique_structures": len({r["canonical_structure_group_id"] for r in dev}),
            "final673_input_coverage_records_without_label_access": len(final),
            "cohort_counts": dict(Counter(r["source_cohort"] for r in dev)),
            "description": name,
        }

    return {
        "U0": tier("2D structure and C0 descriptors/fingerprints", lambda r: r["2D_available"]),
        "U1": tier("U0 plus safe PM6 scalar features", lambda r: r["2D_available"] and r["PM6_available"]),
        "U2": tier("DFT/3D structure inputs", lambda r: r["3D_available"] and r["DFT_available"]),
        "U3": tier("full PM6+DFT+TDDFT multitask labels", lambda r: r["PM6_available"] and r["DFT_available"] and r["TDDFT_available"] and r["auxiliary_targets_available"]),
    }


def assert_contract(summary: dict) -> list[str]:
    failures: list[str] = []
    expected = {"new15016": 15016, "old7316": 7316, "external2698": 2698, "final673": 673}
    if summary["record_counts"] != expected:
        failures.append(f"record_counts mismatch: {summary['record_counts']}")
    if summary["total_records"] != 25703:
        failures.append("total_records != 25703")
    if summary["development_records"] != 25030:
        failures.append("development_records != 25030")
    if summary["legacy3371_decomposition"]["sum"] != 3371:
        failures.append("legacy3371 decomposition failed")
    frozen = summary["known_frozen_overlap_contract"]["rdkit_canonical_smiles_intersections"]
    wanted = {
        "new_vs_old": frozen["new_old"],
        "new_vs_external": frozen["new_external"],
        "new_vs_final": frozen["new_final"],
        "old_vs_external": frozen["old_external"],
        "external_vs_final": frozen["external_final"],
    }
    if summary["computed_structure_overlaps"] != wanted:
        failures.append(f"overlap mismatch: {summary['computed_structure_overlaps']} vs {wanted}")
    return failures


def main() -> None:
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    if head != EXPECTED_HEAD:
        raise SystemExit(f"Gate 2-G1A must start at {EXPECTED_HEAD}, got {head}")

    struct_hash = load_structure_hashes(needed_structure_ids())
    rows = new_rows(struct_hash) + old_rows(struct_hash) + legacy_rows_without_final_labels(struct_hash)
    add_group_weights(rows)
    summary = summarize_identity(rows)
    failures = assert_contract(summary)
    status = "UNIFIED25703_SOURCE_AWARE_TRAINING_REQUIRED" if not failures else "BLOCKED_UNIFIED_IDENTITY_INTEGRITY"

    run_dir = ROOT / "runs/gate2g1a_unified_data"
    run_dir.mkdir(parents=True, exist_ok=True)
    master = pd.DataFrame(rows)
    master.to_parquet(run_dir / "unified25703_master.parquet", index=False)
    master[master["source_cohort"] != "final673"].to_parquet(run_dir / "development25030.parquet", index=False)
    (run_dir / "protocol_local_identity_tables").mkdir(exist_ok=True)
    (run_dir / "detailed_conflicts").mkdir(exist_ok=True)
    (run_dir / "detailed_missingness").mkdir(exist_ok=True)

    method_rows = method_inventory(rows)
    target_rows = target_ledger(rows)
    tier = tier_registry(rows)
    source_registry = {
        "status": status,
        "decision_reason": "Nominal target is compatible, but old/external parser and method lineage require source-aware training.",
        "sources": {
            "new15016": {"records": 15016, "path": str(NEW_TABLE), "sha256": sha_file(NEW_TABLE)},
            "old7316": {"records": 7316, "path": str(OLD_TABLE), "sha256": sha_file(OLD_TABLE)},
            "legacy3371_alias": {"records": 3371, "path": str(LEGACY_MANIFEST), "sha256": sha_file(LEGACY_MANIFEST), "policy": "split into external2698 + final673, never appended separately"},
            "structure_registry": {"path": str(STRUCTURE_REGISTRY), "sha256": sha_file(STRUCTURE_REGISTRY)},
        },
        "final673_boundary": {
            "records": 673,
            "label_access": False,
            "training_eligibility": "FORBIDDEN_UNTIL_FROZEN_CONFIRMATION",
            "public_membership": "aggregate counts only",
        },
        "gate3_shortlist_status": "EXPLORATORY_BASELINE_SHORTLIST_FROZEN",
    }

    write_json("data_registry/unified25703_dataset_registry.json", {"status": status, **summary, "failures": failures})
    write_json("data_registry/unified25703_schema_v1.json", {"registry_columns": list(master.columns), "sensitive_local_assets": ["runs/gate2g1a_unified_data/unified25703_master.parquet", "runs/gate2g1a_unified_data/development25030.parquet"]})
    write_json("data_registry/unified25703_source_registry.json", source_registry)
    write_json("data_registry/unified25703_identity_spec_v1.json", summary["identity_method"])
    write_csv("data_registry/unified25703_target_semantics_ledger.csv", target_rows)
    write_csv("data_registry/unified25703_method_basis_inventory.csv", method_rows)
    write_json("data_registry/unified25703_input_tier_registry.json", tier)
    write_json("data_registry/unified25703_overlap_summary.json", {"computed_structure_overlaps": summary["computed_structure_overlaps"], "frozen_contract": summary["known_frozen_overlap_contract"]["rdkit_canonical_smiles_intersections"]})

    report_rows = [
        {"source": k, "records": v}
        for k, v in summary["record_counts"].items()
    ]
    (ROOT / "reports/gate2g1a_unified_data_summary.md").write_text(
        "# Gate 2-G1A unified 25,703-record data universe\n\n"
        f"Decision: **{status}**.\n\n"
        "This gate performs data governance only. No training, GPU work, candidate scoring, or final673 label use is permitted. "
        "Gate 3 candidates remain `EXPLORATORY_BASELINE_SHORTLIST_FROZEN`.\n\n"
        "## Source counts\n\n"
        + md_table(report_rows, ["source", "records"])
        + f"\nDevelopment universe: **{summary['development_records']}** records. Final confirmation set: **{summary['final_confirmation_records']}** sealed records. "
        f"Global unique structures under the Gate 2-G1A identity rule: **{summary['global_unique_structures']}**.\n"
    )
    (ROOT / "reports/gate2g1a_target_harmonization.md").write_text(
        "# Gate 2-G1A target harmonization\n\n"
        + md_table(target_rows, ["source_cohort", "records", "target_field", "epsilon", "unit", "semantics", "status"])
        + "\nConclusion: the target is nominally the same screened Coulomb proxy, but parser/method lineage differs across old/external/new cohorts. Future training must retain source or method information.\n"
    )
    (ROOT / "reports/gate2g1a_method_domain_shift.md").write_text(
        "# Gate 2-G1A method and domain shift\n\n"
        + md_table(method_rows, ["source_cohort", "records", "method", "basis", "program", "geometry_fidelity"])
        + "\nMethod and parser differences must not be hidden in random splits. Use source-aware or cohort-aware training designs in Gate 2-G1B/G1C.\n"
    )
    overlap_rows = [
        {"comparison": k, "computed": v}
        for k, v in summary["computed_structure_overlaps"].items()
    ]
    (ROOT / "reports/gate2g1a_global_identity_governance.md").write_text(
        "# Gate 2-G1A global identity governance\n\n"
        + md_table(overlap_rows, ["comparison", "computed"])
        + "\nAll duplicate structures must stay in one split and receive `group_weight = 1 / global_structure_group_size`. Duplicate targets are retained as records, not averaged.\n"
    )
    tier_rows = [{"tier": k, **v} for k, v in tier.items()]
    (ROOT / "reports/gate2g1a_training_tier_feasibility.md").write_text(
        "# Gate 2-G1A training tier feasibility\n\n"
        + md_table(tier_rows, ["tier", "description", "eligible_development_records", "eligible_development_unique_structures", "final673_input_coverage_records_without_label_access", "cohort_counts"])
        + "\nNo missing quantum results are imputed to enlarge higher tiers.\n"
    )
    (ROOT / "reports/gate2g1a_final673_boundary.md").write_text(
        "# Gate 2-G1A final673 boundary\n\n"
        "final673 remains sealed for model selection and training. This gate records only aggregate count, input coverage, and structure-level conflict counts. "
        "Targets, IDs, SMILES, and per-sample membership are not published. Future evaluation must report both official 673 and structure-purged sensitivity because external2698 and final673 have 18 structure overlaps.\n"
    )
    (ROOT / "reports/gate2g1a_final_decision.md").write_text(
        "# Gate 2-G1A final decision\n\n"
        f"Decision: **{status}**.\n\n"
        "The 25,703-record universe is available as a governed data contract, not as a naive merged training table. "
        "The 25,030-record development pool may be used only with structure-group split discipline and source/method-aware modeling. "
        "final673 remains a sealed confirmation set. Gate 3 experimental progression remains paused.\n"
    )

    sha_paths = [
        "configs/gate2g1a_unified25703_data_contract_v1.json",
        "data_registry/unified25703_dataset_registry.json",
        "data_registry/unified25703_schema_v1.json",
        "data_registry/unified25703_source_registry.json",
        "data_registry/unified25703_identity_spec_v1.json",
        "data_registry/unified25703_target_semantics_ledger.csv",
        "data_registry/unified25703_method_basis_inventory.csv",
        "data_registry/unified25703_input_tier_registry.json",
        "data_registry/unified25703_overlap_summary.json",
        "reports/gate2g1a_unified_data_summary.md",
        "reports/gate2g1a_target_harmonization.md",
        "reports/gate2g1a_method_domain_shift.md",
        "reports/gate2g1a_global_identity_governance.md",
        "reports/gate2g1a_training_tier_feasibility.md",
        "reports/gate2g1a_final673_boundary.md",
        "reports/gate2g1a_final_decision.md",
        "README.md",
    ]
    with (ROOT / "data_registry/gate2g1a_sha256.txt").open("w") as f:
        for rel in sha_paths:
            f.write(f"{sha_file(ROOT / rel)}  {rel}\n")

    print(json.dumps({"status": status, "records": summary["record_counts"], "failures": failures}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
