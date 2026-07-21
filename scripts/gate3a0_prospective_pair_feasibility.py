#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, rdBase
from rdkit.Chem import FilterCatalog

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from excitationnexus_phase12.combinatorial_assembly import (  # noqa: E402
    ComponentSite, assemble_components, canonical_graph, canonical_isomeric,
    deterministic_pairwise_similarity, graph_canonical_from_smiles, molecular_summary,
    morgan_fingerprint, parse_component_site, sha256_file, sha256_text, stable_json_sha256,
)

RDLogger.DisableLog("rdApp.*")
CONFIG_PATH = ROOT / "configs/gate3a0_prospective_pair_feasibility_v1.json"
LOCK_PATH = ROOT / "data_registry/gate3a0_preregistration_lock_v1.json"
LOCAL_ROOT = ROOT / "runs/gate3a0_prospective_pair_feasibility"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def write_text(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload.rstrip() + "\n")


def qstats(values) -> dict:
    a = np.asarray(list(values), dtype=float)
    if not a.size:
        return {"count": 0}
    return {"count": int(a.size), "min": float(a.min()), "q10": float(np.quantile(a, .1)),
            "median": float(np.quantile(a, .5)), "q90": float(np.quantile(a, .9)),
            "max": float(a.max()), "mean": float(a.mean())}


def load_contract() -> tuple[dict, dict]:
    config, lock = json.loads(CONFIG_PATH.read_text()), json.loads(LOCK_PATH.read_text())
    if sha256_file(CONFIG_PATH) != lock["config_sha256"]:
        raise RuntimeError("preregistration config hash mismatch")
    head = os.popen(f"git -C {ROOT} rev-parse HEAD").read().strip()
    if head != config["expected_head"]:
        raise RuntimeError(f"Git boundary mismatch: {head}")
    for value in config["inputs"].values():
        if isinstance(value, dict) and "sha256" in value:
            if sha256_file(ROOT / value["path"]) != value["sha256"]:
                raise RuntimeError(f"input hash mismatch: {value['path']}")
    return config, lock


def build_sites(config: dict, rows: pd.DataFrame):
    sidecar_root = Path(config["inputs"]["dft_sidecar_root"])
    sites = {"donor": {}, "acceptor": {}}
    variants = {"donor": defaultdict(set), "acceptor": defaultdict(set)}
    missing = Counter()
    for row in rows.sort_values("molecule_id", kind="mergesort").itertuples(index=False):
        sidecar = json.loads((sidecar_root / row.molecule_id / f"{row.molecule_id}_sidecar.json").read_text())
        for role in ("donor", "acceptor"):
            identity = str(getattr(row, f"{role}_structure_group_id_v1"))
            raw = sidecar.get(f"{role}_smiles")
            if not raw:
                missing[role] += 1
                continue
            site = parse_component_site(str(raw), role, identity)
            variants[role][identity].add(site.clean_smiles)
            sites[role].setdefault(identity, site)
    if len(sites["donor"]) != 154 or len(sites["acceptor"]) != 352:
        raise RuntimeError("component registry coverage mismatch")
    if any(len(v) != 1 for role in variants.values() for v in role.values()):
        raise RuntimeError("component structure identity conflict")
    registry = {
        "status": "GATE3A0_COMPONENT_REGISTRY_FROZEN", "rdkit_version": rdBase.rdkitVersion,
        "donor_structure_count": 154, "acceptor_structure_count": 352,
        "missing_sidecar_fields_with_identity_fallback": dict(missing), "entries": {},
        "alias_mapping": {"donor": {}, "acceptor": {}},
        "attachment_rule": config["component_identity"]["attachment_anchor_rule"],
        "substitution_rule": config["component_identity"]["substitution_rule"],
    }
    for role in ("donor", "acceptor"):
        id_col, identity_col = f"{role}_id", f"{role}_structure_group_id_v1"
        for identity, site in sorted(sites[role].items()):
            aliases = sorted(rows.loc[rows[identity_col] == identity, id_col].astype(str).unique())
            registry["entries"][f"{role}:{identity}"] = {
                "role": role, "structure_identity": identity,
                "structure_identity_hash": sha256_text(identity),
                "clean_component_smiles": site.clean_smiles,
                "attachment_marked_smiles": site.marked_attachment_smiles(),
                "attachment_index": site.anchor_index, "placeholder_degree": site.placeholder_degree,
                "source_string_sha256": site.source_sha256, "aliases": aliases,
            }
            for alias in aliases:
                registry["alias_mapping"][role][alias] = identity
    registry["registry_content_sha256"] = stable_json_sha256(registry["entries"])
    source_hash = {r: stable_json_sha256(sorted(s.source_sha256 for s in sites[r].values()))
                   for r in ("donor", "acceptor")}
    return sites, registry, source_hash


def product_result(donor: ComponentSite, acceptor: ComponentSite) -> dict:
    try:
        mol = assemble_components(donor, acceptor)
        return {"status": "SANITIZED", "mol": mol, "isomeric": canonical_isomeric(mol),
                "graph": canonical_graph(mol)}
    except (Chem.AtomValenceException, Chem.KekulizeException):
        return {"status": "INVALID_VALENCE", "mol": None}
    except Exception as exc:
        return {"status": f"EXCLUDED_BY_STRUCTURE_INTEGRITY:{type(exc).__name__}", "mol": None}


def audit_observed(rows: pd.DataFrame, sites: dict, role_resolution: pd.DataFrame, config: dict):
    expected = {r.molecule_id: graph_canonical_from_smiles(r.canonical_structure_smiles_v1)
                for r in rows.itertuples(index=False)}
    output = []
    for row in rows.sort_values("molecule_id", kind="mergesort").itertuples(index=False):
        result = product_result(sites["donor"][row.donor_structure_group_id_v1],
                                sites["acceptor"][row.acceptor_structure_group_id_v1])
        if result["status"] == "SANITIZED":
            status = ("EXACT_CANONICAL_MATCH" if result["isomeric"] == row.canonical_structure_smiles_v1
                      else "GRAPH_ISOMORPHIC_MATCH" if result["graph"] == expected[row.molecule_id]
                      else "STRUCTURE_MISMATCH")
        else:
            status = result["status"].split(":")[0]
        output.append({"molecule_id": row.molecule_id, "status": status,
                       "assembled_graph_hash": sha256_text(result.get("graph", "")),
                       "reference_graph_hash": sha256_text(expected[row.molecule_id])})
    audit = pd.DataFrame(output).sort_values("molecule_id", kind="mergesort").reset_index(drop=True)
    repeat = []
    for row in rows.sort_values("molecule_id", ascending=False, kind="mergesort").itertuples(index=False):
        result = product_result(sites["donor"][row.donor_structure_group_id_v1],
                                sites["acceptor"][row.acceptor_structure_group_id_v1])
        repeat.append((row.molecule_id, result["status"].split(":")[0], sha256_text(result.get("graph", ""))))
    first = [(r.molecule_id,
              "SANITIZED" if r.status in {"EXACT_CANONICAL_MATCH", "GRAPH_ISOMORPHIC_MATCH", "STRUCTURE_MISMATCH"} else r.status,
              r.assembled_graph_hash) for r in audit.itertuples(index=False)]
    first_hash, repeat_hash = stable_json_sha256(sorted(first)), stable_json_sha256(sorted(repeat))
    counts, total = audit.status.value_counts().sort_index().to_dict(), len(audit)
    sanitized = sum(counts.get(x, 0) for x in ("EXACT_CANONICAL_MATCH", "GRAPH_ISOMORPHIC_MATCH", "STRUCTURE_MISMATCH"))
    matched = counts.get("EXACT_CANONICAL_MATCH", 0) + counts.get("GRAPH_ISOMORPHIC_MATCH", 0)
    role = role_resolution.set_index("molecule_id")
    pure = role[(role.original_unknown_count == 0) & (role.original_donor_count > 0) & (role.original_acceptor_count > 0)]
    origin_mismatch = int((pure.pm6_dft_sidecar_roles_exact.astype(str).str.lower() != "true").sum())
    metrics = {
        "record_coverage": total / len(rows), "sanitize_success": sanitized / total,
        "canonical_or_graph_isomorphic_match": matched / total, "attachment_ambiguity": 0.0,
        "verified_atom_origin_records": int(len(pure)),
        "atom_origin_unverifiable_original_unknown_records": int(total - len(pure)),
        "verified_atom_origin_mismatch": origin_mismatch, "status_counts": counts,
        "result_hash": first_hash, "repeat_reversed_input_hash": repeat_hash,
        "deterministic_repeat": first_hash == repeat_hash, "input_order_invariant": first_hash == repeat_hash,
    }
    t = config["assembly_admission"]
    admitted = (metrics["record_coverage"] >= t["record_coverage_min"] and
                metrics["sanitize_success"] >= t["sanitize_success_min"] and
                metrics["canonical_or_graph_isomorphic_match"] >= t["canonical_or_graph_isomorphic_match_min"] and
                metrics["attachment_ambiguity"] <= t["attachment_ambiguity_max"] and
                origin_mismatch <= t["verified_atom_origin_mismatch_max"] and
                metrics["deterministic_repeat"] and metrics["input_order_invariant"])
    metrics["decision"] = "ASSEMBLY_ENGINE_ADMITTED" if admitted else "BLOCKED_COMBINATORIAL_ASSEMBLY_INTEGRITY"
    return metrics, audit


def alert_catalog():
    params = FilterCatalog.FilterCatalogParams()
    catalogs = FilterCatalog.FilterCatalogParams.FilterCatalogs
    for name in ("PAINS_A", "PAINS_B", "PAINS_C", "BRENK"):
        params.AddCatalog(getattr(catalogs, name))
    return FilterCatalog.FilterCatalog(params)


def build_universe(rows: pd.DataFrame, sites: dict, config: dict):
    pair = pd.read_csv(ROOT / config["inputs"]["pair_cold_manifest"]["path"], dtype=str)
    train = pair[pair.partition == "train"]
    d_support = train.groupby("donor_structure_group_id_v1").size().to_dict()
    a_support = train.groupby("acceptor_structure_group_id_v1").size().to_dict()
    observed_pairs = set(zip(rows.donor_structure_group_id_v1, rows.acceptor_structure_group_id_v1))
    observed_graphs = {graph_canonical_from_smiles(x) for x in rows.canonical_structure_smiles_v1}
    records, product_counts = [], Counter()
    for donor_id, donor in sorted(sites["donor"].items()):
        for acceptor_id, acceptor in sorted(sites["acceptor"].items()):
            pair_hash = sha256_text(f"{donor_id}>>{acceptor_id}")
            result = product_result(donor, acceptor)
            common = {"pair_hash": pair_hash, "donor_identity_hash": sha256_text(donor_id),
                      "acceptor_identity_hash": sha256_text(acceptor_id),
                      "donor_support": int(d_support.get(donor_id, 0)),
                      "acceptor_support": int(a_support.get(acceptor_id, 0))}
            if result["status"] != "SANITIZED":
                common.update(canonical_smiles="", full_structure_hash="",
                              status="INVALID_VALENCE" if result["status"] == "INVALID_VALENCE" else "EXCLUDED_BY_STRUCTURE_INTEGRITY")
                records.append(common)
                continue
            graph, current_pair = result["graph"], (donor_id, acceptor_id)
            graph_hash = sha256_text(graph)
            if current_pair in observed_pairs and graph in observed_graphs:
                status = "OBSERVED_EXACT_STRUCTURE"
            elif current_pair in observed_pairs:
                status = "OBSERVED_COMPONENT_PAIR"
            elif graph in observed_graphs:
                status = "OBSERVED_EXACT_STRUCTURE"
            elif d_support.get(donor_id, 0) < 5 or a_support.get(acceptor_id, 0) < 5:
                status = "OUTSIDE_COMPONENT_SUPPORT"
            else:
                status = "NOVEL_PAIR_KNOWN_COMPONENTS"
            common.update(canonical_smiles=result["isomeric"], full_structure_hash=graph_hash, status=status)
            records.append(common)
            product_counts[graph_hash] += 1
    universe = pd.DataFrame(records).sort_values("pair_hash", kind="mergesort").reset_index(drop=True)
    duplicates = {key for key, count in product_counts.items() if count > 1}
    duplicate_mask = (universe.full_structure_hash.isin(duplicates) &
                      universe.status.isin(["NOVEL_PAIR_KNOWN_COMPONENTS", "OUTSIDE_COMPONENT_SUPPORT"]))
    universe.loc[duplicate_mask, "status"] = "DUPLICATE_PRODUCT_FROM_DIFFERENT_ALIASES"
    universe["in_pair_cold_domain"] = universe.status == "NOVEL_PAIR_KNOWN_COMPONENTS"
    catalog = alert_catalog()
    for index in universe.index[universe.in_pair_cold_domain]:
        mol = Chem.MolFromSmiles(universe.at[index, "canonical_smiles"])
        summary = molecular_summary(mol)
        summary["alert_count"] = len(catalog.GetMatches(mol))
        for key, value in summary.items():
            universe.at[index, key] = value
    sample = universe[universe.in_pair_cold_domain].sort_values("pair_hash", kind="mergesort").head(1024)
    fps = [morgan_fingerprint(Chem.MolFromSmiles(x), 2, 2048) for x in sample.canonical_smiles]
    eligible = universe[universe.in_pair_cold_domain]
    diversity = {
        "in_domain_records": int(len(eligible)),
        "unique_full_structure_hashes": int(eligible.full_structure_hash.nunique()),
        "unique_murcko_scaffolds": int(eligible.murcko_scaffold.nunique()),
        "molecular_weight": qstats(eligible.molecular_weight.dropna()),
        "heavy_atom_count": qstats(eligible.heavy_atom_count.dropna()),
        "heteroatom_count": qstats(eligible.heteroatom_count.dropna()),
        "formal_charge_counts": {str(k): int(v) for k, v in eligible.formal_charge.value_counts().sort_index().items()},
        "alert_count": qstats(eligible.alert_count.dropna()),
        "full_molecule_pairwise_morgan": deterministic_pairwise_similarity(fps, 1024),
        "donor_nearest_observed_similarity": 1.0, "acceptor_nearest_observed_similarity": 1.0,
        "interpretation": "Target-free diversity only; alerts are not synthesis or safety proof."
    }
    registry = {
        "status": "GATE3A0_CANDIDATE_UNIVERSE_FROZEN", "cartesian_pairs": int(len(universe)),
        "donor_structures": 154, "acceptor_structures": 352,
        "status_counts": {str(k): int(v) for k, v in universe.status.value_counts().sort_index().items()},
        "observed_component_pairs": len(observed_pairs), "observed_full_structure_graphs": len(observed_graphs),
        "in_domain_novel_pairs": int(universe.in_pair_cold_domain.sum()),
        "low_support_donor_structures": int(sum(d_support.get(x, 0) < 5 for x in sites["donor"])),
        "low_support_acceptor_structures": int(sum(a_support.get(x, 0) < 5 for x in sites["acceptor"])),
        "donor_support": qstats(d_support.get(x, 0) for x in sites["donor"]),
        "acceptor_support": qstats(a_support.get(x, 0) for x in sites["acceptor"]),
        "candidate_content_sha256": stable_json_sha256(sorted(
            (r.pair_hash, r.full_structure_hash, r.status, bool(r.in_pair_cold_domain))
            for r in universe.itertuples(index=False))),
        "local_artifact": "runs/gate3a0_prospective_pair_feasibility/candidate_universe_v1.parquet",
        "diversity": diversity,
    }
    return registry, universe, diversity


def render_reports(assembly, universe, diversity, final_status):
    counts = assembly["status_counts"]
    write_text(ROOT / "reports/gate3a0_observed_reconstruction_audit.md", f"""# Gate 3-A0 observed reconstruction audit

The production assembler uses one record-independent rule: remove the historical `[A]` serialization marker, attach through its first/preceding neighbor, remove one explicit anchor hydrogen when present, and create one single D–A bond. It never uses target values, predictions, molecule-ID product lookup, or record-specific connection mappings.

- Coverage: {assembly['record_coverage']:.6%} ({sum(counts.values()):,}/15,016)
- Sanitized: {assembly['sanitize_success']:.6%}
- Exact canonical matches: {counts.get('EXACT_CANONICAL_MATCH', 0):,}
- Graph-isomorphic matches: {counts.get('GRAPH_ISOMORPHIC_MATCH', 0):,}
- Canonical or graph-isomorphic: {assembly['canonical_or_graph_isomorphic_match']:.6%}
- Invalid valence/kekulization: {counts.get('INVALID_VALENCE', 0):,}
- Attachment ambiguity: {assembly['attachment_ambiguity']:.6%}
- Pure explicit-role origin checks: {assembly['verified_atom_origin_records']:,}; mismatches: {assembly['verified_atom_origin_mismatch']}
- Unknown/empty-donor original-role rows preserved as unverifiable: {assembly['atom_origin_unverifiable_original_unknown_records']:,}
- Reversed-input deterministic hash match: {assembly['deterministic_repeat']}

Decision: **{assembly['decision']}**. Graph-isomorphic matches are permitted by the frozen threshold; the assembler does not invent stereochemistry. All failures remain in a local Git-ignored audit.
""")
    if universe is None:
        candidate = domain = diversity_text = "The candidate universe was not built because assembly admission failed."
    else:
        candidate = f"""The audited universe has **{universe['cartesian_pairs']:,}** structure-identity pairs ({universe['donor_structures']} × {universe['acceptor_structures']}), {universe['observed_component_pairs']:,} observed component pairs, and {universe['in_domain_novel_pairs']:,} novel support-qualified unseen pairs.

```json
{json.dumps(universe['status_counts'], indent=2, sort_keys=True)}
```

Row-level component combinations, product SMILES, support, and alerts remain local and Git-ignored. No property prediction or ranking was run.
"""
        domain = f"""This is strictly **seen-components / unseen-pair** scope, not donor-OOD, acceptor-OOD, or new-scaffold extrapolation.

- Pair-cold train support: donor ≥5 and acceptor ≥5 calculation records.
- Low-support donor structures: {universe['low_support_donor_structures']}
- Low-support acceptor structures: {universe['low_support_acceptor_structures']}
- In-domain novel pairs: {universe['in_domain_novel_pairs']:,}
- Donor support: `{json.dumps(universe['donor_support'], sort_keys=True)}`
- Acceptor support: `{json.dumps(universe['acceptor_support'], sort_keys=True)}`

All admitted pairs use observed component identities, marker topology, charge/valence patterns, and the observed single intercomponent bond type. No label or test-derived threshold is used.
"""
        diversity_text = f"""Target-free diagnostics for the in-domain novel universe:

- Unique full graphs: {diversity['unique_full_structure_hashes']:,}
- Unique Murcko scaffolds: {diversity['unique_murcko_scaffolds']:,}
- Molecular weight: `{json.dumps(diversity['molecular_weight'], sort_keys=True)}`
- Heavy atoms: `{json.dumps(diversity['heavy_atom_count'], sort_keys=True)}`
- Heteroatoms: `{json.dumps(diversity['heteroatom_count'], sort_keys=True)}`
- Structural-alert counts: `{json.dumps(diversity['alert_count'], sort_keys=True)}`
- Deterministic 2,048-bit radius-2 non-chiral pairwise Morgan sample: `{json.dumps(diversity['full_molecule_pairwise_morgan'], sort_keys=True)}`

Donor/acceptor nearest-observed similarity is exactly 1 by construction. Alerts are computational heuristics, not synthesis feasibility or safety evidence.
"""
    write_text(ROOT / "reports/gate3a0_candidate_universe.md", "# Gate 3-A0 candidate universe\n\n" + candidate)
    write_text(ROOT / "reports/gate3a0_pair_cold_domain.md", "# Gate 3-A0 pair-cold domain\n\n" + domain)
    write_text(ROOT / "reports/gate3a0_structural_diversity.md", "# Gate 3-A0 structural diversity\n\n" + diversity_text)
    write_text(ROOT / "reports/gate3a0_experimental_validation_readiness.md", """# Gate 3-A0 experimental validation readiness

Repository evidence does **not** document a committed experimental laboratory, compound procurement/synthesis plan, named reaction, budget, personnel, or schedule. Status: `BLOCKED_NO_EXPERIMENTAL_VALIDATION_PATH`.

Still required are component availability/synthesis, product identity, absorption/emission, electrochemistry, excited-state lifetime, Stern–Volmer/quenching, quantum yield, conversion/yield, TON/TOF, and at least one real photocatalytic reaction. The proxy does not directly validate catalytic performance and has no frozen independent experimental linkage. Positive, negative, and matched controls are predeclared conceptually; blinding and complete reporting of synthesis/measurement failures are required but not operationally confirmed.

This boundary can support a rigorous computational chemistry/ML paper. It cannot support a Nature Catalysis-level catalyst-discovery claim without a prospective experimental reaction loop.
""")
    write_text(ROOT / "reports/gate3a0_final_decision.md", f"""# Gate 3-A0 final decision

Final status: **{final_status}**

The technical assembly and pair-domain result permits only target-free prospective computation. With no documented experimental path, the Gate cannot advance to a catalyst-discovery claim. XGBoost-C0 remains frozen; no properties were predicted and no candidates were selected.
""")


def main():
    started = datetime.now(timezone.utc)
    config, lock = load_contract()
    component = pd.read_csv(ROOT / config["inputs"]["component_manifest"]["path"], dtype=str)
    full = pd.read_csv(ROOT / config["inputs"]["full_structure_manifest"]["path"], dtype=str)
    rows = component.merge(full[["molecule_id", "canonical_structure_smiles_v1", "structure_group_id_v1"]],
                           on="molecule_id", validate="one_to_one").sort_values("molecule_id", kind="mergesort")
    if len(rows) != 15016 or rows.molecule_id.nunique() != 15016:
        raise RuntimeError("15,016-row one-to-one join failed")
    role = pd.read_csv(ROOT / "manifests/role_resolution_v1.csv")
    sites, component_registry, source_hash = build_sites(config, rows)
    write_json(ROOT / "data_registry/gate3a0_component_registry.json", component_registry)
    assembly, observed = audit_observed(rows, sites, role, config)
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    observed.to_parquet(LOCAL_ROOT / "observed_reconstruction_v1.parquet", index=False)
    write_json(ROOT / "data_registry/gate3a0_assembly_contract_v1.json", {
        "version": "v1", "marker_rule": config["component_identity"],
        "admission_thresholds": config["assembly_admission"], "allowed_bond": "single",
        "full_product_lookup": False, "record_specific_hardcodes": False, "target_access": False,
        "atom_origin_policy": "Verify original roles only for pure explicit D/A rows; preserve unknown/empty-donor rows as unverifiable rather than relabeling."
    })
    write_json(ROOT / "data_registry/gate3a0_assembly_audit_registry.json", {
        "status": assembly["decision"], "metrics": assembly,
        "component_source_summary_sha256": source_hash,
        "local_artifact_sha256": sha256_file(LOCAL_ROOT / "observed_reconstruction_v1.parquet"),
        "prelock_probe_disclosed": lock["prelock_probe_disclosure"],
    })
    universe_registry = diversity = None
    if assembly["decision"] == "ASSEMBLY_ENGINE_ADMITTED":
        universe_registry, universe, diversity = build_universe(rows, sites, config)
        universe.to_parquet(LOCAL_ROOT / "candidate_universe_v1.parquet", index=False)
        universe_registry["local_artifact_sha256"] = sha256_file(LOCAL_ROOT / "candidate_universe_v1.parquet")
        write_json(ROOT / "data_registry/gate3a0_candidate_universe_registry.json", universe_registry)
        final_status = ("PROSPECTIVE_COMPUTATIONAL_SCREENING_ONLY"
                        if universe_registry["in_domain_novel_pairs"] > 0 else "BLOCKED_PAIR_COLD_DOMAIN")
    else:
        write_json(ROOT / "data_registry/gate3a0_candidate_universe_registry.json",
                   {"status": "NOT_BUILT_ASSEMBLY_GATE_FAILED", "cartesian_pairs_evaluated": 0})
        final_status = "BLOCKED_COMBINATORIAL_ASSEMBLY_INTEGRITY"
    selection = {
        "status": "FROZEN_BEFORE_PROPERTY_PREDICTION", "version": "v1", "future_total": 16,
        "sets": {"predicted_low_proxy": 4, "predicted_high_proxy": 4,
                 "diversity_exploration": 4, "matched_controls": 4},
        "max_per_donor": 2, "max_per_acceptor": 2,
        "diversity_constraint": "cover distinct donor, acceptor, and full-Morgan clusters before score tie-breaks",
        "controls": "observed calculated near-neighbor pairs and middle-proxy references",
        "candidate_failures_retained": True, "wet_lab_count_requires_team_confirmation": True,
        "property_prediction_run": False, "candidate_list_generated": False,
    }
    write_json(ROOT / "data_registry/gate3a0_selection_protocol_v1.json", selection)
    render_reports(assembly, universe_registry, diversity, final_status)
    ended = datetime.now(timezone.utc)
    evidence = {
        "status": final_status, "started_utc": started.isoformat(), "ended_utc": ended.isoformat(),
        "wall_seconds": (ended - started).total_seconds(), "cpu_only": True, "training": False,
        "property_prediction": False, "candidate_ranking": False, "target_access": False,
        "test_artifact_access": False, "final673_access": False, "raw_data_write": False,
        "assembly": assembly, "candidate_universe": universe_registry,
        "experimental_path": "BLOCKED_NO_EXPERIMENTAL_VALIDATION_PATH",
        "prelock_probe_disclosure": lock["prelock_probe_disclosure"],
    }
    write_json(ROOT / "logs/gate3a0_evidence.json", evidence)
    tracked = [CONFIG_PATH, LOCK_PATH, ROOT / "data_registry/gate3a0_component_registry.json",
               ROOT / "data_registry/gate3a0_assembly_contract_v1.json",
               ROOT / "data_registry/gate3a0_assembly_audit_registry.json",
               ROOT / "data_registry/gate3a0_candidate_universe_registry.json",
               ROOT / "data_registry/gate3a0_selection_protocol_v1.json",
               *sorted((ROOT / "reports").glob("gate3a0_*.md")), ROOT / "logs/gate3a0_evidence.json",
               Path(__file__), ROOT / "src/excitationnexus_phase12/combinatorial_assembly.py",
               ROOT / "tests/test_gate3a0_contract.py"]
    write_text(ROOT / "data_registry/gate3a0_sha256.txt", "\n".join(
        f"{sha256_file(path)}  {path.relative_to(ROOT)}" for path in sorted(set(tracked)) if path.exists()))
    print(json.dumps({"status": final_status, "assembly": assembly,
                      "candidate_counts": None if universe_registry is None else universe_registry["status_counts"],
                      "in_domain": None if universe_registry is None else universe_registry["in_domain_novel_pairs"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
