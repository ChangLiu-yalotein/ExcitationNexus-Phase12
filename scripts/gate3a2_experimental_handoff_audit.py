#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, FilterCatalog, Lipinski, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Contrib.SA_Score import sascorer

from excitationnexus_phase12.combinatorial_assembly import canonical_graph, sha256_text
from excitationnexus_phase12.prospective_scoring import sha256, write_json

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")
CONFIG = ROOT / "configs/gate3a2_experimental_handoff_audit_v1.json"
LOCK = ROOT / "data_registry/gate3a2_preregistration_lock_v1.json"


def resolve(value: str) -> Path:
    return ROOT / value


def utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_value(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_config() -> dict:
    return json.loads(CONFIG.read_text())


def verify_inputs(config: dict) -> None:
    for item in config["inputs"].values():
        path = resolve(item["path"])
        if not path.is_file() or sha256(path) != item["sha256"]:
            raise RuntimeError(f"frozen input hash mismatch: {item['path']}")
    if git_value("rev-parse", "HEAD") != config["base_git_head"] or git_value("rev-parse", "origin/main") != config["base_git_head"]:
        raise RuntimeError("Gate 3-A1 Git boundary mismatch")
    if git_value("rev-parse", f"{config['required_tag']}^{{}}") != config["base_git_head"]:
        raise RuntimeError("Gate 3-A1 tag mismatch")
    shortlist = json.loads(resolve(config["inputs"]["shortlist_registry"]["path"]).read_text())
    if shortlist["status"] != "COMPUTATIONAL_SHORTLIST_FROZEN" or len(shortlist["items"]) != 16:
        raise RuntimeError("Gate 3-A1 shortlist registry failure")
    scoring = json.loads(resolve(config["inputs"]["scoring_unlock"]["path"]).read_text())
    if scoring["status"] != "CANDIDATE_SCORING_CONSUMED_ONCE" or scoring["second_invocation"] != "FAIL_CLOSED":
        raise RuntimeError("Gate 3-A1 scoring lock failure")
    local = pd.read_parquet(resolve(config["inputs"]["local_shortlist"]["path"]))
    if len(local) != 16 or local.full_structure_hash.nunique() != 16:
        raise RuntimeError("local shortlist integrity failure")
    if local.category.value_counts().to_dict() != config["shortlist_contract"]["categories"]:
        raise RuntimeError("shortlist category boundary changed")
    public_pairs = {x["anonymous_pair_hash"] for x in shortlist["items"]}
    if public_pairs != set(local.pair_hash.astype(str)):
        raise RuntimeError("public/local shortlist identity mismatch")


def preregister(config: dict) -> None:
    verify_inputs(config)
    allowed_changes = {
        "configs/gate3a2_experimental_handoff_audit_v1.json",
        "scripts/gate3a2_experimental_handoff_audit.py",
        "tests/test_gate3a2_contract.py",
    }
    current = {line[3:] for line in git_value("status", "--porcelain").splitlines()}
    if not current.issubset(allowed_changes):
        raise RuntimeError(f"unexpected preregistration worktree changes: {sorted(current - allowed_changes)}")
    locked = {
        str(path.relative_to(ROOT)): sha256(path)
        for path in (CONFIG, ROOT / "scripts/gate3a2_experimental_handoff_audit.py", ROOT / "tests/test_gate3a2_contract.py")
    }
    aggregate = hashlib.sha256("".join(f"{k}:{v}\n" for k, v in sorted(locked.items())).encode()).hexdigest()
    shortlist = pd.read_parquet(resolve(config["inputs"]["local_shortlist"]["path"]))
    content_hash = hashlib.sha256("\n".join(sorted(shortlist.pair_hash.astype(str))).encode()).hexdigest()
    write_json(LOCK, {
        "status": "FROZEN_BEFORE_CANDIDATE_CHEMICAL_AUDIT",
        "locked_utc": utc(), "files": locked, "aggregate_sha256": aggregate,
        "shortlist_count": 16, "shortlist_pair_set_sha256": content_hash,
        "replacement_forbidden": True, "rescoring_forbidden": True,
        "external_candidate_structure_queries": False,
        "test_access": False, "final673_access": False,
        "post_lock_policy": "Any scientific rule change requires explicit v2; v1 is never overwritten.",
    })
    (ROOT / "reports/gate3a2_preregistration.md").write_text(
        "# Gate 3-A2 preregistration\n\n"
        "Status: **FROZEN BEFORE CANDIDATE CHEMICAL AUDIT**. The 16 Gate 3-A1 items and four categories "
        "are immutable: audit failures are retained and never replaced from the 36,523-pair universe. "
        "No model, score, threshold, prediction, test artifact, or final673 asset is opened.\n\n"
        "PAINS/BRENK, RDKit SA score, tautomer/protonation/stereochemistry flags, and route-family suggestions "
        "are heuristics for chemist review, not synthesis, safety, availability, or activity evidence. "
        "Candidate structures are not transmitted to external search services; literature and commercial "
        "status therefore remain SOURCE_UNVERIFIED unless a stable pre-existing repository source exists.\n"
    )
    print(json.dumps({"status": "FROZEN_BEFORE_CANDIDATE_CHEMICAL_AUDIT", "shortlist": 16}, indent=2))


def filter_catalog(catalog_name: str):
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(getattr(FilterCatalog.FilterCatalogParams.FilterCatalogs, catalog_name))
    return FilterCatalog.FilterCatalog(params)


def entry_lookup(component_registry: dict) -> dict:
    result = {}
    for entry in component_registry["entries"].values():
        result[(entry["role"], entry["structure_identity_hash"])] = entry
    return result


def mapped_assembly(donor: dict, acceptor: dict) -> tuple[Chem.Mol, dict]:
    dm = Chem.MolFromSmiles(donor["clean_component_smiles"])
    am = Chem.MolFromSmiles(acceptor["clean_component_smiles"])
    if dm is None or am is None:
        raise RuntimeError("frozen component no longer parses")
    for i, atom in enumerate(dm.GetAtoms(), start=1):
        atom.SetAtomMapNum(i)
    offset = dm.GetNumAtoms()
    for i, atom in enumerate(am.GetAtoms(), start=offset + 1):
        atom.SetAtomMapNum(i)
    rw = Chem.RWMol(Chem.CombineMols(dm, am))
    d_anchor = int(donor["attachment_index"])
    a_anchor = offset + int(acceptor["attachment_index"])
    for index in (d_anchor, a_anchor):
        atom = rw.GetAtomWithIdx(index)
        if atom.GetNumExplicitHs() > 0:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
    rw.AddBond(d_anchor, a_anchor, Chem.BondType.SINGLE)
    product = rw.GetMol()
    product.ClearComputedProps()
    Chem.SanitizeMol(product)
    disconnected = Chem.MolToSmiles(Chem.CombineMols(dm, am), canonical=True, isomericSmiles=True)
    mapped_product = Chem.MolToSmiles(product, canonical=True, isomericSmiles=True)
    d_atom, a_atom = product.GetAtomWithIdx(d_anchor), product.GetAtomWithIdx(a_anchor)
    local = {
        "donor_anchor_map": int(d_atom.GetAtomMapNum()),
        "acceptor_anchor_map": int(a_atom.GetAtomMapNum()),
        "donor_anchor_element": d_atom.GetSymbol(),
        "acceptor_anchor_element": a_atom.GetSymbol(),
        "donor_anchor_aromatic": bool(d_atom.GetIsAromatic()),
        "acceptor_anchor_aromatic": bool(a_atom.GetIsAromatic()),
        "donor_anchor_hybridization": str(d_atom.GetHybridization()),
        "acceptor_anchor_hybridization": str(a_atom.GetHybridization()),
        "mapped_disconnection": f"{mapped_product}>>{disconnected}",
    }
    return product, local


def route_families(local: dict) -> list[str]:
    d, a = local["donor_anchor_element"], local["acceptor_anchor_element"]
    aromatic = local["donor_anchor_aromatic"] or local["acceptor_anchor_aromatic"]
    pair = {d, a}
    if pair == {"C"}:
        routes = ["C-C cross-coupling"]
        if aromatic:
            routes.append("Suzuki-type coupling")
        return routes
    if pair == {"C", "N"}:
        return ["Buchwald-Hartwig-type C-N coupling", "SNAr"]
    if pair == {"C", "O"}:
        return ["C-O cross-coupling", "SNAr"]
    return ["bond-forming route unresolved"]


def chemical_flags(mol: Chem.Mol, pains, brenk) -> dict:
    radicals = int(sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()))
    fragments = len(Chem.GetMolFrags(mol))
    chiral = Chem.FindMolChiralCenters(mol, includeUnassigned=True, useLegacyImplementation=False)
    unassigned_chiral = sum(label == "?" for _, label in chiral)
    stereo_bonds = sum(
        bond.GetBondType() == Chem.BondType.DOUBLE and bond.GetStereo() == Chem.BondStereo.STEREONONE
        for bond in mol.GetBonds()
    )
    enumerator = rdMolStandardize.TautomerEnumerator()
    enumerator.SetMaxTautomers(100)
    tautomer_count = len(enumerator.Enumerate(mol))
    pains_matches = [x.GetDescription() for x in pains.GetMatches(mol)]
    brenk_matches = [x.GetDescription() for x in brenk.GetMatches(mol)]
    sa = float(sascorer.calculateScore(mol))
    return {
        "sanitize": True,
        "formal_charge": int(Chem.GetFormalCharge(mol)),
        "radical_electrons": radicals,
        "fragment_count": fragments,
        "unassigned_chiral_centers": int(unassigned_chiral),
        "unspecified_double_bond_stereo_count": int(stereo_bonds),
        "tautomer_count_capped_100": int(tautomer_count),
        "protonation_site_heuristic": int(Lipinski.NumHDonors(mol) + Lipinski.NumHAcceptors(mol)),
        "pains_count": len(pains_matches),
        "brenk_count": len(brenk_matches),
        "pains_descriptions": ";".join(pains_matches),
        "brenk_descriptions": ";".join(brenk_matches),
        "sa_score_heuristic": sa,
        "molecular_weight": float(Descriptors.MolWt(mol)),
        "heavy_atoms": int(mol.GetNumHeavyAtoms()),
        "heteroatoms": int(rdMolDescriptors.CalcNumHeteroatoms(mol)),
        "rings": int(rdMolDescriptors.CalcNumRings(mol)),
        "rotatable_bonds": int(rdMolDescriptors.CalcNumRotatableBonds(mol)),
        "aromatic_atoms": int(sum(atom.GetIsAromatic() for atom in mol.GetAtoms())),
    }


def audit(config: dict) -> None:
    if not LOCK.is_file():
        raise RuntimeError("Gate 3-A2 preregistration lock missing")
    lock = json.loads(LOCK.read_text())
    if sha256(CONFIG) != lock["files"]["configs/gate3a2_experimental_handoff_audit_v1.json"]:
        raise RuntimeError("Gate 3-A2 config changed after lock")
    verify_inputs(config)
    output = resolve(config["local_output"])
    if output.exists():
        raise RuntimeError("Gate 3-A2 local audit already exists")
    shortlist = pd.read_parquet(resolve(config["inputs"]["local_shortlist"]["path"])).sort_values("pair_hash", kind="mergesort")
    component = json.loads(resolve(config["inputs"]["component_registry"]["path"]).read_text())
    lookup = entry_lookup(component)
    universe = pd.read_parquet(resolve(config["inputs"]["candidate_universe"]["path"]), columns=["full_structure_hash", "status"])
    observed_hashes = set(universe.loc[universe.status.eq("OBSERVED_EXACT_STRUCTURE"), "full_structure_hash"].astype(str))
    pains = filter_catalog("PAINS")
    brenk = filter_catalog("BRENK")
    records = []
    for row in shortlist.itertuples(index=False):
        mol = Chem.MolFromSmiles(str(row.canonical_smiles))
        if mol is None:
            raise RuntimeError("shortlist molecule no longer sanitizes")
        flags = chemical_flags(mol, pains, brenk)
        graph_hash = sha256_text(canonical_graph(mol))
        if graph_hash != row.full_structure_hash:
            raise RuntimeError("shortlist full-structure hash mismatch")
        donor = lookup.get(("donor", str(row.donor_identity_hash)))
        acceptor = lookup.get(("acceptor", str(row.acceptor_identity_hash)))
        if donor is None or acceptor is None:
            raise RuntimeError("component identity hash does not bind to frozen registry")
        novel = row.category != "matched_observed_control"
        observed_overlap = row.full_structure_hash in observed_hashes
        if novel == observed_overlap:
            raise RuntimeError("novel/control observed-overlap contract failure")
        local = {
            "donor_component_source_sha256": donor["source_string_sha256"],
            "acceptor_component_source_sha256": acceptor["source_string_sha256"],
            "donor_clean_component_smiles": donor["clean_component_smiles"],
            "acceptor_clean_component_smiles": acceptor["clean_component_smiles"],
            "canonical_smiles": row.canonical_smiles,
            "mapped_disconnection": "",
            "route_family_suggestions": "",
            "route_status": "NOT_APPLICABLE_CONTROL" if not novel else "ROUTE_UNRESOLVED_REQUIRES_CHEMIST_REVIEW",
        }
        assembly_match = True
        if novel:
            product, mapped = mapped_assembly(donor, acceptor)
            clean = Chem.Mol(product)
            for atom in clean.GetAtoms():
                atom.SetAtomMapNum(0)
            assembly_match = sha256_text(canonical_graph(clean)) == row.full_structure_hash
            if not assembly_match:
                raise RuntimeError("mapped assembly changed frozen full structure")
            local.update(mapped)
            local["route_family_suggestions"] = ";".join(route_families(mapped)[:3])
        structural_risk = (
            not assembly_match or flags["radical_electrons"] != 0 or flags["fragment_count"] != 1
            or not math.isfinite(flags["sa_score_heuristic"])
        )
        if not novel:
            classification = "COMPUTATIONAL_CONTROL_ONLY"
        elif structural_risk:
            classification = "STRUCTURAL_RISK"
        else:
            classification = "ROUTE_UNRESOLVED"
        records.append({
            "pair_hash": row.pair_hash, "full_structure_hash": row.full_structure_hash,
            "category": row.category, "classification": classification,
            "novel_seen_component_pair": novel, "observed_structure_overlap": observed_overlap,
            "assembly_match": assembly_match, "donor_identity_hash": row.donor_identity_hash,
            "acceptor_identity_hash": row.acceptor_identity_hash,
            "donor_source_status": "OBSERVED_COMPONENT;SOURCE_UNVERIFIED",
            "acceptor_source_status": "OBSERVED_COMPONENT;SOURCE_UNVERIFIED",
            **flags, **local,
        })
    result = pd.DataFrame(records)
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output, index=False)
    if len(result) != 16 or result.pair_hash.nunique() != 16 or result.full_structure_hash.nunique() != 16:
        raise RuntimeError("candidate audit coverage failure")
    public = []
    for index, row in enumerate(result.sort_values(["category", "pair_hash"], kind="mergesort").itertuples(index=False), start=1):
        public.append({
            "audit_id": f"G3A2-{index:02d}", "anonymous_pair_hash": row.pair_hash,
            "category": row.category, "classification": row.classification,
            "assembly_match": bool(row.assembly_match), "observed_structure_overlap": bool(row.observed_structure_overlap),
            "pains_alert": bool(row.pains_count), "brenk_alert": bool(row.brenk_count),
            "source_status": "SOURCE_UNVERIFIED",
        })
    classifications = result.classification.value_counts().sort_index().to_dict()
    route_counter = Counter()
    for text in result.loc[result.novel_seen_component_pair, "route_family_suggestions"]:
        route_counter.update(x for x in text.split(";") if x)
    risk = {
        "pains_positive": int((result.pains_count > 0).sum()),
        "brenk_positive": int((result.brenk_count > 0).sum()),
        "radical_positive": int((result.radical_electrons > 0).sum()),
        "multi_fragment": int((result.fragment_count > 1).sum()),
        "unassigned_stereo_positive": int(((result.unassigned_chiral_centers + result.unspecified_double_bond_stereo_count) > 0).sum()),
        "tautomer_multiple": int((result.tautomer_count_capped_100 > 1).sum()),
        "sa_score": {q: float(np.quantile(result.sa_score_heuristic, float(q))) for q in ("0", "0.5", "0.9", "1")},
        "molecular_weight": {q: float(np.quantile(result.molecular_weight, float(q))) for q in ("0", "0.5", "0.9", "1")},
    }
    write_json(ROOT / "data_registry/gate3a2_candidate_audit_registry.json", {
        "status": "SHORTLIST_CHEMICAL_AUDIT_FROZEN", "candidate_count": 16,
        "novel_candidates": 12, "observed_controls": 4, "classifications": classifications,
        "risk_counts": risk, "route_family_summary": dict(sorted(route_counter.items())),
        "items": public, "local_artifact": config["local_output"],
        "local_artifact_sha256": sha256(output), "smiles_published": False,
        "mapped_disconnections_published": False, "heuristic_only": True,
    })
    unique_donors = result.donor_identity_hash.nunique()
    unique_acceptors = result.acceptor_identity_hash.nunique()
    write_json(ROOT / "data_registry/gate3a2_component_source_registry.json", {
        "status": "SOURCE_UNVERIFIED", "shortlist_unique_donor_components": int(unique_donors),
        "shortlist_unique_acceptor_components": int(unique_acceptors),
        "observed_component_registry_evidence": True, "literature_confirmed": 0,
        "commercially_confirmed": 0, "database_only_confirmed": 0,
        "source_unverified_components": int(unique_donors + unique_acceptors),
        "external_candidate_structure_queries": 0, "stable_external_links": 0,
        "reason": config["source_policy"]["reason"],
        "interpretation": "SOURCE_UNVERIFIED does not mean unavailable or unreported",
    })
    reaction = {field: "NOT_DOCUMENTED_OR_EXPERIMENTALLY_CONFIRMED" for field in config["reaction_definition_fields"]}
    resources = {field: "NOT_DOCUMENTED_OR_EXPERIMENTALLY_CONFIRMED" for field in config["resource_fields"]}
    write_json(ROOT / "data_registry/gate3a2_experimental_readiness_registry.json", {
        "reaction_definition": reaction, "resource_definition": resources,
        "reaction_fields_confirmed": 0, "reaction_fields_required": len(reaction),
        "resource_fields_confirmed": 0, "resource_fields_required": len(resources),
        "primary_decision": "BLOCKED_REACTION_NOT_DEFINED",
        "parallel_blockers": ["BLOCKED_NO_EXPERIMENTAL_PARTNER", "CHEMIST_REVIEW_REQUIRED", "BLOCKED_NO_EXPERIMENTAL_VALIDATION_PATH"],
        "codex_filled_experimental_details": False,
    })
    (ROOT / "reports/gate3a2_shortlist_chemical_integrity.md").write_text(
        "# Gate 3-A2 shortlist chemical integrity\n\n"
        f"All 16 frozen items reconcile to their full-structure and component hashes; all 12 novel pairs reproduce "
        f"the frozen D-A attachment bond and all four controls overlap observed structures. Classification counts: "
        f"`{json.dumps(classifications, sort_keys=True)}`. No item was removed or replaced.\n\n"
        f"PAINS-positive: {risk['pains_positive']}; BRENK-positive: {risk['brenk_positive']}; "
        f"radical-positive: {risk['radical_positive']}; multi-fragment: {risk['multi_fragment']}; "
        f"SA-score median: {risk['sa_score']['0.5']:.3f}; molecular-weight median: "
        f"{risk['molecular_weight']['0.5']:.1f}. These values, tautomer/stereo flags, and all structural alerts are "
        "heuristics for chemist review, not synthesis, safety, stability, or experimental facts.\n"
    )
    (ROOT / "reports/gate3a2_route_and_source_audit.md").write_text(
        "# Gate 3-A2 route and component-source audit\n\n"
        f"All 12 novel products have atom-mapped local disconnections, retained only in the Git-ignored local artifact. "
        f"Route-family suggestions are aggregate heuristics: `{json.dumps(dict(sorted(route_counter.items())), sort_keys=True)}`. "
        "No precursor handle, leaving-group, reagent, condition, yield, or literature route is established, so every novel "
        "candidate remains `ROUTE_UNRESOLVED_REQUIRES_CHEMIST_REVIEW`.\n\n"
        f"The shortlist uses {unique_donors} donor and {unique_acceptors} acceptor structures already present in the frozen "
        "component registry. Candidate structures were not sent to third-party search services. Literature, database, and "
        "commercial availability therefore remain `SOURCE_UNVERIFIED`; this does not mean absent or unavailable.\n"
    )
    (ROOT / "reports/gate3a2_experimental_handoff_readiness.md").write_text(
        "# Gate 3-A2 experimental handoff readiness\n\n"
        "None of the 15 reaction-definition fields or 11 resource fields has written experimental-person confirmation. "
        "There is no frozen reaction, substrate/product objective, light setup, controls, analytical endpoint, replicate plan, "
        "laboratory, synthesis/testing owner, budget, timeline, EHS owner, or data-return contract. Codex did not invent them.\n\n"
        "Primary blocker: `BLOCKED_REACTION_NOT_DEFINED`. Parallel blockers: "
        "`BLOCKED_NO_EXPERIMENTAL_PARTNER` and `CHEMIST_REVIEW_REQUIRED`. Failed synthesis and failed measurement "
        "retention remains a required future contract, not an implemented procedure.\n"
    )
    (ROOT / "reports/gate3a2_final_decision.md").write_text(
        "# Gate 3-A2 final decision\n\n"
        "Decision: **BLOCKED_REACTION_NOT_DEFINED**.\n\n"
        "The 16-item computational shortlist is intact and structurally auditable, but no synthesis route has chemist "
        "confirmation and no real photocatalytic reaction or experimental resource plan is defined. The work remains "
        "`PROSPECTIVE_COMPUTATIONAL_SCREENING_ONLY`; `BLOCKED_NO_EXPERIMENTAL_VALIDATION_PATH` cannot be lifted "
        "until actual prospective experimental results are returned. No candidate is described as synthesizable, "
        "experimentally validated, active, or high-performance.\n"
    )
    write_json(ROOT / "logs/gate3a2_evidence.json", {
        "status": "BLOCKED_REACTION_NOT_DEFINED", "candidate_count": 16,
        "shortlist_changed": False, "replacement_candidates": 0, "scorer_called": False,
        "training": False, "prediction": False, "gpu": False, "test_access": False,
        "final673_access": False, "candidate_labels_read": False,
        "local_sensitive_artifact_git_ignored": True,
        "persistent_status": "PROSPECTIVE_COMPUTATIONAL_SCREENING_ONLY",
        "persistent_blocker": "BLOCKED_NO_EXPERIMENTAL_VALIDATION_PATH",
        "ended_utc": utc(),
    })
    print(json.dumps({"decision": "BLOCKED_REACTION_NOT_DEFINED", "classifications": classifications, "risk_counts": risk}, indent=2))


def finalize() -> None:
    required = [
        "configs/gate3a2_experimental_handoff_audit_v1.json",
        "data_registry/gate3a2_preregistration_lock_v1.json",
        "data_registry/gate3a2_candidate_audit_registry.json",
        "data_registry/gate3a2_component_source_registry.json",
        "data_registry/gate3a2_experimental_readiness_registry.json",
        "reports/gate3a2_preregistration.md",
        "reports/gate3a2_shortlist_chemical_integrity.md",
        "reports/gate3a2_route_and_source_audit.md",
        "reports/gate3a2_experimental_handoff_readiness.md",
        "reports/gate3a2_final_decision.md",
        "logs/gate3a2_evidence.json",
        "scripts/gate3a2_experimental_handoff_audit.py",
        "tests/test_gate3a2_contract.py",
    ]
    (ROOT / "data_registry/gate3a2_sha256.txt").write_text(
        "\n".join(f"{sha256(ROOT / path)}  {path}" for path in required) + "\n"
    )
    print(json.dumps({"status": "GATE3A2_SHA_FROZEN", "files": len(required)}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["preregister", "audit", "finalize"])
    args = parser.parse_args()
    config = load_config()
    if args.stage == "preregister":
        preregister(config)
    elif args.stage == "audit":
        audit(config)
    else:
        finalize()


if __name__ == "__main__":
    main()
