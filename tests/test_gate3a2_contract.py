import json
from pathlib import Path

import pandas as pd

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")


def config():
    return json.loads((ROOT / "configs/gate3a2_experimental_handoff_audit_v1.json").read_text())


def test_gate3a1_shortlist_is_exactly_frozen():
    c = config()
    local = pd.read_parquet(ROOT / c["inputs"]["local_shortlist"]["path"])
    public = json.loads((ROOT / c["inputs"]["shortlist_registry"]["path"]).read_text())
    assert len(local) == 16
    assert local.category.value_counts().to_dict() == c["shortlist_contract"]["categories"]
    assert set(local.pair_hash.astype(str)) == {x["anonymous_pair_hash"] for x in public["items"]}
    assert local.full_structure_hash.nunique() == 16


def test_no_replacement_or_rescoring():
    c = config()
    assert c["shortlist_contract"]["replacement_forbidden"]
    assert c["shortlist_contract"]["rescoring_forbidden"]
    source = (ROOT / "scripts/gate3a2_experimental_handoff_audit.py").read_text()
    assert "gate3a1_prospective_scoring.py score" not in source
    assert "XGBRegressor" not in source


def test_candidate_specific_assets_stay_local():
    c = config()
    assert not c["publication_boundary"]["canonical_smiles"]
    assert not c["publication_boundary"]["atom_mapped_disconnections"]
    assert not c["publication_boundary"]["specific_candidate_identity"]
    assert not c["publication_boundary"]["procurement_notes"]
    assert c["publication_boundary"]["anonymous_hashes_and_aggregate_counts_only"]


def test_heuristics_are_not_experimental_claims():
    c = config()
    assert c["chemical_audit"]["heuristic_not_experimental_fact"]
    assert c["route_policy"]["yield_reagent_condition_claims_forbidden"]
    assert c["route_policy"]["default_without_chemist_confirmation"] == "ROUTE_UNRESOLVED_REQUIRES_CHEMIST_REVIEW"


def test_sources_fail_closed_to_unverified():
    c = config()
    assert not c["source_policy"]["candidate_specific_external_structure_queries"]
    assert c["source_policy"]["literature_status_without_stable_evidence"] == "SOURCE_UNVERIFIED"
    assert c["source_policy"]["commercial_status_without_stable_evidence"] == "SOURCE_UNVERIFIED"
    assert c["source_policy"]["source_unverified_does_not_mean_absent"]


def test_reaction_and_resource_fields_are_complete():
    c = config()
    assert len(c["reaction_definition_fields"]) == 15
    assert len(c["resource_fields"]) == 11
    assert "reaction_type" in c["reaction_definition_fields"]
    assert "experimental_partner_or_lab" in c["resource_fields"]
    assert "failed_synthesis_and_measurement_retention" in c["resource_fields"]


def test_allowed_candidate_classifications():
    c = config()
    assert set(c["candidate_classifications"]) == {
        "READY_FOR_CHEMIST_REVIEW", "ROUTE_PLAUSIBLE_UNVERIFIED", "ROUTE_UNRESOLVED",
        "STRUCTURAL_RISK", "SOURCE_UNVERIFIED", "COMPUTATIONAL_CONTROL_ONLY",
    }


def test_persistent_scientific_boundary():
    c = config()
    assert c["mandatory_status"] == "PROSPECTIVE_COMPUTATIONAL_SCREENING_ONLY"
    assert c["persistent_blocker"] == "BLOCKED_NO_EXPERIMENTAL_VALIDATION_PATH"
    assert not c["training"] and not c["prediction"] and not c["gpu"]
    assert not c["test_access"] and not c["final673_access"]


def test_existing_scorer_is_fail_closed():
    scoring = json.loads((ROOT / "data_registry/gate3a1_scoring_unlock_v1.json").read_text())
    assert scoring["status"] == "CANDIDATE_SCORING_CONSUMED_ONCE"
    assert scoring["second_invocation"] == "FAIL_CLOSED"


def test_post_audit_public_registry_has_no_structures():
    path = ROOT / "data_registry/gate3a2_candidate_audit_registry.json"
    if not path.exists():
        return
    registry = json.loads(path.read_text())
    assert registry["candidate_count"] == 16
    assert not registry["smiles_published"]
    assert not registry["mapped_disconnections_published"]
    for row in registry["items"]:
        assert "smiles" not in row
        assert "mapped_disconnection" not in row


def test_post_audit_decision_requires_experiment():
    path = ROOT / "data_registry/gate3a2_experimental_readiness_registry.json"
    if not path.exists():
        return
    registry = json.loads(path.read_text())
    assert registry["primary_decision"] == "BLOCKED_REACTION_NOT_DEFINED"
    assert registry["reaction_fields_confirmed"] == 0
    assert registry["resource_fields_confirmed"] == 0
    assert not registry["codex_filled_experimental_details"]
