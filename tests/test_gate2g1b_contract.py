import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gate2g1b_preregistration_boundaries():
    cfg = json.loads((ROOT / "configs/gate2g1b_source_aware_splits_v1.json").read_text())
    assert cfg["execution"] == "CPU_ONLY_SPLIT_GOVERNANCE_NO_TRAINING_NO_GPU"
    assert cfg["final673_label_access"] is False
    assert cfg["gate3_shortlist_status"] == "EXPLORATORY_BASELINE_SHORTLIST_FROZEN"


def test_gate2g1b_counts_when_present():
    path = ROOT / "data_registry/gate2g1b_aggregate_counts.json"
    if not path.exists():
        return
    reg = json.loads(path.read_text())
    assert reg["universe_records"] == 25703
    assert reg["development_registry_records_before_quarantine"] == 25030
    assert reg["final673_records"] == 673
    assert reg["eligible_development_records_after_quarantine"] < 25030
    assert reg["final673_structure_overlap_quarantine_development_records"] > 0


def test_gate2g1b_leakage_assertions_when_present():
    path = ROOT / "data_registry/gate2g1b_leakage_assertions.json"
    if not path.exists():
        return
    reg = json.loads(path.read_text())
    zero_keys = [
        "final673_label_reads",
        "gpu_usage",
        "final_overlap_quarantine_training_intersection",
        "missing_or_duplicate_unified_oof_assignment",
        "global_structure_leakage",
        "acceptor_cold_acceptor_leakage",
        "donor_cold_donor_leakage",
        "pair_cold_pair_leakage",
        "scaffold_cold_scaffold_leakage",
        "source_holdout_cross_source_structure_leakage",
    ]
    for key in zero_keys:
        assert reg[key] == 0
    assert reg["row_shuffle_assignment_hash_invariant"] is True
    assert reg["independent_implementation_assignment_hash_equal"] is True
    assert reg["gate2e1_record_mean_error_rejected"] is True


def test_gate2g1b_provenance_policy_when_present():
    path = ROOT / "data_registry/gate2g1b_provenance_policy_v1.json"
    if not path.exists():
        return
    reg = json.loads(path.read_text())
    assert reg["provenance_blind_model_required"] is True
    assert reg["observable_method_provenance_token_model_allowed"] is True
    assert reg["raw_dataset_source_token"] == "diagnostic_ablation_only_not_automatic_deployment_champion"
    assert reg["final673_token_allowed"] is False
