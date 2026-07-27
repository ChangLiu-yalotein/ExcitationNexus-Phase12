import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gate2g1c_preregistration_boundaries():
    cfg = json.loads((ROOT / "configs/gate2g1c_unified25008_strong_2d_v1.json").read_text())
    assert cfg["final673_label_access"] is False
    assert cfg["candidate_rescoring"] is False
    assert cfg["arms"]["chemprop_v2_dmppn_blind"] == "REQUIRED_BUT_ENV_DEPENDENCY_GATED"
    assert "final673 membership" in cfg["forbidden_inputs"]


def test_gate2g1c_firewall_when_present():
    path = ROOT / "data_registry/gate2g1c_firewall_assertions.json"
    if not path.exists():
        return
    reg = json.loads(path.read_text())
    assert reg["quarantine_records_in_dataset"] == 0
    assert reg["final673_label_reads"] == 0
    assert reg["candidate_assets_accessed"] is False
    assert reg["final673_token_inputs"] == 0


def test_gate2g1c_dependency_truthfulness_when_present():
    path = ROOT / "data_registry/gate2g1c_chemprop_dependency_registry.json"
    if not path.exists():
        return
    reg = json.loads(path.read_text())
    if reg["status"] == "BLOCKED_CHEMPROP_V2_UNAVAILABLE":
        assert reg["v1_not_substituted_for_v2"] is True
        decision = json.loads((ROOT / "data_registry/gate2g1c_decision_registry.json").read_text())
        assert decision["chemprop_blocked"] is True
        assert decision["status"] != "GATE2G1C_UNIFIED_STRONG_2D_BENCHMARK_DONE"


def test_gate2g1c_dependency_recovery_amendment_when_present():
    path = ROOT / "data_registry/gate2g1c_chemprop_dependency_recovery.json"
    if not path.exists():
        return
    reg = json.loads(path.read_text())
    assert reg["status"] == "GATE2G1C_CHEMPROP_V2_DEPENDENCY_RECOVERED"
    assert reg["chemprop_version"] == "2.3.0"
    assert reg["torch_version"] == "2.8.0+cu128"
    assert reg["cuda_available"] is True
    assert reg["cuda_device_count"] == 7
    assert reg["pip_check"] == "No broken requirements found."
    assert reg["chemprop_v1_used"] is False
    assert reg["ml_environment_modified"] is False
    assert reg["final673_label_reads"] == 0
    assert reg["candidate_assets_accessed"] is False
    assert reg["formal_g1c_chemprop_arms_started"] is False
    assert reg["cpu_smoke"]["finite_test_rmse"] is True
    assert reg["gpu_smoke"]["finite_test_rmse"] is True
    assert reg["gpu_smoke"]["gpu6_used"] is False


def test_gate2g1c_chemprop_formal_recovery_when_present():
    input_path = ROOT / "data_registry/gate2g1c_chemprop_formal_input_registry.json"
    graph_path = ROOT / "data_registry/gate2g1c_chemprop_graph_compatibility.json"
    smoke_path = ROOT / "data_registry/gate2g1c_chemprop_formal_admission_smoke.json"
    if not (input_path.exists() and graph_path.exists() and smoke_path.exists()):
        return
    inp = json.loads(input_path.read_text())
    graph = json.loads(graph_path.read_text())
    smoke = json.loads(smoke_path.read_text())
    assert inp["status"] == "GATE2G1C_CHEMPROP_FORMAL_INPUTS_PREPARED"
    assert inp["eligible_records"] == 25008
    assert inp["quarantine_records_in_frame"] == 0
    assert inp["final673_label_reads"] == 0
    assert inp["raw_source_cohort_token_used"] is False
    assert inp["formal_training_started"] is False
    assert graph["status"] == "CHEMPROP_GRAPH_COMPATIBILITY_PASS"
    assert graph["records_checked"] == 25008
    assert graph["parse_failures"] == 0
    assert graph["graph_failures"] == 0
    assert graph["rdkit_version_chemprop_env"] != graph["rdkit_version_ml_env_gate2g1b"]
    assert smoke["status"] == "GATE2G1C_CHEMPROP_FORMAL_ADMISSION_SMOKE_PASS"
    assert smoke["outer_validation_records_used"] == 0
    assert smoke["final673_label_reads"] == 0
    assert smoke["candidate_assets_accessed"] is False
    assert smoke["cpu_smoke"]["finite_validation_loss"] is True
    assert smoke["gpu_smoke"]["finite_validation_loss"] is True
    assert smoke["gpu_smoke"]["gpu6_used"] is False
