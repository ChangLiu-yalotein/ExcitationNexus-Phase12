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
