import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_gate2g0_preregistration_boundaries():
    config = json.loads((ROOT / "configs/gate2g0_model_benchmark_consolidation_v1.json").read_text())
    assert config["execution"] == "CPU_ONLY_NO_TRAINING_NO_TEST_EVALUATION"
    assert config["historical_and_new15016_rankings_separate"] is True
    assert config["shortlist_status"] == "EXPLORATORY_BASELINE_SHORTLIST_FROZEN"
    assert config["final673_access"] is False


def test_gate2g0_outputs_are_separate_ledgers_when_present():
    historical = ROOT / "data_registry/gate2g0_historical_benchmark.csv"
    new = ROOT / "data_registry/gate2g0_new15016_iid_benchmark.csv"
    if historical.exists() and new.exists():
        assert "new15016" not in historical.read_text()
        assert "Layer G" not in new.read_text()


def test_shortlist_reclassification_is_not_rescoring():
    config = json.loads((ROOT / "configs/gate2g0_model_benchmark_consolidation_v1.json").read_text())
    assert config["shortlist_experimental_progression"] == "PAUSED"
