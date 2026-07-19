import hashlib
import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def test_gate1a2_v3_preregistration_lock_and_scope():
    lock = json.loads((ROOT / "data_registry/gate1a2_preregistration_lock_v3.json").read_text())
    assert lock["status"] == "FROZEN_BEFORE_INFERENCE_AND_TRAINING"
    for relative, expected in lock["files"].items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected
    config = json.loads((ROOT / "configs/gate1a2_b21_seed42_reproduction_v3.json").read_text())
    assert config["training_contract"]["formal_training_runs"] == 1
    assert config["training_contract"]["seed"] == 42
    assert config["model_contract"]["parameter_count"] == 1065570
    assert config["final673_access"] is False
    assert config["new15016_access"] is False


def test_gate1a2_numeric_reproduction_and_paired_counts():
    published = ROOT / "runs/gate1a2_b21_seed42/published"
    result = json.loads((published / "gate1a2_metrics.json").read_text())
    assert result["status"] == "REPRODUCED_NUMERIC"
    formal = result["formal_seed42_training"]
    assert formal["metrics"]["n_records"] == 1098
    assert formal["absolute_mae_delta_vs_original"] <= 0.001
    assert result["scope"]["formal_training_runs"] == 1
    assert not any(result["scope"][key] for key in ("other_seeds", "new15016", "final673", "b2_0", "b2_2a"))
    paired = pd.read_csv(published / "gate1a2_cheap_vs_b21_paired_1097.csv")
    assert len(paired) == 1097
    assert paired["molecule_id"].is_unique


def test_push_pending_checkpoint_is_non_destructive():
    script = (ROOT / "scripts/push_pending_checkpoint.sh").read_text()
    assert "only main may be pushed" in script
    assert "git push -u origin main" in script
    assert "--force" not in script
    assert "git reset" not in script
    assert "git rebase" not in script
