import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from gate2e0_common import standardized_linear_residual_rmse, weighted_corr, weighted_spearman


def test_frozen_task_counts_and_no_primary_in_extraction():
    config = json.loads((ROOT / "configs/gate2e0_multitask_target_audit_v1.json").read_text())
    assert len(config["secondary"]) == 12
    assert len(config["masked"]) == 4
    assert config["primary"] not in config["secondary"] + config["masked"]
    assert config["expected_train_validation_union"] == 15015
    assert config["test_artifact_access"] is False
    assert config["final673_access"] is False
    assert config["gpu_use"] is False


def test_group_weighted_statistics_hand_calculation():
    x = np.array([0.0, 0.0, 2.0])
    y = np.array([0.0, 0.0, 4.0])
    w = np.array([0.5, 0.5, 1.0])
    assert np.isclose(weighted_corr(x, y, w), 1.0)
    assert np.isclose(weighted_spearman(x, y, w), 1.0)
    assert standardized_linear_residual_rmse(x, y, w) < 1e-12


def test_fraction_mask_counting_does_not_impute():
    values = np.array([0.2, np.nan, 0.8, np.nan])
    weights = np.array([0.5, 0.5, 1.0, 1.0])
    mask = np.isfinite(values)
    assert np.isclose(weights[mask].sum(), 1.5)
    assert mask.sum() == 2


def test_report_only_family_excluded_from_optimization():
    config = json.loads((ROOT / "configs/gate2e0_multitask_target_audit_v1.json").read_text())
    optimization = {config["primary"], *config["secondary"], *config["masked"]}
    assert optimization.isdisjoint(config["report_only_deterministic"])
    assert optimization.isdisjoint(config["disabled"])
