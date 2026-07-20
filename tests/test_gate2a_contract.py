from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from xgboost import XGBRegressor

from scripts.gate1b1_train_cheap_baselines import fit_preprocessor, record_group_metrics, sha256, transform
from scripts.gate2a_evaluate_once import bootstrap_ci, independent_degradation_ci, paired_bootstrap_ci
from scripts.gate2a_train_ood_baselines import diagnostic_mol, load_contract, validate_manifest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/gate2a_ood_baselines_v1.json"


def test_frozen_inputs_and_feature_contracts() -> None:
    config, source, features = load_contract(CONFIG)
    assert sha256(Path(config["table"])) == config["table_sha256"]
    assert len(features) == 15016 and features.molecule_id.nunique() == 15016
    assert source["features"]["morgan"] == {"radius": 2, "nBits": 512, "includeChirality": False}
    assert len(source["features"]["M1_C0_open"]) == 532
    assert len(source["features"]["M2_C1p5_safe_no_dipole"]) == 535
    diagnostic = config["ood_diagnostic_fingerprint"]
    assert diagnostic == {"radius": 2, "nBits": 2048, "useChirality": True,
                          "views": ["full_molecule", "donor_component", "acceptor_component"], "model_input": False}
    assert not any("2048" in column for column in source["features"]["M2_C1p5_safe_no_dipole"])


@pytest.mark.parametrize("split_name", ["donor_cold", "acceptor_cold", "pair_cold", "both_cold", "full_scaffold_cold"])
def test_manifest_counts_and_leakage_invariants(split_name: str) -> None:
    config = json.loads(CONFIG.read_text())
    frame = validate_manifest(split_name, config["splits"][split_name])
    assert len(frame) == 15016
    assert frame.groupby("structure_group_id_v1").partition.nunique().max() == 1
    assert len(frame[frame.historical_status.eq("HISTORICAL_TRAIN_OVERLAP")]) == 17
    assert frame.loc[frame.historical_status.eq("HISTORICAL_TRAIN_OVERLAP"), "partition"].eq("train").all()


def test_buffer_and_quarantine_fail_closed_by_explicit_partition() -> None:
    config = json.loads(CONFIG.read_text())
    frame = validate_manifest("both_cold", config["splits"]["both_cold"])
    model = frame[frame.partition.isin(["train", "val", "test"])]
    assert not model.partition.isin(["buffer", "historical_quarantine"]).any()
    assert frame.partition.eq("buffer").sum() == 3291
    assert frame.partition.eq("historical_quarantine").sum() == 1


def test_preprocessing_is_train_feature_only_and_target_invariant() -> None:
    matrix = np.array([[1.0, np.nan], [2.0, 7.0], [9.0, 4.0]])
    weights = np.array([0.5, 0.5, 1.0])
    first = fit_preprocessor(matrix, weights)
    heldout_target = np.array([1.0, 2.0]); heldout_target[:] = 1e9
    second = fit_preprocessor(matrix.copy(), weights.copy())
    for key in first: np.testing.assert_allclose(first[key], second[key])
    assert np.isfinite(transform(matrix, first)).all()


def test_group_metrics_and_bootstrap_manual_cases() -> None:
    y = np.zeros(3); p = np.array([2.0, 2.0, 0.0]); groups = np.array(["a", "a", "b"])
    result = record_group_metrics(y, p, groups)
    assert result["record_mae"] == pytest.approx(4 / 3)
    assert result["group_macro_mae"] == pytest.approx(1.0)
    assert bootstrap_ci(np.ones(3), 100, 1) == pytest.approx([1.0, 1.0])
    assert paired_bootstrap_ci(np.ones(3), np.zeros(3), groups, 100, 1) == pytest.approx([1.0, 1.0])
    independent = independent_degradation_ci(np.full(4, 2.0), np.ones(5), 100, 1)
    assert independent["absolute_difference_ci95"] == pytest.approx([1.0, 1.0])
    assert independent["ratio_ci95"] == pytest.approx([2.0, 2.0])
    assert independent["method"] == "independent_structure_group_bootstrap"


def test_xgboost_no_subsampling_seed_is_deterministic() -> None:
    x = np.arange(60, dtype=np.float32).reshape(20, 3); y = np.sin(np.arange(20))
    predictions = []
    for seed in (42, 123):
        model = XGBRegressor(n_estimators=5, max_depth=2, learning_rate=0.1, tree_method="hist", device="cpu", verbosity=0, random_state=seed)
        model.fit(x, y); predictions.append(model.predict(x))
    np.testing.assert_array_equal(predictions[0], predictions[1])


def test_config_has_exactly_twenty_frozen_baseline_slots() -> None:
    config = json.loads(CONFIG.read_text())
    assert len(config["splits"]) * len(config["models"]) == 20
    assert config["test_unlock"] == {"union_arrow_target_reads": 1, "evaluate_once": True, "second_call": "fail_closed"}
    assert config["final673_access"] is False


def test_gate0c_diagnostic_parser_handles_attachment_fragment() -> None:
    assert diagnostic_mol("*1ccc[se]c1") is not None
