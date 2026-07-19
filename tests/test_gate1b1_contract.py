from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.gate1b1_audit_features import assert_firewall
from scripts.gate1b1_train_cheap_baselines import (
    fit_preprocessor,
    record_group_metrics,
    sha256,
    transform,
    weighted_median,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/gate1b1_new_iid_cheap_baselines_v1.json"
LOCK = ROOT / "data_registry/gate1b1_preregistration_lock_v1.json"


def test_preregistration_lock_and_frozen_inputs() -> None:
    config = json.loads(CONFIG.read_text())
    lock = json.loads(LOCK.read_text())
    for relative, expected in lock["files"].items():
        assert sha256(ROOT / relative) == expected
    assert sha256(Path(config["table"])) == config["table_sha256"]
    assert sha256(Path(config["manifest"])) == config["manifest_sha256"]
    assert sha256(Path(config["feature_cache"])) == config["feature_cache_sha256"]


def test_manifest_counts_weights_and_quarantine_boundary() -> None:
    config = json.loads(CONFIG.read_text())
    frame = pd.read_csv(config["manifest"])
    assert frame["partition"].value_counts().to_dict() == config["split_counts"]
    assert frame["molecule_id"].nunique() == 15016
    assert frame.groupby("structure_group_id_v1")["partition"].nunique().max() == 1
    sums = frame.groupby("partition")["group_weight"].sum()
    assert np.isclose(sums["train"], 10248.0)
    assert np.isclose(sums["val"], 2195.0)
    assert np.isclose(sums["test"], 2195.0)
    assert len(frame[frame["partition"].eq("historical_quarantine")]) == 1
    assert not frame[frame["partition"].eq("historical_quarantine")]["partition"].isin(["train", "val", "test"]).any()
    overlap = frame[frame["historical_status"].eq("HISTORICAL_TRAIN_OVERLAP")]
    assert len(overlap) == 17 and overlap["partition"].eq("train").all()


def test_feature_join_is_id_bound_and_coordinate_free() -> None:
    config = json.loads(CONFIG.read_text())
    manifest = pd.read_csv(config["manifest"])
    features = pd.read_parquet(config["feature_cache"])
    joined = manifest[["molecule_id"]].merge(features, on="molecule_id", validate="one_to_one")
    shuffled = manifest[["molecule_id"]].sample(frac=1, random_state=9)
    shuffled_join = shuffled.merge(features, on="molecule_id", validate="one_to_one").set_index("molecule_id")
    columns = config["features"]["M2_C1p5_safe_no_dipole"]
    assert len(joined) == 15016
    assert np.array_equal(
        joined.set_index("molecule_id").sort_index()[columns].to_numpy(),
        shuffled_join.sort_index()[columns].to_numpy(),
    )
    assert not any("coord" in column.lower() or "position" in column.lower() for column in columns)


@pytest.mark.parametrize("column", [
    "tddft_value", "multiwfn_field", "target_label", "pm6_energy_raw",
    "pm6_dipole", "partition", "final673_member", "coulomb_attraction_eV",
])
def test_target_firewall_rejects_forbidden_fields(column: str) -> None:
    with pytest.raises(ValueError):
        assert_firewall([column])


def test_weighted_median_preprocessor_and_val_target_invariance() -> None:
    matrix = np.array([[1.0, np.nan], [3.0, 2.0], [9.0, 4.0]])
    weights = np.array([0.5, 0.5, 1.0])
    assert weighted_median(np.array([1.0, 3.0, 9.0]), weights) == 3.0
    first = fit_preprocessor(matrix, weights)
    # Preprocessing accepts features and train weights only; changing any held-out target is irrelevant.
    heldout_targets = np.array([0.0, 1.0])
    heldout_targets[:] = 10_000.0
    second = fit_preprocessor(matrix.copy(), weights.copy())
    for name in first:
        np.testing.assert_allclose(first[name], second[name])
    assert np.isfinite(transform(matrix, first)).all()


def test_record_and_group_macro_manual_case() -> None:
    y = np.zeros(3)
    prediction = np.array([2.0, 2.0, 0.0])
    groups = np.array(["duplicate", "duplicate", "singleton"])
    metrics = record_group_metrics(y, prediction, groups)
    assert metrics["record_mae"] == pytest.approx(4.0 / 3.0)
    assert metrics["group_macro_mae"] == pytest.approx(1.0)
    assert metrics["record_rmse"] == pytest.approx(np.sqrt(8.0 / 3.0))
    assert metrics["group_macro_rmse"] == pytest.approx(np.sqrt(2.0))
