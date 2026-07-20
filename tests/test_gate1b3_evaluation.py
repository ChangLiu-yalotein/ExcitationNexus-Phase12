from __future__ import annotations

import pandas as pd
import pytest

from excitationnexus_phase12.gate1b3_evaluation import Gate1B3TestDataset, group_bootstrap_difference


def test_evaluation_dataset_requires_explicit_unlock():
    with pytest.raises(PermissionError, match="TEST_TARGET_FIREWALL_LOCKED"):
        Gate1B3TestDataset(None, None, pd.DataFrame(), pd.DataFrame(), {"status": "READY_ONLY"})


def test_group_bootstrap_known_direction_and_determinism():
    frame = pd.DataFrame({
        "structure_group_id_v1": ["a", "a", "b", "c"],
        "primary_true": [0., 0., 0., 0.],
        "better": [0., 0., 1., 1.],
        "worse": [2., 2., 2., 2.],
    })
    first = group_bootstrap_difference(frame, "better", "worse", iterations=1000)
    second = group_bootstrap_difference(frame, "better", "worse", iterations=1000)
    assert first == second
    assert first["point_difference_eV"] < 0
    assert first["ci_excludes_zero"] is True


def test_group_bootstrap_uses_group_macro_not_record_weighting():
    frame = pd.DataFrame({
        "structure_group_id_v1": ["duplicate", "duplicate", "singleton"],
        "primary_true": [0., 0., 0.],
        "first": [2., 2., 0.],
        "second": [0., 0., 1.],
    })
    result = group_bootstrap_difference(frame, "first", "second", iterations=100)
    assert result["point_difference_eV"] == pytest.approx(0.5)
