import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from excitationnexus_phase12.prospective_scoring import (
    C0_COLUMNS, assert_c0_contract, c0_matrix, deterministic_rank_percentile,
    fit_preprocessor, identity_cap_ok, ordered_hash, transform,
)

ROOT = Path("/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training")


def test_c0_exact_contract():
    assert len(C0_COLUMNS) == 532
    assert C0_COLUMNS[:3] == ["pair_MolWt", "pair_MolLogP", "pair_MolMR"]
    assert C0_COLUMNS[-1] == "pair_morgan_511"
    assert_c0_contract(C0_COLUMNS)


def test_firewall_rejects_changed_contract():
    with pytest.raises(RuntimeError):
        assert_c0_contract(C0_COLUMNS + ["tddft_target"])


def test_feature_determinism_and_row_binding():
    smiles = ["CCO", "c1ccccc1", "CC(=O)O"]
    a = c0_matrix(smiles)
    order = [2, 0, 1]
    b = c0_matrix([smiles[i] for i in order])
    assert np.array_equal(a[order], b)
    assert np.isfinite(a).all()


def test_group_weighted_preprocessing_manual():
    x = np.array([[0.0], [2.0], [10.0]])
    w = np.array([0.5, 0.5, 1.0])
    prep = fit_preprocessor(x, w)
    assert np.isclose(prep["means"][0], 5.5)
    assert np.isfinite(transform(x, prep)).all()


def test_rank_hash_tie_break_is_deterministic():
    prediction = np.array([1.0, 1.0, 0.0])
    hashes = np.array(["b", "a", "c"])
    rank = deterministic_rank_percentile(prediction, hashes)
    assert rank.tolist() == [1.0, 0.5, 0.0]
    shuffled = np.array([2, 0, 1])
    restored = np.empty(3)
    restored[shuffled] = deterministic_rank_percentile(prediction[shuffled], hashes[shuffled])
    assert np.array_equal(rank, restored)


def test_identity_cap():
    row = pd.Series({"donor_identity_hash": "d", "acceptor_identity_hash": "a"})
    assert identity_cap_ok(row, {}, {})
    assert not identity_cap_ok(row, {"d": 2}, {})
    assert not identity_cap_ok(row, {}, {"a": 2})


def test_preregistered_scientific_boundaries():
    config = json.loads((ROOT / "configs/gate3a1_prospective_scoring_v1.json").read_text())
    assert config["expected"]["novel_in_domain_candidates"] == 36523
    assert config["stability"]["models"] == 20
    assert config["stability"]["extreme_inclusion_frequency_min"] == 0.8
    assert config["selection"] == {
        "predicted_low_proxy": 4,
        "predicted_high_proxy": 4,
        "diversity_exploration": 4,
        "matched_observed_controls": 4,
        "max_per_donor": 2,
        "max_per_acceptor": 2,
        "extreme_order": "stable frequency gate, then final prediction, then pair hash",
        "exploration": "greedy full-Morgan max-min diversity; first choose minimum nearest-observed similarity; hash tie-break",
        "controls": "nearest observed full-Morgan structure to each exploration item under global identity caps; no target/error criterion",
        "insufficient_policy": "do not lower the 0.80 stability threshold; shrink the affected group and return RANKING_UNSTABLE_NO_SHORTLIST",
    }
    assert config["final_boundary"] == [
        "PROSPECTIVE_COMPUTATIONAL_SCREENING_ONLY",
        "BLOCKED_NO_EXPERIMENTAL_VALIDATION_PATH",
    ]


def test_gate3a0_candidate_boundary_and_ignored_assets():
    config = json.loads((ROOT / "configs/gate3a1_prospective_scoring_v1.json").read_text())
    candidate = pd.read_parquet(ROOT / config["inputs"]["candidate_universe"]["path"], columns=["pair_hash", "status", "in_pair_cold_domain"])
    selected = candidate[candidate.status.eq("NOVEL_PAIR_KNOWN_COMPONENTS") & candidate.in_pair_cold_domain.astype(bool)]
    assert len(selected) == 36523
    assert selected.pair_hash.nunique() == 36523
    assert len(candidate[candidate.status.eq("INVALID_VALENCE")]) == 308
    assert len(candidate[candidate.status.eq("OUTSIDE_COMPONENT_SUPPORT")]) == 1462
    assert len(candidate[candidate.status.eq("DUPLICATE_PRODUCT_FROM_DIFFERENT_ALIASES")]) == 958


def test_column_order_hash_is_stable():
    assert ordered_hash(C0_COLUMNS) == ordered_hash(list(C0_COLUMNS))


def test_scoring_second_invocation_fail_closed_contract():
    source = (ROOT / "scripts/gate3a1_prospective_scoring.py").read_text()
    assert "os.O_CREAT | os.O_EXCL" in source
    assert "candidate scoring is fail-closed after first invocation" in source
    assert "test_artifact_access" in source and "final673_access" in source
