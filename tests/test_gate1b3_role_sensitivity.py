"""Contract tests for the frozen Gate 1-B3 role sensitivity."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

from excitationnexus_phase12.gate1b3_role_sensitivity import candidate_role_tensor, reconcile

ROOT = Path(__file__).resolve().parents[1]


def test_candidate_changes_only_selected_unknown_atoms() -> None:
    original = torch.tensor([2, 1, 2, 1])
    candidate = candidate_role_tensor(original, "1;3")
    assert candidate.tolist() == [0, 1, 0, 1]
    assert original.tolist() == [2, 1, 2, 1]
    assert torch.equal(candidate, candidate_role_tensor(original, "1;3"))


def test_candidate_rejects_non_unknown_atom() -> None:
    with pytest.raises(ValueError, match="not original unknown"):
        candidate_role_tensor(torch.tensor([2, 1]), "2")


def test_reconciliation_is_fail_closed() -> None:
    reconcile(torch.tensor([1.0]).numpy(), torch.tensor([1.000001]).numpy(), 2e-6)
    with pytest.raises(RuntimeError, match="BLOCKED_SENSITIVITY_MISMATCH"):
        reconcile(torch.tensor([1.0]).numpy(), torch.tensor([1.01]).numpy(), 2e-6)


def test_frozen_contract_excludes_unresolved_and_test_parquet_truth() -> None:
    config = json.loads((ROOT / "configs/gate1b3_role_sensitivity_v1.json").read_text())
    assert config["included_records"] == 198
    assert config["excluded_status"] == "UNRESOLVED_AMBIGUOUS"
    assert config["excluded_records"] == 189
    assert config["test_y_true_source"] == "frozen Gate1B3 test-once prediction artifact only"
    assert config["standard_test_evaluator_allowed"] is False
    assert config["training_allowed"] is False


def test_standard_evaluator_remains_locked() -> None:
    source = (ROOT / "src/excitationnexus_phase12/gate1b3_evaluation.py").read_text()
    assert "one-time test output already exists; refusing second evaluation" in source


def test_all_six_checkpoint_hashes_are_registered() -> None:
    registry = json.loads((ROOT / "data_registry/gate1b3_model_registry.json").read_text())
    hashes = [run["checkpoint_sha256"] for wave in ("wave1", "wave2")
              for run in registry[wave]["runs"].values()]
    assert len(hashes) == 6
    assert len(set(hashes)) == 6
    assert all(len(value) == 64 for value in hashes)


def test_completed_sensitivity_reconciles_and_did_not_train() -> None:
    result = json.loads((ROOT / "logs/gate1b3_role_sensitivity.json").read_text())
    assert result["candidate_records"] == 198
    assert result["unresolved_excluded"] == 189
    assert result["quarantine_inference_records"] == 0
    assert max(result["original_prediction_max_abs_reconciliation_eV"].values()) < 2e-6
    assert result["test_y_true_source"] == "frozen Gate1B3 test-once artifact"
    assert result["test_parquet_target_read"] is False
    assert result["training_performed"] is False
    assert result["parameters_updated"] is False
