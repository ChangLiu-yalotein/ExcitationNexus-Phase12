from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch, Data

from excitationnexus_phase12.gate1c1 import (
    deterministic_similarity_merge,
    group_bootstrap_error_difference,
    perturb_positions,
    sha256_file,
)

ROOT = Path(__file__).resolve().parents[1]


def config() -> dict:
    return json.loads((ROOT / "configs/gate1c1_diagnosis_v1.json").read_text())


def test_preregistration_and_frozen_test_source() -> None:
    lock = json.loads((ROOT / "data_registry/gate1c1_preregistration_lock_v1.json").read_text())
    assert sha256_file(ROOT / lock["config_path"]) == lock["config_sha256"]
    assert lock["test_inference_allowed"] is False
    assert lock["validation_counterfactual_allowed"] is True
    assert lock["final673_access"] is False
    frozen = config()["frozen_inputs"]["gate1b3_test_predictions"]
    assert sha256_file(ROOT / frozen["path"]) == frozen["sha256"]


def test_group_bootstrap_manual_point_estimate() -> None:
    frame = pd.DataFrame({"structure_group_id_v1":["a","a","b"], "primary_true":[0.,0.,0.],
                          "first":[1.,3.,4.], "second":[0.,0.,1.]})
    result = group_bootstrap_error_difference(frame, "first", "second", iterations=100, seed=7)
    # Per-group differences are mean([1,3])=2 and 4-1=3, hence macro mean 2.5.
    assert result["point_difference_eV"] == 2.5
    assert result["groups"] == 2


def test_subgroup_minimum_and_deterministic_merge() -> None:
    frame = pd.DataFrame({"bin":["a"]*2+["b"]*3+["c"]*6,
                          "structure_group_id_v1":[f"g{x}" for x in range(11)]})
    first, metadata = deterministic_similarity_merge(frame,"bin",["a","b","c"],minimum_records=5,minimum_groups=5)
    second, _ = deterministic_similarity_merge(frame,"bin",["a","b","c"],minimum_records=5,minimum_groups=5)
    assert first.equals(second)
    assert metadata[0]["source_bins"] == ["a","b"]
    assert all(x["records"] >= 5 and x["groups"] >= 5 for x in metadata)
    assert config()["subgroup_policy"]["minimum_records"] == 50
    assert config()["subgroup_policy"]["minimum_structure_groups"] == 30


def synthetic_batch() -> Batch:
    return Batch.from_data_list([Data(z=torch.tensor([6,7]), pos=torch.tensor([[0.,0.,0.],[1.,0.,0.]]),
                                           role=torch.tensor([0,1]), edge_index=torch.tensor([[0,1],[1,0]]),
                                           molecule_id="synthetic")])


def test_counterfactual_is_deterministic_and_global_transform_preserves_distance() -> None:
    batch = synthetic_batch()
    one,_ = perturb_positions(batch,"gaussian_noise_0.05A")
    two,_ = perturb_positions(batch,"gaussian_noise_0.05A")
    assert torch.equal(one,two)
    transformed,_ = perturb_positions(batch,"global_rotation_translation")
    assert torch.allclose(torch.cdist(batch.pos,batch.pos),torch.cdist(transformed,transformed),atol=1e-6)


def test_counterfactual_contract_is_validation_only() -> None:
    contract = config()["validation_counterfactuals"]
    assert contract["partition"] == "val only"
    assert contract["test_counterfactuals_forbidden"] is True
    assert config()["forbidden"][1] == "new test inference or test counterfactual"


def test_decision_matrix_requires_exactly_one_branch() -> None:
    matrix = config()["decision_matrix"]
    assert matrix["exactly_one_required"] is True
    assert matrix["priority"] == ["SCALE_3D","FUSE_2D_3D","STOP_PURE_3D"]

def test_final_diagnosis_covers_all_test_records_without_test_inference() -> None:
    result = json.loads((ROOT / "logs/gate1c1_evidence.json").read_text())
    assert result["status"] == "GATE1C1_DONE_STOP_PURE_3D"
    assert sum(item["records"] for item in result["subgroups"]["similarity"].values()) == 2319
    assert result["decision_evidence"]["powered_3d_model_stratum_win_count"] == 2
    assert result["decision_evidence"]["powered_3d_winning_subgroup_count"] == 1
    assert result["test_artifact_analysis_only"] is True
    assert result["new_test_predictions_created"] is False
    assert result["training_performed"] is False
    assert result["final673_accessed"] is False


def test_counterfactual_evidence_is_validation_only_and_invariant() -> None:
    result = json.loads((ROOT / "logs/gate1c1_validation_counterfactuals.json").read_text())
    assert result["partition"] == "val"
    assert result["records"] == 2309
    assert result["test_inference_performed"] is False
    assert result["training_performed"] is False
    assert max(result["original_reconciliation_max_eV"].values()) < 2e-6
    for model in result["ensembles"].values():
        assert model["global_rotation_translation"]["max_abs_delta_prediction_eV"] < 2e-6

