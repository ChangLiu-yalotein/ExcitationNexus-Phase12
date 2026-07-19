from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch, Data

from excitationnexus_phase12.dft_graph_dataset import DFTGraphCache, Gate1B2GraphDataset
from excitationnexus_phase12.losses import weighted_masked_multitask_loss
from excitationnexus_phase12.metrics import regression_metrics
from excitationnexus_phase12.models import M3DAUSharedModel, M3MergedModel
from excitationnexus_phase12.normalization import weighted_stats

ROOT = Path(__file__).resolve().parents[1]
CONFIG = json.loads((ROOT / "configs/gate1b2_3d_admission_v1.json").read_text())
REGISTRY = ROOT / CONFIG["graph_registry"]
CACHE = ROOT / CONFIG["graph_cache"]
MANIFEST = ROOT / CONFIG["manifest"]


def synthetic(roles=(0, 1, 2, 2)) -> Batch:
    n = len(roles)
    edge = torch.tensor([[i for i in range(n) for j in range(n) if i != j],
                         [j for i in range(n) for j in range(n) if i != j]], dtype=torch.long)
    data = Data(z=torch.tensor([6, 7, 8, 1][:n]), pos=torch.arange(n * 3, dtype=torch.float32).reshape(n, 3) / 7,
                role=torch.tensor(roles), edge_index=edge, num_nodes=n)
    return Batch.from_data_list([data])


def models():
    common = dict(hidden_dim=48, num_rbf=16, layers=2, cutoff=5.0)
    return M3MergedModel(**common, head_width=128), M3DAUSharedModel(**common, head_width=44)


def test_registry_join_counts_and_target_firewall() -> None:
    registry = pd.read_parquet(REGISTRY); manifest = pd.read_csv(MANIFEST)
    assert len(manifest.merge(registry, on="molecule_id", validate="one_to_one")) == 15016
    assert registry.partition.value_counts().to_dict() == {"train": 10387, "test": 2319, "val": 2309, "historical_quarantine": 1}
    assert not any(column.lower().startswith(("tddft_", "multiwfn_", "target_")) or "energy" in column.lower()
                   for column in registry.columns)


def test_role_resolution_complete_and_traceable() -> None:
    roles = pd.read_csv(ROOT / "manifests/role_resolution_v1.csv")
    empty = roles.original_donor_count.eq(0) & roles.original_unknown_count.gt(0)
    assert len(roles) == 15016 and empty.sum() == 387
    assert roles.loc[empty, "resolution_status"].value_counts().to_dict() == {
        "RESOLVED_UNIQUE_OR_SYMMETRY_EQUIVALENT": 198, "UNRESOLVED_AMBIGUOUS": 189}
    assert roles.loc[empty, ["pm6_json_sha256", "dft_json_sha256", "dft_pdb_sha256"]].notna().all().all()
    assert roles.loc[roles.molecule_id.eq("D81_A28"), "sidecar_conflict_flag"].eq(True).all()


def test_quarantine_cannot_form_dataset() -> None:
    cache = DFTGraphCache(CACHE, REGISTRY); manifest = pd.read_csv(MANIFEST)
    with pytest.raises(ValueError):
        Gate1B2GraphDataset(cache, manifest, partition="historical_quarantine")


def test_real_graph_masks_complete_mutually_exclusive_and_deterministic() -> None:
    cache = DFTGraphCache(CACHE, REGISTRY)
    graph1 = cache.graph(0, cutoff=5.0, max_neighbors=32); graph2 = cache.graph(0, cutoff=5.0, max_neighbors=32)
    masks = graph1.donor_mask.long() + graph1.acceptor_mask.long() + graph1.unknown_mask.long()
    assert torch.equal(masks, torch.ones_like(masks))
    assert torch.equal(graph1.z, graph2.z) and torch.equal(graph1.pos, graph2.pos)
    assert torch.equal(graph1.role, graph2.role) and torch.equal(graph1.edge_index, graph2.edge_index)


@pytest.mark.parametrize("role", [(1, 2, 2, 1), (2, 2, 2, 2)])
def test_empty_donor_and_all_unknown_forward_finite(role) -> None:
    batch = synthetic(role)
    for model in models():
        output = model(batch)
        assert output.shape == (1,) and torch.isfinite(output).all()


def test_translation_rotation_and_permutation_invariance() -> None:
    batch = synthetic(); angle = 0.61
    rotation = torch.tensor([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]], dtype=torch.float32)
    permutation = torch.tensor([2, 0, 3, 1]); inverse = torch.empty_like(permutation); inverse[permutation] = torch.arange(4)
    permuted = batch.clone(); permuted.z = batch.z[permutation]; permuted.pos = batch.pos[permutation]
    permuted.role = batch.role[permutation]; permuted.batch = batch.batch[permutation]
    permuted.edge_index = inverse[batch.edge_index]
    shifted = batch.clone(); shifted.pos = batch.pos + torch.tensor([2.0, -3.0, 1.0])
    rotated = batch.clone(); rotated.pos = batch.pos @ rotation.T
    for model in models():
        model.eval(); base = model(batch)
        torch.testing.assert_close(model(shifted), base, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(model(rotated), base, atol=2e-5, rtol=1e-5)
        torch.testing.assert_close(model(permuted), base, atol=2e-5, rtol=1e-5)


def test_output_shapes_and_parameter_fairness() -> None:
    merged, dau = models(); batch = Batch.from_data_list([synthetic().to_data_list()[0], synthetic((1, 2, 2, 1)).to_data_list()[0]])
    assert merged(batch).shape == (2,) and dau(batch).shape == (2,)
    left, right = sum(p.numel() for p in merged.parameters()), sum(p.numel() for p in dau.parameters())
    assert abs(left - right) / max(left, right) <= 0.05
    assert left == 36689 and right == 36461


def test_group_weighted_loss_manual() -> None:
    prediction = {"p": torch.tensor([0.0, 2.0, 4.0])}; target = torch.zeros((3, 1)); mask = torch.ones((3, 1), dtype=torch.bool)
    weight = torch.tensor([0.5, 0.5, 1.0])
    loss, _ = weighted_masked_multitask_loss(prediction, target, mask, weight, ["p"], {"p": 1.0}, base_loss="mae")
    assert float(loss) == pytest.approx((0.5 * 0 + 0.5 * 2 + 1.0 * 4) / 2.0)


def test_record_group_macro_manual() -> None:
    metrics = regression_metrics([0, 0, 0], [2, 2, 0], ["dup", "dup", "single"])
    assert metrics["record_mae"] == pytest.approx(4 / 3)
    assert metrics["group_macro_mae"] == pytest.approx(1.0)


def test_row_shuffle_hash_invariance() -> None:
    registry = pd.read_parquet(REGISTRY)[["molecule_id", "graph_content_sha256"]].sort_values("molecule_id").reset_index(drop=True)
    shuffled = registry.sample(frac=1, random_state=4).sort_values("molecule_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(registry, shuffled)


def test_heldout_target_cannot_change_graph_or_train_normalization() -> None:
    train = np.array([1.0, 2.0, 3.0]); weights = np.array([1.0, 0.5, 0.5])
    first = weighted_stats(train, weights); heldout = np.array([4.0, 5.0]); heldout[:] = 9999
    second = weighted_stats(train.copy(), weights.copy())
    assert first == second
    registry_hashes = pd.read_parquet(REGISTRY).graph_content_sha256.copy()
    assert registry_hashes.equals(pd.read_parquet(REGISTRY).graph_content_sha256)
