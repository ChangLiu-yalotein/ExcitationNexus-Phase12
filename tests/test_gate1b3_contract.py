from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from torch_geometric.data import Batch, Data

from excitationnexus_phase12.dft_graph_dataset import DFTGraphCache
from excitationnexus_phase12.edge_cache import ShardedEdgeCache
from excitationnexus_phase12.gate1b3_pipeline import Gate1B3GraphDataset, read_targets
from excitationnexus_phase12.graph_builder import _directed_radius_graph
from excitationnexus_phase12.losses import weighted_masked_multitask_loss
from excitationnexus_phase12.metrics import regression_metrics
from excitationnexus_phase12.models import M3DAUSharedModel, M3MergedModel
from excitationnexus_phase12.normalization import weighted_stats

ROOT = Path(__file__).resolve().parents[1]


def synthetic_graph(role=(0, 1, 2)) -> Data:
    pos = torch.tensor([[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    role_tensor = torch.tensor(role, dtype=torch.long)
    return Data(z=torch.tensor([6, 7, 8]), pos=pos, role=role_tensor,
                edge_index=_directed_radius_graph(pos, 5.0, 32), num_nodes=3)


def test_frozen_parameter_counts_and_difference():
    merged, dau = M3MergedModel(), M3DAUSharedModel()
    a, b = sum(p.numel() for p in merged.parameters()), sum(p.numel() for p in dau.parameters())
    assert (a, b) == (36689, 36461)
    assert abs(a - b) / max(a, b) <= .05


@pytest.mark.parametrize("roles", [(1, 1, 2), (2, 2, 2)])
def test_empty_donor_and_all_unknown_finite(roles):
    batch = Batch.from_data_list([synthetic_graph(roles)])
    assert torch.isfinite(M3MergedModel()(batch)).all()
    assert torch.isfinite(M3DAUSharedModel()(batch)).all()


@pytest.mark.parametrize("model", [M3MergedModel(), M3DAUSharedModel()])
def test_translation_rotation_and_synchronous_permutation(model):
    graph = synthetic_graph(); base = model(Batch.from_data_list([graph]))
    translated = copy.deepcopy(graph); translated.pos += torch.tensor([2., -3., .5])
    angle = .73; rotation = torch.tensor([[np.cos(angle), -np.sin(angle), 0.],
                                          [np.sin(angle), np.cos(angle), 0.], [0., 0., 1.]], dtype=graph.pos.dtype)
    rotated = copy.deepcopy(graph); rotated.pos = rotated.pos @ rotation.T
    permutation = torch.tensor([2, 0, 1]); inverse = torch.empty_like(permutation); inverse[permutation] = torch.arange(3)
    permuted = copy.deepcopy(graph); permuted.z = graph.z[permutation]; permuted.pos = graph.pos[permutation]
    permuted.role = graph.role[permutation]; permuted.edge_index = inverse[graph.edge_index]
    assert torch.allclose(base, model(Batch.from_data_list([translated])), atol=2e-5)
    assert torch.allclose(base, model(Batch.from_data_list([rotated])), atol=2e-5)
    assert torch.allclose(base, model(Batch.from_data_list([permuted])), atol=2e-5)


def test_group_weighted_loss_and_metrics_hand_calculation():
    pred = torch.tensor([1., 3., 6.]); target = torch.tensor([[0.], [0.], [2.]])
    weight = torch.tensor([.5, .5, 1.]); mask = torch.ones((3, 1), dtype=torch.bool)
    loss, _ = weighted_masked_multitask_loss({"primary": pred}, target, mask, weight,
                                              ["primary"], {"primary": 1.}, base_loss="mae")
    assert float(loss) == pytest.approx((.5 + 1.5 + 4.) / 2.)
    metrics = regression_metrics([0., 0., 2.], [1., 3., 6.], ["g1", "g1", "g2"])
    assert metrics["record_mae"] == pytest.approx(8 / 3)
    assert metrics["group_macro_mae"] == pytest.approx(3.)


def test_train_only_weighted_normalization_unchanged_by_heldout():
    initial = weighted_stats(pd.Series([1., 3.]), pd.Series([.5, .5]))
    heldout_a, heldout_b = np.array([4., 5.]), np.array([4000., -5000.])
    assert not np.array_equal(heldout_a, heldout_b)
    assert initial == weighted_stats(pd.Series([1., 3.]), pd.Series([.5, .5]))


def test_test_target_and_quarantine_firewalls():
    with pytest.raises(PermissionError, match="TEST_TARGET_FIREWALL_LOCKED"):
        read_targets("unused.parquet", "target", ["x"], requested_partition="test")
    with pytest.raises(PermissionError, match="train/val"):
        Gate1B3GraphDataset(None, None, pd.DataFrame(), partition="historical_quarantine", targets=pd.DataFrame())


def test_checkpoint_resume_deterministic(tmp_path):
    torch.manual_seed(7); x = torch.tensor([[1., 2.]]); y = torch.tensor([[3.]])
    model = torch.nn.Linear(2, 1); opt = torch.optim.AdamW(model.parameters(), lr=.01)
    loss = (model(x) - y).square().mean(); loss.backward(); opt.step(); opt.zero_grad()
    checkpoint = {"model": copy.deepcopy(model.state_dict()), "optimizer": copy.deepcopy(opt.state_dict())}
    path = tmp_path / "resume.pt"; torch.save(checkpoint, path)
    loss = (model(x) - y).square().mean(); loss.backward(); opt.step(); expected = copy.deepcopy(model.state_dict())
    resumed = torch.nn.Linear(2, 1); resumed_opt = torch.optim.AdamW(resumed.parameters(), lr=.01)
    state = torch.load(path, weights_only=False); resumed.load_state_dict(state["model"]); resumed_opt.load_state_dict(state["optimizer"])
    loss = (resumed(x) - y).square().mean(); loss.backward(); resumed_opt.step()
    assert all(torch.equal(expected[k], resumed.state_dict()[k]) for k in expected)


def test_real_cached_edges_equal_dynamic_if_cache_available():
    torch.set_num_threads(1)
    registry_path = ROOT / "data_registry/gate1b3_edge_cache_registry.json"
    if not registry_path.exists(): pytest.skip("edge cache build pending")
    source = DFTGraphCache(ROOT / "runs/gate1b2_3d_admission/dft_graph_cache_v1.npz",
                           ROOT / "data_registry/dft_3d_graph_registry_v1.parquet")
    cached = ShardedEdgeCache(registry_path)
    for row in source.registry.iloc[[0, 97, 1001, 5001, 10001, 15015]].itertuples(index=False):
        a, b = int(row.atom_offset_start), int(row.atom_offset_end)
        dynamic = _directed_radius_graph(torch.from_numpy(source.pos[a:b].copy()).float(), 5., 32)
        assert torch.equal(dynamic, cached.edge_index(str(row.molecule_id)))


def test_cache_corruption_fails(tmp_path):
    registry_path = ROOT / "data_registry/gate1b3_edge_cache_registry.json"
    if not registry_path.exists(): pytest.skip("edge cache build pending")
    registry = json.loads(registry_path.read_text()); (tmp_path / "data_registry").mkdir(); (tmp_path / "shards").mkdir()
    (tmp_path / "shards/bad.npz").write_bytes(b"corrupt")
    registry["cache_root"] = "shards"; registry["shards"] = [{"file": "bad.npz", "sha256": "0" * 64}]
    local = tmp_path / "data_registry/registry.json"; local.write_text(json.dumps(registry))
    with pytest.raises(ValueError, match="hash failure"):
        ShardedEdgeCache(local)
