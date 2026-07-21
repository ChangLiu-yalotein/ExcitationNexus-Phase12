import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from gate2e0_common import weighted_corr, weighted_spearman


def test_extraction_once_and_protocol_local():
    config = json.loads((ROOT / "configs/gate2e0_multitask_target_audit_v1.json").read_text())
    registry = json.loads((ROOT / "data_registry/gate2e0_auxiliary_extraction_registry.json").read_text())
    assert registry["arrow_reads"] == 1 and registry["union_count"] == 15015
    assert not registry["primary_column_read"] and not registry["generic_union_file_written"]
    assert not registry["test_artifact_accessed"] and not registry["final673_accessed"]
    for protocol, spec in config["protocols"].items():
        manifest = pd.read_csv(ROOT / spec["manifest"])
        forbidden = set(manifest.loc[~manifest.partition.isin(["train", "val"]), "molecule_id"])
        for partition in ("train", "val"):
            item = registry["protocols"][protocol][partition]
            labels = pd.read_parquet(ROOT / item["artifact_path"])
            assert len(labels) == spec[partition] and labels.molecule_id.is_unique
            assert set(labels.molecule_id).isdisjoint(forbidden)
            assert set(labels.columns) == {"molecule_id", *config["secondary"], *config["masked"]}


def test_t_index_known_redundancy_and_fraction_unassigned():
    frame = pd.read_parquet(ROOT / "runs/gate2e0_multitask_target_audit/auxiliary_labels/iid_train_aux_labels.parquet")
    residual = frame.tddft_t_index_angstrom - (frame.tddft_D_index_angstrom - frame.tddft_H_CT_angstrom)
    assert residual.abs().max() <= 0.0010000001
    graph = json.loads((ROOT / "data_registry/gate2e0_target_graph_v2.json").read_text())
    assert "tddft_t_index_angstrom" not in graph["secondary_optimization"]
    assert graph["report_only_redundant"] == ["tddft_t_index_angstrom"]
    for particle in ("hole", "electron"):
        total = frame[f"tddft_{particle}_on_donor_fraction"] + frame[f"tddft_{particle}_on_acceptor_fraction"]
        assert (total.dropna().sub(1.0).abs() > 1e-6).any()


def test_row_order_invariant_weighted_relationships():
    rng = np.random.default_rng(20260720)
    x = rng.normal(size=100); y = 2 * x + rng.normal(scale=0.2, size=100); w = rng.uniform(0.1, 1.0, size=100)
    order = rng.permutation(len(x))
    assert np.isclose(weighted_corr(x, y, w), weighted_corr(x[order], y[order], w[order]))
    assert np.isclose(weighted_spearman(x, y, w), weighted_spearman(x[order], y[order], w[order]))


def test_final_decision_contract():
    evidence = json.loads((ROOT / "logs/gate2e0_evidence.json").read_text())
    assert evidence["scientific_decision"] == "MULTITASK_TARGET_GRAPH_ADMITTED"
    assert evidence["admitted_secondary"] == 11 and evidence["admitted_masked"] == 4
    assert evidence["source_arrow_reads"] == 1
    assert not evidence["model_training"] and not evidence["prediction_generation"] and not evidence["gpu_used"]
    assert not evidence["test_artifact_accessed"] and not evidence["final673_accessed"]
