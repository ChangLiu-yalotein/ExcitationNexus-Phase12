from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text())


def test_sequence_length_correction_was_frozen_before_forward():
    path = ROOT / "configs/gate2d2_v2_sequence_length_correction.json"
    correction = json.loads(path.read_text())
    lock = load("data_registry/gate2d2_v2_sequence_length_correction_lock.json")
    smoke = load("logs/gate2d2_v2_long_sequence_smoke.json")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == lock["config_sha256"] == smoke["token_length_correction_sha256"]
    assert lock["molecular_forward_count_before_lock"] == 0
    assert lock["validation_metric_count_before_lock"] == 0
    assert correction["frozen_tokenizer_maxima"] == {"full": 417, "donor": 208, "acceptor": 378}
    assert smoke["status"] == "GATE2D2_V2_LONG_SEQUENCE_FORWARD_PASSED"
    assert smoke["repeat_max_abs"] <= 1e-6
    assert smoke["single_vs_padded_batch_max_abs"] <= 1e-5
    assert {kind: value["max_length"] for kind, value in smoke["summary"].items()} == correction["frozen_tokenizer_maxima"]


def test_embedding_and_model_registries_are_complete_and_firewalled():
    embedding = load("data_registry/gate2d2_v2_embedding_registry.json")
    models = load("data_registry/gate2d2_v2_model_registry.json")
    assert embedding["status"] == "GATE2D2_V2_EMBEDDINGS_FROZEN"
    assert {kind: value["identities"] for kind, value in embedding["categories"].items()} == {"full": 14639, "donor": 154, "acceptor": 352}
    assert embedding["batch_size_sample_max_abs"] <= 1e-5
    assert embedding["trainable_parameters"] == embedding["optimizer_parameters"] == 0
    assert models["status"] == "GATE2D2_V2_VALIDATION_MODELS_FROZEN"
    assert models["new_models"] == 12
    assert set(models["protocols"]) == {"iid", "donor_cold", "acceptor_cold", "pair_cold", "both_cold", "full_scaffold_cold"}
    for registry in (embedding, models):
        assert not registry["test_artifacts_accessed"]
        assert not registry["main_parquet_accessed"]
        assert not registry["final673_accessed"]


def test_final_decision_uses_preregistered_thresholds():
    metrics = load("logs/gate2d2_v2_validation_metrics.json")
    primary = metrics["primary"]
    assert metrics["decision"] == "REPRESENTATION_SIGNAL_INCONCLUSIVE"
    assert primary["acceptor_C_minus_A"]["point"] < 0
    assert primary["acceptor_C_minus_A"]["ci95"][1] > 0
    assert primary["acceptor_C_minus_B"]["point"] <= -0.001
    assert primary["acceptor_C_minus_B"]["ci95"][1] < 0
    assert primary["iid_C_minus_A"]["ci95"][1] > 0.002
    assert not metrics["test_artifacts_accessed"]
    assert not metrics["main_parquet_accessed"]
    assert not metrics["final673_accessed"]
