from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts/gate2d2_audit_model_asset.py"


def load_module():
    spec = importlib.util.spec_from_file_location("gate2d2_audit", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_tokenizer_happy_path_and_unknown_fail_closed():
    module = load_module()
    vocab = {"C": 0, "*": 1, "<bos>": 2, "<eos>": 3}
    assert module.audit_strings(["CC", "C*C"], vocab)["tokenizer_success"]
    failed = module.audit_strings(["[12*]C"], vocab)
    assert not failed["tokenizer_success"]
    assert failed["unknown_token_sequences"] == 1


def test_sequence_limit_fail_closed():
    module = load_module()
    vocab = {"C": 0}
    failed = module.audit_strings(["C" * module.MAX_SEQUENCE_LENGTH], vocab)
    assert failed["over_max_sequence_length"] == 1
    assert not failed["tokenizer_success"]


def test_gate2d2_firewall_and_fixed_asset():
    config = json.loads((ROOT / "configs/gate2d2_frozen_molformer_admission_v1.json").read_text())
    assert config["model"]["repo_id"] == "ibm-research/MoLFormer-XL-both-10pct"
    assert len(config["model"]["revision"]) == 40
    assert config["validation_only"] is True
    assert config["main_parquet_access"] is False
    assert config["test_artifact_access"] is False
    assert config["final673_access"] is False
    assert config["encoder"]["requires_grad"] is False
    assert config["encoder"]["optimizer_parameter_count"] == 0
    assert config["encoder"]["pooling"] == "attention-mask-aware mean pooling of final hidden state"


def test_feature_arms_are_equal_dimension():
    config = json.loads((ROOT / "configs/gate2d2_frozen_molformer_admission_v1.json").read_text())
    assert {arm["columns"] for arm in config["feature_arms"].values()} == {532}
    assert config["feature_arms"]["B_MF_Full_512"]["pca_dimensions"] == 512
    assert config["feature_arms"]["C_MF_Role_512"]["donor_pca_dimensions"] == 256
    assert config["feature_arms"]["C_MF_Role_512"]["acceptor_pca_dimensions"] == 256


def test_blocked_audit_contains_no_sensitive_access():
    path = ROOT / "logs/gate2d2_embedding_audit.json"
    if not path.exists():
        return
    audit = json.loads(path.read_text())
    assert audit["test_artifacts_accessed"] is False
    assert audit["main_parquet_accessed"] is False
    assert audit["final673_accessed"] is False
    assert audit["input_strings_modified"] is False


def test_preregistered_donor_pca_is_mathematically_infeasible():
    module_path = ROOT / "scripts/gate2d2_build_protocol_features.py"
    spec = importlib.util.spec_from_file_location("gate2d2_features", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.max_identifiable_pcs(154) == 153
    assert module.max_identifiable_pcs(124) == 123
    assert module.max_identifiable_pcs(154) < 256


def test_blocker_registries_fail_closed_before_embedding_or_models():
    pca = json.loads((ROOT / "data_registry/gate2d2_pca_registry.json").read_text())
    emb = json.loads((ROOT / "data_registry/gate2d2_embedding_registry.json").read_text())
    models = json.loads((ROOT / "data_registry/gate2d2_model_registry.json").read_text())
    assert pca["status"] == "BLOCKED_PREREGISTERED_PCA_INFEASIBLE"
    assert all(not item["donor_pca_feasible"] for item in pca["protocols"].values())
    assert emb["raw_embeddings_created"] is False
    assert emb["remote_code_executed"] is False
    assert models["new_models"] == 0
    assert models["validation_predictions_created"] == 0
    assert not pca["test_artifacts_accessed"]
    assert not pca["main_parquet_accessed"]
    assert not pca["final673_accessed"]
