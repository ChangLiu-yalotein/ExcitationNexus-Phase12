from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts.gate2d1_common import ROOT, TARGET, config_and_verify, content_hash, paired_cluster_bootstrap, sha
from scripts.gate2d1_build_role_aware_features import parse

def test_preregistration_and_arm_budget():
    c=config_and_verify(); a=c["representations"]
    assert [a[x]["columns"] for x in a]==[532,1556,1556]
    assert a["B_C0_Wide_1536"]["full_bits"]==a["C_RA2D_1536"]["full_bits"]+a["C_RA2D_1536"]["donor_bits"]+a["C_RA2D_1536"]["acceptor_bits"]==1536
    assert c["fingerprint"]=={"algorithm":"RDKit Morgan","radius":2,"use_chirality":False,"training_bit_budget_B_C":1536}

def test_protocol_local_label_supplements():
    c=config_and_verify(); reg=json.loads((ROOT/"data_registry/gate2d1_train_label_extraction_registry.json").read_text())
    assert reg["union_count"]==1017 and reg["arrow_reads"]==1 and not reg["final673_accessed"]
    for name,info in reg["protocols"].items():
        m=pd.read_csv(ROOT/c["protocols"][name]["manifest"]); x=pd.read_parquet(ROOT/info["artifact_path"])
        assert len(x)==info["supplement_count"] and x.molecule_id.is_unique and np.isfinite(x[TARGET]).all()
        assert set(x.molecule_id)<=set(m.loc[m.partition.eq("train"),"molecule_id"])
        assert not set(x.molecule_id)&set(m.loc[~m.partition.eq("train"),"molecule_id"])
        assert sha(info["artifact_path"])==info["artifact_sha256"]

def test_component_parser_preserves_wildcard_and_non_kekulizing_fallback():
    mol,mode=parse("*c1ccccc1")
    assert mol is not None and any(a.GetAtomicNum()==0 for a in mol.GetAtoms())
    mol2,mode2=parse("*1ccc[se]c1")
    assert mol2 is not None and any(a.GetAtomicNum()==0 for a in mol2.GetAtoms())

def test_feature_content_hash_input_order_invariant():
    ids=np.array(["a","b","c"]); x=np.arange(6,dtype=np.float32).reshape(3,2); order=np.array([2,0,1]); sort=np.argsort(ids[order])
    assert content_hash(ids,x)==content_hash(ids[order][sort],x[order][sort])

def test_paired_cluster_bootstrap_manual_and_order_invariant():
    f=pd.DataFrame({TARGET:[0.,0.,0.],"g":["a","a","b"]}); left=np.array([2.,2.,0.]); right=np.array([1.,1.,1.])
    a=paired_cluster_bootstrap(f,left,right,"g",500,7); order=np.array([2,0,1]); b=paired_cluster_bootstrap(f.iloc[order].reset_index(drop=True),left[order],right[order],"g",500,7)
    assert a==b and a["point"]==pytest.approx(0.)

def test_no_test_or_source_table_paths_in_main_config():
    c=config_and_verify(); text=(ROOT/"configs/gate2d1_role_aware_2d_v1.json").read_text().lower()
    assert not c["test_artifact_access"] and not c["main_parquet_access_after_extraction"] and not c["final673_access"]
    assert "test_predictions_once" not in text and "molecule_values_v3" not in text
    assert all(x["arm_a_validation"].endswith("val_predictions.csv") for x in c["protocols"].values())

def test_feature_schema_when_frozen():
    p=ROOT/"data_registry/gate2d1_feature_schema_v1.json"
    if not p.exists(): pytest.skip("features not frozen yet")
    x=json.loads(p.read_text()); assert x["records"]==15016 and x["arms"]=={"A_C0_512_reference":532,"B_C0_Wide_1536":1556,"C_RA2D_1536":1556}
    assert x["c0_exact_match_records"]==15016 and x["target_columns"]==[] and x["input_order_invariant"]

def test_model_registry_validation_only_when_frozen():
    p=ROOT/"data_registry/gate2d1_model_registry.json"
    if not p.exists(): pytest.skip("models not frozen yet")
    x=json.loads(p.read_text()); assert x["new_models"]==12 and not x["test_artifacts_accessed"] and not x["main_parquet_accessed"] and not x["final673_accessed"]
