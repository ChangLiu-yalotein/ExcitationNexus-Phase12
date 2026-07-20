from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts.gate2c_common import finite_quantile, score_sets, bootstrap_mean, two_way_bootstrap, sha256, verify_config

ROOT=Path(__file__).resolve().parents[1]

def frame():
    return pd.DataFrame({"molecule_id":["m1","m2","m3"],"structure_group_id_v1":["s1","s1","s2"],"donor_structure_group_id_v1":["d1","d1","d2"],"acceptor_structure_group_id_v1":["a1","a2","a1"],"prediction":[0.,2.,1.],"truth":[0.,0.,0.]})

def test_preregistration_and_authorized_minimal_extraction_contract():
    c=json.loads((ROOT/"configs/gate2c_uq_applicability_audit_v1.json").read_text()); verify_config(c)
    lock=json.loads((ROOT/"data_registry/gate2c_preregistration_lock_v1.json").read_text())
    reg=json.loads((ROOT/"data_registry/gate2c_validation_label_registry.json").read_text())
    assert lock["authorization"]=="AUTHORIZED_MINIMAL_VALIDATION_LABEL_EXTRACTION"
    assert reg["source_columns_read"]==["molecule_id","tddft_coulomb_attraction_eV_eps3p5_proxy"]
    assert reg["arrow_reads"]==1 and reg["test_rows_requested_as_test"]==0 and not reg["final673_accessed"]
    assert sha256("runs/gate2c_uq/calibration_labels_v1.parquet")==c["inputs"]["validation_labels"]["sha256"]

def test_finite_conformal_quantile_and_unattainable_boundary():
    x=np.arange(1,10,dtype=float)
    assert finite_quantile(x,.8)["q"]==8
    assert finite_quantile(x,.9)["q"]==9
    assert finite_quantile(x,.95)["status"]=="UNATTAINABLE_FINITE_SAMPLE"

def test_cluster_max_scores_are_not_record_scores():
    scores=score_sets(frame(),"donor_structure_group_id_v1")
    np.testing.assert_array_equal(np.sort(scores["record"]),[0,1,2])
    np.testing.assert_array_equal(np.sort(scores["structure"]),[1,2])
    np.testing.assert_array_equal(np.sort(scores["identity"]),[1,2])

def test_cluster_bootstrap_deterministic_and_order_independent():
    x=np.array([.2,.5,.9]); assert bootstrap_mean(x,100,3)==bootstrap_mean(x[::-1],100,3)
    f=frame(); mapping={"m1":1.,"m2":0.,"m3":1.}; values=f.molecule_id.map(mapping).to_numpy(); shuffled=f.sample(frac=1,random_state=7)
    a=two_way_bootstrap(f,values,200,9); b=two_way_bootstrap(shuffled,shuffled.molecule_id.map(mapping).to_numpy(),200,9)
    assert a==b

def test_protocol_validation_test_overlap_and_forbidden_partitions():
    c=json.loads((ROOT/"configs/gate2c_uq_applicability_audit_v1.json").read_text())
    for spec in c["protocols"].values():
        m=pd.read_csv(ROOT/spec["manifest"]); assert len(m)==15016
        assert not set(m.loc[m.partition.eq("val"),"molecule_id"]) & set(m.loc[m.partition.eq("test"),"molecule_id"])
        assert set(m.partition.unique()) <= {"train","val","test","buffer","historical_quarantine"}

def test_no_test_truth_in_calibrator_or_similarity_registry_when_present():
    p=ROOT/"data_registry/gate2c_calibrator_registry.json"
    if not p.exists(): pytest.skip("calibrator not fit yet")
    text=p.read_text().lower(); data=json.loads(text)
    assert "molecule_id" not in text and "per_sample" not in text
    assert not data["test_residuals_used"]

def test_evaluate_once_fail_closed_contract_when_consumed():
    p=ROOT/"logs/gate2c_coverage_metrics.json"
    if not p.exists(): pytest.skip("test lock not consumed yet")
    unlock=json.loads((ROOT/"data_registry/gate2c_test_unlock_v1.json").read_text())
    assert unlock["consumed"] and unlock["coverage_metrics_sha256"]==sha256(p)

def test_source_firewall_strings():
    evaluation=(ROOT/"scripts/gate2c_evaluate_uq_once.py").read_text().lower()
    assert "molecule_values_v3" not in evaluation and "pyarrow" not in evaluation
    assert "import gate2a_evaluate_once" not in evaluation
    c=json.loads((ROOT/"configs/gate2c_uq_applicability_audit_v1.json").read_text())
    assert c["gate2a_evaluator_calls"]==0 and not c["final673_access"]
