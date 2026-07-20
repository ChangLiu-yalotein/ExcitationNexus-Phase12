from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from scripts.gate1b1_train_cheap_baselines import sha256
from scripts.gate2b_hierarchical_ood_audit import cluster_table, diagnostic_mol, oneway_bootstrap, two_way_bootstrap, verify_inputs, worst_identities

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/gate2b_hierarchical_ood_audit_v1.json"

def synthetic():
    return pd.DataFrame({"molecule_id":["a","b","c"], "structure_group_id_v1":["s1","s1","s2"], "donor_structure_group_id_v1":["d1","d1","d2"], "acceptor_structure_group_id_v1":["a1","a2","a1"], "primary_true":[0.,0.,0.], "model":[2.,2.,0.]})

def test_frozen_artifact_contract_and_no_source_table_access():
    c=json.loads(CONFIG.read_text()); verify_inputs(c)
    assert c["artifact_only"] and not c["main_parquet_access"] and not c["new_predictions"]
    assert c["gate2a_evaluator_calls"]==0 and not c["final673_access"]
    assert (ROOT/"data_registry/gate2a_test_unlock_v1.json").exists()

def test_duplicated_records_do_not_become_independent_clusters():
    f=synthetic(); t=cluster_table(f,"model","structure_group_id_v1")
    assert len(t)==2 and t.set_index("structure_group_id_v1").loc["s1","records"]==2
    assert t.mae.mean()==pytest.approx(1.) and np.abs(f.model).mean()==pytest.approx(4/3)

def test_oneway_cluster_bootstrap_manual_constant_case():
    f=synthetic(); f["model"]=1.; np.testing.assert_allclose(oneway_bootstrap(f,"model","donor_structure_group_id_v1",100,7),1.)

def test_two_way_bootstrap_deterministic_and_order_invariant():
    f=synthetic(); a=two_way_bootstrap(f,"model",500,11); b=two_way_bootstrap(f.sample(frac=1,random_state=99),"model",500,11); np.testing.assert_array_equal(a,b)

def test_bootstrap_seed_changes_draws_but_stays_finite():
    f=synthetic(); a=oneway_bootstrap(f,"model","structure_group_id_v1",100,1); b=oneway_bootstrap(f,"model","structure_group_id_v1",100,2)
    assert not np.array_equal(a,b) and np.isfinite(a).all() and np.isfinite(b).all()

def test_prediction_artifacts_are_immutable_inputs():
    c=json.loads(CONFIG.read_text())
    for key in ("iid_predictions","ood_predictions","ood_metrics"):
        item=c["inputs"][key]; assert sha256(ROOT/item["path"])==item["sha256"]

def test_protocol_specific_inference_units():
    c=json.loads(CONFIG.read_text()); expected={"iid":"structure_group_id_v1","donor_cold":"donor_structure_group_id_v1","acceptor_cold":"acceptor_structure_group_id_v1","pair_cold":"pair_group_id_v1","both_cold":"two_way_donor_acceptor","full_scaffold_cold":"full_scaffold_group_id_v1"}
    assert {k:v["cluster"] for k,v in c["manifests"].items()}==expected
    assert c["bootstrap"]["replicates"]==10000 and c["both_cold_bootstrap"]["method"]=="two_way_pigeonhole_multiplicity_weighting"


def test_diagnostic_parser_and_anonymous_worst_identity_output():
    assert diagnostic_mol("*1ccc[se]c1") is not None
    worst=worst_identities(synthetic(),"model","donor_structure_group_id_v1")
    assert worst[0]["donor_structure_group_id_v1"]=="d1" and "molecule_id" not in worst[0]


def test_structure_group_secondary_is_distinct_from_cold_primary():
    c=json.loads(CONFIG.read_text())
    assert c["manifests"]["acceptor_cold"]["cluster"]!="structure_group_id_v1"
    assert c["manifests"]["iid"]["cluster"]=="structure_group_id_v1"
