import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from gate2e1_evaluate_validation_once import cluster_bootstrap_diff

def test_inner_split_zero_leakage_and_forced_fit():
 r=json.loads((ROOT/'data_registry/gate2e1_inner_split_registry.json').read_text())
 for protocol,item in r['protocols'].items():
  d=pd.read_parquet(ROOT/item['path']); unit=item['unit']; assert not(set(d.loc[d.inner_partition.eq('inner_fit'),unit])&set(d.loc[d.inner_partition.eq('inner_checkpoint'),unit]));assert item['historical_overlap_checkpoint']==0
def test_eighteen_models_frozen_before_validation():
 r=json.loads((ROOT/'data_registry/gate2e1_model_registry.json').read_text());assert r['model_count']==18;assert r['status']=='FROZEN_BEFORE_OFFICIAL_VALIDATION';p=r['parameter_contract'];assert p['S0']['primary_shapes']==p['M11']['primary_shapes']==p['M15']['primary_shapes']
def test_validation_unlock_once_and_no_test():
 u=json.loads((ROOT/'data_registry/gate2e1_validation_unlock_v1.json').read_text());e=json.loads((ROOT/'logs/gate2e1_evidence.json').read_text());assert u['consumed'] and u['completed'];assert e['official_validation_invocations']==1 and e['second_invocation_fail_closed'];assert not e['test_accessed'] and not e['main_parquet_accessed'] and not e['final673_accessed']
def test_cluster_bootstrap_order_invariant():
 y=np.array([0.,1.,2.,3.,4.]);a=np.array([.1,.8,2.4,2.8,4.2]);b=np.array([.2,.7,2.1,3.5,4.1]);g=np.array(['a','a','b','c','c']);one=cluster_bootstrap_diff(y,a,b,g,1000,20260721);order=np.array([4,1,3,0,2]);two=cluster_bootstrap_diff(y[order],a[order],b[order],g[order],1000,20260721);assert one==two
def test_decision_is_threshold_derived_not_champion_selection():
 v=json.loads((ROOT/'logs/gate2e1_validation_metrics.json').read_text());assert v['decision']=='MULTITASK_SIGNAL_INCONCLUSIVE';assert v['masked_decision']=='MASKED_FRAGMENT_SIGNAL_INCONCLUSIVE';assert v['comparisons']['acceptor_cold']['M11_minus_S0']['ci95'][1]>0;assert v['comparisons']['acceptor_cold']['M15_minus_M11']['ci95'][1]>0
