import json,sys
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/"scripts"))
from gate2e2a_common import MultiTaskNet,assign_units,bootstrap_diff,old_wrong_unit_targets,reference_weighted_unit_targets,weighted_unit_targets

def test_fixed_contract_and_forbidden_scope():
 c=json.loads((ROOT/'configs/gate2e2a_multitask_crossfit_recovery_v1.json').read_text());assert c['arms']==['S0','M11'];assert c['seeds']==[42,123,456];assert c['outer']['folds']==5;assert 'M15' in c['forbidden'];assert 'main parquet' in c['forbidden']
def test_weighted_unit_target_dual_implementation_rejects_old_bug():
 d=pd.DataFrame({'u':['a','a','a','b'],'y':[0.,0.,10.,3.],'w':[.5,.5,1.,1.]});d=d.rename(columns={'y':'tddft_coulomb_attraction_eV_eps3p5_proxy','w':'group_weight'})
 p=weighted_unit_targets(d,'u');r=reference_weighted_unit_targets(d,'u');old=old_wrong_unit_targets(d,'u');assert p==r;assert p['a']==5.;assert old['a']!=p['a']
def test_assignment_row_order_invariant():
 t={f'u{i}':float(i) for i in range(100)};assert assign_units(t,5,20260721)==assign_units(dict(reversed(list(t.items()))),5,20260721)
def test_primary_path_fairness():
 from gate2e2a_common import tasks
 a=MultiTaskNet('S0',tasks('S0'));b=MultiTaskNet('M11',tasks('M11'))
 sa={n:list(p.shape) for n,p in a.named_parameters() if n.startswith(('trunk.','primary_head.'))};sb={n:list(p.shape) for n,p in b.named_parameters() if n.startswith(('trunk.','primary_head.'))};assert sa==sb
def test_cluster_bootstrap_order_invariant():
 d=pd.DataFrame({"y":[0.,1.,2.,3.],"a":[.1,.9,2.2,2.8],"b":[.2,1.2,1.8,3.3],"u":["x","x","y","z"]}); x=bootstrap_diff(d,"a","b","u",1000,20260721); z=bootstrap_diff(d.sample(frac=1,random_state=8),"a","b","u",1000,20260721); assert x==z

def test_frozen_outputs_are_training_only():
 p=json.loads((ROOT/"logs/gate2e2a_crossfit_metrics.json").read_text()); e=json.loads((ROOT/"logs/gate2e2a_evidence.json").read_text()); r=json.loads((ROOT/"data_registry/gate2e2a_model_registry.json").read_text()); assert p["decision"]=="MULTITASK_CROSSFIT_INCONCLUSIVE"; assert not any([p["official_validation_accessed"],p["test_accessed"],p["main_parquet_accessed"],p["final673_accessed"]]); assert r["neural_models"]==60 and r["xgb_models"]==10 and r["all_finite"]; assert e["training_only_crossfit"]
