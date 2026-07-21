import json,sys
from pathlib import Path
import numpy as np,pandas as pd
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from gate2f1_crossfit import ARM_EXTRA,bootstrap_diff,transform_pair_delta

def test_arm_contract_and_no_gap():
 c=json.loads((ROOT/'configs/gate2f1_multifidelity_delta_crossfit_v1.json').read_text())
 assert c['arms']==['C0','PM6_3','DFT_3','PAIR_6','DELTA_6','DELTA_3','ROLE_9']
 assert all('gap' not in x for cols in ARM_EXTRA.values() for x in cols)
 assert [532+len(ARM_EXTRA[x]) for x in c['arms']]==[532,535,535,538,538,535,541]

def test_pair_delta_bidirectional_reconstruction():
 pm6=np.array([[1.,2.,3.],[4.,5.,6.]]);dft=np.array([[2.,4.,8.],[3.,8.,9.]])
 pair,delta=transform_pair_delta(pm6,dft)
 assert np.array_equal(pair[:,:3]+delta[:,3:],pair[:,3:])
 assert np.array_equal(pair[:,3:]-pair[:,:3],delta[:,3:])

def test_cluster_bootstrap_order_invariant():
 f=pd.DataFrame({'unit':['b','a','b','a'],'y':[0.,0.,1.,1.],'a':[.2,.1,.8,.7],'b':[.3,.4,.6,.9]})
 x=bootstrap_diff(f,'a','b','unit',1000,7);z=bootstrap_diff(f.sample(frac=1,random_state=3),'a','b','unit',1000,7)
 assert x==z

def test_firewall():
 c=json.loads((ROOT/'configs/gate2f1_multifidelity_delta_crossfit_v1.json').read_text())
 assert not c['official_validation_access'] and not c['test_access'] and not c['main_parquet_access'] and not c['final673_access']
