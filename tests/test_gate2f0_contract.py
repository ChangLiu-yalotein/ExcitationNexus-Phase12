import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'scripts'))
from gate2f0_audit import interface_features
def test_frozen_firewall():
 c=json.loads((ROOT/'configs/gate2f0_acceptor_delta_feasibility_v1.json').read_text());assert c['cpu_only'] and not c['training'] and not c['prediction_generation'];assert not c['test_access'] and not c['final673_access'] and not c['main_parquet_access'];assert 'TDDFT raw files' in c['forbidden']
def test_delta_sign_and_units_contract():
 assert (-.2)-(-.3)==.09999999999999998;c=json.loads((ROOT/'configs/gate2f0_acceptor_delta_feasibility_v1.json').read_text());assert c['delta_pairs']==['homo_hartree','lumo_hartree','gap_hartree','dipole_magnitude_debye']
def test_interface_invariance_and_empty_donor():
 z=np.array([6,6,8]);x=np.array([[0.,0.,0.],[2.,0.,0.],[5.,0.,0.]]);r=np.array(['donor','acceptor','unknown']);a=interface_features(z,x,r,[3.5,4.,5.]);q=np.array([[0.,-1.,0.],[1.,0.,0.],[0.,0.,1.]]);b=interface_features(z,x@q+np.array([8.,-3.,2.]),r,[3.5,4.,5.]);assert all(np.isclose(a[k],b[k],equal_nan=True) for k in a);r[:]='unknown';c=interface_features(z,x,r,[3.5,4.,5.]);assert np.isnan(c['da_min_distance']) and c['donor_present']==0
