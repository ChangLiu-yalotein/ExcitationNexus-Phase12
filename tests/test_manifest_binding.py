import pandas as pd
from excitationnexus_phase12.contracts import MANIFEST_FILES, verify_frozen_inputs

EXPECTED={
'iid_group_seed42_v1':{'train':10387,'val':2309,'test':2319,'historical_quarantine':1},
'donor_cold_v1':{'train':10530,'val':2234,'test':2251,'historical_quarantine':1},
'acceptor_cold_v1':{'train':10543,'val':2235,'test':2237,'historical_quarantine':1},
'pair_cold_v1':{'train':10387,'val':2319,'test':2309,'historical_quarantine':1},
'both_cold_external_test_v1':{'train':9345,'val':1792,'test':587,'buffer':3291,'historical_quarantine':1},
'full_scaffold_cold_v1':{'train':10511,'val':2254,'test':2250,'historical_quarantine':1}}

def test_frozen_hashes(paths):
    assert len(verify_frozen_inputs(paths['table'],paths['manifests']))==7

def test_six_manifest_counts(paths):
    for name,file in MANIFEST_FILES.items():
        got=pd.read_csv(paths['manifests']/file).partition.value_counts().to_dict()
        assert got==EXPECTED[name]

def test_iid_historical_filename_metadata(paths):
    import json
    x=json.load(open(paths['root']/'configs/gate0d_data_contract_v1.json'))
    assert x['iid_metadata']=={'selected_candidate_seed':123,'filename_seed_label_is_historical':True}
