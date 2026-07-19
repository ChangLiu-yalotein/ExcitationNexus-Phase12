import pandas as pd
from excitationnexus_phase12.dataset import load_bound_table

def test_join_is_one_to_one_and_order_independent(paths,tmp_path):
    m=paths['manifests']/'split_iid_group_seed42_v1.csv'
    a=load_bound_table(paths['table'],m).set_index('molecule_id').sort_index()
    shuffled=pd.read_parquet(paths['table']).sample(frac=1,random_state=7)
    q=tmp_path/'shuffled.parquet'; shuffled.to_parquet(q,index=False)
    b=load_bound_table(q,m).set_index('molecule_id').sort_index()
    assert len(a)==15016 and a.index.equals(b.index)
    assert a.canonical_smiles.equals(b.canonical_smiles)
