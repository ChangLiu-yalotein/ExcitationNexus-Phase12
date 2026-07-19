import pandas as pd

def test_group_size_2_3_4_sum_one():
    d=pd.DataFrame([(n,i,1/n) for n in (2,3,4) for i in range(n)],columns=['group','row','weight'])
    assert d.groupby('group').weight.sum().round(12).eq(1).all()

def test_real_manifest_group_weights(paths):
    d=pd.read_csv(paths['manifests']/'split_iid_group_seed42_v1.csv')
    assert d.groupby('structure_group_id_v1').group_weight.sum().sub(1).abs().max()<1e-10
