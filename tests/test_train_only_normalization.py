import pandas as pd
from excitationnexus_phase12.normalization import fit_train_only_normalization,weighted_stats

def test_train_only_and_group_weighted():
    d=pd.DataFrame({'partition':['train','train','val','test','buffer','historical_quarantine'],
                    'group_weight':[.5,.5,1,1,1,1],'x':[0,2,999,999,999,999],'y':[2,4,999,999,999,999]})
    a=fit_train_only_normalization(d,['x'],['y'],manifest_sha256='m',table_sha256='t')
    d.loc[d.partition.ne('train'),'y']=-999999
    b=fit_train_only_normalization(d,['x'],['y'],manifest_sha256='m',table_sha256='t')
    assert a==b and a['targets']['y']['mean']==3

def test_zero_std_and_empty_are_safe():
    assert weighted_stats([1,1],[1,1])['std']==1
    assert weighted_stats([float('nan')],[1])['std']==1
