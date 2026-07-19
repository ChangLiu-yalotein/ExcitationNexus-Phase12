import pandas as pd
import pytest
from excitationnexus_phase12.contracts import TaskGraph
from excitationnexus_phase12.dataset import Phase12Dataset

def frame():
    return pd.DataFrame({'molecule_id':['a','b','c','d'],'partition':['train','val','buffer','historical_quarantine'],
      'structure_group_id_v1':['1','2','3','4'],'donor_structure_group_id_v1':['d']*4,
      'acceptor_structure_group_id_v1':['a']*4,'pair_group_id_v1':['p']*4,'role_aware_group_id_v1':['r']*4,
      'group_weight':[1.]*4,'num_atoms_total':[2]*4})

def test_explicit_partition_excludes_buffer_and_quarantine(paths):
    ds=Phase12Dataset(frame(),partition='train',view='table_only',raw_root=paths['raw'],task_graph=TaskGraph((),(),(),(),()))
    assert len(ds)==1 and ds[0]['molecule_id']=='a'

def test_forbidden_partition_rejected(paths):
    with pytest.raises(ValueError): Phase12Dataset(frame(),partition='buffer',view='table_only',raw_root=paths['raw'],task_graph=TaskGraph((),(),(),(),()))
