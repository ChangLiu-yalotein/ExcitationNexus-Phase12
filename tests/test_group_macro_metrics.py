import math
from excitationnexus_phase12.metrics import regression_metrics

def test_record_and_group_macro_known_difference():
    # group A has two errors 1,1; group B has one error 4
    m=regression_metrics([0,0,0],[1,1,4],['A','A','B'])
    assert math.isclose(m['record_mae'],2) and math.isclose(m['group_macro_mae'],2.5)

def test_role_inconsistent_replicates_not_target_averaged():
    # Averaging target/pred first would incorrectly give zero group error.
    m=regression_metrics([0,2],[2,0],['same','same'])
    assert m['group_macro_mae']==2

def test_constant_target_r2_reason():
    m=regression_metrics([1,1],[1,2],['a','b'])
    assert math.isnan(m['record_r2']['value']) and m['record_r2']['reason']=='CONSTANT_TARGET'
