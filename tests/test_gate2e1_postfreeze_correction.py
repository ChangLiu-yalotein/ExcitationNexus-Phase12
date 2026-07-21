import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def test_postfreeze_weighting_violation_is_fail_closed():
 c=json.loads((ROOT/'data_registry/gate2e1_postfreeze_integrity_correction_v1.json').read_text());e=json.loads((ROOT/'logs/gate2e1_evidence.json').read_text())
 assert c['status']=='BLOCKED_MULTITASK_PIPELINE_INTEGRITY';assert c['protocols']['iid']['assignment_identical'];assert not c['protocols']['acceptor_cold']['assignment_identical'];assert c['protocols']['acceptor_cold']['symmetric_difference_records']==146;assert not c['rerun_permitted'];assert e['status']=='BLOCKED_MULTITASK_PIPELINE_INTEGRITY';assert e['rerun_performed'] is False
