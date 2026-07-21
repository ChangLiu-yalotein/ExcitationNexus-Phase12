import json,sys
from pathlib import Path
import numpy as np, pandas as pd, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'scripts'))
from gate2e1_common import MultiTaskNet,arm_tasks,masked_loss,parameter_contract,task_weights,weighted_stats

def test_config_fixed_arms_and_firewall():
 c=json.loads((ROOT/'configs/gate2e1_physics_multitask_admission_v1.json').read_text()); assert list(c['arms'])==['S0','M11','M15']; assert c['features']['total']==532; assert not c['test_access']; assert not c['final673_access']; assert not c['official_validation_access_during_training']
def test_primary_path_parameter_fairness():
 p=parameter_contract(); assert p['S0']['primary_shapes']==p['M11']['primary_shapes']==p['M15']['primary_shapes']; assert p['S0']['trunk']==p['M11']['trunk']==p['M15']['trunk']; assert p['S0']['primary_head']==p['M11']['primary_head']==p['M15']['primary_head']
def test_masked_weighted_mae_hand_case():
 tasks=arm_tasks('M15'); out={t:torch.tensor([0.,2.,4.],requires_grad=True) for t in tasks}; y=torch.full((3,len(tasks)),float('nan')); y[:,0]=torch.tensor([0.,0.,0.]); y[0,1]=1.; y[2,1]=2.; w=torch.tensor([.5,.5,1.]); loss,per=masked_loss(out,y,w,tasks,task_weights('M15')); assert torch.isfinite(loss); assert torch.isclose(per[tasks[0]],torch.tensor(2.5)); assert torch.isclose(per[tasks[1]],torch.tensor(5/3))
def test_group_weighted_normalization_hand_case():
 d=pd.DataFrame({'x':[0.,0.,2.],'group_weight':[.5,.5,1.]}); s=weighted_stats(d,['x'])['x']; assert np.isclose(s['mean'],1.); assert np.isclose(s['std'],1.)
def test_cpu_forward_backward_and_masks():
 for arm in ('S0','M11','M15'):
  torch.manual_seed(42); tasks=arm_tasks(arm); m=MultiTaskNet(arm,tasks); x=torch.randn(8,532); y=torch.randn(8,len(tasks)); y[::2,-1]=float('nan') if arm=='M15' else y[::2,-1]; w=torch.ones(8); loss,_=masked_loss(m(x),y,w,tasks,task_weights(arm)); loss.backward(); assert torch.isfinite(loss); assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
