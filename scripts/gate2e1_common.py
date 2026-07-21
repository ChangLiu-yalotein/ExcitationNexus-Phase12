#!/usr/bin/env python3
from __future__ import annotations

import hashlib, json, random
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
PRIMARY = "tddft_coulomb_attraction_eV_eps3p5_proxy"

def resolve(p):
    p=Path(p); return p if p.is_absolute() else ROOT/p
def sha(p):
    h=hashlib.sha256()
    with resolve(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def read_json(p): return json.loads(resolve(p).read_text())
def write_json(p,v):
    p=resolve(p); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def config(): return read_json('configs/gate2e1_physics_multitask_admission_v1.json')
def target_graph(): return read_json('data_registry/gate2e0_target_graph_v2.json')
def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)

def load_primary():
    sys_path = str(ROOT/'scripts')
    import sys
    if sys_path not in sys.path: sys.path.insert(0,sys_path)
    from gate2e0_common import load_config, load_primary_labels
    return load_primary_labels(load_config())

def load_protocol(protocol, include_validation=False):
    c=config(); spec=c['protocols'][protocol]
    if sha(spec['manifest'])!=spec['manifest_sha256']: raise RuntimeError('manifest hash mismatch')
    if sha(c['features']['path'])!=c['features']['sha256']: raise RuntimeError('feature hash mismatch')
    m=pd.read_csv(resolve(spec['manifest']))
    allowed=['train','val'] if include_validation else ['train']
    if not include_validation and not set(m.loc[m.partition.isin(allowed),'partition']).issubset({'train'}): raise RuntimeError('training firewall')
    rows=m.loc[m.partition.isin(allowed)].copy()
    f=pd.read_parquet(resolve(c['features']['path']))
    desc=[x for x in f.columns if x.startswith('pair_') and not x.startswith('pair_morgan_') and not x.startswith('pm6_')]
    bits=[f'pair_morgan_{i}' for i in range(512)]
    if len(desc)!=20 or len(bits)!=512 or any(x not in f for x in bits): raise RuntimeError('C0 schema mismatch')
    rows=rows.merge(f[['molecule_id',*desc,*bits]],on='molecule_id',validate='one_to_one')
    rows=rows.merge(load_primary(),on='molecule_id',validate='one_to_one')
    reg=read_json(c['auxiliary_registry']); part_frames=[]
    for part in allowed:
        item=reg['protocols'][protocol][part]
        if sha(item['artifact_path'])!=item['artifact_sha256']: raise RuntimeError('aux hash mismatch')
        a=pd.read_parquet(resolve(item['artifact_path']))
        part_frames.append(a)
    aux=pd.concat(part_frames,ignore_index=True)
    rows=rows.merge(aux,on='molecule_id',validate='one_to_one')
    if len(rows)!=sum(m.partition.eq(x).sum() for x in allowed): raise RuntimeError('join count')
    if not np.isfinite(rows[desc+bits+[PRIMARY]].to_numpy(float)).all(): raise RuntimeError('nonfinite input/primary')
    return rows,desc,bits

def weighted_stats(frame,tasks):
    out={}
    for t in tasks:
        q=frame[[t,'group_weight']].dropna(); w=q.group_weight.to_numpy(float); x=q[t].to_numpy(float)
        mean=float(np.sum(w*x)/np.sum(w)); var=float(np.sum(w*(x-mean)**2)/np.sum(w)); std=var**.5
        out[t]={'mean':mean,'std':std if std>1e-12 else 1.0,'valid':len(q),'effective_weight':float(w.sum())}
    return out

def feature_stats(frame,desc):
    w=frame.group_weight.to_numpy(float); x=frame[desc].to_numpy(float)
    mean=np.sum(w[:,None]*x,axis=0)/w.sum(); std=np.sqrt(np.sum(w[:,None]*(x-mean)**2,axis=0)/w.sum()); std=np.where(std>1e-12,std,1.0)
    return {'columns':desc,'mean':mean.tolist(),'std':std.tolist()}

def arrays(frame,desc,bits,fstats,tstats,tasks):
    xd=(frame[desc].to_numpy(np.float32)-np.asarray(fstats['mean'],np.float32))/np.asarray(fstats['std'],np.float32)
    xb=frame[bits].to_numpy(np.float32); x=np.concatenate([xd,xb],axis=1)
    y=np.full((len(frame),len(tasks)),np.nan,np.float32)
    for j,t in enumerate(tasks):
        y[:,j]=(frame[t].to_numpy(np.float32)-tstats[t]['mean'])/tstats[t]['std']
    return x,y,frame.group_weight.to_numpy(np.float32)

class ResidualBlock(nn.Module):
    def __init__(self):
        super().__init__(); self.linear1=nn.Linear(512,512); self.act=nn.SiLU(); self.drop=nn.Dropout(.1); self.linear2=nn.Linear(512,512); self.norm=nn.LayerNorm(512)
    def forward(self,x): return self.norm(x+self.linear2(self.drop(self.act(self.linear1(x)))))
class Trunk(nn.Module):
    def __init__(self):
        super().__init__(); self.input=nn.Linear(532,512); self.norm=nn.LayerNorm(512); self.act=nn.SiLU(); self.drop=nn.Dropout(.1); self.blocks=nn.ModuleList([ResidualBlock(),ResidualBlock()]); self.output=nn.Linear(512,256)
    def forward(self,x):
        x=self.drop(self.act(self.norm(self.input(x))))
        for b in self.blocks:x=b(x)
        return self.act(self.output(x))
class Head(nn.Module):
    def __init__(self): super().__init__(); self.net=nn.Sequential(nn.Linear(256,128),nn.SiLU(),nn.Linear(128,1))
    def forward(self,x): return self.net(x).squeeze(-1)
class MultiTaskNet(nn.Module):
    def __init__(self,arm,tasks):
        super().__init__(); self.arm=arm; self.tasks=tasks; self.trunk=Trunk(); self.primary_head=Head(); self.auxiliary_heads=nn.ModuleDict({t:Head() for t in tasks[1:]})
    def forward(self,x):
        z=self.trunk(x); out={self.tasks[0]:self.primary_head(z)}
        out.update({t:self.auxiliary_heads[t](z) for t in self.tasks[1:]}); return out

def arm_tasks(arm):
    g=target_graph(); tasks=[g['primary']['column']]
    if arm in ('M11','M15'): tasks+=g['secondary_optimization']
    if arm=='M15': tasks+=g['masked_auxiliary']
    return tasks
def task_weights(arm):
    g=target_graph(); w={PRIMARY:1.0}
    if arm in ('M11','M15'): w.update({t:g['secondary_per_task_weight'] for t in g['secondary_optimization']})
    if arm=='M15': w.update({t:g['masked_per_task_weight'] for t in g['masked_auxiliary']})
    return w
def masked_loss(outputs,y,w,tasks,weights):
    total=torch.zeros((),device=y.device); per={}
    for j,t in enumerate(tasks):
        mask=torch.isfinite(y[:,j]); denom=w[mask].sum()
        if not mask.any() or denom<=0: continue
        loss=(w[mask]*(outputs[t][mask]-y[mask,j]).abs()).sum()/denom; per[t]=loss; total=total+weights[t]*loss
    return total,per
def group_macro_mae(y,p,groups):
    d=pd.DataFrame({'y':y,'p':p,'g':groups}); return float(d.assign(e=lambda z:(z.y-z.p).abs()).groupby('g').e.mean().mean())
def parameter_contract():
    out={}
    for arm in ('S0','M11','M15'):
        set_seed(42); m=MultiTaskNet(arm,arm_tasks(arm)); out[arm]={'total':sum(p.numel() for p in m.parameters()),'trunk':sum(p.numel() for p in m.trunk.parameters()),'primary_head':sum(p.numel() for p in m.primary_head.parameters()),'primary_shapes':{n:list(p.shape) for n,p in m.named_parameters() if n.startswith('trunk.') or n.startswith('primary_head.')}}
    if not(out['S0']['primary_shapes']==out['M11']['primary_shapes']==out['M15']['primary_shapes']): raise RuntimeError('primary path mismatch')
    return out
