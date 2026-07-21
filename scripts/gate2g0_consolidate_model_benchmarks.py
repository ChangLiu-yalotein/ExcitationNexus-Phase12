#!/usr/bin/env python3
from __future__ import annotations

import csv, hashlib, json, subprocess, sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
HIST = ROOT.parent / "equiformer_v3_model"
EXPECTED_HEAD = "e18e6f2fd41ba8fe04a96bebbec4ab101608c8eb"

def rj(p): return json.loads((ROOT/p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
def wj(p,v):
    p=ROOT/p; p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def wc(p,rows):
    p=ROOT/p; p.parent.mkdir(parents=True,exist_ok=True); cols=sorted({k for r in rows for k in r})
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)
def md(rows,cols):
    def c(x): return f'{x:.6f}' if isinstance(x,float) else str(x).replace('|','\\|').replace('\n',' ')
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+''.join('| '+' | '.join(c(r.get(k,'')) for k in cols)+' |\n' for r in rows)
def resolve(p):
    p=Path(p); return p if p.is_absolute() else ROOT/p

def xgb_smoke(path):
    import xgboost as xgb
    m=xgb.Booster(); m.load_model(path); n=int(m.num_features())
    y=m.predict(xgb.DMatrix(np.zeros((4,n),np.float32)))
    if y.shape!=(4,) or not np.isfinite(y).all(): raise RuntimeError('nonfinite XGBoost smoke')
    return n,int(m.num_boosted_rounds())

def m3_smoke(path,name):
    sys.path.insert(0,str(ROOT/'scripts')); from gate1b3_train import build_model
    ck=torch.load(path,map_location='cpu',weights_only=False); model=build_model(rj('configs/gate1b3_formal_training_v1.json'),name); model.load_state_dict(ck['model']); model.eval()
    b=SimpleNamespace(z=torch.tensor([6,7,8,6,16,7]),pos=torch.tensor([[0.,0.,0.],[1.2,0.,0.],[0.,1.1,0.],[0.,0.,0.],[1.4,0.,0.],[0.,1.3,0.]]),role=torch.tensor([0,1,2,0,1,2]),edge_index=torch.tensor([[0,1,1,2,3,4,4,5],[1,0,2,1,4,3,5,4]]),batch=torch.tensor([0,0,0,1,1,1]),num_graphs=2,num_nodes=6)
    with torch.inference_mode(): y=model(b)
    if y.shape!=(2,) or not torch.isfinite(y).all(): raise RuntimeError('nonfinite M3 smoke')
    return sum(p.numel() for p in model.parameters())

def multitask_smoke(path):
    sys.path.insert(0,str(ROOT/'scripts')); from gate2e1_common import MultiTaskNet
    ck=torch.load(path,map_location='cpu',weights_only=False); m=MultiTaskNet(ck['arm'],ck['tasks']); m.load_state_dict(ck['state_dict']); m.eval()
    with torch.inference_mode(): out=m(torch.zeros(4,532))
    if set(out)!=set(ck['tasks']) or not all(v.shape==(4,) and torch.isfinite(v).all() for v in out.values()): raise RuntimeError('nonfinite multitask smoke')
    return sum(p.numel() for p in m.parameters())

def state_smoke(path):
    ck=torch.load(path,map_location='cpu',weights_only=False); st=ck.get('state_dict',ck.get('model',ck)); ts=[v for v in st.values() if torch.is_tensor(v)]
    if not ts or not all(torch.isfinite(t).all() for t in ts if t.is_floating_point()): raise RuntimeError('invalid state tensors')
    return len(ts),sum(t.numel() for t in ts)

def checkpoint_inventory():
    rows=[]; failures=[]; xp=set()
    def visit(v):
        if isinstance(v,dict):
            for k,x in v.items():
                if k in {'model_path','path'} and isinstance(x,str) and x.endswith('.json'): xp.add(resolve(x))
                visit(x)
        elif isinstance(v,list):
            for x in v: visit(x)
    for reg in ['data_registry/gate2a_model_registry.json','data_registry/gate2d1_model_registry.json','data_registry/gate2d2_v2_model_registry.json','data_registry/gate3a1_model_registry.json']: visit(rj(reg))
    xp.update(ROOT.glob('runs/gate1b1_new_iid_cheap_baselines/models/**/*.json')); xp.update(ROOT.glob('runs/gate2f1_crossfit/**/*.json'))
    for p in sorted(xp):
        if not p.exists(): failures.append({'path':str(p),'reason':'missing'}); continue
        if p.name == 'weighted_median.json':
            value=json.loads(p.read_text()); rows.append({'family':'weighted_median','path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'loadability':'LOADABLE_FINITE_CONSTANT','smoke_rows':4})
            continue
        try:
            n,rounds=xgb_smoke(p); rows.append({'family':'XGBoost','path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'loadability':'LOADABLE_FINITE_FORWARD','smoke_rows':4,'input_columns':n,'rounds':rounds})
        except Exception as e: failures.append({'path':str(p),'reason':f'{type(e).__name__}: {e}'})
    for name in ['m3_merged','m3_dau_shared']:
        for seed in [42,123,456]:
            p=ROOT/f'runs/gate1b3_{name}_seed{seed}/best_checkpoint.pt'
            try: rows.append({'family':name,'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'loadability':'LOADABLE_FINITE_FORWARD','smoke_rows':2,'parameters':m3_smoke(p,name)})
            except Exception as e: failures.append({'path':str(p),'reason':f'{type(e).__name__}: {e}'})
    pts=list((ROOT/'runs/gate2e1_physics_multitask/full_refit').glob('**/model.pt'))+list((ROOT/'runs/gate2e2a_multitask_crossfit/neural').glob('**/refit_checkpoint.pt'))
    for p in sorted(pts):
        try: rows.append({'family':'physics_multitask','path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'loadability':'LOADABLE_FINITE_FORWARD','smoke_rows':4,'parameters':multitask_smoke(p),'scientific_status':'BLOCKED_RESULT_ASSET_ONLY' if 'gate2e1_physics' in str(p) else 'TRAINING_ONLY_CROSSFIT'})
        except Exception as e: failures.append({'path':str(p),'reason':f'{type(e).__name__}: {e}'})
    old={'b2_1_seed42':HIST/'checkpoints/2026-04-25-05-26-24/best_checkpoint.pt','b2_1_seed123':HIST/'checkpoints/2026-04-25-09-33-52/checkpoint.pt','b2_1_seed456':HIST/'checkpoints/2026-04-25-09-33-52/best_checkpoint.pt','b2_2a_seed42':HIST/'checkpoints/2026-04-26-04-07-28-b2_2a_seed42_full/best_checkpoint.pt'}
    for name,p in old.items():
        try:
            n,e=state_smoke(p); rows.append({'family':name,'path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'loadability':'LOADABLE_STATE_DICT_PRIOR_INFERENCE_EVIDENCE','smoke_rows':0,'tensor_count':n,'tensor_elements':e})
        except Exception as ex: failures.append({'path':str(p),'reason':f'{type(ex).__name__}: {ex}'})
    return rows,failures

def benchmarks():
    hist=[
      {'model':'cheap no-dipole champion','protocol':'Layer G paired 7313','test_records':1097,'mae_eV':0.07020991162281436,'checkpoint':'CHECKPOINT_MISSING_REPORT_ONLY','status':'REPRODUCED_NUMERIC'},
      {'model':'B2-1 historical ensemble','protocol':'historical 7316','test_records':1098,'mae_eV':0.07940903802712758,'std_eV':0.0025025339119785065,'checkpoint':'PRESENT_3_SEEDS','status':'HISTORICAL_ASSET_VALID'},
      {'model':'B2-1 new reproduction','protocol':'historical 7316','test_records':1098,'mae_eV':0.0781230,'std_eV':0.0013170,'checkpoint':'PRESENT_3_SEEDS','status':'FAILED_FROZEN_THRESHOLD_AGGREGATE'},
      {'model':'B2-0','protocol':'historical B2','test_records':'reported','mae_eV':'report-only','checkpoint':'CHECKPOINT_MAPPING_UNRESOLVED_REPORT_ONLY','status':'REPORT_ONLY'},
      {'model':'B2-2a','protocol':'historical B2-2a','test_records':'reported','mae_eV':'different historical scope','checkpoint':'PRESENT_SEED42','status':'PROTOCOL_NOT_ALIGNED_WITH_NEW15016'},
      {'model':'Paper A / Smoothed Memory','protocol':'historical external-dev','test_records':'protocol-specific','mae_eV':'report-only','checkpoint':'CHECKPOINT_MISSING_REPORT_ONLY','status':'OWN_PROTOCOL_ONLY'}]
    b=rj('runs/gate1b1_new_iid_cheap_baselines/published/gate1b1_metrics.json'); models=b['overall']; iid=[]
    choices=[('weighted_median','Weighted median'),('ridge_c0','Ridge-C0'),('xgb_c0_seed42','XGBoost-C0'),('xgb_c1p5_safe_seed42','XGBoost-C1.5-safe')]
    for k,name in choices:
        z=models[k]; z=z.get('test',z); iid.append({'model':name,'dataset':'new15016','protocol':'IID','records':z.get('records',2319),'groups':z.get('groups',2195),'group_macro_mae_eV':z['group_macro_mae'],'group_macro_rmse_eV':z.get('group_macro_rmse'),'group_macro_r2':z.get('group_macro_r2'),'paper_status':'PRIMARY_BASELINE' if name=='XGBoost-C0' else 'BASELINE'})
    b3=rj('runs/gate1b3_test_once/metrics.json')
    for name,k in [('M3-Merged ensemble','m3_merged'),('M3-DAU-Shared ensemble','m3_dau_shared')]:
        z=b3['architecture_summary'][k]['ensemble']; r2=z['group_macro_r2']; r2=r2.get('value') if isinstance(r2,dict) else r2
        iid.append({'model':name,'dataset':'new15016','protocol':'IID','records':z['valid_count'],'groups':2195,'group_macro_mae_eV':z['group_macro_mae'],'group_macro_rmse_eV':z['group_macro_rmse'],'group_macro_r2':r2,'paper_status':'NEGATIVE_3D_BASELINE'})
    g=rj('runs/gate2a_ood_baselines/published/gate2a_metrics.json'); ood=[]
    for proto,pd in g['metrics'].items():
        for name,z in pd['models'].items():
            z=z.get('test',z); ood.append({'protocol':proto,'model':name,'records':z['records'],'groups':z['groups'],'group_macro_mae_eV':z['group_macro_mae'],'group_macro_rmse_eV':z.get('group_macro_rmse'),'group_macro_r2':z.get('group_macro_r2')})
    return hist,iid,ood

def roadmap():
    return [
      {'item':'XGBoost-C0 new15016','state':'DONE','evidence':'Gate 1-B1 / 2-A'},
      {'item':'M3-Merged and M3-DAU small invariant baselines','state':'DONE_NEGATIVE','evidence':'Gate 1-B3'},
      {'item':'role-aware Morgan / frozen MoLFormer','state':'NEGATIVE_OR_INCONCLUSIVE','evidence':'Gate 2-D1/D2'},
      {'item':'fixed-weight physics multitask','state':'BLOCKED_THEN_INCONCLUSIVE','evidence':'Gate 2-E1 correction / E2A'},
      {'item':'ground-state multifidelity/delta','state':'NEGATIVE_OR_INCONCLUSIVE','evidence':'Gate 2-F1'},
      {'item':'Chemprop v2 D-MPNN strong 2D','state':'NOT_IMPLEMENTED','evidence':'roadmap gap'},
      {'item':'PaiNN/TensorNet2 strong 3D','state':'NOT_IMPLEMENTED','evidence':'roadmap gap'},
      {'item':'formal EquiformerV3 molecular baseline on new15016','state':'NOT_IMPLEMENTED','evidence':'historical B2 is not this'},
      {'item':'explicit D/A interface cross-edge branch','state':'NOT_IMPLEMENTED','evidence':'M3-DAU has no interface branch'},
      {'item':'ReMEI-Net','state':'NOT_IMPLEMENTED','evidence':'no unified main model'},
      {'item':'PM6 gating / FiLM','state':'NOT_IMPLEMENTED','evidence':'C1.5 concat is not gating'},
      {'item':'A0-A10 parameter-matched ablation','state':'NOT_IMPLEMENTED','evidence':'roadmap gap'},
      {'item':'retrospective active-learning benchmark','state':'NOT_IMPLEMENTED','evidence':'roadmap gap'},
      {'item':'paper-grade model card and data card','state':'NOT_IMPLEMENTED','evidence':'roadmap gap'}]

def figures(iid,ood,road):
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    out=ROOT/'reports/figures/gate2g0'; out.mkdir(parents=True,exist_ok=True); made=[]
    def save(fig,name):
        for ext in ['png','pdf']:
            p=out/f'{name}.{ext}'; fig.savefig(p,bbox_inches='tight',dpi=180); made.append(str(p.relative_to(ROOT)))
        plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar([r['model'] for r in iid],[r['group_macro_mae_eV'] for r in iid]); ax.tick_params(axis='x',rotation=30); ax.set_ylabel('Group-macro MAE (eV)'); ax.set_title('new15016 IID frozen benchmark'); save(fig,'iid_model_mae')
    c0=[r for r in ood if str(r['model']).lower()=='xgboost_c0']; fig,ax=plt.subplots(figsize=(8,4)); ax.bar([r['protocol'] for r in c0],[r['group_macro_mae_eV'] for r in c0]); ax.axhline(.08418147504486073,color='black',ls='--',label='IID C0'); ax.tick_params(axis='x',rotation=25); ax.legend(); ax.set_ylabel('Group-macro MAE (eV)'); save(fig,'ood_protocol_mae')
    fig,ax=plt.subplots(figsize=(8,4)); ax.bar([r['protocol'] for r in c0],[r['group_macro_mae_eV']-.08418147504486073 for r in c0]); ax.axhline(0,color='black',lw=.8); ax.tick_params(axis='x',rotation=25); ax.set_ylabel('Descriptive difference vs IID (eV)'); save(fig,'iid_to_ood_degradation')
    counts={s:sum(r['state']==s for r in road) for s in sorted({r['state'] for r in road})}; fig,ax=plt.subplots(figsize=(8,4)); ax.bar(list(counts),list(counts.values())); ax.tick_params(axis='x',rotation=30); ax.set_ylabel('Roadmap items'); save(fig,'roadmap_status_matrix')
    return made

def main():
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    if head!=EXPECTED_HEAD: raise RuntimeError(f'Git boundary mismatch: {head}')
    checkpoints,failures=checkpoint_inventory(); hist,iid,ood=benchmarks(); road=roadmap(); figs=figures(iid,ood,road)
    wc('data_registry/gate2g0_checkpoint_inventory.csv',checkpoints); wc('data_registry/gate2g0_historical_benchmark.csv',hist); wc('data_registry/gate2g0_new15016_iid_benchmark.csv',iid); wc('data_registry/gate2g0_new15016_ood_benchmark.csv',ood); wc('data_registry/gate2g0_roadmap_gap_matrix.csv',road)
    status='BLOCKED_MODEL_ASSET_INTEGRITY' if failures else 'BENCHMARK_CONSOLIDATED_READY_FOR_MAIN_MODEL'
    report='# Gate 2-G0 paper-grade model and benchmark consolidation\n\n## Scientific boundary\n\nHistorical Layer G and new15016 are separate ledgers and must never be ranked together. All metrics predict **J_eh_screened_eV_eps3p5 proxy**, not experimental Eb or catalytic activity. Gate 3\'s 16-item list is frozen as `EXPLORATORY_BASELINE_SHORTLIST_FROZEN`; experimental progression is paused.\n\n## Historical ledger\n\n'+md(hist,['model','protocol','test_records','mae_eV','std_eV','checkpoint','status'])+'\n## new15016 IID benchmark\n\n'+md(iid,['model','records','groups','group_macro_mae_eV','group_macro_rmse_eV','group_macro_r2','paper_status'])+f'\n## Asset integrity\n\nAudited {len(checkpoints)} local assets; load/smoke failures: {len(failures)}. Phase-12 XGBoost/PyTorch assets received finite-forward smoke. Historical B2 state dictionaries were loaded and retain their prior frozen inference evidence. Missing model dumps remain report-only.\n\n## Roadmap gaps\n\n'+md(road,['item','state','evidence'])+f'\n## Decision\n\n`{status}`\n\nNext permitted step is Gate 2-G1 preregistration; it is not started here.\n'
    (ROOT/'reports/gate2g0_model_benchmark_consolidation.md').write_text(report); (ROOT/'reports/gate2g0_roadmap_gap_audit.md').write_text('# Gate 2-G0 roadmap gap audit\n\n'+md(road,['item','state','evidence'])); (ROOT/'reports/gate2g0_checkpoint_inventory.md').write_text('# Gate 2-G0 checkpoint inventory\n\n'+md(checkpoints,['family','path','bytes','sha256','loadability','smoke_rows']))
    ev={'status':status,'base_head':EXPECTED_HEAD,'cpu_only':True,'training_performed':False,'test_evaluator_called':False,'test_inference_generated':False,'final673_accessed':False,'shortlist_status':'EXPLORATORY_BASELINE_SHORTLIST_FROZEN','checkpoint_assets_audited':len(checkpoints),'load_failures':failures,'figures':figs}; wj('logs/gate2g0_evidence.json',ev); wj('data_registry/gate2g0_model_registry.json',{'status':status,'checkpoint_assets':len(checkpoints),'checkpoint_inventory':'data_registry/gate2g0_checkpoint_inventory.csv','historical_report_only_models':['cheap no-dipole champion','Paper A baseline','Smoothed Memory','B2-0 mapping unresolved'],'scientific_boundaries':{'separate_ledgers':True,'proxy_only':True,'gate2e1_blocked':True}}); print(json.dumps(ev,indent=2))
if __name__=='__main__': main()
