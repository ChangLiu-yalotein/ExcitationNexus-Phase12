#!/usr/bin/env python3
from __future__ import annotations
import csv, hashlib, json
from pathlib import Path
import numpy as np
import xgboost as xgb

ROOT=Path('/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training')
paths=sorted((ROOT/'runs/gate2e2a_multitask_crossfit/xgb').glob('**/*.ubj'))+sorted((ROOT/'runs/gate2f1_multifidelity_delta_crossfit').glob('**/*.ubj'))
if len(paths)!=80: raise RuntimeError(f'expected 80 UBJ models, found {len(paths)}')
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
rows=[]
for p in paths:
    model=xgb.Booster(); model.load_model(p); width=int(model.num_features()); pred=model.predict(xgb.DMatrix(np.zeros((4,width),np.float32)))
    if pred.shape!=(4,) or not np.isfinite(pred).all(): raise RuntimeError(f'nonfinite smoke {p}')
    rows.append({'family':'XGBoost-UBJ','path':str(p),'sha256':sha(p),'bytes':p.stat().st_size,'loadability':'LOADABLE_FINITE_FORWARD','smoke_rows':4,'input_columns':width,'rounds':int(model.num_boosted_rounds())})
inventory=ROOT/'data_registry/gate2g0_checkpoint_inventory.csv'
with inventory.open() as f: old=list(csv.DictReader(f))
all_rows=old+rows; columns=sorted({k for r in all_rows for k in r})
with inventory.open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=columns); w.writeheader(); w.writerows(all_rows)
e=json.loads((ROOT/'logs/gate2g0_evidence.json').read_text()); e['checkpoint_assets_audited']=len(all_rows); e['ubj_assets_audited']=80; e['load_failures']=[]
(ROOT/'logs/gate2g0_evidence.json').write_text(json.dumps(e,indent=2,sort_keys=True)+'\n')
r=json.loads((ROOT/'data_registry/gate2g0_model_registry.json').read_text()); r['checkpoint_assets']=len(all_rows); r['ubj_assets_audited']=80
(ROOT/'data_registry/gate2g0_model_registry.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n')
print(json.dumps({'ubj_assets':80,'total_assets':len(all_rows),'failures':0},indent=2))
