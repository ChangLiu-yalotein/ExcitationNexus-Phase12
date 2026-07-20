#!/usr/bin/env python3
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
TARGET="tddft_coulomb_attraction_eV_eps3p5_proxy"
PROTOCOLS=("iid","donor_cold","acceptor_cold","pair_cold","both_cold","full_scaffold_cold")
ARMS=("A_C0_512_reference","B_C0_Wide_1536","C_RA2D_1536")

def resolve(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p
def sha(path):
    h=hashlib.sha256()
    with resolve(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def read_json(path): return json.loads(resolve(path).read_text())
def write_json(path,value):
    def convert(x):
        if isinstance(x,np.generic): return x.item()
        if isinstance(x,np.ndarray): return x.tolist()
        raise TypeError(f"not JSON serializable: {type(x).__name__}")
    p=resolve(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False,default=convert)+"\n")
def safe(value):
    x=float(value); return x if np.isfinite(x) else None

def config_and_verify():
    c=read_json("configs/gate2d1_role_aware_2d_v1.json"); lock=read_json("data_registry/gate2d1_preregistration_lock_v1.json")
    if sha("configs/gate2d1_role_aware_2d_v1.json")!=lock["config_sha256"]: raise RuntimeError("preregistration config changed")
    for x in c["inputs"].values():
        if sha(x["path"])!=x["sha256"]: raise RuntimeError(f"input hash mismatch: {x['path']}")
    for x in c["protocols"].values():
        if sha(x["manifest"])!=x["sha256"] or sha(x["arm_a_validation"])!=x["arm_a_sha256"]: raise RuntimeError("protocol input hash mismatch")
    if not c["validation_only"] or c["main_parquet_access_after_extraction"] or c["test_artifact_access"] or c["final673_access"]: raise RuntimeError("firewall config invalid")
    return c

def load_labels(c,name,manifest):
    base=pd.read_parquet(resolve(c["inputs"]["base_train_labels"]["path"])).rename(columns={"target":TARGET})
    valall=pd.read_parquet(resolve(c["inputs"]["validation_labels"]["path"]))
    frames=[base,valall]
    if name!="iid":
        reg=read_json(c["inputs"]["train_label_extraction_registry"]["path"]); info=reg["protocols"][name]
        if sha(info["artifact_path"])!=info["artifact_sha256"] or info["manifest_sha256"]!=c["protocols"][name]["sha256"]: raise RuntimeError("protocol label binding mismatch")
        frames.append(pd.read_parquet(resolve(info["artifact_path"])))
    merged=pd.concat(frames,ignore_index=True)
    spread=merged.groupby("molecule_id")[TARGET].agg(lambda x:float(x.max()-x.min()))
    if (spread>1e-12).any(): raise RuntimeError("conflicting label artifacts")
    merged=merged.sort_values("molecule_id",kind="mergesort").drop_duplicates("molecule_id")
    train_ids=manifest.loc[manifest.partition.eq("train"),["molecule_id"]]
    val_ids=manifest.loc[manifest.partition.eq("val"),["molecule_id"]]
    train=train_ids.merge(merged,on="molecule_id",validate="one_to_one"); val=val_ids.merge(valall,on="molecule_id",validate="one_to_one")
    if len(train)!=len(train_ids) or len(val)!=len(val_ids) or train[TARGET].isna().any() or val[TARGET].isna().any(): raise RuntimeError("protocol train/validation label coverage failed")
    forbidden=set(manifest.loc[~manifest.partition.eq("train"),"molecule_id"])
    if set(train.molecule_id)&forbidden: raise RuntimeError("protocol-local train label leakage")
    return train,val

def weighted_median(values,weights):
    order=np.argsort(values,kind="mergesort"); x=np.asarray(values)[order]; w=np.asarray(weights)[order]
    return float(x[np.searchsorted(np.cumsum(w),.5*w.sum(),side="left")])
def fit_prep(x,w):
    med=np.array([weighted_median(x[:,j],w) for j in range(x.shape[1])]); z=np.where(np.isfinite(x),x,med); mean=np.sum(z*w[:,None],axis=0)/w.sum(); var=np.sum((z-mean)**2*w[:,None],axis=0)/w.sum(); scale=np.sqrt(np.maximum(var,0)); scale[scale<1e-12]=1
    return med,mean,scale
def transform(x,prep):
    med,mean,scale=prep; return ((np.where(np.isfinite(x),x,med)-mean)/scale).astype(np.float32)
def content_hash(ids,*arrays):
    h=hashlib.sha256(); h.update(("\n".join(map(str,ids))+"\n").encode())
    for x in arrays: h.update(np.ascontiguousarray(x).tobytes())
    return h.hexdigest()
def arm_matrix(cache,arm):
    if arm=="A_C0_512_reference": return np.concatenate([cache["descriptors"],cache["full512"]],axis=1)
    if arm=="B_C0_Wide_1536": return np.concatenate([cache["descriptors"],cache["full1536"]],axis=1)
    if arm=="C_RA2D_1536": return np.concatenate([cache["descriptors"],cache["full512"],cache["donor512"],cache["acceptor512"]],axis=1)
    raise KeyError(arm)
def metric_frame(frame,pred,cluster):
    e=np.asarray(pred)-frame[TARGET].to_numpy(float); out={"records":len(frame),"record_mae":float(np.abs(e).mean()),"record_rmse":float(np.sqrt(np.mean(e**2))),"p90_absolute_error":float(np.quantile(np.abs(e),.9)),"structure_group_macro_mae":float(pd.Series(np.abs(e)).groupby(frame.structure_group_id_v1.to_numpy()).mean().mean())}
    if cluster=="two_way_donor_acceptor":
        out["donor_identity_macro_mae"]=float(pd.Series(np.abs(e)).groupby(frame.donor_structure_group_id_v1.to_numpy()).mean().mean()); out["acceptor_identity_macro_mae"]=float(pd.Series(np.abs(e)).groupby(frame.acceptor_structure_group_id_v1.to_numpy()).mean().mean())
    else:
        vals=pd.Series(np.abs(e)).groupby(frame[cluster].to_numpy()).mean(); out["identity_macro_mae"]=float(vals.mean()); out["identity_count"]=len(vals); out["worst_decile_identity_mae"]=float(vals[vals>=vals.quantile(.9)].mean())
    return out
def paired_cluster_bootstrap(frame,left,right,cluster,reps=10000,seed=20260720):
    delta=np.abs(np.asarray(left)-frame[TARGET].to_numpy(float))-np.abs(np.asarray(right)-frame[TARGET].to_numpy(float)); table=pd.DataFrame({"cluster":frame[cluster].to_numpy(),"delta":delta}).groupby("cluster",sort=True).delta.mean().to_numpy(); table=np.sort(table); rng=np.random.default_rng(seed); out=np.empty(reps)
    for i in range(reps): out[i]=table[rng.integers(0,len(table),len(table))].mean()
    return {"point":float(table.mean()),"ci95":np.quantile(out,[.025,.975]).astype(float).tolist(),"clusters":len(table)}
