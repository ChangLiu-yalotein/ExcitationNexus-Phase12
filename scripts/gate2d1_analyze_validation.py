#!/usr/bin/env python3
"""Validation-only hierarchical analysis and frozen Gate 2-D1 admission decision."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem
from scipy.stats import spearmanr
import xgboost as xgb
try:
    from scripts.gate2d1_common import ROOT, PROTOCOLS, TARGET, arm_matrix, config_and_verify, paired_cluster_bootstrap, read_json, resolve, safe, sha, transform, write_json
except ModuleNotFoundError:
    from gate2d1_common import ROOT, PROTOCOLS, TARGET, arm_matrix, config_and_verify, paired_cluster_bootstrap, read_json, resolve, safe, sha, transform, write_json

def two_way(frame,left,right,reps,seed):
    delta=np.abs(np.asarray(left)-frame[TARGET].to_numpy())-np.abs(np.asarray(right)-frame[TARGET].to_numpy()); work=frame.assign(delta=delta).sort_values("molecule_id",kind="mergesort")
    ds=sorted(work.donor_structure_group_id_v1.unique()); ac=sorted(work.acceptor_structure_group_id_v1.unique()); di=pd.Categorical(work.donor_structure_group_id_v1,categories=ds).codes; ai=pd.Categorical(work.acceptor_structure_group_id_v1,categories=ac).codes; values=work.delta.to_numpy(); rng=np.random.default_rng(seed); out=np.empty(reps)
    for i in range(reps):
        total=0
        while total==0:
            dm=np.bincount(rng.integers(0,len(ds),len(ds)),minlength=len(ds)); am=np.bincount(rng.integers(0,len(ac),len(ac)),minlength=len(ac)); w=dm[di]*am[ai]; total=w.sum()
        out[i]=np.sum(w*values)/total
    return {"point":float(values.mean()),"ci95":np.quantile(out,[.025,.975]).astype(float).tolist(),"donor_clusters":len(ds),"acceptor_clusters":len(ac)}
def jaccard_nearest(query,train):
    inter=query.astype(np.int16)@train.astype(np.int16).T; q=query.sum(1)[:,None]; t=train.sum(1)[None,:]; union=q+t-inter; return np.max(np.divide(inter,union,out=np.zeros_like(inter,dtype=float),where=union>0),axis=1)
def hetero(text):
    m=Chem.MolFromSmiles(str(text)) or Chem.MolFromSmiles(str(text),sanitize=False)
    if m is None: return None
    atoms=list(m.GetAtoms()); return sum(a.GetAtomicNum() not in (0,1,6) for a in atoms)/len(atoms) if atoms else None
def block_stats(booster,contrib,blocks):
    gain=booster.get_score(importance_type="gain"); total_gain=sum(gain.values()) or 1.; total_shap=np.abs(contrib[:,:-1]).sum() or 1.; out={}
    for name,(lo,hi) in blocks.items():
        g=sum(gain.get(f"f{i}",0.) for i in range(lo,hi)); s=float(np.abs(contrib[:,lo:hi]).sum())
        out[name]={"gain_sum":float(g),"gain_fraction":float(g/total_gain),"mean_absolute_tree_shap":float(np.abs(contrib[:,lo:hi]).mean()),"absolute_shap_fraction":s/total_shap}
    return out

def main():
    c=config_and_verify(); registry=read_json("data_registry/gate2d1_model_registry.json"); schema=read_json("data_registry/gate2d1_feature_schema_v1.json")
    if registry["new_models"]!=12: raise RuntimeError("model registry incomplete")
    metrics={"status":"GATE2D1_VALIDATION_ANALYSIS_FROZEN","protocols":{},"primary":{},"test_artifacts_accessed":False,"main_parquet_accessed":False,"final673_accessed":False}
    frames={}
    for name in PROTOCOLS:
        p=registry["protocols"][name]; path=resolve(p["paired_validation_path"])
        if sha(path)!=p["paired_validation_sha256"]: raise RuntimeError("paired validation hash mismatch")
        f=pd.read_parquet(path).sort_values("molecule_id",kind="mergesort").reset_index(drop=True); frames[name]=f; cluster=c["protocol_clusters"][name]
        comparisons={}
        for left,right in (("C_RA2D_1536","B_C0_Wide_1536"),("C_RA2D_1536","A_C0_512_reference"),("B_C0_Wide_1536","A_C0_512_reference")):
            comparisons[f"{left}_minus_{right}"]=two_way(f,f[left],f[right],c["bootstrap"]["replicates"],c["bootstrap"]["seed"]) if cluster=="two_way_donor_acceptor" else paired_cluster_bootstrap(f,f[left],f[right],cluster,c["bootstrap"]["replicates"],c["bootstrap"]["seed"])
        metrics["protocols"][name]={"arms":p["arms"],"comparisons":comparisons,"primary_cluster":cluster}
    primary=metrics["protocols"]["acceptor_cold"]["comparisons"]["C_RA2D_1536_minus_B_C0_Wide_1536"]; iid=metrics["protocols"]["iid"]["comparisons"]["C_RA2D_1536_minus_B_C0_Wide_1536"]

    raw=np.load(resolve(c["local_feature_cache"]),allow_pickle=False); cache={k:raw[k] for k in raw.files if k!="molecule_id"}; ids=pd.read_parquet(resolve(c["inputs"]["structure_registry"]["path"]),columns=["molecule_id"]).sort_values("molecule_id",kind="mergesort").molecule_id.astype(str).to_numpy(); index={mid:i for i,mid in enumerate(ids)}
    man=pd.read_csv(resolve(c["protocols"]["acceptor_cold"]["manifest"])); train=man.loc[man.partition.eq("train")]; val=frames["acceptor_cold"]; ti=np.array([index[x] for x in train.molecule_id]); vi=np.array([index[x] for x in val.molecule_id]); val["acceptor_similarity"]=jaccard_nearest(cache["acceptor512"][vi],cache["acceptor512"][ti])
    comp=pd.read_csv(resolve(c["inputs"]["component_registry"]["path"]))[["molecule_id","acceptor_canonical_structure_smiles_v1","acceptor_scaffold_group_id_v1"]]; val=val.merge(comp,on="molecule_id",validate="one_to_one"); val["abs_B"]=(val.B_C0_Wide_1536-val[TARGET]).abs(); val["abs_C"]=(val.C_RA2D_1536-val[TARGET]).abs(); val["delta_CB"]=val.abs_C-val.abs_B
    identity=val.groupby("acceptor_structure_group_id_v1",sort=True).agg(records=("molecule_id","size"),similarity=("acceptor_similarity","mean"),mae_B=("abs_B","mean"),mae_C=("abs_C","mean"),delta_CB=("delta_CB","mean"),target_min=(TARGET,"min"),target_max=(TARGET,"max"),smiles=("acceptor_canonical_structure_smiles_v1","first"),scaffold=("acceptor_scaffold_group_id_v1","first")).reset_index(); identity["target_range"]=identity.target_max-identity.target_min; identity["heteroatom_fraction"]=[hetero(x) for x in identity.smiles]
    worst=identity.nlargest(10,"mae_B")[["acceptor_structure_group_id_v1","records","similarity","mae_B","mae_C","delta_CB","target_range"]].to_dict("records")
    low=identity.loc[identity.similarity<=identity.similarity.quantile(.25)]; mechanism={"acceptor_identities":len(identity),"delta_vs_similarity_spearman":safe(spearmanr(identity.delta_CB,identity.similarity).statistic),"delta_vs_identity_size_spearman":safe(spearmanr(identity.delta_CB,identity.records).statistic),"delta_vs_target_range_spearman":safe(spearmanr(identity.delta_CB,identity.target_range).statistic),"delta_vs_heteroatom_fraction_spearman":safe(spearmanr(identity.delta_CB,identity.heteroatom_fraction).statistic),"low_similarity_quartile_mean_delta_CB":float(low.delta_CB.mean()),"identities_improved_fraction":float((identity.delta_CB<0).mean()),"worst_B_identities_improved":int((identity.nlargest(10,"mae_B").delta_CB<0).sum()),"worst_10_anonymous":worst,"unique_acceptor_scaffolds":int(identity.scaffold.nunique())}
    hashes=pd.Series([hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in cache["acceptor512"]],index=ids); collision=pd.DataFrame({"molecule_id":ids,"fp_hash":hashes.values}).merge(man[["molecule_id","partition","acceptor_structure_group_id_v1"]],on="molecule_id"); group_hash=collision.drop_duplicates("acceptor_structure_group_id_v1"); duplicated=set(group_hash.loc[group_hash.fp_hash.duplicated(False),"acceptor_structure_group_id_v1"]); mechanism["acceptor_fingerprint_collision_structures"]=len(duplicated); mechanism["collision_structures_in_validation"]=int(identity.acceptor_structure_group_id_v1.isin(duplicated).sum()); mechanism["collision_blocker"]=False

    shape={}; matrices={arm:arm_matrix(cache,arm) for arm in ("B_C0_Wide_1536","C_RA2D_1536")}
    for arm,blocks in {"B_C0_Wide_1536":{"full_descriptors":(0,20),"full_fingerprint":(20,1556)},"C_RA2D_1536":{"full_descriptors":(0,20),"full_fingerprint":(20,532),"donor_fingerprint":(532,1044),"acceptor_fingerprint":(1044,1556)}}.items():
        info=registry["protocols"]["acceptor_cold"]["arms"][arm]; prep_np=np.load(resolve(info["preprocessor_path"]),allow_pickle=False); prep=(prep_np["medians"],prep_np["means"],prep_np["scales"]); xval=transform(matrices[arm][vi].astype(float),prep); booster=xgb.Booster(); booster.load_model(resolve(info["model_path"])); contrib=booster.predict(xgb.DMatrix(xval),pred_contribs=True); shap_path=resolve(c["local_run_root"])/"acceptor_cold"/arm/"validation_tree_shap.npz"; np.savez_compressed(shap_path,contributions=contrib)
        shape[arm]={"blocks":block_stats(booster,contrib,blocks),"shap_path":str(shap_path.relative_to(ROOT)),"shap_sha256":sha(shap_path)}
    mechanism["feature_importance_and_shap"]=shape
    metrics["primary"]={"acceptor_C_minus_B":primary,"iid_C_minus_B":iid,"thresholds":c["admission"]}
    integrity=not mechanism["collision_blocker"] and schema["parse"]["donor_success"]==15016 and schema["parse"]["acceptor_success"]==15016 and schema["c0_exact_match_records"]==15016
    admitted=primary["point"]<=c["admission"]["acceptor_point_delta_max_eV"] and primary["ci95"][1]<c["admission"]["acceptor_bootstrap_ci_upper_max_eV"] and iid["ci95"][1]<=c["admission"]["iid_noninferiority_ci_upper_max_eV"] and integrity
    a=registry["protocols"]["acceptor_cold"]["arms"]; ma={k:v["validation"]["identity_macro_mae"] for k,v in a.items()}; capacity=ma["B_C0_Wide_1536"]<ma["A_C0_512_reference"] and ma["C_RA2D_1536"]<ma["A_C0_512_reference"] and not admitted
    inconclusive=primary["point"]<0 and primary["ci95"][1]>=0
    decision="ROLE_AWARE_2D_ADMITTED" if admitted else "CAPACITY_ONLY_EFFECT" if capacity else "REPRESENTATION_SIGNAL_INCONCLUSIVE" if inconclusive else "ROLE_AWARE_2D_NOT_ADMITTED"
    metrics["decision"]=decision; metrics["integrity_blocker"]=not integrity; write_json("logs/gate2d1_validation_metrics.json",metrics); write_json("logs/gate2d1_evidence.json",{"status":"GATE2D1_DONE","decision":decision,"primary":metrics["primary"],"new_models":12,"validation_only":True,"test_artifacts_accessed":False,"main_parquet_accessed_after_authorized_extraction":False,"final673_accessed":False,"feature_artifact_sha256":schema["artifact_sha256"],"model_registry_sha256":sha("data_registry/gate2d1_model_registry.json")}); write_json("logs/gate2d1_acceptor_mechanism.json",mechanism)
    print(json.dumps({"decision":decision,"primary":metrics["primary"],"mechanism":{k:v for k,v in mechanism.items() if k not in ("worst_10_anonymous","feature_importance_and_shap")}},indent=2))
if __name__=="__main__": main()
