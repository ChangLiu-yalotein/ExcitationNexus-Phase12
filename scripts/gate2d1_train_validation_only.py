#!/usr/bin/env python3
"""Fit exactly twelve independent Arm B/C models and evaluate validation only."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
try:
    from scripts.gate2d1_common import ROOT, PROTOCOLS, TARGET, arm_matrix, config_and_verify, fit_prep, load_labels, metric_frame, read_json, resolve, sha, transform, write_json
except ModuleNotFoundError:
    from gate2d1_common import ROOT, PROTOCOLS, TARGET, arm_matrix, config_and_verify, fit_prep, load_labels, metric_frame, read_json, resolve, sha, transform, write_json

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--physical-gpu",type=int,required=True); args=ap.parse_args()
    c=config_and_verify(); schema=read_json("data_registry/gate2d1_feature_schema_v1.json"); cache_path=resolve(c["local_feature_cache"]); registry_path=ROOT/"data_registry/gate2d1_model_registry.json"
    if sha(cache_path)!=schema["artifact_sha256"] or registry_path.exists(): raise RuntimeError("feature cache mismatch or models already frozen")
    raw=np.load(cache_path,allow_pickle=False); cache={k:raw[k] for k in raw.files if k!="molecule_id"}
    ids=pd.read_parquet(resolve(c["inputs"]["structure_registry"]["path"]),columns=["molecule_id"]).sort_values("molecule_id",kind="mergesort").molecule_id.astype(str).to_numpy()
    if any(len(x)!=len(ids) for x in cache.values()): raise RuntimeError("feature cache row count mismatch")
    index={mid:i for i,mid in enumerate(ids)}
    desc=[f"pair_{x}" for x in c["descriptor_names"]]; feature_names={"B_C0_Wide_1536":desc+[f"full_wide_{i}" for i in range(1536)],"C_RA2D_1536":desc+[f"full_{i}" for i in range(512)]+[f"donor_{i}" for i in range(512)]+[f"acceptor_{i}" for i in range(512)]}
    matrices={arm:arm_matrix(cache,arm) for arm in ("B_C0_Wide_1536","C_RA2D_1536")}
    registry={"status":"GATE2D1_VALIDATION_MODELS_FROZEN","new_models":0,"physical_gpu":args.physical_gpu,"cuda_visible_devices":os.environ.get("CUDA_VISIBLE_DEVICES"),"protocols":{},"test_artifacts_accessed":False,"main_parquet_accessed":False,"final673_accessed":False,"models_shared_between_protocols":False,"target_derived_preprocessing":False}
    for name in PROTOCOLS:
        spec=c["protocols"][name]; manifest=pd.read_csv(resolve(spec["manifest"])); train_y,val_y=load_labels(c,name,manifest)
        identity_cols=["molecule_id","structure_group_id_v1","donor_structure_group_id_v1","acceptor_structure_group_id_v1","pair_group_id_v1","full_scaffold_group_id_v1","group_weight"]
        train=manifest.loc[manifest.partition.eq("train"),identity_cols].merge(train_y,on="molecule_id",validate="one_to_one"); val=manifest.loc[manifest.partition.eq("val"),identity_cols].merge(val_y,on="molecule_id",validate="one_to_one")
        ti=np.array([index[x] for x in train.molecule_id]); vi=np.array([index[x] for x in val.molecule_id]); weights=train.group_weight.to_numpy(float); cluster=c["protocol_clusters"][name]
        arm_a=pd.read_csv(resolve(spec["arm_a_validation"])); arm_a=val[[*identity_cols,TARGET]].merge(arm_a,on="molecule_id",validate="one_to_one")
        protocol={"train_records":len(train),"validation_records":len(val),"train_weight_sum":float(weights.sum()),"manifest_sha256":spec["sha256"],"arm_a_prediction_sha256":spec["arm_a_sha256"],"arms":{"A_C0_512_reference":{"validation":metric_frame(arm_a,arm_a.prediction.to_numpy(),cluster)}}}
        combined=val[[*identity_cols,TARGET]].copy(); combined["A_C0_512_reference"]=arm_a.set_index("molecule_id").loc[combined.molecule_id,"prediction"].to_numpy()
        for arm in ("B_C0_Wide_1536","C_RA2D_1536"):
            out=resolve(c["local_run_root"])/name/arm
            if out.exists(): raise RuntimeError(f"partial/existing model output: {out}")
            out.mkdir(parents=True); prep=fit_prep(matrices[arm][ti].astype(float),weights); prep_path=out/"preprocessor.npz"; np.savez_compressed(prep_path,medians=prep[0],means=prep[1],scales=prep[2],feature_names=np.asarray(feature_names[arm]))
            xtr=transform(matrices[arm][ti].astype(float),prep); xval=transform(matrices[arm][vi].astype(float),prep)
            params=dict(c["xgboost"]); model=XGBRegressor(**params); started=time.perf_counter(); model.fit(xtr,train[TARGET].to_numpy(float),sample_weight=weights); seconds=time.perf_counter()-started
            pred=model.predict(xval); model_path=out/"model.json"; model.save_model(model_path); pred_path=out/"validation_predictions.parquet"; pd.DataFrame({"molecule_id":val.molecule_id,"prediction":pred}).sort_values("molecule_id",kind="mergesort").to_parquet(pred_path,index=False)
            metrics=metric_frame(val,pred,cluster); protocol["arms"][arm]={"validation":metrics,"model_path":str(model_path.relative_to(ROOT)),"model_sha256":sha(model_path),"preprocessor_path":str(prep_path.relative_to(ROOT)),"preprocessor_sha256":sha(prep_path),"prediction_path":str(pred_path.relative_to(ROOT)),"prediction_sha256":sha(pred_path),"training_wall_seconds":seconds,"feature_columns":len(feature_names[arm]),"fit_partition":"train","inference_partition":"val","test_accessed":False}
            combined[arm]=pred; registry["new_models"]+=1
        pred_all=resolve(c["local_run_root"])/name/"validation_paired.parquet"; combined.sort_values("molecule_id",kind="mergesort").to_parquet(pred_all,index=False); protocol["paired_validation_path"]=str(pred_all.relative_to(ROOT)); protocol["paired_validation_sha256"]=sha(pred_all); registry["protocols"][name]=protocol
    write_json(registry_path,registry)
    print(json.dumps({"status":registry["status"],"new_models":registry["new_models"],"physical_gpu":args.physical_gpu,"protocols":{n:{a:round(x["validation"].get("identity_macro_mae",x["validation"].get("acceptor_identity_macro_mae",0)),8) for a,x in p["arms"].items()} for n,p in registry["protocols"].items()}},indent=2))
if __name__=="__main__": main()
