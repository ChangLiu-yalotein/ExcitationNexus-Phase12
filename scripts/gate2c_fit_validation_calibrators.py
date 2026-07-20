#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from gate2c_common import ROOT, PROTOCOLS, attach_similarity, build_similarity, cluster_column, finite_quantile, load_validation, manifests, read_json, risk, safe_float, score_sets, sha256, verify_config, write_json

def main():
    config=read_json("configs/gate2c_uq_applicability_audit_v1.json"); verify_config(config); mans=manifests(config)
    out_path=ROOT/"data_registry/gate2c_calibrator_registry.json"; sim_path=ROOT/"runs/gate2c_uq/similarity_validation_and_iid_test_v1.parquet"
    if out_path.exists(): raise RuntimeError("Gate 2-C calibrators already frozen; refusing refit")
    if sim_path.exists():
        similarity=__import__("pandas").read_parquet(sim_path)
        expected=sum(int(m.partition.eq("val").sum()) for m in mans.values())+int(mans["iid"].partition.eq("test").sum())
        if len(similarity)!=expected or similarity[["full","donor","acceptor"]].isna().any().any(): raise RuntimeError("incomplete deterministic similarity cache")
        sim_meta={"rows":len(similarity),"parse_failures":{name:0 for name in PROTOCOLS},"fingerprint":config["fingerprint"],"resumed_after_precalibration_process_exit":True}
    else:
        similarity,sim_meta=build_similarity(config,mans); sim_path.parent.mkdir(parents=True,exist_ok=True); similarity.to_parquet(sim_path,index=False)
    registry={"status":"GATE2C_VALIDATION_CALIBRATORS_FROZEN","methods":["record","structure","identity"],"nominal_coverage":config["nominal_coverage"],"protocols":{},
              "validation_similarity":{"path":str(sim_path.relative_to(ROOT)),"sha256":sha256(sim_path),**sim_meta},"test_residuals_used":False,"new_point_predictions":False,"final673_accessed":False}
    for name in PROTOCOLS:
        frame=attach_similarity(config,name,"val",load_validation(config,name,mans[name]),similarity); cluster=cluster_column(config,name); scores=score_sets(frame,cluster)
        quantiles={method:{str(level):finite_quantile(values,level) for level in config["nominal_coverage"]} for method,values in scores.items()}
        abs_error=(frame.prediction-frame.truth).abs(); thresholds={}
        for fraction in config["retained_fractions"]:
            thresholds[str(fraction)]=None if fraction==1 else float(np.quantile(frame.ad_score,1-fraction,method="higher"))
        high=float(np.quantile(abs_error,.9,method="higher")); base=risk(frame,cluster)
        selected=frame.loc[frame.ad_score>=thresholds["0.8"]]; selected_risk=risk(selected,cluster)
        registry["protocols"][name]={"validation_records":len(frame),"primary_cluster":cluster,"score_counts":{k:len(v) for k,v in scores.items()},"quantiles":quantiles,
            "ad":{"score":config["protocols"][name]["ad"],"thresholds":thresholds,"high_error_threshold":high,"validation_all":base,"validation_at_80pct_threshold":selected_risk,
                  "validation_retained_fraction":len(selected)/len(frame),"score_error_spearman":safe_float(spearmanr(frame.ad_score,abs_error).statistic)}}
    write_json(out_path,registry)
    print(json.dumps({"status":registry["status"],"calibrator_registry_sha256":sha256(out_path),"similarity_sha256":registry["validation_similarity"]["sha256"]},indent=2))
if __name__=="__main__": main()
