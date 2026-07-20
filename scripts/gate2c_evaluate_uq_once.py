#!/usr/bin/env python3
"""One-time aggregate UQ evaluation on already frozen test prediction artifacts."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from gate2c_common import ROOT, PROTOCOLS, attach_similarity, cluster_column, coverage_summary, load_test, manifests, read_json, selective_metrics, sha256, verify_config, write_json

def main():
    config=read_json("configs/gate2c_uq_applicability_audit_v1.json"); verify_config(config)
    unlock_path=ROOT/"data_registry/gate2c_test_unlock_v1.json"; output_path=ROOT/"logs/gate2c_coverage_metrics.json"
    if not unlock_path.exists(): raise RuntimeError("Gate 2-C test unlock absent")
    if output_path.exists(): raise RuntimeError("Gate 2-C evaluate-once lock consumed; fail-closed")
    unlock=read_json(unlock_path); calibrators=read_json("data_registry/gate2c_calibrator_registry.json")
    if unlock.get("calibrator_registry_sha256")!=sha256("data_registry/gate2c_calibrator_registry.json"): raise RuntimeError("calibrator registry changed after unlock")
    similarity=__import__("pandas").read_parquet(ROOT/calibrators["validation_similarity"]["path"]); mans=manifests(config)
    reps=config["bootstrap"]["replicates"]; seed=config["bootstrap"]["seed"]
    result={"status":"GATE2C_TEST_COVERAGE_EVALUATED_ONCE","protocols":{},"point_predictions_generated":False,"main_parquet_reads":0,"gate2a_evaluator_calls":0,"final673_accessed":False}
    for pi,name in enumerate(PROTOCOLS):
        frame=attach_similarity(config,name,"test",load_test(config,name,mans[name]),similarity); cluster=cluster_column(config,name); cal=calibrators["protocols"][name]
        iqr=float(np.quantile(frame.truth,.75)-np.quantile(frame.truth,.25)); methods={}
        for mi,(method,levels) in enumerate(cal["quantiles"].items()):
            methods[method]={}
            for li,level in enumerate(config["nominal_coverage"]):
                qinfo=levels[str(level)]
                if qinfo["status"]!="ATTAINABLE": methods[method][str(level)]={**qinfo,"coverage":None}; continue
                q=float(qinfo["q"]); summary=coverage_summary(frame,q,cluster,reps,seed+pi*1000+mi*100+li)
                methods[method][str(level)]={**qinfo,"coverage":summary,"interval_width":2*q,"normalized_width_over_target_iqr":2*q/iqr if iqr else None,"undercoverage_gap":level-summary["record_marginal"]}
        ad=selective_metrics(frame,cluster,cal["ad"]["thresholds"],cal["ad"]["high_error_threshold"])
        val=cal["ad"]; v0=val["validation_all"]; v8=val["validation_at_80pct_threshold"]; t0=ad["fixed_validation_thresholds"]["1.0"]; t8=ad["fixed_validation_thresholds"]["0.8"]
        ad["preregistered_useful_rule_pass"] = bool(v8["record_mae"]<v0["record_mae"] and v8["identity_macro_mae"]<v0["identity_macro_mae"] and val["score_error_spearman"]<=-.1 and t8.get("record_mae",np.inf)<t0["record_mae"] and t8.get("identity_macro_mae",np.inf)<t0["identity_macro_mae"])
        result["protocols"][name]={"test_records":len(frame),"target_iqr":iqr,"primary_cluster":cluster,"interval_methods":methods,"applicability_domain":ad}
    tol=config["coverage_tolerance"]; labels=[]
    iid=result["protocols"]["iid"]; iid_ok=True
    for method in ("record","structure","identity"):
        for level in config["nominal_coverage"]:
            item=iid["interval_methods"][method][str(level)]
            iid_ok &= item["coverage"] is not None and item["coverage"]["record_marginal"] >= level-tol["iid_absolute_gap"]
    adequate=[]
    for name in ("donor_cold","acceptor_cold","pair_cold","full_scaffold_cold"):
        item=result["protocols"][name]["interval_methods"]["identity"]["0.9"]
        count=item.get("coverage",{}).get("identity_clusters",0) if item.get("coverage") else 0
        if count>=30: adequate.append(item["coverage"]["identity_macro"]>=.9-tol["ood_identity_90_gap"])
    if iid_ok and adequate and all(adequate): labels.append("UQ_EMPIRICALLY_CALIBRATED_OOD")
    elif iid_ok: labels.append("UQ_CALIBRATED_IID_ONLY")
    acc=result["protocols"]["acceptor_cold"]["interval_methods"]["identity"]["0.9"]
    if acc.get("coverage"):
        cov=acc["coverage"]
        if cov["identity_macro_ci95"][0]<tol["acceptor_ci_lower_min"] or cov["worst_decile_identity_coverage"]<tol["worst_decile_min"]: labels.append("ACCEPTOR_UQ_UNDERCOVERAGE")
    labels.append("BOTH_COLD_UQ_UNSUPPORTED")
    labels.append("AD_SCORE_USEFUL" if all(result["protocols"][n]["applicability_domain"]["preregistered_useful_rule_pass"] for n in PROTOCOLS) else "AD_SCORE_NOT_VALIDATED")
    result["decision_labels"]=labels; write_json(output_path,result)
    consumed={**unlock,"consumed":True,"coverage_metrics_sha256":sha256(output_path)}; write_json("data_registry/gate2c_test_unlock_v1.json",consumed)
    print(json.dumps({"status":result["status"],"decision_labels":labels,"coverage_metrics_sha256":sha256(output_path)},indent=2))
if __name__=="__main__": main()
