#!/usr/bin/env python3
"""Render aggregate applicability-domain evidence without reopening test artifacts."""
from gate2c_common import ROOT, read_json, write_json

def main():
    coverage=read_json("logs/gate2c_coverage_metrics.json")
    metrics={"status":"GATE2C_AD_DIAGNOSTICS_FROM_FROZEN_AGGREGATES","protocols":{n:x["applicability_domain"] for n,x in coverage["protocols"].items()},"decision_labels":[x for x in coverage["decision_labels"] if x.startswith("AD_")],"test_artifacts_reopened":False}
    write_json("logs/gate2c_applicability_metrics.json",metrics)
    lines=["# Gate 2-C applicability-domain audit","","All cutoffs were frozen on validation. Test risk–coverage curves are diagnostic and did not select a threshold.",""]
    for name,x in metrics["protocols"].items():
        all_=x["fixed_validation_thresholds"]["1.0"]; at80=x["fixed_validation_thresholds"]["0.8"]
        lines += [f"## {name}","",f"- Validation-locked 80% cutoff retained {at80.get('retained_fraction',0):.3f} of test records.",f"- Record MAE: {all_['record_mae']:.6f} → {at80.get('record_mae',float('nan')):.6f} eV.",f"- Identity-macro MAE: {all_['identity_macro_mae']:.6f} → {at80.get('identity_macro_mae',float('nan')):.6f} eV.",f"- AD score vs absolute error Spearman: {x['ad_score_vs_absolute_error_spearman']}",f"- Fixed preregistered usefulness rule: {x['preregistered_useful_rule_pass']}",""]
    lines += ["## Decision","",f"`{metrics['decision_labels'][0]}`",""]
    (ROOT/"reports/gate2c_applicability_domain.md").write_text("\n".join(lines)+"\n")
if __name__=="__main__": main()
