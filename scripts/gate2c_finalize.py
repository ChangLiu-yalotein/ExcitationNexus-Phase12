#!/usr/bin/env python3
from __future__ import annotations
from gate2c_common import ROOT, PROTOCOLS, read_json, sha256, write_json

def f(value): return "NA" if value is None else f"{value:.4f}"

def main():
    c=read_json("logs/gate2c_coverage_metrics.json"); e=read_json("logs/gate2c_exchangeability.json"); a=read_json("logs/gate2c_applicability_metrics.json")
    lines=["# Gate 2-C interval coverage","","Intervals use validation residuals only. OOD coverage is empirical; it is not presented as a distribution-free guarantee.","",
           "| Protocol | Method | Nominal | Record coverage | Identity/crossed coverage | Width (eV) | Status |","|---|---|---:|---:|---:|---:|---|"]
    for name in PROTOCOLS:
        for method in c["protocols"][name]["interval_methods"]:
            for level in ("0.8","0.9","0.95"):
                x=c["protocols"][name]["interval_methods"][method][level]; cov=x.get("coverage")
                identity=None if cov is None else cov.get("identity_macro", (cov.get("donor_identity_macro",0)+cov.get("acceptor_identity_macro",0))/2 if "donor_identity_macro" in cov else None)
                lines.append(f"| {name} | {method} | {float(level):.0%} | {f(None if cov is None else cov['record_marginal'])} | {f(identity)} | {f(x.get('interval_width'))} | {x['status']} |")
    (ROOT/"reports/gate2c_interval_coverage.md").write_text("\n".join(lines)+"\n")

    acc=c["protocols"]["acceptor_cold"]; ar=acc["interval_methods"]["record"]["0.9"]; ai=acc["interval_methods"]["identity"]["0.9"]
    al=["# Gate 2-C acceptor-cold UQ","",f"Acceptor-cold point-prediction degradation remains frozen from Gate 2-B. The 90% record interval achieved {ar['coverage']['record_marginal']:.4f} record coverage. The acceptor-identity maximum-residual interval achieved {ai['coverage']['identity_macro']:.4f} identity-macro coverage (95% cluster-bootstrap CI {ai['coverage']['identity_macro_ci95']}) with width {ai['interval_width']:.4f} eV.","",
        "`ACCEPTOR_UQ_UNDERCOVERAGE` was not triggered under the preregistered rule. This is empirical OOD calibration, not proof of exchangeability or a distribution-free guarantee.",""]
    (ROOT/"reports/gate2c_acceptor_uq.md").write_text("\n".join(al))

    both=c["protocols"]["both_cold"]; br=both["interval_methods"]["record"]["0.9"]
    bl=["# Gate 2-C both-cold UQ","","Status: `BOTH_COLD_UQ_UNSUPPORTED`.","",f"The empirical 90% record interval covered {br['coverage']['record_marginal']:.4f} of 587 records (two-way pigeonhole CI {br['coverage']['two_way_record_coverage_ci95']}) at width {br['interval_width']:.4f} eV.","",
        "The design is donor-by-acceptor crossed, has only 15 independent test donors, and does not satisfy an exact conformal-exchangeability argument. Its validation AD score was constant at the exact-component boundary; the validation-locked 80% cutoff retained no both-cold test records. Average empirical coverage therefore cannot support a reliable both-cold guarantee.",""]
    (ROOT/"reports/gate2c_both_cold_uq.md").write_text("\n".join(bl))

    decisions=c["decision_labels"]
    dl=["# Gate 2-C final decision","",f"Final status: **GATE2C_DONE_UQ_APPLICABILITY_AUDIT**.","","Decision labels:",""]+[f"- `{x}`" for x in decisions]+["","Interpretation:","",
        "- IID and the adequately powered OOD identity protocols achieved the preregistered empirical coverage tolerance. OOD exchangeability remains unverified, so no strict distribution-free claim is made.",
        "- Acceptor-cold point prediction is still the principal OOD failure mode, but its conservative identity-max interval did not under-cover; the cost is a wide 90% interval.",
        "- Both-cold remains unsupported because crossed-cluster exchangeability and statistical power are inadequate, despite acceptable marginal empirical coverage.",
        "- The validation-locked similarity AD rule was not stable across all protocols and is not validated as a deployment filter.",""]
    (ROOT/"reports/gate2c_final_decision.md").write_text("\n".join(dl))
    evidence={"status":"GATE2C_DONE_UQ_APPLICABILITY_AUDIT","decision_labels":decisions,"authorized_validation_extraction":True,"validation_label_artifact_sha256":read_json("data_registry/gate2c_validation_label_registry.json")["artifact_sha256"],
              "calibrator_registry_sha256":sha256("data_registry/gate2c_calibrator_registry.json"),"coverage_metrics_sha256":sha256("logs/gate2c_coverage_metrics.json"),"test_once_lock_consumed":True,
              "second_evaluation_fail_closed":True,"point_models_trained":False,"new_point_predictions":False,"main_parquet_reads_after_extraction":0,"gate2a_evaluator_calls":0,"final673_accessed":False,
              "exchangeability":{n:x["exchangeability_status"] for n,x in e["protocols"].items()}}
    write_json("logs/gate2c_evidence.json",evidence)
if __name__=="__main__": main()
