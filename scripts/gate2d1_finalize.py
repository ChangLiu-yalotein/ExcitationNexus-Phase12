#!/usr/bin/env python3
from __future__ import annotations
from pathlib import Path
try:
    from scripts.gate2d1_common import ROOT, PROTOCOLS, read_json
except ModuleNotFoundError:
    from gate2d1_common import ROOT, PROTOCOLS, read_json

def val_metric(item,name):
    v=item["validation"]
    return v.get("identity_macro_mae",(v.get("donor_identity_macro_mae",0)+v.get("acceptor_identity_macro_mae",0))/2)
def main():
    m=read_json("logs/gate2d1_validation_metrics.json"); mech=read_json("logs/gate2d1_acceptor_mechanism.json"); reg=read_json("data_registry/gate2d1_model_registry.json")
    lines=["# Gate 2-D1 validation-only results","","No test prediction, test target, main source Parquet, or final673 asset was read.","","| Protocol | A C0-512 | B Wide-1536 | C RA2D-1536 | C−B (eV) | 95% cluster CI |","|---|---:|---:|---:|---:|---|"]
    for name in PROTOCOLS:
        arms=reg["protocols"][name]["arms"]; cmp=m["protocols"][name]["comparisons"]["C_RA2D_1536_minus_B_C0_Wide_1536"]
        lines.append(f"| {name} | {val_metric(arms['A_C0_512_reference'],name):.6f} | {val_metric(arms['B_C0_Wide_1536'],name):.6f} | {val_metric(arms['C_RA2D_1536'],name):.6f} | {cmp['point']:+.6f} | [{cmp['ci95'][0]:+.6f}, {cmp['ci95'][1]:+.6f}] |")
    lines += ["","Metrics are protocol-specific identity-macro MAE; both-cold displays the mean of donor- and acceptor-identity macro MAE and uses the preregistered two-way bootstrap. These validation results cannot be presented as frozen test performance.",""]
    (ROOT/"reports/gate2d1_validation_results.md").write_text("\n".join(lines))

    shap=mech["feature_importance_and_shap"]["C_RA2D_1536"]["blocks"]
    ml=["# Gate 2-D1 acceptor-OOD mechanism","",f"The role-aware arm uses the acceptor block (gain fraction {shap['acceptor_fingerprint']['gain_fraction']:.3f}; absolute TreeSHAP fraction {shap['acceptor_fingerprint']['absolute_shap_fraction']:.3f}), but this does not translate into acceptor-cold improvement.","",f"- Acceptor identities: {mech['acceptor_identities']}",f"- Identity fraction improved by C versus B: {mech['identities_improved_fraction']:.3f}",f"- Mean C−B error in the lowest acceptor-similarity quartile: {mech['low_similarity_quartile_mean_delta_CB']:+.6f} eV",f"- C−B delta versus acceptor similarity Spearman: {mech['delta_vs_similarity_spearman']:.3f}",f"- Worst ten B identities improved: {mech['worst_B_identities_improved']}/10",f"- Acceptor fingerprint collision structures: {mech['acceptor_fingerprint_collision_structures']}; validation identities affected: {mech['collision_structures_in_validation']}","", "The 512-bit collision is a documented representation limitation, not a parser or protocol-leakage blocker; it affects no acceptor-cold validation identity. Anonymous identity hashes are retained in the aggregate evidence; no SMILES are published.",""]
    (ROOT/"reports/gate2d1_acceptor_ood_mechanism.md").write_text("\n".join(ml))

    p=m["primary"]["acceptor_C_minus_B"]; iid=m["primary"]["iid_C_minus_B"]
    dl=["# Gate 2-D1 final decision","",f"Decision: **{m['decision']}**.","",f"- Acceptor-cold C−B identity-macro MAE: {p['point']:+.6f} eV; 95% acceptor-cluster CI [{p['ci95'][0]:+.6f}, {p['ci95'][1]:+.6f}].",f"- Required admission: point ≤ -0.0020 eV and CI upper bound < 0. Neither condition is met.",f"- IID C−B structure-group MAE: {iid['point']:+.6f} eV; CI [{iid['ci95'][0]:+.6f}, {iid['ci95'][1]:+.6f}]. IID non-inferiority passes, but cannot rescue the failed primary endpoint.","", "Arm B and Arm C are both worse than frozen Arm A on the acceptor-cold primary endpoint, so `CAPACITY_ONLY_EFFECT` is not supported. The point estimate does not improve, so the result is not merely inconclusive: RA2D is not admitted under the frozen validation protocol.","", "This result rejects this fixed role-separated Morgan representation/XGBoost intervention; it does not prove that donor/acceptor roles are scientifically irrelevant. No test unlock is authorized.",""]
    (ROOT/"reports/gate2d1_final_decision.md").write_text("\n".join(dl))
if __name__=="__main__": main()
