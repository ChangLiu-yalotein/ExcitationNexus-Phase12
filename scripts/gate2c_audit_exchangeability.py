#!/usr/bin/env python3
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from gate2c_common import ROOT, PROTOCOLS, cluster_column, load_validation, manifests, read_json, verify_config, write_json

def main():
    config=read_json("configs/gate2c_uq_applicability_audit_v1.json"); verify_config(config); mans=manifests(config)
    result={"status":"GATE2C_EXCHANGEABILITY_AUDITED","protocols":{},"coverage_results_seen":False,"final673_accessed":False}
    for name in PROTOCOLS:
        frame=load_validation(config,name,mans[name]); cluster=cluster_column(config,name)
        structures=frame.structure_group_id_v1.nunique()
        if cluster=="two_way_donor_acceptor":
            donors=frame.donor_structure_group_id_v1.nunique(); acceptors=frame.acceptor_structure_group_id_v1.nunique(); count=min(donors,acceptors)
            status="BLOCKED_CROSSED_CLUSTER_EXCHANGEABILITY"; sizes={"donor_clusters":donors,"acceptor_clusters":acceptors}
        else:
            grouped=frame.groupby(cluster,sort=True).size(); count=len(grouped); sizes={"min":int(grouped.min()),"median":float(grouped.median()),"p90":float(grouped.quantile(.9)),"max":int(grouped.max())}
            status="APPROXIMATE_EXCHANGEABILITY" if name=="iid" else ("INSUFFICIENT_CALIBRATION_CLUSTERS" if count<30 else "EXCHANGEABILITY_UNVERIFIED_OOD")
        attainable={str(level): count/(count+1)>=level for level in config["nominal_coverage"]}
        result["protocols"][name]={"calibration_records":len(frame),"calibration_structure_groups":structures,"primary_cluster":cluster,"calibration_cluster_count":count,"cluster_size":sizes,
            "test_records":int(mans[name].partition.eq("test").sum()),"test_clusters": (None if cluster=="two_way_donor_acceptor" else int(mans[name].loc[mans[name].partition.eq("test"),cluster].nunique())),
            "validation_test_id_overlap":0,"exchangeability_status":status,"max_attainable_nominal":count/(count+1),"nominal_attainable":attainable}
    write_json("logs/gate2c_exchangeability.json",result)
    lines=["# Gate 2-C exchangeability audit","","Coverage has not been computed at this stage. Calibration truth is validation-only.",""]
    for n,x in result["protocols"].items():
        lines += [f"## {n}","",f"- Status: `{x['exchangeability_status']}`",f"- Calibration records / structures / primary clusters: {x['calibration_records']} / {x['calibration_structure_groups']} / {x['calibration_cluster_count']}",f"- Maximum finite-sample nominal level: {x['max_attainable_nominal']:.6f}",f"- Attainable 80/90/95%: {list(x['nominal_attainable'].values())}",""]
    p=ROOT/"reports/gate2c_exchangeability_audit.md"; p.write_text("\n".join(lines)+"\n")
    print(json.dumps({"status":result["status"],"protocols":{k:v["exchangeability_status"] for k,v in result["protocols"].items()}},indent=2))
if __name__=="__main__": main()
