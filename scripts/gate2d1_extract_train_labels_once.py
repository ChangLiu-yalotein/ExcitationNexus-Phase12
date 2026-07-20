#!/usr/bin/env python3
"""One-time, two-column extraction of protocol-local missing train labels."""
from __future__ import annotations
import argparse, hashlib, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow.dataset as ds

ROOT=Path(__file__).resolve().parents[1]
TARGET="tddft_coulomb_attraction_eV_eps3p5_proxy"

def resolve(x):
    p=Path(x); return p if p.is_absolute() else ROOT/p
def sha(path):
    h=hashlib.sha256()
    with resolve(path).open("rb") as f:
        for b in iter(lambda:f.read(1<<20),b""): h.update(b)
    return h.hexdigest()
def write_json(path,value):
    p=resolve(path); p.parent.mkdir(parents=True,exist_ok=True); p.write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/gate2d1_train_label_extraction_v1.json"); args=ap.parse_args()
    c=json.loads(resolve(args.config).read_text()); outdir=resolve(c["output_directory"]); registry=resolve(c["registry"])
    if registry.exists() or outdir.exists(): raise RuntimeError("extraction already attempted/frozen; refusing second source read")
    if sha(c["source_table"])!=c["source_table_sha256"]: raise RuntimeError("source table hash mismatch")
    for x in c["existing_labels"].values():
        if sha(x["path"])!=x["sha256"]: raise RuntimeError("existing label artifact hash mismatch")
    base=pd.read_parquet(resolve(c["existing_labels"]["iid_train_val"]["path"])).rename(columns={"target":TARGET})
    val=pd.read_parquet(resolve(c["existing_labels"]["gate2c_validation_union"]["path"]))
    if not base.molecule_id.is_unique or not val.molecule_id.is_unique: raise RuntimeError("existing label IDs not unique")
    available=set(base.molecule_id)|set(val.molecule_id); missing={}; manifests={}
    for name,spec in c["protocols"].items():
        if sha(spec["manifest"])!=spec["sha256"]: raise RuntimeError(f"manifest hash mismatch: {name}")
        m=pd.read_csv(resolve(spec["manifest"])); manifests[name]=m
        ids=set(m.loc[m.partition.eq("train"),"molecule_id"])-available
        if len(ids)!=spec["expected_missing_train"]: raise RuntimeError(f"missing train count mismatch: {name}")
        missing[name]=ids
    union=set().union(*missing.values())
    if len(union)!=c["expected_union_count"] or any(not any(mid in ids for ids in missing.values()) for mid in union): raise RuntimeError("union identity contract failed")
    table=ds.dataset(c["source_table"],format="parquet").to_table(columns=c["columns"],filter=ds.field("molecule_id").isin(sorted(union))).to_pandas()
    if len(table)!=len(union) or not table.molecule_id.is_unique or set(table.molecule_id)!=union: raise RuntimeError("source extraction ID integrity failed")
    if table[TARGET].isna().any() or not np.isfinite(table[TARGET].to_numpy(float)).all(): raise RuntimeError("missing/non-finite extracted label")
    outdir.mkdir(parents=True)
    evidence={"status":c["authorization"],"completed_utc":datetime.now(timezone.utc).isoformat(),"arrow_reads":1,"source_columns":c["columns"],"source_sha256":c["source_table_sha256"],"union_count":len(union),"generic_union_file_written":False,"protocols":{},"final673_accessed":False}
    for name,ids in missing.items():
        m=manifests[name]; block=table.loc[table.molecule_id.isin(ids),c["columns"]].sort_values("molecule_id",kind="mergesort").reset_index(drop=True)
        train=set(m.loc[m.partition.eq("train"),"molecule_id"]); forbidden=set(m.loc[~m.partition.eq("train"),"molecule_id"])
        if set(block.molecule_id)-train or set(block.molecule_id)&forbidden: raise RuntimeError(f"protocol-local leakage: {name}")
        path=outdir/f"{name}_train_labels.parquet"; block.to_parquet(path,index=False)
        full=pd.concat([base, val, block],ignore_index=True).drop_duplicates("molecule_id",keep="first")
        joined=m.loc[m.partition.eq("train"),["molecule_id"]].merge(full,on="molecule_id",validate="one_to_one")
        if len(joined)!=int(m.partition.eq("train").sum()) or joined[TARGET].isna().any(): raise RuntimeError(f"final train coverage failed: {name}")
        shuffled=block.sample(frac=1,random_state=20260720).sort_values("molecule_id",kind="mergesort").reset_index(drop=True)
        tmp=outdir/f".{name}.shuffle_check.parquet"; shuffled.to_parquet(tmp,index=False)
        order_invariant=sha(tmp)==sha(path); tmp.unlink()
        if not order_invariant: raise RuntimeError(f"shuffle hash invariant failed: {name}")
        evidence["protocols"][name]={"supplement_count":len(block),"final_train_coverage":len(joined),"missing":0,"duplicates":0,"non_finite":0,"forbidden_partition_overlap":0,"outside_manifest":0,"artifact_path":str(path.relative_to(ROOT)),"artifact_sha256":sha(path),"manifest_sha256":c["protocols"][name]["sha256"],"input_order_invariant":True}
    write_json(registry,evidence)
    print(json.dumps({"status":evidence["status"],"union_count":len(union),"arrow_reads":1,"per_protocol":{k:{"supplement":v["supplement_count"],"train_coverage":v["final_train_coverage"],"sha256":v["artifact_sha256"]} for k,v in evidence["protocols"].items()},"final673_accessed":False},indent=2))
if __name__=="__main__": main()
