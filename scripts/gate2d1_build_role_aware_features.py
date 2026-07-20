#!/usr/bin/env python3
"""Build the frozen, target-free Gate 2-D1 representation cache."""
from __future__ import annotations
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger, rdBase
from rdkit.Chem import rdFingerprintGenerator
try:
    from scripts.gate2d1_common import ROOT, config_and_verify, content_hash, resolve, sha, write_json
except ModuleNotFoundError:
    from gate2d1_common import ROOT, config_and_verify, content_hash, resolve, sha, write_json

def parse(text):
    text=str(text); m=Chem.MolFromSmiles(text)
    if m is not None: return m,"NORMAL"
    m=Chem.MolFromSmiles(text,sanitize=False)
    if m is None: return None,"PARSE_FAILED"
    ops=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    return (m,"NON_KEKULIZING_FALLBACK") if Chem.SanitizeMol(m,sanitizeOps=ops,catchErrors=True)==Chem.SanitizeFlags.SANITIZE_NONE else (None,"PARSE_FAILED")
def fp_array(mol,generator,bits):
    out=np.zeros(bits,dtype=np.uint8); DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol),out); return out
def array_sha(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def main():
    c=config_and_verify(); output=resolve(c["local_feature_cache"]); registry=ROOT/"data_registry/gate2d1_feature_schema_v1.json"; audit=ROOT/"logs/gate2d1_feature_audit.json"
    if output.exists() or registry.exists(): raise RuntimeError("feature cache already frozen")
    RDLogger.DisableLog("rdApp.*")
    frozen=pd.read_parquet(resolve(c["inputs"]["c0_feature_cache"]["path"])); structures=pd.read_parquet(resolve(c["inputs"]["structure_registry"]["path"])); components=pd.read_csv(resolve(c["inputs"]["component_registry"]["path"]))
    source=structures[["molecule_id","canonical_structure_smiles_v1","structure_group_id_v1"]].merge(components[["molecule_id","donor_canonical_structure_smiles_v1","acceptor_canonical_structure_smiles_v1","donor_structure_group_id_v1","acceptor_structure_group_id_v1"]],on="molecule_id",validate="one_to_one").merge(frozen,on="molecule_id",validate="one_to_one").sort_values("molecule_id",kind="mergesort").reset_index(drop=True)
    if len(source)!=15016 or not source.molecule_id.is_unique: raise RuntimeError("feature source join failed")
    desc_names=[f"pair_{x}" for x in c["descriptor_names"]]; c0_names=[f"pair_morgan_{i}" for i in range(512)]
    descriptors=source[desc_names].to_numpy(np.float32); frozen512=source[c0_names].to_numpy(np.uint8)
    g512=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=512,includeChirality=False); g1536=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=1536,includeChirality=False)
    caches={"full512":{},"full1536":{},"donor512":{},"acceptor512":{}}; parse_modes={"full":{},"donor":{},"acceptor":{}}
    def encoded(text,kind,bits):
        key=str(text); name=f"{kind}{bits}" if kind=="full" else f"{kind}512"
        if key not in caches[name]:
            m,mode=parse(key); parse_modes[kind][key]=mode
            if m is None: raise RuntimeError(f"{kind} component parse failure")
            caches[name][key]=fp_array(m,g1536 if bits==1536 else g512,bits)
        return caches[name][key]
    full512=np.stack([encoded(x,"full",512) for x in source.canonical_structure_smiles_v1]); full1536=np.stack([encoded(x,"full",1536) for x in source.canonical_structure_smiles_v1]); donor512=np.stack([encoded(x,"donor",512) for x in source.donor_canonical_structure_smiles_v1]); acceptor512=np.stack([encoded(x,"acceptor",512) for x in source.acceptor_canonical_structure_smiles_v1])
    if not np.array_equal(full512,frozen512): raise RuntimeError("Arm A full-512 fingerprint does not exactly reproduce frozen C0")
    ids=source.molecule_id.astype(str).to_numpy(); digest=content_hash(ids,descriptors,full512,full1536,donor512,acceptor512)
    shuffled=source.sample(frac=1,random_state=20260720).sort_values("molecule_id",kind="mergesort")
    if not np.array_equal(shuffled.molecule_id.to_numpy(),ids): raise RuntimeError("sort determinism failed")
    output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(output,molecule_id=ids,descriptors=descriptors,full512=full512,full1536=full1536,donor512=donor512,acceptor512=acceptor512)
    def collisions(values,groups):
        table=pd.DataFrame({"group":groups,"hash":[hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest() for x in values]}).drop_duplicates("group")
        return {"structures":len(table),"unique_fingerprints":int(table.hash.nunique()),"collision_count":int(len(table)-table.hash.nunique())}
    donor_stats=collisions(donor512,source.donor_structure_group_id_v1); acceptor_stats=collisions(acceptor512,source.acceptor_structure_group_id_v1)
    schema={"status":"GATE2D1_TARGET_FREE_FEATURES_FROZEN","artifact_path":str(output.relative_to(ROOT)),"artifact_sha256":sha(output),"content_sha256":digest,"records":len(source),"rdkit_version":rdBase.rdkitVersion,"radius":2,"use_chirality":False,"arms":{"A_C0_512_reference":532,"B_C0_Wide_1536":1556,"C_RA2D_1536":1556},"blocks":{"descriptors":{"shape":list(descriptors.shape),"sha256":array_sha(descriptors)},"full512":{"shape":list(full512.shape),"sha256":array_sha(full512)},"full1536":{"shape":list(full1536.shape),"sha256":array_sha(full1536)},"donor512":{"shape":list(donor512.shape),"sha256":array_sha(donor512)},"acceptor512":{"shape":list(acceptor512.shape),"sha256":array_sha(acceptor512)}},"c0_exact_match_records":15016,"input_order_invariant":True,"target_columns":[],"donor":donor_stats,"acceptor":acceptor_stats,"bit_density":{"full512":float(full512.mean()),"full1536":float(full1536.mean()),"donor512":float(donor512.mean()),"acceptor512":float(acceptor512.mean())},"parse":{"full_success":len(source),"donor_success":len(source),"acceptor_success":len(source),"full_fallback_unique":sum(x=="NON_KEKULIZING_FALLBACK" for x in parse_modes["full"].values()),"donor_fallback_unique":sum(x=="NON_KEKULIZING_FALLBACK" for x in parse_modes["donor"].values()),"acceptor_fallback_unique":sum(x=="NON_KEKULIZING_FALLBACK" for x in parse_modes["acceptor"].values()),"donor_wildcard_records":int(source.donor_canonical_structure_smiles_v1.str.contains(r"\*",regex=True).sum()),"acceptor_wildcard_records":int(source.acceptor_canonical_structure_smiles_v1.str.contains(r"\*",regex=True).sum()),"wildcard_replaced_with_carbon":False},"alias_consistency":True,"ambiguous_role_records_used_for_component_inference":0}
    write_json(registry,schema); write_json(audit,{**schema,"artifact_path":"LOCAL_GIT_IGNORED","source_hashes":{k:v["sha256"] for k,v in c["inputs"].items() if k in ("c0_feature_cache","structure_registry","component_registry")}})
    lines=["# Gate 2-D1 feature integrity","",f"Status: **{schema['status']}**.","",f"- Records: {schema['records']:,}; Arm columns: 532 / 1,556 / 1,556.",f"- Frozen C0-512 exact matches: {schema['c0_exact_match_records']:,}.",f"- Donor/acceptor parse success: {schema['parse']['donor_success']:,} / {schema['parse']['acceptor_success']:,}.",f"- Donor fingerprint structures/unique/collisions: {donor_stats['structures']} / {donor_stats['unique_fingerprints']} / {donor_stats['collision_count']}.",f"- Acceptor fingerprint structures/unique/collisions: {acceptor_stats['structures']} / {acceptor_stats['unique_fingerprints']} / {acceptor_stats['collision_count']}.",f"- Bit density full512/full1536/donor/acceptor: {schema['bit_density']}.","- Attachment wildcards were preserved; no atom-role candidate was used; cache contains no targets.",""]
    (ROOT/"reports/gate2d1_feature_integrity.md").write_text("\n".join(lines))
    print(json.dumps({"status":schema["status"],"artifact_sha256":schema["artifact_sha256"],"content_sha256":digest,"arms":schema["arms"],"donor":donor_stats,"acceptor":acceptor_stats},indent=2))
if __name__=="__main__": main()
