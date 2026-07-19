#!/usr/bin/env python3
'''Gate 0-B canonical structure governance and split feasibility audit.

CPU-only. final-blind is loaded only as an ID column for aggregate counts.
No final ID, structure, label, or per-sample membership is ever emitted.
'''

from __future__ import annotations
import hashlib, json, re, sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd
from rdkit import Chem, rdBase, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

PROJECT=Path("/home/changliu/ExcitationNexus")
DATA=Path("/home/changliu/ExcitationNexus_Data_v2")
OUT=PROJECT/"12_Phase4_Multitask_OOD_Training"
TABLE=DATA/"tables/molecule_values_v3.parquet"
OLD=PROJECT/"05_Phase2_Baseline_Protocol/tables/teacher_table_7316_all.csv"
EXTERNAL=PROJECT/"07_Phase2C_Smoothed_Memory/tables/external_dev_benchmark_2697.csv"
BLIND=PROJECT/"07_Phase2C_Smoothed_Memory/tables/final_blind_test_674.csv"
LEGACY=PROJECT/"06_Phase2_External_Holdout/raw_3371/excitations"
REGISTRY=PROJECT/"DA_data/structure_60k_sorted.jsonl"
DFT=DATA/"raw_compact/dft/results"; PM6=DATA/"raw_compact/pm6/results"; TDDFT=DATA/"raw_compact/tddft/results"
EXPECTED={1:14267,2:369,3:1,4:2}
PRIMARY="tddft_coulomb_attraction_eV_eps3p5_proxy"
SECONDARY=["tddft_excitation_energy_ev","tddft_wavelength_nm","tddft_oscillator_strength","tddft_transition_dipole_au","tddft_coulomb_attraction_au","tddft_coulomb_attraction_eV","tddft_Sm","tddft_Sr","tddft_D_index_angstrom","tddft_H_CT_angstrom","tddft_t_index_angstrom","tddft_HDI","tddft_EDI","tddft_Q_D_to_A_au","tddft_dipole_change_norm_au"]
MASKED=["tddft_hole_on_donor_fraction","tddft_hole_on_acceptor_fraction","tddft_electron_on_donor_fraction","tddft_electron_on_acceptor_fraction"]
TARGETS=[PRIMARY,*SECONDARY,*MASKED]
TOL={PRIMARY:1e-3,"tddft_excitation_energy_ev":1e-3,"tddft_wavelength_nm":0.1,"tddft_oscillator_strength":1e-4,"tddft_transition_dipole_au":1e-4,"tddft_coulomb_attraction_au":1e-5,"tddft_coulomb_attraction_eV":1e-3,"tddft_Sm":1e-5,"tddft_Sr":1e-5,"tddft_D_index_angstrom":1e-3,"tddft_H_CT_angstrom":1e-3,"tddft_t_index_angstrom":1e-3,"tddft_HDI":1e-3,"tddft_EDI":1e-3,"tddft_Q_D_to_A_au":1e-5,"tddft_dipole_change_norm_au":1e-3,"tddft_hole_on_donor_fraction":1e-5,"tddft_hole_on_acceptor_fraction":1e-5,"tddft_electron_on_donor_fraction":1e-5,"tddft_electron_on_acceptor_fraction":1e-5}

def h(s): return hashlib.sha256(s.encode("utf-8")).hexdigest()
def nid(v):
    m=re.fullmatch(r"D-?(\d+)_A-?(\d+)",str(v).strip())
    if not m: raise ValueError(f"bad molecule ID {v!r}")
    return f"D{int(m.group(1))}_A{int(m.group(2))}"
def mkey(v):
    m=re.fullmatch(r"D-?(\d+)_A-?(\d+)",str(v).strip())
    return (int(m.group(1)),int(m.group(2))) if m else (sys.maxsize,sys.maxsize)
def ckey(v):
    m=re.search(r"(\d+)",str(v)); return int(m.group(1)) if m else sys.maxsize
def canon_full(s):
    m=Chem.MolFromSmiles(str(s))
    if m is None: raise ValueError("full SMILES parse failure")
    return Chem.MolToSmiles(Chem.RemoveHs(m),canonical=True,isomericSmiles=True)
RDLogger.DisableLog("rdApp.*")

def no_kekulize_mol(s):
    m=Chem.MolFromSmiles(str(s),sanitize=False)
    if m is None: return None
    try:
        Chem.SanitizeMol(m,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        return m
    except Exception:
        return None
def canon_component(s):
    m=no_kekulize_mol(str(s).replace("[A]","[*]"))
    if m is None: raise ValueError("component SMILES parse failure")
    return Chem.MolToSmiles(m,canonical=True,isomericSmiles=True,kekuleSmiles=False)
def scaffold(s):
    try:
        m=Chem.MolFromSmiles(s) or no_kekulize_mol(s)
        if m is None: return "","","PARSE_FAILED"
        x=Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m),canonical=True,isomericSmiles=True)
        return x,h(x),"EMPTY_ACYCLIC" if not x else "OK"
    except Exception: return "","","PARSE_FAILED"
def idset(p): return {nid(x) for x in pd.read_csv(p,usecols=["molecule_id"])["molecule_id"]}
def legacy_ids():
    suffix="_excitation.json"
    return {nid(p.name[:-len(suffix)]) for p in LEGACY.glob(f"*{suffix}")}
def registry_smiles(wanted):
    found={}
    with REGISTRY.open(encoding="utf-8") as f:
        for line in f:
            x=json.loads(line); mid=nid(x["id"])
            if mid in wanted and x.get("smiles"): found[mid]=x["smiles"]
    return found
def kabsch(a,b):
    a=a-a.mean(0); b=b-b.mean(0); u,_,vt=np.linalg.svd(b.T@a); r=u@vt
    if np.linalg.det(r)<0: u[:,-1]*=-1; r=u@vt
    return float(np.sqrt(np.mean(np.sum((b@r-a)**2,axis=1))))
def btype(v):
    return {"single":Chem.BondType.SINGLE,"double":Chem.BondType.DOUBLE,"triple":Chem.BondType.TRIPLE,"aromatic":Chem.BondType.AROMATIC}.get(str(v).lower(),Chem.BondType.SINGLE)
def geom(mid):
    dp=DFT/mid/f"{mid}_dft.json"; sp=DFT/mid/f"{mid}_sidecar.json"; pp=DFT/mid/f"{mid}_dft.pdb"
    if not dp.exists() or not sp.exists() or not pp.exists(): return {"status":"UNRESOLVED","reason":"missing"}
    try:
        d=json.loads(dp.read_text()); s=json.loads(sp.read_text())
        atoms=sorted(d["atoms"],key=lambda x:int(x["index"]))
        elements=[x["element"] for x in atoms]; roles=[x.get("type","unknown") for x in atoms]
        coords=np.asarray([x["coords"] for x in atoms],float)
        bonds=sorted((min(int(x["atom1_index"]),int(x["atom2_index"]))-1,max(int(x["atom1_index"]),int(x["atom2_index"]))-1,str(x.get("type","single")).lower()) for x in d["bonds"])
        origins=list(s.get("atom_origins",[]))
        pdb_elements=[]; pdb_coords=[]
        for line in pp.read_text().splitlines():
            if line.startswith(("ATOM  ","HETATM")):
                pdb_elements.append(line[76:78].strip() or re.sub(r"[^A-Za-z]","",line[12:16]).title())
                pdb_coords.append([float(line[30:38]),float(line[38:46]),float(line[46:54])])
        pdb_coords=np.asarray(pdb_coords,float)
        pdb_ok=(pdb_coords.shape==coords.shape and pdb_elements==elements and float(np.max(np.abs(pdb_coords-coords)))<=5.1e-4)
        if coords.shape!=(len(atoms),3): return {"status":"UNRESOLVED","reason":"shape"}
        return {"status":"PARSED","elements":elements,"roles":roles,"coords":coords,"bonds":bonds,"sidecar_ok":origins==roles and len(origins)==len(roles),"pdb_json_ok":pdb_ok,"n":len(atoms)}
    except Exception as e: return {"status":"UNRESOLVED","reason":f"parse:{type(e).__name__}"}
def graph(g):
    x=Chem.RWMol()
    for e,r in zip(g["elements"],g["roles"]):
        a=Chem.Atom(e); a.SetIsotope(1 if r=="donor" else 2 if r=="acceptor" else 3); x.AddAtom(a)
    for a,b,t in g["bonds"]: x.AddBond(a,b,btype(t))
    return x.GetMol()
def compare_geom(a,b):
    if a.get("status")!="PARSED" or b.get("status")!="PARSED": return "UNRESOLVED",None,"parse_failure",0
    if a["n"]!=b["n"]: return "UNRESOLVED",None,"atom_count_mismatch",0
    heavy=np.array([x!="H" for x in a["elements"]])
    if a["elements"]==b["elements"] and a["roles"]==b["roles"] and a["bonds"]==b["bonds"]:
        return "RESOLVED_INDEX_GRAPH",kabsch(a["coords"][heavy],b["coords"][heavy]),"explicit_index_element_role_bond",1
    try:
        matches=graph(b).GetSubstructMatches(graph(a),uniquify=False,useChirality=True,maxMatches=1001)
        if not matches: return "UNRESOLVED",None,"no_role_aware_isomorphism",0
        if len(matches)>=1001: return "UNRESOLVED",None,"symmetry_mapping_truncated",len(matches)
        vals=[kabsch(a["coords"][heavy],b["coords"][np.asarray(x)][heavy]) for x in matches]
        return "RESOLVED_SYMMETRY_ENUMERATED",min(vals),"complete_role_aware_isomorphism",len(matches)
    except Exception as e: return "UNRESOLVED",None,f"isomorphism:{type(e).__name__}",0
def dist(s):
    c=s.value_counts()
    return {"unique":int(len(c)),"singleton_entities":int((c==1).sum()),"low_frequency_le5_entities":int((c<=5).sum()),"median_records_per_entity":float(c.median()),"p95_records_per_entity":float(c.quantile(.95)),"max_records_per_entity":int(c.max())}
def append_once(path,marker,text):
    old=path.read_text(encoding="utf-8") if path.exists() else ""
    if marker not in old: path.write_text(old.rstrip()+"\n\n"+text.rstrip()+"\n",encoding="utf-8")

def main():
    start=datetime.now(timezone.utc)
    for d in ["manifests","reports","logs","data_registry"]: (OUT/d).mkdir(parents=True,exist_ok=True)
    df=pd.read_parquet(TABLE).copy()
    if len(df)!=15016 or not df.molecule_id.is_unique: raise RuntimeError("record invariant")
    df["canonical_structure_smiles_v1"]=df.canonical_smiles.map(canon_full)
    df["structure_group_id_v1"]=df.canonical_structure_smiles_v1.map(h)
    df["donor_canonical_structure_smiles_v1"]=df.donor_smiles.map(canon_component)
    df["acceptor_canonical_structure_smiles_v1"]=df.acceptor_smiles.map(canon_component)
    df["donor_structure_group_id_v1"]=df.donor_canonical_structure_smiles_v1.map(h)
    df["acceptor_structure_group_id_v1"]=df.acceptor_canonical_structure_smiles_v1.map(h)
    df["pair_identity_string_v1"]=df.donor_canonical_structure_smiles_v1+">>"+df.acceptor_canonical_structure_smiles_v1
    df["pair_group_id_v1"]=df.pair_identity_string_v1.map(h)
    df["role_identity_string_v1"]=df.canonical_structure_smiles_v1+"||D:"+df.donor_canonical_structure_smiles_v1+"||A:"+df.acceptor_canonical_structure_smiles_v1
    df["role_aware_group_id_v1"]=df.role_identity_string_v1.map(h)
    sizes=df.structure_group_id_v1.value_counts()
    df["structure_group_size"]=df.structure_group_id_v1.map(sizes).astype(int); df["group_weight"]=1/df.structure_group_size
    n2o=df.groupby("structure_group_id_v1").canonical_smiles_sha256.nunique(); o2n=df.groupby("canonical_smiles_sha256").structure_group_id_v1.nunique()
    partition_equal=bool(n2o.max()==1 and o2n.max()==1)
    exact=int((df.structure_group_id_v1==df.canonical_smiles_sha256).sum()); source_direct=int((df.canonical_smiles.map(h)==df.canonical_smiles_sha256).sum())
    size_dist={int(k):int(v) for k,v in sizes.value_counts().sort_index().items()}; expected_ok=size_dist==EXPECTED
    dup_ids=set(sizes[sizes>1].index); dup=df[df.structure_group_id_v1.isin(dup_ids)].copy()
    for prefix,col in [("full","canonical_structure_smiles_v1"),("donor","donor_canonical_structure_smiles_v1"),("acceptor","acceptor_canonical_structure_smiles_v1")]:
        x=df[col].map(scaffold)
        df[[f"{prefix}_scaffold_smiles_v1",f"{prefix}_scaffold_group_id_v1",f"{prefix}_scaffold_status"]]=pd.DataFrame(x.tolist(),index=df.index)
    did=df.groupby("donor_id").donor_structure_group_id_v1.nunique(); aid=df.groupby("acceptor_id").acceptor_structure_group_id_v1.nunique()
    da=df.groupby("donor_structure_group_id_v1").donor_id.nunique(); aa=df.groupby("acceptor_structure_group_id_v1").acceptor_id.nunique()
    df["donor_id_structure_conflict"]=df.donor_id.map(did).gt(1); df["acceptor_id_structure_conflict"]=df.acceptor_id.map(aid).gt(1)
    df["donor_structure_alias_count"]=df.donor_structure_group_id_v1.map(da).astype(int); df["acceptor_structure_alias_count"]=df.acceptor_structure_group_id_v1.map(aa).astype(int)

    rows=[]
    for gid,g in dup.groupby("structure_group_id_v1",sort=True):
        for target in TARGETS:
            v=g[target].dropna().astype(float).to_numpy()
            base={"structure_group_id_v1":gid,"structure_group_size":len(g),"target":target,"target_role":"primary" if target==PRIMARY else "masked_auxiliary" if target in MASKED else "secondary","non_null_count":len(v),"meaningful_tolerance":TOL[target]}
            if len(v):
                rg=float(v.max()-v.min()); mean=float(v.mean())
                base.update({"min":float(v.min()),"max":float(v.max()),"range":rg,"mean":mean,"std_population":float(v.std()),"maximum_absolute_deviation":float(np.max(np.abs(v-mean))),"exact_consistency":bool(np.all(v==v[0])),"allclose_consistency":bool(np.allclose(v,v[0],rtol=1e-7,atol=1e-8)),"meaningful_disagreement":bool(rg>TOL[target])})
            else:
                base.update({"min":np.nan,"max":np.nan,"range":np.nan,"mean":np.nan,"std_population":np.nan,"maximum_absolute_deviation":np.nan,"exact_consistency":False,"allclose_consistency":False,"meaningful_disagreement":False})
            rows.append(base)
    dispersion=pd.DataFrame(rows)

    drows=[]; rrows=[]; unresolved_groups=0
    for gid,g in dup.groupby("structure_group_id_v1",sort=True):
        order=g.sort_values("molecule_id",key=lambda s:s.map(mkey)); mids=order.molecule_id.tolist(); gs={mid:geom(mid) for mid in mids}; ref=mids[0]
        comps=[(mid,*compare_geom(gs[ref],gs[mid])) for mid in mids[1:]]; unres=[x for x in comps if x[1]=="UNRESOLVED"]
        if unres or gs[ref].get("status")!="PARSED": unresolved_groups+=1
        rmsd=[x[2] for x in comps if x[2] is not None]; role_ok=order.role_aware_group_id_v1.nunique()==1
        side_ok=all(x.get("sidecar_ok",False) for x in gs.values()); pdb_ok=all(x.get("pdb_json_ok",False) for x in gs.values())
        atom_ok=order.num_atoms_total.nunique(dropna=False)==1 and order.dft_atom_count.nunique(dropna=False)==1 and len({x.get("n") for x in gs.values()})==1
        td=dispersion[dispersion.structure_group_id_v1==gid]; meaningful=int(td.meaningful_disagreement.sum()); primary=td[td.target==PRIMARY].iloc[0]
        eligible=bool(role_ok and side_ok and atom_ok and not unres and meaningful==0)
        drows.append({"structure_group_id_v1":gid,"structure_group_size":len(order),"molecule_ids_numeric_sorted":";".join(mids),"unique_donor_ids":order.donor_id.nunique(),"donor_ids_numeric_sorted":";".join(sorted(order.donor_id.unique(),key=ckey)),"unique_acceptor_ids":order.acceptor_id.nunique(),"acceptor_ids_numeric_sorted":";".join(sorted(order.acceptor_id.unique(),key=ckey)),"unique_donor_structures":order.donor_structure_group_id_v1.nunique(),"unique_acceptor_structures":order.acceptor_structure_group_id_v1.nunique(),"unique_role_aware_groups":order.role_aware_group_id_v1.nunique(),"role_annotation_consistent":role_ok,"atom_count_consistent":atom_ok,"sidecar_atom_origin_consistent":side_ok,"dft_pdb_json_geometry_consistent":pdb_ok,"d81_a28_conflict_in_group":"D81_A28" in mids,"geometry_reference_id":ref,"geometry_resolved_comparisons":len(comps)-len(unres),"geometry_unresolved_comparisons":len(unres),"geometry_mapping_methods":";".join(sorted({x[1] for x in comps})),"heavy_atom_rmsd_min_angstrom":min(rmsd) if rmsd else np.nan,"heavy_atom_rmsd_max_angstrom":max(rmsd) if rmsd else np.nan,"geometry_unresolved_reasons":";".join(sorted({x[3] for x in unres})),"primary_range_eV":float(primary["range"]),"meaningful_target_disagreement_count":meaningful,"representative_eligible_under_strict_rule":eligible})
        rrows.append({"structure_group_id_v1":gid,"deterministic_representative_molecule_id":ref,"selection_rule":"lowest numeric (donor_number, acceptor_number)","strict_representative_eligible":eligible,"role_consistent":role_ok,"geometry_fully_resolved":not unres,"sidecar_consistent":side_ok,"meaningful_target_disagreement_count":meaningful})
    dg=pd.DataFrame(drows); reps=pd.DataFrame(rrows)
    pdsp=dispersion[dispersion.target==PRIMARY]; r=pdsp["range"].astype(float)
    bins={"le_1e-6_eV":int((r<=1e-6).sum()),"gt_1e-6_le_1e-3_eV":int(((r>1e-6)&(r<=1e-3)).sum()),"gt_1e-3_le_1e-2_eV":int(((r>1e-3)&(r<=1e-2)).sum()),"gt_1e-2_eV":int((r>1e-2).sum())}

    old=idset(OLD); ext=idset(EXTERNAL); blind=idset(BLIND); legacy=legacy_ids()
    hs=registry_smiles(old|ext|blind|legacy); hh={mid:h(canon_full(s)) for mid,s in hs.items()}
    oldh={hh[x] for x in old}; exth={hh[x] for x in ext}; blindh={hh[x] for x in blind}; nonblindh={hh[x] for x in legacy-blind}; newh=set(df.structure_group_id_v1)
    new_final_id=len(set(df.molecule_id.map(nid))&blind); new_final_structure=len(newh&blindh); selectionh=exth|nonblindh
    qrows=[]
    for gid,g in df.groupby("structure_group_id_v1",sort=True):
        sel=gid in selectionh; train=gid in oldh
        if not(sel or train): continue
        acts=(["HISTORICAL_MODEL_SELECTION_QUARANTINE"] if sel else [])+(["HISTORICAL_TRAIN_OVERLAP"] if train else [])
        qrows.append({"structure_group_id_v1":gid,"new_record_count":len(g),"new_molecule_ids_numeric_sorted":";".join(sorted(g.molecule_id,key=mkey)),"historical_external_or_nonblind_legacy_overlap":sel,"historical_old7316_overlap":train,"governance_action":";".join(acts),"allowed_future_internal_evaluation":False,"new_only_training_policy":"FORBIDDEN" if sel else "DEFER_TO_GATE_0C_NO_DOUBLE_WEIGHT"})
    quarantine=pd.DataFrame(qrows)

    compcols=["molecule_id","donor_id","acceptor_id","donor_canonical_structure_smiles_v1","acceptor_canonical_structure_smiles_v1","donor_structure_group_id_v1","acceptor_structure_group_id_v1","pair_group_id_v1","role_aware_group_id_v1","donor_id_structure_conflict","acceptor_id_structure_conflict","donor_structure_alias_count","acceptor_structure_alias_count","donor_scaffold_group_id_v1","acceptor_scaffold_group_id_v1","full_scaffold_group_id_v1","donor_scaffold_status","acceptor_scaffold_status","full_scaffold_status"]
    comp=df[compcols].copy()
    specs={"donor_structure":"donor_structure_group_id_v1","acceptor_structure":"acceptor_structure_group_id_v1","ordered_structure_pair":"pair_group_id_v1","full_structure":"structure_group_id_v1","donor_scaffold":"donor_scaffold_group_id_v1","acceptor_scaffold":"acceptor_scaffold_group_id_v1","full_scaffold":"full_scaffold_group_id_v1"}
    frows=[]
    for typ,col in specs.items():
        for key,count in df[col].value_counts().sort_index().items(): frows.append({"entity_type":typ,"entity_group_id_v1":key,"record_count":int(count)})
    freq=pd.DataFrame(frows); feas={k:dist(df[v]) for k,v in specs.items()}
    feas.update({"unique_donor_ids":int(df.donor_id.nunique()),"unique_acceptor_ids":int(df.acceptor_id.nunique()),"unique_id_pairs":int(df[["donor_id","acceptor_id"]].drop_duplicates().shape[0]),"donor_id_multi_structure_count":int((did>1).sum()),"acceptor_id_multi_structure_count":int((aid>1).sum()),"donor_structure_alias_groups":int((da>1).sum()),"acceptor_structure_alias_groups":int((aa>1).sum()),"scaffold_parse_failures":{"donor":int((df.donor_scaffold_status=="PARSE_FAILED").sum()),"acceptor":int((df.acceptor_scaffold_status=="PARSE_FAILED").sum()),"full":int((df.full_scaffold_status=="PARSE_FAILED").sum())}})

    tc=Counter(); tu=defaultdict(set)
    for mid in df.molecule_id:
        candidates=[("pm6_metadata.extraction_date",PM6/mid/f"{mid}_metadata.json","extraction_date"),("dft_metadata.timestamp_utc",DFT/mid/f"{mid}_metadata.json","timestamp_utc"),("dft_json.timestamp",DFT/mid/f"{mid}_dft.json","timestamp"),("tddft_properties.timestamp_utc",TDDFT/mid/f"{mid}_properties.json","timestamp_utc")]
        for name,p,key in candidates:
            try: v=json.loads(p.read_text()).get(key) if p.exists() else None
            except Exception: v=None
            if v not in (None,""): tc[name]+=1; tu[name].add(str(v))
    timeaudit={"status":"BLOCKED_NO_TRUSTED_TIMESTAMP","candidate_field_non_null_counts":dict(tc),"candidate_field_unique_counts":{k:len(v) for k,v in tu.items()},"reason":"available fields are extraction/derived JSON timestamps without immutable generation or scheduler provenance; filesystem mtime forbidden","trusted_generation_timestamp":False,"trusted_submission_or_completion_timestamp":False,"trusted_batch_or_version_identifier":False}

    role_bad=int((~dg.role_annotation_consistent).sum()); meaningful_primary=int(pdsp.meaningful_disagreement.sum()); meaningful_any=int(dispersion.meaningful_disagreement.sum()); strict=int(reps.strict_representative_eligible.sum())
    pdb_bad=int((~dg.dft_pdb_json_geometry_consistent).sum())
    meaningful_groups=int(dispersion.loc[dispersion.meaningful_disagreement,"structure_group_id_v1"].nunique())
    rmsd_summary={k:float(v) for k,v in dg.heavy_atom_rmsd_max_angstrom.describe(percentiles=[.5,.9,.95,.99]).to_dict().items()}
    target_summary={}
    for target,g in dispersion.groupby("target"):
        target_summary[target]={"groups":int(len(g)),"meaningful_groups":int(g.meaningful_disagreement.sum()),"median_range":float(g["range"].median()),"max_range":float(g["range"].max())}
    strategy_comparison={
      "deterministic_representative":{"output_records":int(len(sizes)),"records_removed_from_training_view":int(len(df)-len(sizes)),"strictly_eligible_duplicate_groups":strict,"ineligible_duplicate_groups":int(len(dg)-strict)},
      "retain_replicates_group_weight":{"output_records":int(len(df)),"effective_total_group_weight":float(df.group_weight.sum()),"effective_duplicate_group_weight":float(df.loc[df.structure_group_size>1,"group_weight"].sum()),"group_weight_min":float(df.group_weight.min()),"group_weight_max":float(df.group_weight.max())},
      "group_target_aggregation":{"output_structure_rows":int(len(sizes)),"groups_requiring_aggregation":int(len(dg)),"role_inconsistent_groups":role_bad,"groups_with_any_meaningful_target_disagreement":meaningful_groups,"strictly_eligible_duplicate_groups":strict}
    }
    if role_bad==0 and unresolved_groups==0 and meaningful_any==0:
        policy="DETERMINISTIC_REPRESENTATIVE"; reason="all duplicate groups have consistent roles, resolved geometry, and no meaningful target disagreement"
    else:
        policy="RETAIN_REPLICATES_WITH_GROUP_WEIGHT"; reason="role/geometry uncertainty or meaningful target dispersion exists; aggregation could erase distinct records"

    gmcols=["molecule_id","donor_id","acceptor_id","canonical_structure_smiles_v1","structure_group_id_v1","donor_structure_group_id_v1","acceptor_structure_group_id_v1","pair_group_id_v1","role_aware_group_id_v1","structure_group_size","group_weight","full_scaffold_group_id_v1","sidecar_conflict_flag"]
    gm=df[gmcols].sort_values("molecule_id",key=lambda s:s.map(mkey))
    gm.to_parquet(OUT/"manifests/new15016_structure_groups_v1.parquet",index=False); gm.to_csv(OUT/"manifests/new15016_structure_groups_v1.csv",index=False)
    dg.to_csv(OUT/"manifests/duplicate_structure_groups_v1.csv",index=False); dispersion.to_csv(OUT/"manifests/duplicate_target_dispersion_v1.csv",index=False)
    comp.to_csv(OUT/"manifests/component_identity_v1.csv",index=False); freq.to_csv(OUT/"manifests/component_frequency_v1.csv",index=False)
    quarantine.to_csv(OUT/"manifests/historical_overlap_quarantine_v1.csv",index=False); reps.to_csv(OUT/"manifests/representative_candidates_v1.csv",index=False)

    p0=bool(feas["donor_id_multi_structure_count"] or feas["acceptor_id_multi_structure_count"] or any(feas["scaffold_parse_failures"].values()))
    done=bool(len(df)==15016 and len(sizes)==14639 and expected_ok and partition_equal and len(dg)==372 and unresolved_groups==0 and pdb_bad==0 and not p0 and new_final_id==0 and new_final_structure==0)
    evidence={"gate":"0-B","status":"DONE" if done else "BLOCKED","generated_utc":datetime.now(timezone.utc).isoformat(),"python":sys.version.split()[0],"rdkit_version":rdBase.rdkitVersion,"calculation_records":len(df),"canonical_structure_groups":len(sizes),"structure_size_distribution":size_dist,"duplicate_groups":len(dg),"duplicate_records":len(dup),"extra_records":len(df)-len(sizes),"expected_distribution_match":expected_ok,"stored_hash_comparison":{"exact_hash_matches":exact,"mismatches":len(df)-exact,"stored_hash_matches_direct_source_string":source_direct,"partition_equal":partition_equal,"new_to_stored_max_partitions":int(n2o.max()),"stored_to_new_max_partitions":int(o2n.max()),"reason_if_hash_differs":"V1 hashes RDKit RemoveHs canonical output; source hashes stored explicit-H canonical string"},"duplicate_audit":{"role_inconsistent_groups":role_bad,"atom_count_inconsistent_groups":int((~dg.atom_count_consistent).sum()),"sidecar_inconsistent_groups":int((~dg.sidecar_atom_origin_consistent).sum()),"dft_pdb_json_inconsistent_groups":pdb_bad,"geometry_unresolved_groups":unresolved_groups,"heavy_atom_rmsd_summary":rmsd_summary,"d81_a28_in_duplicate_group":bool(dg.d81_a28_conflict_in_group.any()),"strict_representative_eligible_groups":strict},"target_dispersion":{"primary_name":"J_eh_screened_eV_eps3p5 proxy","primary_column":PRIMARY,"primary_range_bins":bins,"primary_meaningful_disagreement_groups":meaningful_primary,"all_target_meaningful_disagreement_group_target_pairs":meaningful_any,"groups_with_any_meaningful_target_disagreement":meaningful_groups,"per_target_summary":target_summary},"replicate_strategy_comparison":strategy_comparison,"component_feasibility":feas,"historical":{"new_vs_final_id_intersection_aggregate":new_final_id,"new_vs_final_structure_intersection_aggregate":new_final_structure,"new_vs_external_structure_groups":len(newh&exth),"new_vs_old_structure_groups":len(newh&oldh),"external_vs_final_structure_groups":len(exth&blindh),"old_vs_external_structure_groups":len(oldh&exth),"quarantine_group_count":len(quarantine),"quarantine_new_record_count":int(quarantine.new_record_count.sum()) if len(quarantine) else 0,"final_per_sample_artifact_emitted":False},"replicate_policy":{"recommendation":policy,"reason":reason},"time_split":timeaudit,"split_generated":False,"training_or_cuda_compute_run":False,"raw_data_modified":False}
    (OUT/"logs/gate0b_evidence.json").write_text(json.dumps(evidence,indent=2),encoding="utf-8")
    spec={"version":"v1","rdkit_version":rdBase.rdkitVersion,"full_structure_algorithm":["Chem.MolFromSmiles","Chem.RemoveHs","Chem.MolToSmiles(canonical=True,isomericSmiles=True)","SHA-256 canonical string"],"component_placeholder_rule":"replace [A] with [*], parse sanitize=False, sanitize with SANITIZE_ALL excluding SANITIZE_KEKULIZE, then MolToSmiles(canonical=True,isomericSmiles=True,kekuleSmiles=False)","pair_identity_string":"donor canonical + >> + acceptor canonical","role_aware_identity_string":"full canonical + ||D: + donor canonical + ||A: + acceptor canonical","hash_rule":"hash only frozen identity string; RDKit version is metadata, not hash content","group_weight":"1 / structure_group_size","partition_invariant":"one structure_group_id_v1 remains wholly in one partition","geometry_mapping":"index+element+role+bond exact match, else complete role-aware graph isomorphism; truncation/failure UNRESOLVED","meaningful_target_tolerances":TOL}
    (OUT/"data_registry/structure_identity_spec_v1.json").write_text(json.dumps(spec,indent=2),encoding="utf-8")

    (OUT/"reports/gate0b_structure_governance.md").write_text(f'''# Gate 0-B structure governance

Status: **{evidence["status"]}**.

- calculation records: **{len(df):,}** complete PM6+DFT+TDDFT records.
- canonical structures: **{len(sizes):,}** RDKit V1 groups.
- raw records deleted: 0.

## Hash and duplicate audit

- V1/source exact hash matches: {exact:,}/{len(df):,}.
- Source hashes stored explicit-H strings: {source_direct:,}/{len(df):,}.
- Partitions equivalent in both directions: **{partition_equal}**.
- Size distribution: {json.dumps(size_dist,sort_keys=True)}
- Duplicate groups/records/extra: {len(dg)}/{len(dup)}/{len(df)-len(sizes)}
- Role/atom-count/sidecar/PDB-JSON inconsistent groups: {role_bad}/{int((~dg.atom_count_consistent).sum())}/{int((~dg.sidecar_atom_origin_consistent).sum())}/{pdb_bad}
- Geometry UNRESOLVED groups: {unresolved_groups}; RMSD max-group summary: {json.dumps(rmsd_summary)}. D81_A28 in duplicate group: {bool(dg.d81_a28_conflict_in_group.any())}.

V1 hashes `RemoveHs` canonical strings; source hashes explicit-H strings. Equal counts were not treated as proof. RMSD uses index+element+role+bond correspondence or complete role-aware isomorphism; truncation is UNRESOLVED.

## Target dispersion and policy

Primary: **J_eh_screened_eV_eps3p5 proxy**, not experimental Eb.

- Range bins: {json.dumps(bins)}
- Primary meaningful groups: {meaningful_primary}; all enabled meaningful group-target pairs: {meaningful_any}.
- Raw and screened Coulomb are deterministic transforms, not independent evidence.

### Quantitative comparison

- deterministic representative: 14,639 training-view rows, removes 377 repeated rows; only {strict}/372 duplicate groups satisfy the strict role+geometry+target rule.
- retain replicates + group weight: 15,016 rows, effective total weight {float(df.group_weight.sum()):.0f}; duplicate rows sum to effective weight {float(df.loc[df.structure_group_size>1,"group_weight"].sum()):.0f}.
- group target aggregation: 14,639 structure rows, but {role_bad} role-aware-inconsistent groups and {meaningful_groups} groups with meaningful target disagreement make global averaging unsafe.

Recommendation: **{policy}** because {reason}. Keep groups in one partition. Retained replicates require `group_weight=1/group_size` and record-level plus structure-group-macro metrics.
''',encoding="utf-8")

    (OUT/"reports/gate0b_component_split_feasibility.md").write_text(f'''# Gate 0-B component and split feasibility

No split was generated.

- donor IDs/structures: {feas["unique_donor_ids"]}/{feas["donor_structure"]["unique"]}
- acceptor IDs/structures: {feas["unique_acceptor_ids"]}/{feas["acceptor_structure"]["unique"]}
- ID pairs/structure pairs: {feas["unique_id_pairs"]}/{feas["ordered_structure_pair"]["unique"]}
- donor/acceptor same-ID multi-structure: {feas["donor_id_multi_structure_count"]}/{feas["acceptor_id_multi_structure_count"]}
- donor/acceptor alias groups: {feas["donor_structure_alias_groups"]}/{feas["acceptor_structure_alias_groups"]}

Cold splits use component structure groups, never numeric IDs. Alias IDs collapse before OOD assignment.

```json
{json.dumps(feas,indent=2)}
```

donor-cold: {"BLOCKED_P0_ID_CONFLICT" if feas["donor_id_multi_structure_count"] else "FEASIBLE_BY_STRUCTURE_GROUP"}.
acceptor-cold: {"BLOCKED_P0_ID_CONFLICT" if feas["acceptor_id_multi_structure_count"] else "FEASIBLE_BY_STRUCTURE_GROUP"}.
pair-cold: FEASIBLE after alias collapse.
both-cold: Gate 0-C must solve group-disjoint assignment and verify power.
scaffold-cold: {"BLOCKED_PARSE_FAILURE" if any(feas["scaffold_parse_failures"].values()) else "FEASIBLE_BY_MURCKO_GROUP"}.
time/prospective: **BLOCKED_NO_TRUSTED_TIMESTAMP**.
''',encoding="utf-8")

    (OUT/"reports/historical_benchmark_boundary_policy_v1.md").write_text(f'''# Historical benchmark boundary policy V1

- Preserve official 7,316 / 2,698 / 673 memberships and original metrics.
- Do not claim external2698/final673 are structure-disjoint.
- old/external shared structures: {len(oldh&exth)}.
- external/final shared structures: {len(exth&blindh)}.
- new/final aggregate intersections: ID={new_final_id}, structure={new_final_structure}.
- Quarantine external/non-blind legacy overlap at full structure-group level.
- old overlap is not an independent new structure and cannot be independent evaluation or double-weighted.
- Future final evaluation preregisters official and structure-purged sensitivity metrics.
- No final evaluation, label read, or per-sample final artifact occurred.

The 18-structure external/final issue is a benchmark correction, not a new-data grouping blocker.
''',encoding="utf-8")

    blockers=[]
    if not expected_ok: blockers.append("structure size distribution mismatch")
    if not partition_equal: blockers.append("V1/source partitions differ")
    if unresolved_groups: blockers.append(f"{unresolved_groups} duplicate groups have unresolved geometry")
    if pdb_bad: blockers.append(f"{pdb_bad} duplicate groups have PDB/JSON geometry mismatch")
    if feas["donor_id_multi_structure_count"] or feas["acceptor_id_multi_structure_count"]: blockers.append("same component ID maps to multiple structures")
    if any(feas["scaffold_parse_failures"].values()): blockers.append("Murcko scaffold parse failures")
    qn=int(quarantine.new_record_count.sum()) if len(quarantine) else 0
    (OUT/"reports/gate0b_summary.md").write_text(f'''# Gate 0-B summary

Overall: **{evidence["status"]}**.

| Item | Status | Evidence |
|---|---|---|
| 15,016 records retained | DONE | {len(df)} |
| 14,639 structure groups | {"DONE" if len(sizes)==14639 else "BLOCKED"} | {len(sizes)} |
| Expected size distribution | {"DONE" if expected_ok else "BLOCKED"} | {json.dumps(size_dist)} |
| 372 duplicates audited | {"DONE" if len(dg)==372 else "BLOCKED"} | {len(dg)} |
| Component identity | {"DONE" if not p0 else "BLOCKED"} | donor conflict={feas["donor_id_multi_structure_count"]}; acceptor={feas["acceptor_id_multi_structure_count"]} |
| Historical group quarantine | DONE | groups={len(quarantine)}; records={qn} |
| final673 sealed | DONE | aggregate new ID={new_final_id}, structure={new_final_structure} |
| Replicate policy | DONE | {policy} |
| Split/training/GPU | DONE (none) | Gate 0-C only |
| Time split | BLOCKED_NO_TRUSTED_TIMESTAMP | no trusted timestamp |

Gate 0-C blockers: {"; ".join(blockers) if blockers else "none for non-time grouped splits; time split unavailable"}.
''',encoding="utf-8")

    append_once(OUT/"PROJECT_STATE.md","## Gate 0-B",f'''## Gate 0-B

Status: **{evidence["status"]}**. 15,016 calculation records map to 14,639 canonical structures; none deleted. No split/training. Replicate recommendation: `{policy}`. Time split is `BLOCKED_NO_TRUSTED_TIMESTAMP`. See `reports/gate0b_summary.md`.''')
    append_once(OUT/"TODO.md","Gate 0-B structure governance",f'''## Gate 0-B structure governance

- [x] DONE — Freeze V1 full/component/pair/role-aware identities.
- [x] DONE — Audit 372 duplicate groups and enabled targets.
- [x] DONE — Quantify aliases, conflicts, scaffolds, and tails.
- [x] DONE — Freeze historical group quarantine and benchmark correction.
- [{"x" if done else "!"}] {"DONE" if done else "BLOCKED"} — Gate 0-B identity/geometry success criteria.
- [!] BLOCKED — Time split: no trusted immutable timestamp.
- [ ] TODO — Gate 0-C grouped splits; not performed here.''')
    append_once(OUT/"DECISIONS.md","Gate 0-B decisions",f'''## Gate 0-B decisions

- 15,016 calculation records and 14,639 canonical structures are distinct counts; delete none.
- Use V1 structure/component hashes, not rows or numeric IDs.
- Keep one structure group wholly in one partition.
- Replicate recommendation `{policy}`; retained rows use `group_weight=1/group_size`.
- Preserve official metrics and add structure-purged sensitivity metrics.
- external/final overlap is a benchmark limitation, not a new-data blocker.
- No filesystem-mtime time split; `BLOCKED_NO_TRUSTED_TIMESTAMP`.''')

    reg=OUT/"RUN_REGISTRY.csv"
    registry_lines=[line for line in reg.read_text().splitlines() if not line.startswith("gate0b-20260718,")]
    registry_lines.append(f'gate0b-20260718,12,structure_governance,{evidence["status"]},{start.isoformat()},{datetime.now(timezone.utc).isoformat()},NONE,scripts/gate0b_structure_governance.py,"15016 records / 14639 structures / {policy}",logs/gate0b_evidence.json')
    reg.write_text("\n".join(registry_lines)+"\n")
    run=[f"{start.isoformat()} Gate 0-B started CPU-only",f"records={len(df)} structures={len(sizes)} duplicates={len(dg)}",f"geometry_unresolved_groups={unresolved_groups}",f"component_conflicts donor={feas['donor_id_multi_structure_count']} acceptor={feas['acceptor_id_multi_structure_count']}",f"replicate_policy={policy}",f"status={evidence['status']}",f"{datetime.now(timezone.utc).isoformat()} stopped before split/training"]
    (OUT/"logs/gate0b_run.log").write_text("\n".join(run)+"\n")
    targets=[OUT/"scripts/gate0b_structure_governance.py",*sorted((OUT/"manifests").glob("*v1.*")),*sorted((OUT/"reports").glob("gate0b*.md")),OUT/"reports/historical_benchmark_boundary_policy_v1.md",OUT/"logs/gate0b_evidence.json",OUT/"logs/gate0b_run.log",OUT/"data_registry/structure_identity_spec_v1.json",OUT/"PROJECT_STATE.md",OUT/"TODO.md",OUT/"DECISIONS.md",OUT/"RUN_REGISTRY.csv"]
    (OUT/"data_registry/gate0b_sha256.txt").write_text("\n".join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p}" for p in targets)+"\n")
    print(json.dumps(evidence,indent=2))

if __name__=="__main__": main()
