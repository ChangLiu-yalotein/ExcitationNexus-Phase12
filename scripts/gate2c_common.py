#!/usr/bin/env python3
"""Shared, artifact-bound utilities for Gate 2-C.

This module never reads the source target Parquet.  Validation truth comes only from
the one-time, local calibration artifact; test truth comes only from frozen point-
prediction artifacts.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[1]
TARGET = "tddft_coulomb_attraction_eV_eps3p5_proxy"
PROTOCOLS = ("iid", "donor_cold", "acceptor_cold", "pair_cold", "both_cold", "full_scaffold_cold")


def resolve(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else ROOT / path


def read_json(path: str | Path) -> dict:
    return json.loads(resolve(path).read_text())


def write_json(path: str | Path, value: object) -> None:
    path = resolve(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with resolve(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""): h.update(chunk)
    return h.hexdigest()


def safe_float(value) -> float | None:
    value = float(value)
    return value if np.isfinite(value) else None


def verify_config(config: dict) -> None:
    for item in config["inputs"].values():
        if sha256(item["path"]) != item["sha256"]: raise RuntimeError(f"input hash mismatch: {item['path']}")
    for item in config["ood_validation_predictions"].values():
        if sha256(item["path"]) != item["sha256"]: raise RuntimeError(f"validation prediction hash mismatch: {item['path']}")
    for item in config["protocols"].values():
        if sha256(item["manifest"]) != item["sha256"]: raise RuntimeError(f"manifest hash mismatch: {item['manifest']}")
    lock = read_json("data_registry/gate2c_preregistration_lock_v1.json")
    if sha256("configs/gate2c_uq_applicability_audit_v1.json") != lock["config_sha256"]:
        raise RuntimeError("preregistration config changed after lock")
    if config["main_parquet_access_after_extraction"] or not config["no_new_point_predictions"]:
        raise RuntimeError("Gate 2-C firewall invalid")


def manifests(config: dict) -> dict[str, pd.DataFrame]:
    result = {}
    for name, spec in config["protocols"].items():
        frame = pd.read_csv(resolve(spec["manifest"]))
        if len(frame) != 15016 or not frame.molecule_id.is_unique: raise RuntimeError(f"bad manifest: {name}")
        if set(frame.loc[frame.partition.eq("val"), "molecule_id"]) & set(frame.loc[frame.partition.eq("test"), "molecule_id"]):
            raise RuntimeError(f"validation/test overlap: {name}")
        result[name] = frame
    return result


def load_validation(config: dict, name: str, manifest: pd.DataFrame) -> pd.DataFrame:
    pred_spec = config["inputs"]["iid_validation_predictions"] if name == "iid" else config["ood_validation_predictions"][name]
    prediction = pd.read_csv(resolve(pred_spec["path"]))
    labels = pd.read_parquet(resolve(config["inputs"]["validation_labels"]["path"]))
    ids = manifest.loc[manifest.partition.eq("val")].copy()
    frame = ids.merge(prediction, on="molecule_id", validate="one_to_one").merge(labels, on="molecule_id", validate="one_to_one")
    if len(frame) != len(ids) or not np.isfinite(frame[["prediction", TARGET]].to_numpy()).all(): raise RuntimeError(f"validation join failed: {name}")
    return frame.rename(columns={TARGET: "truth"}).sort_values("molecule_id", kind="mergesort").reset_index(drop=True)


def load_test(config: dict, name: str, manifest: pd.DataFrame) -> pd.DataFrame:
    if name == "iid":
        frame = pd.read_csv(resolve(config["inputs"]["iid_test_predictions"]["path"])).rename(columns={"xgb_c0_seed42": "prediction", "primary_true": "truth"})
    else:
        frame = pd.read_csv(resolve(config["inputs"]["ood_test_predictions"]["path"]))
        frame = frame.loc[frame.split_name.eq(name)].rename(columns={"xgb_c0": "prediction", "primary_true": "truth"})
    identity = manifest.loc[manifest.partition.eq("test"), ["molecule_id", "partition", "structure_group_id_v1", "donor_structure_group_id_v1", "acceptor_structure_group_id_v1", "pair_group_id_v1", "full_scaffold_group_id_v1"]]
    duplicate = [c for c in identity.columns if c != "molecule_id" and c in frame.columns]
    frame = frame.drop(columns=duplicate).merge(identity, on="molecule_id", validate="one_to_one")
    if len(frame) != len(identity) or not frame.partition.eq("test").all(): raise RuntimeError(f"test artifact mismatch: {name}")
    return frame.sort_values("molecule_id", kind="mergesort").reset_index(drop=True)


def finite_quantile(scores: np.ndarray, nominal: float) -> dict:
    scores = np.sort(np.asarray(scores, dtype=np.float64)); n = len(scores)
    rank = int(math.ceil((n + 1) * nominal))
    if rank > n: return {"status": "UNATTAINABLE_FINITE_SAMPLE", "n": n, "rank": rank, "max_attainable": n / (n + 1), "q": None}
    return {"status": "ATTAINABLE", "n": n, "rank": rank, "max_attainable": n / (n + 1), "q": float(scores[rank - 1])}


def score_sets(frame: pd.DataFrame, cluster: str) -> dict[str, np.ndarray]:
    work = frame.assign(abs_residual=(frame.prediction - frame.truth).abs())
    out = {"record": work.abs_residual.to_numpy(), "structure": work.groupby("structure_group_id_v1", sort=True).abs_residual.max().to_numpy()}
    if cluster != "two_way_donor_acceptor": out["identity"] = work.groupby(cluster, sort=True).abs_residual.max().to_numpy()
    else:
        out["donor_identity_sensitivity"] = work.groupby("donor_structure_group_id_v1", sort=True).abs_residual.max().to_numpy()
        out["acceptor_identity_sensitivity"] = work.groupby("acceptor_structure_group_id_v1", sort=True).abs_residual.max().to_numpy()
    return out


def mol(smiles: str):
    item = Chem.MolFromSmiles(smiles)
    if item is not None: return item
    item = Chem.MolFromSmiles(smiles, sanitize=False)
    if item is None: return None
    ops = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
    return item if Chem.SanitizeMol(item, sanitizeOps=ops, catchErrors=True) == Chem.SanitizeFlags.SANITIZE_NONE else None


def build_similarity(config: dict, all_manifests: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """Build target-free validation and IID-test nearest-train similarities."""
    RDLogger.DisableLog("rdApp.*")
    struct = pd.read_parquet(resolve(config["inputs"]["structure_registry"]["path"]))[["molecule_id", "canonical_structure_smiles_v1"]]
    comp = pd.read_csv(resolve(config["inputs"]["component_registry"]["path"]))[["molecule_id", "donor_canonical_structure_smiles_v1", "acceptor_canonical_structure_smiles_v1"]]
    strings = struct.merge(comp, on="molecule_id", validate="one_to_one").set_index("molecule_id")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=config["fingerprint"]["radius"], fpSize=config["fingerprint"]["n_bits"], includeChirality=config["fingerprint"]["use_chirality"])
    cache = {}
    def fp(text):
        if text not in cache:
            item = mol(str(text)); cache[text] = None if item is None else gen.GetFingerprint(item)
        return cache[text]
    def nearest(query_ids, train_ids, column):
        train_text = strings.loc[sorted(set(train_ids)), column].dropna().astype(str).unique().tolist()
        train_fp = [fp(x) for x in train_text]; train_fp = [x for x in train_fp if x is not None]
        if not train_fp: raise RuntimeError(f"no train fingerprints for {column}")
        vals = []
        for mid in query_ids:
            q = fp(strings.at[mid, column]); vals.append(np.nan if q is None else max(DataStructs.BulkTanimotoSimilarity(q, train_fp)))
        return vals
    rows=[]; failures={}
    for name in PROTOCOLS:
        man=all_manifests[name]; train=man.loc[man.partition.eq("train"),"molecule_id"].tolist()
        query=man.loc[man.partition.eq("val"),"molecule_id"].tolist()
        if name=="iid": query += man.loc[man.partition.eq("test"),"molecule_id"].tolist()
        block=pd.DataFrame({"protocol":name,"molecule_id":query,"partition":["val"]*int(man.partition.eq("val").sum()) + (["test"]*int(man.partition.eq("test").sum()) if name=="iid" else [])})
        for short,col in (("full","canonical_structure_smiles_v1"),("donor","donor_canonical_structure_smiles_v1"),("acceptor","acceptor_canonical_structure_smiles_v1")):
            block[short]=nearest(query,train,col)
        failures[name]=int(block[["full","donor","acceptor"]].isna().any(axis=1).sum()); rows.append(block)
    result=pd.concat(rows,ignore_index=True).sort_values(["protocol","partition","molecule_id"],kind="mergesort")
    if result[["full","donor","acceptor"]].isna().any().any(): raise RuntimeError(f"fingerprint parse failures: {failures}")
    return result, {"rows":len(result),"parse_failures":failures,"fingerprint":config["fingerprint"]}


def attach_similarity(config: dict, name: str, partition: str, frame: pd.DataFrame, local: pd.DataFrame) -> pd.DataFrame:
    if partition == "test" and name != "iid":
        source=pd.read_parquet(resolve(config["inputs"]["ood_test_similarity"]["path"]))
        source=source.loc[source.split_name.eq(name)].rename(columns={"nearest_train_full_morgan2048_chiral":"full","nearest_train_donor_morgan2048_chiral":"donor","nearest_train_acceptor_morgan2048_chiral":"acceptor"})
    else: source=local.loc[local.protocol.eq(name)&local.partition.eq(partition)]
    out=frame.merge(source[["molecule_id","full","donor","acceptor"]],on="molecule_id",validate="one_to_one")
    ad=config["protocols"][name]["ad"]
    out["ad_score"] = np.minimum(out.donor,out.acceptor) if ad=="min_donor_acceptor" else out[ad]
    return out


def cluster_column(config: dict, name: str) -> str:
    return config["protocols"][name]["cluster"]


def bootstrap_mean(values: np.ndarray, reps: int, seed: int) -> list[float]:
    values=np.sort(np.asarray(values,dtype=float)); rng=np.random.default_rng(seed); n=len(values); draws=np.empty(reps)
    for i in range(reps): draws[i]=values[rng.integers(0,n,n)].mean()
    return np.quantile(draws,[.025,.975]).astype(float).tolist()


def two_way_bootstrap(frame: pd.DataFrame, values: np.ndarray, reps: int, seed: int) -> list[float]:
    work=frame.assign(_value=np.asarray(values)).sort_values("molecule_id",kind="mergesort")
    ds=sorted(work.donor_structure_group_id_v1.unique()); ac=sorted(work.acceptor_structure_group_id_v1.unique())
    di=pd.Categorical(work.donor_structure_group_id_v1,categories=ds).codes; ai=pd.Categorical(work.acceptor_structure_group_id_v1,categories=ac).codes
    vals=work._value.to_numpy(float); rng=np.random.default_rng(seed); out=np.empty(reps)
    for i in range(reps):
        total=0
        while total==0:
            dm=np.bincount(rng.integers(0,len(ds),len(ds)),minlength=len(ds)); am=np.bincount(rng.integers(0,len(ac),len(ac)),minlength=len(ac))
            w=dm[di]*am[ai]; total=w.sum()
        out[i]=np.sum(w*vals)/total
    return np.quantile(out,[.025,.975]).astype(float).tolist()


def coverage_summary(frame: pd.DataFrame, q: float, cluster: str, reps: int, seed: int) -> dict:
    work=frame.copy(); work["covered"]=(work.prediction-work.truth).abs()<=q
    structure=work.groupby("structure_group_id_v1",sort=True).covered.mean()
    structure_sim=work.groupby("structure_group_id_v1",sort=True).covered.all()
    base={"record_marginal":float(work.covered.mean()),"structure_group_macro":float(structure.mean()),"structure_simultaneous":float(structure_sim.mean()),"records":len(work),"structures":len(structure)}
    if cluster=="two_way_donor_acceptor":
        donor=work.groupby("donor_structure_group_id_v1",sort=True).covered.mean(); accept=work.groupby("acceptor_structure_group_id_v1",sort=True).covered.mean()
        base.update({"donor_identity_macro":float(donor.mean()),"acceptor_identity_macro":float(accept.mean()),"donor_clusters":len(donor),"acceptor_clusters":len(accept),
                     "two_way_record_coverage_ci95":two_way_bootstrap(work,work.covered.to_numpy(float),reps,seed),
                     "donor_only_ci95":bootstrap_mean(donor.to_numpy(),reps,seed+1),"acceptor_only_ci95":bootstrap_mean(accept.to_numpy(),reps,seed+2),
                     "worst_decile_identity_coverage":float(min(donor.quantile(.1),accept.quantile(.1)))})
    else:
        identity=work.groupby(cluster,sort=True).covered.mean(); simultaneous=work.groupby(cluster,sort=True).covered.all()
        base.update({"identity_macro":float(identity.mean()),"cluster_simultaneous":float(simultaneous.mean()),"identity_clusters":len(identity),
                     "identity_macro_ci95":bootstrap_mean(identity.to_numpy(),reps,seed),"worst_decile_identity_coverage":float(identity.quantile(.1)),
                     "coverage_by_cluster_size_spearman":safe_float(spearmanr(work.groupby(cluster).size(),identity).statistic)})
    return base


def risk(frame: pd.DataFrame, cluster: str) -> dict:
    work=frame.assign(abs_error=(frame.prediction-frame.truth).abs())
    record=float(work.abs_error.mean())
    if cluster=="two_way_donor_acceptor":
        identity=float((work.groupby("donor_structure_group_id_v1").abs_error.mean().mean()+work.groupby("acceptor_structure_group_id_v1").abs_error.mean().mean())/2)
    else: identity=float(work.groupby(cluster).abs_error.mean().mean())
    return {"record_mae":record,"identity_macro_mae":identity}


def selective_metrics(frame: pd.DataFrame, cluster: str, thresholds: dict[str,str|float], high_error: float) -> dict:
    out={"fixed_validation_thresholds":{},"test_diagnostic_curve":{},"high_error_threshold":high_error}
    ordered=frame.sort_values(["ad_score","molecule_id"],ascending=[False,True],kind="mergesort").reset_index(drop=True)
    errors=(ordered.prediction-ordered.truth).abs().to_numpy(); cumulative=np.cumsum(errors)/np.arange(1,len(errors)+1)
    out["aurc_record"] = float(cumulative.mean())
    for key,threshold in thresholds.items():
        selected=frame if threshold is None else frame.loc[frame.ad_score>=float(threshold)]
        out["fixed_validation_thresholds"][key]={"threshold":threshold,"retained_records":len(selected),"retained_fraction":len(selected)/len(frame),**risk(selected,cluster)} if len(selected) else {"threshold":threshold,"retained_records":0}
    for fraction in (1,.9,.8,.7,.5):
        n=max(1,int(math.ceil(len(ordered)*fraction))); out["test_diagnostic_curve"][str(fraction)]={"retained_records":n,**risk(ordered.iloc[:n],cluster)}
    abs_error=(frame.prediction-frame.truth).abs(); rho=safe_float(spearmanr(frame.ad_score,abs_error).statistic)
    positive=abs_error>high_error
    try:
        from sklearn.metrics import roc_auc_score
        auc=float(roc_auc_score(positive.astype(int),-frame.ad_score)) if positive.nunique()>1 else None
    except Exception: auc=None
    out.update({"ad_score_vs_absolute_error_spearman":rho,"high_error_detection_roc_auc":auc,"high_error_fraction":float(positive.mean())})
    return out
