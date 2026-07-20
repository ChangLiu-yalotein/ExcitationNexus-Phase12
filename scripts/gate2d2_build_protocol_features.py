#!/usr/bin/env python3
"""Fail-closed PCA feasibility gate for the preregistered Gate 2-D2 feature arms."""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
PROTOCOLS=("iid","donor_cold","acceptor_cold","pair_cold","both_cold","full_scaffold_cold")
STATUS="BLOCKED_PREREGISTERED_PCA_INFEASIBLE"
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def max_identifiable_pcs(n_unique): return max(int(n_unique)-1,0)
def main():
 cp=ROOT/"configs/gate2d2_frozen_molformer_admission_v1.json"; lp=ROOT/"data_registry/gate2d2_preregistration_lock_v1.json"
 c=json.loads(cp.read_text()); lock=json.loads(lp.read_text())
 if sha(cp)!=lock["config_sha256"]: raise RuntimeError("preregistration config changed")
 d1=json.loads((ROOT/"configs/gate2d1_role_aware_2d_v1.json").read_text())
 comp=pd.read_csv(ROOT/c["inputs"]["component_registry"]["path"],usecols=["molecule_id","donor_structure_group_id_v1","acceptor_structure_group_id_v1"])
 required_donor=c["feature_arms"]["C_MF_Role_512"]["donor_pca_dimensions"]; required_acceptor=c["feature_arms"]["C_MF_Role_512"]["acceptor_pca_dimensions"]
 protocols={}
 for name in PROTOCOLS:
  spec=d1["protocols"][name]; manifest=pd.read_csv(ROOT/spec["manifest"],usecols=["molecule_id","partition"])
  train=manifest.loc[manifest.partition.eq("train"),["molecule_id"]].merge(comp,on="molecule_id",validate="one_to_one")
  nd=int(train.donor_structure_group_id_v1.nunique()); na=int(train.acceptor_structure_group_id_v1.nunique())
  protocols[name]={"train_records":len(train),"unique_donor_structures":nd,"unique_acceptor_structures":na,"max_identifiable_donor_pcs":max_identifiable_pcs(nd),"max_identifiable_acceptor_pcs":max_identifiable_pcs(na),"requested_donor_pcs":required_donor,"requested_acceptor_pcs":required_acceptor,"donor_pca_feasible":max_identifiable_pcs(nd)>=required_donor,"acceptor_pca_feasible":max_identifiable_pcs(na)>=required_acceptor}
 if any(x["donor_pca_feasible"] for x in protocols.values()): raise RuntimeError("expected all protocol donor PCA arms to be infeasible")
 feasibility={"status":STATUS,"reason":"The preregistered unique-component/equal-weight donor PCA requests 256 components, but every protocol has fewer than 257 unique train donor structures.","mathematical_rule":"centered PCA identifiable rank <= n_unique_samples - 1","protocols":protocols,"silent_fixes_forbidden":["duplicate donor rows to satisfy sklearn shape","zero-pad non-identifiable principal components","fit PCA on held-out protocol components","change 256/256 allocation after preregistration","substitute another encoder"],"test_artifacts_accessed":False,"main_parquet_accessed":False,"final673_accessed":False}
 write(ROOT/"data_registry/gate2d2_pca_registry.json",feasibility)
 write(ROOT/"data_registry/gate2d2_embedding_registry.json",{"status":"NOT_EXTRACTED_DUE_TO_PCA_CONTRACT_BLOCKER","model_revision":c["model"]["revision"],"remote_code_executed":False,"valid_weights_downloaded":False,"raw_embeddings_created":False,"test_artifacts_accessed":False,"main_parquet_accessed":False,"final673_accessed":False})
 write(ROOT/"data_registry/gate2d2_model_registry.json",{"status":"NO_REGRESSION_MODELS_TRAINED_DUE_TO_PCA_CONTRACT_BLOCKER","new_models":0,"validation_predictions_created":0,"test_artifacts_accessed":False,"main_parquet_accessed":False,"final673_accessed":False})
 write(ROOT/"logs/gate2d2_evidence.json",{"status":STATUS,"git_start_head":"cc4d25167ad5303f0f371353df7d14dfad4f6011","model_asset_locked":True,"tokenizer_gate_passed":True,"preregistration_locked_before_embeddings":True,"pca_feasibility":feasibility,"remote_code_executed":False,"foundation_weights_loaded":False,"embedding_extraction_started":False,"pca_fitted":False,"regression_models_trained":0,"validation_metrics_computed":False,"test_artifacts_accessed":False,"main_parquet_accessed":False,"final673_accessed":False,"partial_weight_download_stopped_after_blocker":True})
 table="".join("| "+n+" | "+str(x["unique_donor_structures"])+" | "+str(x["max_identifiable_donor_pcs"])+" | "+str(x["unique_acceptor_structures"])+" | "+str(x["max_identifiable_acceptor_pcs"])+" |\n" for n,x in protocols.items())
 (ROOT/"reports/gate2d2_embedding_integrity.md").write_text("# Gate 2-D2 embedding integrity\n\nStatus: **NOT_EXTRACTED_DUE_TO_PCA_CONTRACT_BLOCKER**.\n\nThe pinned model and tokenizer passed security/integrity admission, but no remote code or weight was loaded and no embedding was extracted because the downstream preregistered PCA arm is mathematically infeasible. This preserves the rule that expensive assets are not executed after an earlier fail-closed gate.\n")
 (ROOT/"reports/gate2d2_validation_results.md").write_text("# Gate 2-D2 validation results\n\nStatus: **NO_VALIDATION_RUN**.\n\nNo PCA, XGBoost model, validation prediction, metric, bootstrap, or admission decision was produced. This is not a negative representation result.\n")
 (ROOT/"reports/gate2d2_acceptor_mechanism.md").write_text("# Gate 2-D2 acceptor mechanism\n\nStatus: **NOT_EVALUATED**.\n\nThe continuous-representation hypothesis remains untested. No acceptor-cold validation result or mechanism analysis exists for Gate 2-D2 v1.\n")
 (ROOT/"reports/gate2d2_final_decision.md").write_text("# Gate 2-D2 final decision\n\nGate status: **"+STATUS+"**.\n\nThe frozen Arm C requires donor PCA to 256 dimensions while fitting unique protocol-train donor structures with equal weight. Across protocols there are only 124-154 unique train donors, so centered PCA has at most 123-153 identifiable components.\n\n| protocol | unique donor | max donor PCs | unique acceptor | max acceptor PCs |\n|---|---:|---:|---:|---:|\n"+table+"\nNo result label such as FROZEN_CONTINUOUS_REPRESENTATION_NOT_ADMITTED is assigned, because the representation experiment was never run. Repetition, zero-padding, transductive PCA, or post-lock dimension changes would violate the preregistration. A v2 requires an explicit new compression contract; the cleanest candidate is a target-free fixed random projection with the original 512/256/256 output budgets, but it is not authorized here.\n\nNo test artifact, main Parquet, final673, remote model code, embedding, PCA, or regression model was accessed or executed.\n")
 print(json.dumps({"status":STATUS,"protocols":protocols},indent=2))
if __name__=="__main__": main()
