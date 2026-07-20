#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
def write(path,value):
 path.parent.mkdir(parents=True,exist_ok=True); path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+"\n")
def main():
 cp=ROOT/"configs/gate2d2_frozen_molformer_admission_v1.json"; ap=ROOT/"logs/gate2d2_embedding_audit.json"
 c=json.loads(cp.read_text()); a=json.loads(ap.read_text())
 if a["status"]!="MODEL_ASSET_AND_TOKENIZER_ADMITTED": raise RuntimeError("tokenizer gate failed")
 for item in c["inputs"].values():
  p=ROOT/item["path"]
  if sha(p)!=item["sha256"]: raise RuntimeError("input hash mismatch: "+item["path"])
 hub={
 ".gitattributes":{"bytes":1519,"git_blob":"a6344aac8c09253b3b630fb776ae94478aa0275b"},
 "README.md":{"bytes":6279,"git_blob":"274e4069755c7eed9c3cda0a8fcfd6aa9415c416"},
 "config.json":{"bytes":1015,"git_blob":"f24331e0bb3a00ca7557226d3dddc61c78b691f8"},
 "configuration_molformer.py":{"bytes":7101,"git_blob":"9a0824bbc564297a7a8320626790eae2ca8c1ac8"},
 "convert_molformer_original_checkpoint_to_pytorch.py":{"bytes":3545,"git_blob":"30159e4eaab336ab8bce3e17bb3caaddd590d846"},
 "model.safetensors":{"bytes":187248784,"lfs_sha256":"0795977fe7192c4acdaf052f0e8464af57bc4bb59211271c5e61aaba2637b9c6"},
 "modeling_molformer.py":{"bytes":36884,"git_blob":"fe26e4af404657b8bb2951d13797f71ad9284501"},
 "pipeline.jpeg":{"bytes":241085,"git_blob":"e97233825700c8baea008c2ad8969c86b4626de0"},
 "pytorch_model.bin":{"bytes":187391645,"lfs_sha256":"93e3fced64b896fcfea4934505ac80275db7afb7320d0b32ee0c691d99ab8678"},
 "special_tokens_map.json":{"bytes":125,"git_blob":"43246c556f7d57aa9db7a8dbb2bc0530d25eb2e3"},
 "tokenization_molformer.py":{"bytes":9480,"git_blob":"9a2f2ca032372baa7f439f60d245b7ca66069357"},
 "tokenization_molformer_fast.py":{"bytes":6503,"git_blob":"a58a743c6ee41fc7ae4038a290d25e68b66d4cfd"},
 "tokenizer.json":{"bytes":54010,"git_blob":"48105a2ff71e781579b2034852086db3e8b01906"},
 "tokenizer_config.json":{"bytes":1294,"git_blob":"ba1a41edace11006ef54ec8b274b43645936ebb5"},
 "vocab.json":{"bytes":41641,"git_blob":"09c8af343ba46874dea2d3aed48dc26409b87645"}}
 asset={"status":"GATE2D2_MODEL_ASSET_LOCKED","repo_id":c["model"]["repo_id"],"revision":c["model"]["revision"],"license":c["model"]["license"],"hub_client":"huggingface_hub 1.2.3 API fallback; hf CLI unavailable","official_model_card_tested_transformers":"5.12.1","runtime_code_audited_before_execution":True,"runtime_code_finding":a["custom_code_audit"],"preferred_weights":"model.safetensors","hub_files":hub,"local_audit_files":a["source_files"],"audit_sha256":sha(ap)}
 assetp=ROOT/"data_registry/gate2d2_model_asset_lock_v1.json"; write(assetp,asset)
 lock={"status":"GATE2D2_PREREGISTERED_BEFORE_EMBEDDINGS_OR_REGRESSION","created_utc":datetime.now(timezone.utc).isoformat(),"config_path":str(cp.relative_to(ROOT)),"config_sha256":sha(cp),"model_asset_lock_path":str(assetp.relative_to(ROOT)),"model_asset_lock_sha256":sha(assetp),"tokenizer_audit_path":str(ap.relative_to(ROOT)),"tokenizer_audit_sha256":sha(ap),"inputs":c["inputs"],"model_revision":c["model"]["revision"],"arms":c["feature_arms"],"admission":c["admission"],"test_firewall":{"test_artifact_access":False,"main_parquet_access":False,"final673_access":False}}
 lockp=ROOT/"data_registry/gate2d2_preregistration_lock_v1.json"; write(lockp,lock)
 t=a["tokenizer_audit"]; rows="".join("| "+n+" | "+str(x["unique_inputs"])+" | "+str(x["max_length"])+" | "+str(x["over_max_sequence_length"])+" | "+str(x["unknown_token_sequences"])+" | "+str(x["parse_reconstruction_failures"])+" |\n" for n,x in t.items())
 (ROOT/"reports/gate2d2_model_security_and_tokenizer_audit.md").write_text("# Gate 2-D2 model security and tokenizer audit\n\nStatus: **MODEL_ASSET_AND_TOKENIZER_ADMITTED**.\n\n- Repository: "+asset["repo_id"]+" at immutable revision "+asset["revision"]+".\n- License: Apache-2.0; safetensors is available and is the only admitted weight format.\n- Runtime custom code was read before execution. No network, subprocess, shell, dynamic eval/exec, or runtime file-write behavior was found. The conversion utility is excluded because it uses torch.load/torch.save.\n- Repository config has deterministic_eval=false; runtime must override it to true.\n- Pooling: attention-mask-aware mean of final hidden state.\n- Max length=512, truncation disabled. Upstream pretraining dropped >202-token molecules, so those inputs are a documented domain-mismatch limitation.\n\n| input | unique | max tokens | >512 | unknown | reconstruction failures |\n|---|---:|---:|---:|---:|---:|\n"+rows+"\nWildcards are preserved; no string is changed. No target, test artifact, main Parquet, or final673 asset was accessed.\n")
 (ROOT/"reports/gate2d2_preregistration.md").write_text("# Gate 2-D2 preregistration\n\nStatus: **GATE2D2_PREREGISTERED_BEFORE_EMBEDDINGS_OR_REGRESSION**.\n\nThe immutable encoder revision, audited code, fixed pooling/max length, three 532-column arms, protocol-local train-only PCA, fixed XGBoost, six validation-only protocols, bootstrap and admission thresholds are locked. No test artifact, main Parquet, final673, foundation-model gradient, or post-hoc encoder substitution is permitted.\n")
 print(json.dumps({"status":lock["status"],"config_sha256":lock["config_sha256"],"asset_lock_sha256":lock["model_asset_lock_sha256"]},indent=2))
if __name__=="__main__": main()
