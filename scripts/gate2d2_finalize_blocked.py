#!/usr/bin/env python3
from __future__ import annotations
import hashlib
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
STATUS="BLOCKED_PREREGISTERED_PCA_INFEASIBLE"
def append_once(path,marker,text):
 current=path.read_text()
 if marker not in current: path.write_text(current.rstrip()+"\n\n"+text.strip()+"\n")
def sha(path):
 h=hashlib.sha256()
 with path.open("rb") as f:
  for b in iter(lambda:f.read(1<<20),b""): h.update(b)
 return h.hexdigest()
append_once(ROOT/"PROJECT_STATE.md","## Gate 2-D2 result","""## Gate 2-D2 result

- MoLFormer asset security and tokenizer gates passed at immutable revision `a14249e5ad9e3e7c3b1bb604393e914cfcebd2c8`; license is Apache-2.0 and safetensors is available.
- Full/donor/acceptor tokenizer audit covered 14,639 / 154 / 352 unique strings with zero unknown tokens, reconstruction failures, or truncations at the preregistered max length 512.
- Gate 2-D2 v1 stopped before executing remote code or extracting embeddings: the frozen Arm C requests 256 donor PCs from unique protocol-train donors, but only 124-154 unique donors exist (maximum centered PCA ranks 123-153).
- No PCA, XGBoost model, validation prediction, test artifact, main Parquet, or final673 asset was accessed. This is a protocol blocker, not a negative MoLFormer result.
- Final status: `BLOCKED_PREREGISTERED_PCA_INFEASIBLE`. Any retry requires an explicit v2 compression contract; v1 cannot be silently edited.""")
append_once(ROOT/"TODO.md","Gate 2-D2: frozen continuous representation v1","""- [!] Gate 2-D2: frozen continuous representation v1 is `BLOCKED_PREREGISTERED_PCA_INFEASIBLE`; donor PCA-256 cannot be fitted on 124-154 unique protocol-train donors. No embedding or validation model was run.""")
append_once(ROOT/"DECISIONS.md","## Gate 2-D2 decisions","""## Gate 2-D2 decisions

- Lock MoLFormer to revision `a14249e5ad9e3e7c3b1bb604393e914cfcebd2c8`, Apache-2.0, safetensors, audited custom code, deterministic eval, mask-aware final-hidden-state mean pooling, and no truncation at max length 512.
- Preserve the documented pretraining domain mismatch for inputs above 202 tokens; do not silently truncate or change molecular/component strings.
- Freeze Gate 2-D2 v1 as `BLOCKED_PREREGISTERED_PCA_INFEASIBLE`: unique/equal-weight donor PCA-256 exceeds every protocol's identifiable rank.
- Do not manufacture PCA dimensions by duplicating component rows, zero-padding non-identifiable components, using held-out structures, changing the 256/256 allocation after lock, or substituting another encoder.
- Do not interpret the blocker as evidence against continuous representations. A future v2 requires a new preregistration and a mathematically feasible target-free compression rule; no v2 is authorized by this Gate.""")
rr=ROOT/"RUN_REGISTRY.csv"; marker="gate2d2-20260720"
if marker not in rr.read_text():
 end=datetime.now(timezone.utc).isoformat()
 with rr.open("a") as f: f.write('gate2d2-20260720,12,frozen_continuous_representation_admission,BLOCKED_PREREGISTERED_PCA_INFEASIBLE,2026-07-20T15:28:00Z,'+end+',NONE,"asset/code/tokenizer audit + preregistration + PCA feasibility gate","model/tokenizer passed; donor PCA-256 infeasible at 124-154 unique train donors; no embeddings/models",logs/gate2d2_evidence.json\n')
files=[
"configs/gate2d2_frozen_molformer_admission_v1.json",
"data_registry/gate2d2_model_asset_lock_v1.json",
"data_registry/gate2d2_preregistration_lock_v1.json",
"data_registry/gate2d2_embedding_registry.json",
"data_registry/gate2d2_pca_registry.json",
"data_registry/gate2d2_model_registry.json",
"scripts/gate2d2_audit_model_asset.py",
"scripts/gate2d2_freeze_preregistration.py",
"scripts/gate2d2_extract_frozen_embeddings.py",
"scripts/gate2d2_build_protocol_features.py",
"scripts/gate2d2_train_validation_only.py",
"scripts/gate2d2_analyze_validation.py",
"scripts/gate2d2_finalize_blocked.py",
"scripts/git_checkpoint.sh",
"tests/test_gate2d2_contract.py",
"reports/gate2d2_model_security_and_tokenizer_audit.md",
"reports/gate2d2_preregistration.md",
"reports/gate2d2_embedding_integrity.md",
"reports/gate2d2_validation_results.md",
"reports/gate2d2_acceptor_mechanism.md",
"reports/gate2d2_final_decision.md",
"logs/gate2d2_embedding_audit.json",
"logs/gate2d2_evidence.json",
".gitignore",
"PROJECT_STATE.md","TODO.md","DECISIONS.md","RUN_REGISTRY.csv"]
out=[]
for name in files:
 p=ROOT/name
 if not p.is_file(): raise RuntimeError("missing Gate 2-D2 asset: "+name)
 out.append(sha(p)+"  "+name)
(ROOT/"data_registry/gate2d2_sha256.txt").write_text("\n".join(out)+"\n")
print(STATUS)
