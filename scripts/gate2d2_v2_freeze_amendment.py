#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from datetime import datetime,timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
CONFIG=ROOT/'configs/gate2d2_frozen_molformer_admission_v2.json'
MATRIX=ROOT/'data_registry/gate2d2_v2_fixed_projection_matrices.npz'
REGISTRY=ROOT/'data_registry/gate2d2_v2_projection_registry.json'
LOCK=ROOT/'data_registry/gate2d2_v2_preregistration_lock.json'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def write(path,value): path.write_text(json.dumps(value,indent=2,sort_keys=True,allow_nan=False)+'\n')
def matrix_bytes(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def main():
 if MATRIX.exists() or REGISTRY.exists() or LOCK.exists(): raise RuntimeError('v2 projection/amendment already exists; refuse overwrite')
 c=json.loads(CONFIG.read_text())
 if np.__version__!=c['projection']['numpy_version']: raise RuntimeError('NumPy version mismatch')
 hidden=c['encoder']['hidden_dimension']
 rf=np.random.Generator(np.random.PCG64(c['projection']['R_full']['seed'])).normal(0.0,1.0/np.sqrt(512),(hidden,512)).astype(np.float32)
 rc=np.random.Generator(np.random.PCG64(c['projection']['R_component']['seed'])).normal(0.0,1.0/np.sqrt(256),(hidden,256)).astype(np.float32)
 np.savez(MATRIX,R_full=rf,R_component=rc)
 reg={'status':'GATE2D2_V2_FIXED_PROJECTIONS_FROZEN','generator':'PCG64','numpy_version':np.__version__,'generation_dtype':'float64','storage_dtype':'float32','distribution':'N(0,1/output_dimension)','R_full':{'shape':list(rf.shape),'seed':20260720,'content_sha256':matrix_bytes(rf)},'R_component':{'shape':list(rc.shape),'seed':20260721,'content_sha256':matrix_bytes(rc),'shared_by':['donor','acceptor']},'npz_path':str(MATRIX.relative_to(ROOT)),'npz_sha256':sha(MATRIX),'target_or_data_access':False}
 write(REGISTRY,reg)
 lock={'status':'GATE2D2_V2_PREREGISTERED_BEFORE_MODEL_FORWARD','created_utc':datetime.now(timezone.utc).isoformat(),'config_path':str(CONFIG.relative_to(ROOT)),'config_sha256':sha(CONFIG),'projection_registry_path':str(REGISTRY.relative_to(ROOT)),'projection_registry_sha256':sha(REGISTRY),'projection_matrix_path':str(MATRIX.relative_to(ROOT)),'projection_matrix_sha256':sha(MATRIX),'v1_status':'BLOCKED_PREREGISTERED_PCA_INFEASIBLE','v1_files_modified':False,'validation_performance_seen_before_method_choice':False,'admission':c['admission'],'test_firewall':{'main_parquet':False,'test_artifact':False,'final673':False}}
 write(LOCK,lock)
 (ROOT/'reports/gate2d2_v2_amendment.md').write_text('# Gate 2-D2 v2 amendment\n\nStatus: **GATE2D2_V2_PREREGISTERED_BEFORE_MODEL_FORWARD**.\n\nv1 remains permanently `BLOCKED_PREREGISTERED_PCA_INFEASIBLE`; it produced no embedding, model, or validation result. v2 changes only the mathematically infeasible sample-fitted PCA to fixed target-free Gaussian projections generated before any model forward or validation performance. Encoder, revision, tokenizer, pooling, feature budgets, XGBoost, protocols, bootstrap and admission thresholds are unchanged.\n')
 print(json.dumps({'status':lock['status'],'config_sha256':lock['config_sha256'],'matrix_sha256':reg['npz_sha256']},indent=2))
if __name__=='__main__': main()
