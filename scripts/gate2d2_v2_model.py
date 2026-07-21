#!/usr/bin/env python3
from __future__ import annotations
import hashlib,importlib.util,json,os,sys
from pathlib import Path
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
MODEL_DIR=ROOT/'runs/gate2d2_frozen_molformer/model_asset_audit'
VENDOR=ROOT/'runs/gate2d2_frozen_molformer/runtime_vendor'
CONFIG=ROOT/'configs/gate2d2_frozen_molformer_admission_v2.json'
LOCK=ROOT/'data_registry/gate2d2_v2_preregistration_lock.json'

def sha(path):
 h=hashlib.sha256()
 with path.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): h.update(b)
 return h.hexdigest()
def anon(text): return hashlib.sha256(text.encode()).hexdigest()
def verify():
 c=json.loads(CONFIG.read_text()); lock=json.loads(LOCK.read_text())
 if sha(CONFIG)!=lock['config_sha256']: raise RuntimeError('v2 preregistration config changed')
 weight=MODEL_DIR/c['model']['weights_file']
 if not weight.is_file() or sha(weight)!=c['model']['weights_sha256']: raise RuntimeError('safetensors hash mismatch')
 if c['model']['revision']!='a14249e5ad9e3e7c3b1bb604393e914cfcebd2c8': raise RuntimeError('revision mismatch')
 return c
def runtime_imports():
 sys.path.insert(0,str(VENDOR))
 original_find_spec=importlib.util.find_spec
 def find_spec_without_optional_multimedia(name,*args,**kwargs):
  if name in ('torchvision','torchaudio') or name.startswith(('torchvision.','torchaudio.')):
   return None
  return original_find_spec(name,*args,**kwargs)
 importlib.util.find_spec=find_spec_without_optional_multimedia
 try:
  from transformers import AutoModel,AutoTokenizer
 finally:
  importlib.util.find_spec=original_find_spec
 return AutoModel,AutoTokenizer
def load_model(device):
 c=verify(); AutoModel,AutoTokenizer=runtime_imports()
 os.environ['HF_HUB_OFFLINE']='1'; os.environ['TRANSFORMERS_OFFLINE']='1'; os.environ['HF_HUB_DISABLE_TELEMETRY']='1'
 tokenizer=AutoTokenizer.from_pretrained(str(MODEL_DIR),trust_remote_code=True,local_files_only=True)
 model=AutoModel.from_pretrained(str(MODEL_DIR),trust_remote_code=True,local_files_only=True,use_safetensors=True,deterministic_eval=True)
 if model.config.hidden_size!=c['encoder']['hidden_dimension']: raise RuntimeError('BLOCKED_ENCODER_DIMENSION_MISMATCH')
 model.eval().to(device)
 for p in model.parameters(): p.requires_grad_(False)
 if any(p.requires_grad for p in model.parameters()): raise RuntimeError('encoder parameter still trainable')
 return c,tokenizer,model
def pooled(model,batch):
 with torch.inference_mode(): out=model(**{k:v for k,v in batch.items() if k in ('input_ids','attention_mask')})
 hidden=out.last_hidden_state; mask=batch['attention_mask'].unsqueeze(-1).to(hidden.dtype)
 manual=(hidden*mask).sum(1)/mask.sum(1).clamp_min(1)
 if out.pooler_output is not None and not torch.allclose(manual,out.pooler_output,atol=1e-6,rtol=1e-6): raise RuntimeError('pooler mismatch')
 return manual
def unique_inputs():
 s=pd.read_parquet(ROOT/'manifests/new15016_structure_groups_v1.parquet',columns=['canonical_structure_smiles_v1'])
 c=pd.read_csv(ROOT/'manifests/component_identity_v1.csv',usecols=['donor_canonical_structure_smiles_v1','acceptor_canonical_structure_smiles_v1'])
 return {'full':sorted(s.iloc[:,0].astype(str).unique()),'donor':sorted(c.iloc[:,0].astype(str).unique()),'acceptor':sorted(c.iloc[:,1].astype(str).unique())}
