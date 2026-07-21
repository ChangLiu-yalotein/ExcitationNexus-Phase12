from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
def h(x): return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()

def test_v1_is_preserved_and_v2_changes_only_compressor():
 c=json.loads((ROOT/'configs/gate2d2_frozen_molformer_admission_v2.json').read_text())
 assert c['amends']=='gate2d2_frozen_molformer_admission_v1'
 assert c['v1_frozen']['status']=='BLOCKED_PREREGISTERED_PCA_INFEASIBLE'
 assert not c['v1_frozen']['embeddings_created'] and not c['v1_frozen']['models_created']
 assert 'PCA' in c['forbidden'] and c['model']['revision']=='a14249e5ad9e3e7c3b1bb604393e914cfcebd2c8'
 assert {x['columns'] for x in c['feature_arms'].values()}=={532}
 assert c['admission']['acceptor_C_minus_A_max_eV']==-0.003

def test_fixed_projection_shapes_hashes_and_shared_component_matrix():
 reg=json.loads((ROOT/'data_registry/gate2d2_v2_projection_registry.json').read_text())
 z=np.load(ROOT/reg['npz_path'],allow_pickle=False)
 assert z['R_full'].shape==(768,512) and z['R_component'].shape==(768,256)
 assert z['R_full'].dtype==np.float32 and z['R_component'].dtype==np.float32
 assert h(z['R_full'])==reg['R_full']['content_sha256']
 assert h(z['R_component'])==reg['R_component']['content_sha256']
 assert reg['R_component']['shared_by']==['donor','acceptor']

def test_projection_is_byte_reproducible():
 z=np.load(ROOT/'data_registry/gate2d2_v2_fixed_projection_matrices.npz',allow_pickle=False)
 rf=np.random.Generator(np.random.PCG64(20260720)).normal(0,1/np.sqrt(512),(768,512)).astype(np.float32)
 rc=np.random.Generator(np.random.PCG64(20260721)).normal(0,1/np.sqrt(256),(768,256)).astype(np.float32)
 assert np.array_equal(rf,z['R_full']) and np.array_equal(rc,z['R_component'])

def test_v2_firewall_and_long_sequence_contract():
 c=json.loads((ROOT/'configs/gate2d2_frozen_molformer_admission_v2.json').read_text())
 assert not c['main_parquet_access'] and not c['test_artifact_access'] and not c['final673_access']
 assert not c['encoder']['requires_grad'] and c['encoder']['optimizer_parameter_count']==0
 assert c['long_sequence_gate']['required_maxima']=={'full':399,'donor':208,'acceptor':372}
 assert c['long_sequence_gate']['repeat_max_abs_tolerance']==1e-6
 assert c['long_sequence_gate']['single_vs_padded_batch_max_abs_tolerance']==1e-5
