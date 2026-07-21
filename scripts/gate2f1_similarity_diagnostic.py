#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem,DataStructs
from rdkit.Chem import rdFingerprintGenerator
ROOT=Path(__file__).resolve().parents[1]
COMP=ROOT/'manifests/component_identity_v1.csv';COMP_SHA='dca4a5e5661d7336226b16c87624ba3b457cdcd0379ccb2130a48d4f30306515'
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def writej(p,v):Path(p).write_text(json.dumps(v,indent=2,sort_keys=True)+'\n')
def parse(s):
 m=Chem.MolFromSmiles(str(s),sanitize=False)
 if m is None:raise RuntimeError('component parse failed')
 Chem.SanitizeMol(m,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE);return m
def main():
 if sha(COMP)!=COMP_SHA:raise RuntimeError('component source hash mismatch')
 lock={'version':'gate2f1_similarity_diagnostic_lock_v1','timing':'after model freeze and before target-free similarity computation','scope':'user-preregistered secondary lowest-acceptor-similarity-quartile diagnostic','model_or_primary_decision_change':False,'component_identity_sha256':COMP_SHA,'fingerprint':{'radius':2,'n_bits':2048,'use_chirality':True,'metric':'nearest_outer_train_tanimoto'},'test_access':False};writej(ROOT/'data_registry/gate2f1_similarity_diagnostic_lock_v1.json',lock)
 fold=pd.read_parquet(ROOT/'runs/gate2e2a_multitask_crossfit/folds/acceptor_cold_outer.parquet');comp=pd.read_csv(COMP,usecols=['molecule_id','acceptor_structure_group_id_v1','acceptor_canonical_structure_smiles_v1']);z=fold.merge(comp,on='molecule_id',validate='one_to_one')
 gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=2048,includeChirality=True);u=comp[['acceptor_structure_group_id_v1','acceptor_canonical_structure_smiles_v1']].drop_duplicates();fps={r.acceptor_structure_group_id_v1:gen.GetFingerprint(parse(r.acceptor_canonical_structure_smiles_v1)) for r in u.itertuples(index=False)};scores=[]
 for f in range(5):
  tr=sorted(z.loc[~z.outer_fold.eq(f),'acceptor_structure_group_id_v1'].unique());te=sorted(z.loc[z.outer_fold.eq(f),'acceptor_structure_group_id_v1'].unique());tf=[fps[x] for x in tr]
  scores.extend((x,float(max(DataStructs.BulkTanimotoSimilarity(fps[x],tf)))) for x in te)
 sim=pd.DataFrame(scores,columns=['acceptor_structure_group_id_v1','nearest_outer_train_acceptor_similarity']);sim=sim.drop_duplicates();root=ROOT/'runs/gate2f1_multifidelity_delta_crossfit';base=None
 for arm in ['C0','PM6_3','DFT_3','PAIR_6','DELTA_6','DELTA_3','ROLE_9']:
  q=pd.read_parquet(root/'acceptor_cold'/f'{arm}_oof.parquet').rename(columns={'prediction':arm});base=q if base is None else base.merge(q[['molecule_id',arm]],on='molecule_id',validate='one_to_one')
 base=base.merge(sim,on='acceptor_structure_group_id_v1',validate='many_to_one');base['w']=base.group_weight
 rows=[]
 for aid,g in base.groupby('acceptor_structure_group_id_v1',sort=True):
  r={'acceptor_hash':hashlib.sha256(str(aid).encode()).hexdigest(),'similarity':float(g.nearest_outer_train_acceptor_similarity.iloc[0]),'records':len(g)}
  for arm in ['C0','PM6_3','DFT_3','PAIR_6','DELTA_6','DELTA_3','ROLE_9']:r[f'{arm}_mae']=float(np.sum(np.abs(g[arm]-g.y)*g.w)/g.w.sum())
  rows.append(r)
 out=pd.DataFrame(rows).sort_values('acceptor_hash');q=float(out.similarity.quantile(.25));low=out[out.similarity<=q];summary={'identity_count':len(out),'lowest_quartile_threshold':q,'lowest_quartile_identities':len(low),'all_identity_macro':{a:float(out[f'{a}_mae'].mean()) for a in ['C0','PM6_3','DFT_3','PAIR_6','DELTA_6','DELTA_3','ROLE_9']},'lowest_quartile_identity_macro':{a:float(low[f'{a}_mae'].mean()) for a in ['C0','PM6_3','DFT_3','PAIR_6','DELTA_6','DELTA_3','ROLE_9']},'lowest_quartile_pair_minus_c0':float((low.PAIR_6_mae-low.C0_mae).mean()),'lowest_quartile_delta_minus_pair':float((low.DELTA_6_mae-low.PAIR_6_mae).mean()),'selection_or_model_change':False,'test_access':False};out.to_parquet(root/'acceptor_identity_effects_v1.parquet',index=False);summary['local_identity_effects_sha256']=sha(root/'acceptor_identity_effects_v1.parquet');writej(ROOT/'data_registry/gate2f1_similarity_diagnostic_registry.json',summary)
 (ROOT/'reports/gate2f1_acceptor_identity_analysis.md').write_text(f"# Gate 2-F1 acceptor identity analysis\n\nThe primary protocol contains {len(out)} held-out training-only acceptor identities. Paired primary inference uses 10,000 identity-cluster bootstrap replicates. The target-free lowest-similarity quartile contains {len(low)} identities at a nearest-outer-train Tanimoto threshold of {q:.6f}; PAIR-6−C0 there is {summary['lowest_quartile_pair_minus_c0']:+.9f} eV and DELTA-6−PAIR-6 is {summary['lowest_quartile_delta_minus_pair']:+.9f} eV. This secondary subgroup was user-preregistered, did not select an arm, and does not alter the primary decision. Per-identity effects remain local and are represented publicly only by aggregate statistics and a file hash.\n")
 print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
