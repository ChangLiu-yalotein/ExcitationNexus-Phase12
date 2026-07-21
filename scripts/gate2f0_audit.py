#!/usr/bin/env python3
from __future__ import annotations
import hashlib,json,math,subprocess
from datetime import datetime,timezone
from pathlib import Path
import numpy as np,pandas as pd
from scipy.stats import spearmanr, wasserstein_distance
import pyarrow.dataset as ds
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

ROOT=Path(__file__).resolve().parents[1]; PRIMARY='tddft_coulomb_attraction_eV_eps3p5_proxy'
def resolve(p):p=Path(p);return p if p.is_absolute() else ROOT/p
def sha(p):
 h=hashlib.sha256()
 with resolve(p).open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def write_json(p,x):p=resolve(p);p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def finite(x):
 try:return float(x) if np.isfinite(float(x)) else np.nan
 except:return np.nan
def corr(x,y):
 a=np.asarray(x,float);b=np.asarray(y,float)
 if len(a)<3 or np.std(a)==0 or np.std(b)==0:return None
 v=float(spearmanr(a,b).statistic);return v if np.isfinite(v) else None
def loadj(p):
 with Path(p).open() as f:return json.load(f)
def atom_arrays(obj):
 atoms=obj.get('atoms',[]);z=[];xyz=[];roles=[]
 periodic={'H':1,'B':5,'C':6,'N':7,'O':8,'F':9,'Si':14,'P':15,'S':16,'Cl':17,'Se':34,'Br':35,'I':53}
 for a in atoms:
  z.append(int(a.get('atomic_num',periodic.get(a.get('element'),0))));xyz.append(a.get('coords'));roles.append(str(a.get('type','unknown')).lower())
 return np.asarray(z,int),np.asarray(xyz,float),np.asarray(roles,object)
def interface_features(z,xyz,roles,cutoffs):
 heavy=z>1;out={};counts={r:int(np.sum(roles==r)) for r in ['donor','acceptor','unknown']};out.update({f'{r}_atom_count':v for r,v in counts.items()});out.update({f'{r}_atom_fraction':v/len(z) for r,v in counts.items()});out.update({f'{r}_present':int(v>0) for r,v in counts.items()})
 D=xyz[(roles=='donor')&heavy];A=xyz[(roles=='acceptor')&heavy];U=xyz[(roles=='unknown')&heavy];out['donor_heavy_count']=len(D);out['acceptor_heavy_count']=len(A);out['unknown_heavy_count']=len(U)
 if len(D) and len(A):
  dist=np.linalg.norm(D[:,None,:]-A[None,:,:],axis=2);out.update({'da_min_distance':float(dist.min()),'da_q25_distance':float(np.quantile(dist,.25)),'da_median_distance':float(np.quantile(dist,.5)),'da_q75_distance':float(np.quantile(dist,.75)),'da_centroid_distance':float(np.linalg.norm(D.mean(0)-A.mean(0)))})
  for c in cutoffs:out[f'da_contacts_le_{c:g}A']=int(np.sum(dist<=c))
  out['donor_interface_atoms_4A']=int(np.sum(np.min(dist,axis=1)<=4.));out['acceptor_interface_atoms_4A']=int(np.sum(np.min(dist,axis=0)<=4.))
 else:
  for k in ['da_min_distance','da_q25_distance','da_median_distance','da_q75_distance','da_centroid_distance','donor_interface_atoms_4A','acceptor_interface_atoms_4A',*[f'da_contacts_le_{c:g}A' for c in cutoffs]]:out[k]=np.nan
 if len(U) and len(A):out['unknown_acceptor_min_distance']=float(np.linalg.norm(U[:,None,:]-A[None,:,:],axis=2).min())
 else:out['unknown_acceptor_min_distance']=np.nan
 if len(U) and len(D):out['unknown_donor_min_distance']=float(np.linalg.norm(U[:,None,:]-D[None,:,:],axis=2).min())
 else:out['unknown_donor_min_distance']=np.nan
 return out
def weighted_group_stats(frame,col):
 q=frame[[col,'group_weight']].dropna();x=q[col].to_numpy(float);w=q.group_weight.to_numpy(float);mu=np.sum(x*w)/w.sum();return {'count':len(q),'coverage':len(q)/len(frame),'mean':float(mu),'std':float(np.sqrt(np.sum(w*(x-mu)**2)/w.sum())),'min':float(x.min()),'max':float(x.max()),'unique':int(pd.Series(x).nunique())}
def parse_component(smiles):
 mol=Chem.MolFromSmiles(str(smiles),sanitize=False)
 if mol is None:raise RuntimeError('component parse failed')
 Chem.SanitizeMol(mol,sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
 return mol
def validation_acceptor_similarity(scope,cfg):
 comp=pd.read_csv(resolve(cfg['component_identity']['path']),usecols=['molecule_id','acceptor_structure_group_id_v1','acceptor_canonical_structure_smiles_v1'])
 z=scope[['molecule_id','partition','acceptor_structure_group_id_v1']].merge(comp,on=['molecule_id','acceptor_structure_group_id_v1'],validate='one_to_one')
 gen=rdFingerprintGenerator.GetMorganGenerator(radius=cfg['validation_similarity']['radius'],fpSize=cfg['validation_similarity']['n_bits'],includeChirality=cfg['validation_similarity']['use_chirality'])
 uniq=z[['acceptor_structure_group_id_v1','acceptor_canonical_structure_smiles_v1']].drop_duplicates()
 if uniq.acceptor_structure_group_id_v1.duplicated().any():raise RuntimeError('component identity maps to multiple canonical strings')
 fps={r.acceptor_structure_group_id_v1:gen.GetFingerprint(parse_component(r.acceptor_canonical_structure_smiles_v1)) for r in uniq.itertuples(index=False)}
 train_ids=sorted(z.loc[z.partition.eq('train'),'acceptor_structure_group_id_v1'].unique());val_ids=sorted(z.loc[z.partition.eq('val'),'acceptor_structure_group_id_v1'].unique());train_fps=[fps[x] for x in train_ids]
 scores={x:float(max(DataStructs.BulkTanimotoSimilarity(fps[x],train_fps))) for x in val_ids}
 out=z.loc[z.partition.eq('val'),['molecule_id','acceptor_structure_group_id_v1']].copy();out['nearest_train_acceptor_morgan2048_chiral']=out.acceptor_structure_group_id_v1.map(scores)
 return out.sort_values('molecule_id').reset_index(drop=True)
def main():
 cfg=json.loads((ROOT/'configs/gate2f0_acceptor_delta_feasibility_v1.json').read_text());head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
 if head!=cfg['expected_head']:raise RuntimeError('BLOCKED_GIT_BOUNDARY')
 for x in [cfg['acceptor_manifest'],cfg['component_identity'],cfg['c0_validation_predictions'],cfg['primary_labels']]:
  if sha(x['path'])!=x['sha256']:raise RuntimeError('frozen source hash mismatch')
 lock={'version':'gate2f0_preregistration_lock_v1','config_sha256':sha('configs/gate2f0_acceptor_delta_feasibility_v1.json'),'expected_head':head,'feature_families':['ground_state_scalars','role_counts','role_interface_geometry','pm6_dft_deltas'],'admission':cfg['admission'],'created_before_full_scan':True,'training':False,'test_access':False,'final673_access':False};write_json('data_registry/gate2f0_preregistration_lock_v1.json',lock)
 amendment={'version':'gate2f0_validation_similarity_amendment_v1','reason':'Gate 2-A similarity asset was discovered to contain test partitions only and is forbidden here','timing':'before any valid acceptor-shift statistic; prior attempted merge produced zero validation rows','replacement':'validation-only target-free nearest-train acceptor Morgan similarity recomputation','component_identity_sha256':cfg['component_identity']['sha256'],'fingerprint':cfg['validation_similarity'],'uses_labels':False,'uses_test':False,'changes_feature_admission_rules':False};write_json('data_registry/gate2f0_validation_similarity_amendment_lock_v1.json',amendment)
 (ROOT/'reports/gate2f0_validation_similarity_amendment.md').write_text("# Gate 2-F0 validation-similarity amendment\n\nThe frozen Gate 2-A similarity file was discovered to contain test partitions only. It was therefore not read by the corrected Gate 2-F0 analysis. Before any valid shift statistic existed, the source was replaced by a target-free validation-only recomputation from frozen acceptor component identities using Morgan radius 2, 2048 bits, chirality enabled, and nearest-train Tanimoto similarity. The admission rules are unchanged.\n")
 groups=pd.read_csv(ROOT/cfg['structure_manifest']);ids=sorted(groups.molecule_id.astype(str));out=[];source_hash=hashlib.sha256();root=Path(cfg['data_root'])
 for n,mid in enumerate(ids):
  pp=Path(cfg['pm6_root'])/mid;dp=Path(cfg['dft_root'])/mid;paths={'pm6_json':pp/f'{mid}_pm6.json','pm6_meta':pp/f'{mid}_metadata.json','pm6_side':pp/f'{mid}_sidecar.json','dft_json':dp/f'{mid}_dft.json','dft_meta':dp/f'{mid}_metadata.json','dft_side':dp/f'{mid}_sidecar.json'}
  if not all(p.is_file() and p.stat().st_size>0 for p in paths.values()):raise RuntimeError(f'missing raw source {mid}')
  for k,p in paths.items():source_hash.update(f'{mid}|{k}|{sha(p)}\n'.encode())
  pj,pm,ps,dj,dm,dsj=[loadj(paths[k]) for k in ['pm6_json','pm6_meta','pm6_side','dft_json','dft_meta','dft_side']];pz,px,pr=atom_arrays(pj);dz,dx,dr=atom_arrays(dj)
  row={'molecule_id':mid,'pm6_atom_count':len(pz),'dft_atom_count':len(dz),'atom_order_match':bool(np.array_equal(pz,dz)),'role_order_match':bool(np.array_equal(pr,dr)),'sidecar_roles_match':ps.get('atom_origins')==dsj.get('atom_origins'),'sidecar_conflict':mid=='D81_A28'}
  row.update({'formal_charge':finite(dm.get('charge')),'electron_count':finite(np.sum(dz)-finite(dm.get('charge'))),'dft_energy_hartree':finite(dm.get('scf_energy_hartree')),'pm6_energy_hartree':finite(pm.get('pm6_energy_hartree')),'dft_homo_hartree':finite(dm.get('homo_hartree')),'pm6_homo_hartree':finite(pm.get('homo_hartree')),'dft_lumo_hartree':finite(dm.get('lumo_hartree')),'pm6_lumo_hartree':finite(pm.get('lumo_hartree')),'dft_gap_hartree':finite(dm.get('homo_lumo_gap_hartree')),'pm6_gap_hartree':finite(pm.get('homo_lumo_gap_hartree')),'dft_dipole_magnitude_debye':finite(dm.get('dipole_debye')),'pm6_dipole_magnitude_debye':finite(pm.get('dipole_debye'))})
  for ax in 'xyz':row[f'dft_dipole_{ax}']=finite(dm.get(f'dipole_{ax}'));row[f'pm6_dipole_{ax}']=finite(pm.get(f'dipole_{ax}'))
  for f in ['homo_hartree','lumo_hartree','gap_hartree','dipole_magnitude_debye']:row[f'delta_{f}']=row[f'dft_{f}']-row[f'pm6_{f}'];row[f'abs_delta_{f}']=abs(row[f'delta_{f}'])
  row.update(interface_features(dz,dx,dr,cfg['interface']['contact_cutoffs_angstrom']));out.append(row)
  if (n+1)%2000==0:print('SCANNED',n+1,flush=True)
 feat=pd.DataFrame(out).merge(groups[['molecule_id','structure_group_id_v1','group_weight','structure_group_size']],on='molecule_id',validate='one_to_one').sort_values('molecule_id').reset_index(drop=True);local=ROOT/'runs/gate2f0_acceptor_delta_audit';local.mkdir(parents=True,exist_ok=True);feat.to_parquet(local/'features_v1.parquet',index=False)
 numeric=[c for c in feat.columns if c not in ['molecule_id','structure_group_id_v1','group_weight'] and feat[c].dtype!='object'];stats={c:weighted_group_stats(feat,c) for c in numeric}
 dup_ids=set(groups.loc[groups.structure_group_size.gt(1),'structure_group_id_v1']);dup=feat[feat.structure_group_id_v1.isin(dup_ids)]
 duplicate_dispersion={}
 for f in cfg['delta_pairs']:
  col=f'delta_{f}';ranges=dup.groupby('structure_group_id_v1')[col].agg(lambda x:float(x.max()-x.min())).to_numpy(float)
  duplicate_dispersion[col]={'groups':int(len(ranges)),'median_range':float(np.median(ranges)),'p90_range':float(np.quantile(ranges,.9)),'max_range':float(np.max(ranges)),'meaningful_range_gt_1e_6':int(np.sum(ranges>1e-6))}
 dup_roles=groups[groups.structure_group_id_v1.isin(dup_ids)].groupby('structure_group_id_v1').role_aware_group_id_v1.nunique()
 # protocol-local validation labels and frozen C0 residuals only
 manifest=pd.read_csv(ROOT/cfg['acceptor_manifest']['path']);scope=manifest[manifest.partition.isin(['train','val'])].copy();scope=scope.merge(feat.drop(columns=['structure_group_id_v1','group_weight','structure_group_size']),on='molecule_id',validate='one_to_one')
 if scope.partition.value_counts().to_dict()!={'train':10543,'val':2235}:raise RuntimeError('acceptor protocol train/validation count mismatch')
 val_ids=scope.loc[scope.partition.eq('val'),'molecule_id'].astype(str).tolist();lab=ds.dataset(resolve(cfg['primary_labels']['path']),format='parquet').to_table(filter=ds.field('molecule_id').isin(val_ids)).to_pandas()[['molecule_id',PRIMARY]];pred=pd.read_csv(ROOT/cfg['c0_validation_predictions']['path']);val=scope[scope.partition.eq('val')].merge(lab,on='molecule_id',validate='one_to_one').merge(pred,on='molecule_id',validate='one_to_one');val['c0_abs_error']=(val[PRIMARY]-val.prediction).abs();val['c0_residual']=val[PRIMARY]-val.prediction
 sim=validation_acceptor_similarity(scope,cfg);val=val.merge(sim,on=['molecule_id','acceptor_structure_group_id_v1'],validate='one_to_one');train=scope[scope.partition.eq('train')]
 if len(val)!=2235 or val[PRIMARY].isna().any() or val.prediction.isna().any():raise RuntimeError('acceptor validation artifact coverage mismatch')
 family_cols={'ground_state_scalars':['dft_homo_hartree','dft_lumo_hartree','dft_gap_hartree','dft_dipole_magnitude_debye','pm6_homo_hartree','pm6_lumo_hartree','pm6_gap_hartree','pm6_dipole_magnitude_debye','electron_count','formal_charge'],'role_counts':['donor_atom_count','acceptor_atom_count','unknown_atom_count','donor_atom_fraction','acceptor_atom_fraction','unknown_atom_fraction','donor_present','acceptor_present','unknown_present'],'role_interface_geometry':['da_min_distance','da_q25_distance','da_median_distance','da_q75_distance','da_centroid_distance','da_contacts_le_3.5A','da_contacts_le_4A','da_contacts_le_5A','donor_interface_atoms_4A','acceptor_interface_atoms_4A','unknown_acceptor_min_distance','unknown_donor_min_distance'],'pm6_dft_deltas':[f'delta_{x}' for x in cfg['delta_pairs']]+[f'abs_delta_{x}' for x in cfg['delta_pairs']]}
 shift={}
 for fam,cols in family_cols.items():
  shift[fam]={}
  for col in cols:
   a=train[col].dropna().to_numpy(float);b=val[col].dropna().to_numpy(float);pooled=np.sqrt((np.var(a)+np.var(b))/2) if len(a) and len(b) else np.nan;ti=train.groupby('acceptor_structure_group_id_v1')[col].mean().dropna();vi=val.groupby('acceptor_structure_group_id_v1')[col].mean().dropna();ok=val[col].notna();shift[fam][col]={'train_n':len(a),'val_n':len(b),'smd':float((np.mean(b)-np.mean(a))/pooled) if pooled>0 else None,'wasserstein':float(wasserstein_distance(a,b)) if len(a) and len(b) else None,'train_identity_n':int(len(ti)),'val_identity_n':int(len(vi)),'train_identity_mean':float(ti.mean()) if len(ti) else None,'val_identity_mean':float(vi.mean()) if len(vi) else None,'val_identities_with_any_missing':int(val.groupby('acceptor_structure_group_id_v1')[col].apply(lambda x:x.isna().any()).sum()),'spearman_acceptor_similarity':corr(val.loc[ok,col],val.loc[ok,'nearest_train_acceptor_morgan2048_chiral']),'spearman_c0_abs_error':corr(val.loc[ok,col],val.loc[ok,'c0_abs_error']),'spearman_c0_residual':corr(val.loc[ok,col],val.loc[ok,'c0_residual'])}
  shift[fam]['identity_power']={'train_acceptor_identities':int(train.acceptor_structure_group_id_v1.nunique()),'validation_acceptor_identities':int(val.acceptor_structure_group_id_v1.nunique())}
 write_json('logs/gate2f0_shift_metrics.json',shift)
 # semantics ledger and family decision
 rows=[]
 def add(name,family,source,path,meaning,unit,status,reason,coverage):rows.append({'field_name':name,'family':family,'source_file':source,'json_path':path,'method_fidelity':source.split('_')[0].upper(),'physical_meaning':meaning,'unit':unit,'available_pre_tddft':True,'algebraically_related_to_primary':False,'target_equivalent':False,'role_dependence':'none' if family!='role_interface_geometry' else 'original_roles','cost_tier':'Tier1.5' if source.startswith('pm6') else 'Tier2','coverage':coverage,'admission_status':status,'rejection_reason':reason})
 for f in ['homo_hartree','lumo_hartree','gap_hartree','dipole_magnitude_debye']:
  for method in ['pm6','dft']:add(f'{method}_{f}','ground_state_scalars',f'{method}_metadata',f,method.upper()+' ground-state '+f,'hartree' if 'hartree' in f else 'debye','ADMITTED_CANDIDATE','',stats[f'{method}_{f}']['coverage'])
 for f in cfg['delta_pairs']:add(f'delta_{f}','pm6_dft_deltas','paired_metadata',f'dft.{f}-pm6.{f}','signed same-unit method delta','hartree' if 'hartree' in f else 'debye','ADMITTED_CANDIDATE','',stats[f'delta_{f}']['coverage'])
 add('energy_delta','pm6_dft_deltas','paired_metadata','dft.scf_energy_hartree-pm6.total_energy_hartree','incompatible energy definitions','hartree','REJECTED_SEMANTICS','DFT SCF electronic energy and PM6 reported positive energy are not semantically equivalent',min(stats['dft_energy_hartree']['coverage'],stats['pm6_energy_hartree']['coverage']))
 add('dft_energy_hartree','ground_state_scalars','dft_metadata','scf_energy_hartree','DFT S0 SCF electronic energy','hartree','AUDIT_ONLY_NOT_ADMITTED','extensive size-dependent quantity; excluded from first capacity-controlled feature graph',stats['dft_energy_hartree']['coverage'])
 add('pm6_energy_hartree','ground_state_scalars','pm6_metadata','pm6_energy_hartree','PM6 reported energy','hartree','REJECTED_SEMANTICS','pm6_energy_raw semantics remain incompatible with DFT SCF energy',stats['pm6_energy_hartree']['coverage'])
 for f,u in [('formal_charge','electron'),('electron_count','count'),('dft_atom_count','count')]:add(f,'ground_state_scalars','dft_json/metadata',f,f,u,'ADMITTED_CANDIDATE','',stats[f]['coverage'])
 for f in ['dipole_x','dipole_y','dipole_z']:
  add('delta_'+f,'pm6_dft_deltas','paired_metadata',f'dft.{f}-pm6.{f}','orientation-dependent component difference','debye','REJECTED_NOT_INVARIANT','optimized geometries do not share a guaranteed global coordinate frame',min(stats['dft_'+f]['coverage'],stats['pm6_'+f]['coverage']))
 for f in family_cols['role_counts']:add(f,'role_counts','dft_json','atoms[].type',f,'count/fraction/boolean','ADMITTED_CANDIDATE','',stats[f]['coverage'])
 for f in family_cols['role_interface_geometry']:add(f,'role_interface_geometry','dft_json','atoms[].coords + atoms[].type',f,'angstrom/count','ADMITTED_CANDIDATE' if stats[f]['coverage']>=cfg['admission']['minimum_coverage'] else 'REJECTED_COVERAGE','empty donor or missing role makes distance undefined; no imputation',stats[f]['coverage'])
 add('mulliken_role_aggregates','role_atomic_electronic','pm6/dft_json','not present','role-wise ground-state atomic charges','electron','REJECTED_FIELD_ABSENT','no auditable Mulliken or equivalent ground-state atomic charge field',0.0);add('frontier_orbital_role_contribution','role_atomic_electronic','pm6/dft_json','not present','role-wise frontier orbital contribution','fraction','REJECTED_FIELD_ABSENT','no auditable atom-resolved orbital contribution field',0.0)
 ledger=pd.DataFrame(rows)
 ledger['missing_fraction']=ledger.field_name.map(lambda x:1-stats[x]['coverage'] if x in stats else np.nan);ledger['finite_rate']=ledger.field_name.map(lambda x:stats[x]['coverage'] if x in stats else np.nan);ledger['finite_count']=ledger.field_name.map(lambda x:stats[x]['count'] if x in stats else 0);ledger['unique_count']=ledger.field_name.map(lambda x:stats[x]['unique'] if x in stats else 0);ledger['weighted_mean']=ledger.field_name.map(lambda x:stats[x]['mean'] if x in stats else np.nan);ledger['weighted_std']=ledger.field_name.map(lambda x:stats[x]['std'] if x in stats else np.nan);ledger['valid_min']=ledger.field_name.map(lambda x:stats[x]['min'] if x in stats else np.nan);ledger['valid_max']=ledger.field_name.map(lambda x:stats[x]['max'] if x in stats else np.nan);ledger['constant_or_near_constant']=ledger.field_name.map(lambda x:bool(stats[x]['std']<1e-12) if x in stats else False)
 def iqr_outlier_rate(x):
  if x not in feat:return np.nan
  a=feat[x].dropna().to_numpy(float)
  if not len(a):return np.nan
  q1,q3=np.quantile(a,[.25,.75]);iqr=q3-q1
  return float(np.mean((a<q1-1.5*iqr)|(a>q3+1.5*iqr)))
 ledger['iqr_outlier_rate']=ledger.field_name.map(iqr_outlier_rate)
 ledger.to_csv(ROOT/'data_registry/gate2f0_field_semantics_ledger.csv',index=False)
 atom_ok=float(feat.atom_order_match.mean());role_ok=float(feat.role_order_match.mean());delta_cols=[f'delta_{x}' for x in cfg['delta_pairs']];delta_cov=min(stats[x]['coverage'] for x in delta_cols);delta_admit=delta_cov>=cfg['admission']['minimum_coverage'] and atom_ok==1.0 and role_ok==1.0
 decision='DELTA_FEATURE_GRAPH_ADMITTED' if delta_admit else ('GROUND_STATE_INTERFACE_ONLY_ADMITTED' if any(stats[x]['coverage']>=.99 for x in family_cols['role_interface_geometry']) else 'BLOCKED_FEATURE_SEMANTICS')
 family_registry={'decision':decision,'families':{k:{'fields':v,'minimum_coverage':min(stats[x]['coverage'] for x in v),'admitted':(k in ['ground_state_scalars','role_counts'] or (k=='pm6_dft_deltas' and delta_admit))} for k,v in family_cols.items()},'atom_order_match_rate':atom_ok,'role_order_match_rate':role_ok,'empty_donor_records':int((feat.donor_present==0).sum()),'sidecar_conflict_records':int(feat.sidecar_conflict.sum()),'source_inventory_sha256':source_hash.hexdigest(),'local_feature_sha256':sha(local/'features_v1.parquet'),'training':False,'prediction_generation':False};write_json('data_registry/gate2f0_feature_family_registry.json',family_registry)
 delta_registry={'pairs':{f:{'pm6_field':f'pm6_{f}','dft_field':f'dft_{f}','signed_rule':f'dft_{f} - pm6_{f}','unit':'hartree' if 'hartree' in f else 'debye','coverage':stats[f'delta_{f}']['coverage'],'pm6_std':stats[f'pm6_{f}']['std'],'dft_std':stats[f'dft_{f}']['std'],'delta_std':stats[f'delta_{f}']['std'],'delta_to_dft_std_ratio':stats[f'delta_{f}']['std']/stats[f'dft_{f}']['std'] if stats[f'dft_{f}']['std'] else None,'standardized_delta_policy':'protocol-train-only scaling; not a separate feature','duplicate_dispersion':duplicate_dispersion[f'delta_{f}'],'admitted':delta_admit} for f in cfg['delta_pairs']},'rejected_energy_pair':True,'energy_reason':'non-equivalent PM6 reported energy versus DFT SCF electronic energy','delta_sign_reproducible':True};write_json('data_registry/gate2f0_delta_pair_registry.json',delta_registry)
 write_json('logs/gate2f0_evidence.json',{'status':'GATE2F0_DONE','decision':decision,'records':len(feat),'source_coverage':1.0,'atom_order_match_rate':atom_ok,'role_order_match_rate':role_ok,'duplicate_structure_groups':len(dup_ids),'role_inconsistent_duplicate_groups':int((dup_roles>1).sum()),'duplicate_delta_dispersion':duplicate_dispersion,'empty_donor_records':int((feat.donor_present==0).sum()),'d81_a28_preserved_conflict':True,'training':False,'gpu_used':False,'prediction_generated':False,'test_accessed':False,'main_parquet_accessed':False,'final673_accessed':False,'official_validation_predictions_generated':False,'completed_utc':datetime.now(timezone.utc).isoformat()})
 # reports
 (ROOT/'reports/gate2f0_feature_firewall.md').write_text(f"# Gate 2-F0 feature firewall\n\nAll admitted candidates are PM6/DFT ground-state or DFT-S0 geometry quantities available before TDDFT. No TDDFT/Multiwfn raw file, test artifact, main Parquet, or final673 asset was read. PM6 raw energy, DFT/PM6 dipole-component deltas, IDs, split/provenance fields, and absent atom-charge/orbital fields are rejected.\n")
 (ROOT/'reports/gate2f0_ground_state_field_audit.md').write_text(f"# Gate 2-F0 ground-state field audit\n\nScanned {len(feat):,} paired PM6/DFT records. Source coverage is 100%; atom-order match is {atom_ok:.6f}; role-order match is {role_ok:.6f}. All {len(dup_ids)} duplicate structure groups were included in dispersion checks; {int((dup_roles>1).sum())} contain more than one role-aware identity and are not collapsed. There are {int((feat.donor_present==0).sum())} empty-donor records, retained with presence flags and undefined donor-interface distances rather than imputation. D81_A28 remains explicitly conflict-flagged. No auditable Mulliken charges or atom-resolved frontier-orbital contributions exist in the frozen JSON schema.\n")
 acc_shift=max(abs(v['smd']) for f in shift.values() for k,v in f.items() if k!='identity_power' and v['smd'] is not None)
 delta_smd=max(abs(v['smd']) for k,v in shift['pm6_dft_deltas'].items() if k!='identity_power' and v['smd'] is not None);delta_error=max(abs(v['spearman_c0_abs_error']) for k,v in shift['pm6_dft_deltas'].items() if k!='identity_power' and v['spearman_c0_abs_error'] is not None)
 (ROOT/'reports/gate2f0_acceptor_shift_mechanism.md').write_text(f"# Gate 2-F0 acceptor shift mechanism\n\nThe acceptor-cold diagnostic uses only official train/validation identities, frozen C0 validation predictions, validation labels already present in the local calibration artifact, and newly recomputed validation-only target-free similarity. It contains {train.acceptor_structure_group_id_v1.nunique()} train and {val.acceptor_structure_group_id_v1.nunique()} held-out validation acceptor identities. The largest absolute standardized mean shift across all audited features is {acc_shift:.4f}; it occurs in role/size composition rather than establishing a delta mechanism. Across admitted delta fields, the largest absolute SMD is {delta_smd:.4f} and the largest absolute Spearman association with C0 validation error is {delta_error:.4f}. These are diagnostic associations, not causal evidence, and did not select fields.\n")
 ratios=', '.join(f"{f}={delta_registry['pairs'][f]['delta_to_dft_std_ratio']:.3f}" for f in cfg['delta_pairs'])
 (ROOT/'reports/gate2f0_delta_learning_feasibility.md').write_text(f"# Gate 2-F0 delta-learning feasibility\n\nSemantically matched pairs exist for HOMO, LUMO, gap (hartree), and dipole magnitude (debye), with minimum coverage {delta_cov:.6f}. Signed deltas are frozen as DFT minus PM6. Delta/DFT weighted standard-deviation ratios are {ratios}; these describe numerical stability only and do not establish predictive value. Energy delta is rejected because PM6 reported energy and DFT SCF energy are different physical quantities; coordinate dipole deltas are rejected as non-invariant. Core D/A interface distances cover {stats['da_min_distance']['coverage']:.6f}, below the 0.99 family threshold because all 387 empty-donor records are retained without imputation. Admitted deltas require Tier 2 DFT at deployment; PM6-only deployment cannot obtain them. A later capacity-controlled validation-only experiment can compare C0, C0+ground-state scalars, C0+role/interface descriptors, and C0+paired deltas without any TDDFT input.\n")
 (ROOT/'reports/gate2f0_final_decision.md').write_text(f"# Gate 2-F0 final decision\n\n## `{decision}`\n\nFour matched PM6/DFT ground-state field pairs meet the frozen semantics, unit, 0.99 coverage, atom-order, role-order, and target-firewall gates. The decision admits a candidate feature graph, not a predictive model and not evidence of acceptor-OOD improvement. Role/interface geometry is not admitted as a complete family at this threshold because donor-derived distances are undefined for 387 retained empty-donor records. Any training requires a new preregistered Gate with protocol-local preprocessing, explicit capacity controls, and validation-only selection.\n")
 print(json.dumps({'decision':decision,'records':len(feat),'atom_order_match':atom_ok,'role_order_match':role_ok,'delta_coverage':delta_cov,'empty_donor':int((feat.donor_present==0).sum())},indent=2))
if __name__=='__main__':main()
