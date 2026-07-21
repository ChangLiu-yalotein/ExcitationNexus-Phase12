#!/usr/bin/env python3
from __future__ import annotations

import csv, json
from pathlib import Path
ROOT=Path('/home/changliu/ExcitationNexus/12_Phase4_Multitask_OOD_Training')

def write_csv(name,rows):
    p=ROOT/name; p.parent.mkdir(parents=True,exist_ok=True); cols=sorted({k for r in rows for k in r})
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=cols); w.writeheader(); w.writerows(rows)

def markdown(rows,cols):
    def clean(x): return f'{x:.6f}' if isinstance(x,float) else str(x).replace('|','\\|')
    return '| '+' | '.join(cols)+' |\n| '+' | '.join(['---']*len(cols))+' |\n'+''.join('| '+' | '.join(clean(r.get(c,'')) for c in cols)+' |\n' for r in rows)


uq=json.loads((ROOT/'logs/gate2c_coverage_metrics.json').read_text()); uq_rows=[]
for protocol,pdata in uq['protocols'].items():
    method='acceptor_identity_sensitivity' if protocol=='both_cold' else 'identity'
    item=pdata['interval_methods'][method]['0.9']; coverage=item['coverage']
    uq_rows.append({'protocol':protocol,'nominal':0.9,'method':method,'record_coverage':coverage['record_marginal'],'identity_macro_coverage':coverage.get('identity_macro',coverage.get('acceptor_identity_macro')),'cluster_simultaneous_coverage':coverage.get('cluster_simultaneous',coverage.get('record_marginal')),'interval_width_eV':item['interval_width'],'normalized_width':item['normalized_width_over_target_iqr'],'calibration_clusters':item['n'],'exchangeability_note':'crossed empirical only' if protocol=='both_cold' else 'protocol-specific audit'})
params=[
 {'model':'M3-Merged','trainable_parameters':36689,'iid_group_macro_mae_eV':0.08766406500328516,'parameter_status':'exact'},
 {'model':'M3-DAU-Shared','trainable_parameters':36461,'iid_group_macro_mae_eV':0.08854613716585205,'parameter_status':'exact'},
 {'model':'B2-1 dual tower','trainable_parameters':1065570,'iid_group_macro_mae_eV':'historical protocol only','parameter_status':'historical exact; not plotted with new15016'},
 {'model':'XGBoost-C0','trainable_parameters':'not meaningfully comparable','iid_group_macro_mae_eV':0.08418147504486073,'parameter_status':'500 trees'}]
costs=[
 {'model':'XGBoost-C0 deployment','protocol':'all 15015 legal records','train_wall_seconds':1.4949336778372526,'inference_wall_seconds':'registry available','peak_gpu_memory':'not uniformly captured','source':'Gate 3-A1'},
 {'model':'M3-Merged','protocol':'new15016 IID 3 seeds','train_wall_seconds':'per-run registry','inference_wall_seconds':'per-seed frozen artifact','peak_gpu_memory':'per-run registry','source':'Gate 1-B3'},
 {'model':'M3-DAU-Shared','protocol':'new15016 IID 3 seeds','train_wall_seconds':'per-run registry','inference_wall_seconds':'per-seed frozen artifact','peak_gpu_memory':'per-run registry','source':'Gate 1-B3'},
 {'model':'physics multitask E2A','protocol':'training-only cross-fit','train_wall_seconds':'per-fold registry','inference_wall_seconds':'OOF only','peak_gpu_memory':'per-fold registry','source':'Gate 2-E2A'}]
negative=[
 {'method':'M3-Merged / M3-DAU','status':'NEGATIVE','reason':'both weaker than XGBoost-C0 on frozen IID'},
 {'method':'role-aware Morgan','status':'NOT_ADMITTED','reason':'lowest-similarity acceptors worsened'},
 {'method':'frozen MoLFormer','status':'INCONCLUSIVE','reason':'did not beat C0 and failed IID guard'},
 {'method':'fixed-weight multitask','status':'BLOCKED_THEN_INCONCLUSIVE','reason':'corrected training-only cross-fit acceptor CI crossed zero'},
 {'method':'ground-state multifidelity/delta','status':'INCONCLUSIVE_NO_DELTA_GAIN','reason':'small cross-fit effect; delta reparameterization no gain'}]
splits=[
 {'ledger':'historical','dataset':'Layer G / historical 7316','models':'cheap, B2-0/B2-1/B2-2a','comparison':'historical protocols only'},
 {'ledger':'new15016','dataset':'15016 calculations / 14639 structure groups','models':'median, Ridge, XGBoost, M3, validation/cross-fit admissions','comparison':'IID and five frozen OOD protocols'}]
write_csv('data_registry/gate2g0_uq_benchmark.csv',uq_rows); write_csv('data_registry/gate2g0_parameter_inventory.csv',params); write_csv('data_registry/gate2g0_cost_inventory.csv',costs); write_csv('data_registry/gate2g0_negative_results.csv',negative); write_csv('data_registry/gate2g0_dataset_split_map.csv',splits)
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
out=ROOT/'reports/figures/gate2g0'; made=[]
fig,ax=plt.subplots(figsize=(7,4)); ax.scatter([r['interval_width_eV'] for r in uq_rows],[r['identity_macro_coverage'] for r in uq_rows])
for r in uq_rows: ax.annotate(r['protocol'],(r['interval_width_eV'],r['identity_macro_coverage']),fontsize=7)
ax.axhline(.9,color='black',ls='--'); ax.set_xlabel('90% interval width (eV)'); ax.set_ylabel('Identity-macro coverage')
for ext in ['png','pdf']:
    p=out/f'uq_coverage_width.{ext}'; fig.savefig(p,bbox_inches='tight',dpi=180); made.append(str(p.relative_to(ROOT)))
plt.close(fig)
fig,ax=plt.subplots(figsize=(6,4)); p2=params[:2]; ax.scatter([r['trainable_parameters'] for r in p2],[r['iid_group_macro_mae_eV'] for r in p2])
for r in p2: ax.annotate(r['model'],(r['trainable_parameters'],r['iid_group_macro_mae_eV']),fontsize=8)
ax.set_xlabel('Trainable parameters'); ax.set_ylabel('IID group-macro MAE (eV)')
for ext in ['png','pdf']:
    p=out/f'parameter_count_vs_mae.{ext}'; fig.savefig(p,bbox_inches='tight',dpi=180); made.append(str(p.relative_to(ROOT)))
plt.close(fig)
e=json.loads((ROOT/'logs/gate2g0_evidence.json').read_text()); e['figures']=sorted(set(e['figures']+made)); e['supplemental_tables']=['gate2g0_uq_benchmark.csv','gate2g0_parameter_inventory.csv','gate2g0_cost_inventory.csv','gate2g0_negative_results.csv','gate2g0_dataset_split_map.csv']; (ROOT/'logs/gate2g0_evidence.json').write_text(json.dumps(e,indent=2,sort_keys=True)+'\n')
print(json.dumps({'status':e['status'],'supplemental_figures':made},indent=2))
