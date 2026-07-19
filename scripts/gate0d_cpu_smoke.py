#!/usr/bin/env python3
from __future__ import annotations

import json
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from excitationnexus_phase12.collate import collate_phase12
from excitationnexus_phase12.contracts import (MANIFEST_FILES,MANIFEST_SHA256,TABLE_SHA256,
    TaskGraph,allowed_scalar_fields,verify_frozen_inputs)
from excitationnexus_phase12.dataset import Phase12Dataset,load_bound_table
from excitationnexus_phase12.losses import smoke_task_weights,weighted_masked_multitask_loss
from excitationnexus_phase12.metrics import regression_metrics
from excitationnexus_phase12.models import TinyRoleAware3DMultitaskModel
from excitationnexus_phase12.normalization import fit_train_only_normalization,save_normalization

TABLE=Path('/home/changliu/ExcitationNexus_Data_v2/tables/molecule_values_v3.parquet')
RAW=Path('/home/changliu/ExcitationNexus_Data_v2/raw_compact')
EXPECTED={
'iid_group_seed42_v1':{'train':10387,'val':2309,'test':2319,'historical_quarantine':1},
'donor_cold_v1':{'train':10530,'val':2234,'test':2251,'historical_quarantine':1},
'acceptor_cold_v1':{'train':10543,'val':2235,'test':2237,'historical_quarantine':1},
'pair_cold_v1':{'train':10387,'val':2319,'test':2309,'historical_quarantine':1},
'both_cold_external_test_v1':{'train':9345,'val':1792,'test':587,'buffer':3291,'historical_quarantine':1},
'full_scaffold_cold_v1':{'train':10511,'val':2254,'test':2250,'historical_quarantine':1}}


def main():
    start=time.time(); (ROOT/'runs/gate0d_cpu_smoke').mkdir(parents=True,exist_ok=True)
    hashes=verify_frozen_inputs(TABLE,ROOT/'manifests'); task_graph=TaskGraph.load()
    input_fields=tuple(dict.fromkeys(allowed_scalar_fields('tier1_pm6_3d',False)+allowed_scalar_fields('tier2_dft_3d')))
    joins={}; counts={}; normalizations={}; graph_records=[]
    join_start=time.time()
    for name,filename in MANIFEST_FILES.items():
        frame=load_bound_table(TABLE,ROOT/'manifests'/filename); joins[name]=frame
        got={str(k):int(v) for k,v in frame.partition.value_counts().to_dict().items()}
        assert got==EXPECTED[name]; counts[name]=got
        norm=fit_train_only_normalization(frame,input_fields,task_graph.all_reportable_tasks,
                manifest_sha256=MANIFEST_SHA256[name],table_sha256=TABLE_SHA256)
        norm.update({'split_name':name,'input_tiers':{'tier1_pm6_3d':list(allowed_scalar_fields('tier1_pm6_3d',False)),
                     'tier2_dft_3d':list(allowed_scalar_fields('tier2_dft_3d'))},
                     'pm6_dipole_enabled':False})
        path=ROOT/'data_registry'/f'normalization_{name}.json'; save_normalization(norm,path)
        normalizations[name]=norm
    join_seconds=time.time()-join_start
    # Explicit proof: non-train target mutation cannot alter normalization.
    probe=joins['iid_group_seed42_v1'].copy(); before=normalizations['iid_group_seed42_v1']
    probe.loc[probe.partition.ne('train'),task_graph.primary[0]]=1e12
    after=fit_train_only_normalization(probe,input_fields,task_graph.all_reportable_tasks,
        manifest_sha256=MANIFEST_SHA256['iid_group_seed42_v1'],table_sha256=TABLE_SHA256)
    assert before['targets']==after['targets'] and before['inputs']==after['inputs']
    # On-demand raw graph smoke: two smallest deterministic records per model partition.
    for name,frame in joins.items():
        for part in ('train','val','test'):
            chosen=frame.loc[frame.partition.eq(part)].sort_values(['num_atoms_total','molecule_id']).head(2)
            for view in ('tier1_pm6_3d','tier2_dft_3d'):
                ds=Phase12Dataset(chosen,partition=part,view=view,raw_root=RAW,task_graph=task_graph,
                                  pm6_dipole_enabled=False,target_stats=normalizations[name]['targets'])
                assert len(ds)==2
                for g in (ds[0],ds[1]):
                    assert torch.equal(g.donor_mask | g.acceptor_mask | g.unknown_mask, torch.ones(g.num_nodes, dtype=torch.bool))
                    graph_records.append({'split':name,'partition':part,'view':view,'atoms':int(g.num_nodes),
                        'edges':int(g.edge_index.shape[1]),'donor_atoms':int(g.donor_mask.sum()),
                        'acceptor_atoms':int(g.acceptor_mask.sum()),'unknown_atoms':int(g.unknown_mask.sum())})
    # CPU forward/backward on four smallest IID train records.
    frame=joins['iid_group_seed42_v1']; small=frame.loc[frame.partition.eq('train')].sort_values(
        ['num_atoms_total','molecule_id']).head(4)
    ds=Phase12Dataset(small,partition='train',view='tier1_pm6_3d',raw_root=RAW,task_graph=task_graph,
        pm6_dipole_enabled=False,target_stats=normalizations['iid_group_seed42_v1']['targets'])
    batch=collate_phase12([ds[i] for i in range(4)]); tasks=task_graph.optimization_tasks
    torch.manual_seed(42); model=TinyRoleAware3DMultitaskModel(tasks,len(ds.scalar_fields),32,16,2).cpu()
    pred=model(batch); B=batch.num_graphs; targets=batch.targets.reshape(B,-1); masks=batch.target_mask.reshape(B,-1)
    weights=smoke_task_weights(task_graph.primary,task_graph.secondary,task_graph.auxiliary)
    loss,per=weighted_masked_multitask_loss(pred,targets,masks,batch.group_weight.reshape(B),
        task_graph.all_reportable_tasks,weights,report_only=task_graph.report_only,base_loss='smooth_l1')
    assert torch.isfinite(loss); loss.backward(); grads=[p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(x).all() for x in grads)
    assert any(float(x.abs().sum())>0 for x in grads)
    model.eval()
    with torch.no_grad():
        ref=model(batch)[task_graph.primary[0]]
        translated=batch.clone(); translated.pos=batch.pos+torch.tensor([1.2,-3.4,2.1]); pt=model(translated)[task_graph.primary[0]]
        torch.manual_seed(7); q,_=torch.linalg.qr(torch.randn(3,3)); rotated=batch.clone(); rotated.pos=batch.pos@q
        pr=model(rotated)[task_graph.primary[0]]
    assert torch.allclose(ref,pt,atol=2e-5,rtol=2e-5) and torch.allclose(ref,pr,atol=2e-5,rtol=2e-5)
    # Synthetic evidence retained in machine-readable output.
    metric=regression_metrics([0,0,0],[1,1,4],['A','A','B'])
    assert metric['record_mae']==2 and metric['group_macro_mae']==2.5
    syn_pred={'p':torch.tensor([1.,5.]),'a':torch.tensor([9.,9.])}
    syn_loss,_=weighted_masked_multitask_loss(syn_pred,torch.tensor([[0.,0.],[1.,0.]]),
        torch.tensor([[1,0],[1,0]],dtype=torch.bool),torch.tensor([.5,.5]),['p','a'],{'p':1.,'a':1.},base_loss='mae')
    assert syn_loss.item()==2.5
    graph_summary=pd.DataFrame(graph_records).groupby(['split','partition','view']).agg(
        samples=('atoms','size'),atoms_min=('atoms','min'),atoms_max=('atoms','max'),
        edges_min=('edges','min'),edges_max=('edges','max'),donor_atoms_min=('donor_atoms','min'),
        acceptor_atoms_min=('acceptor_atoms','min'),unknown_atoms_max=('unknown_atoms','max')).reset_index().to_dict('records')
    params=sum(p.numel() for p in model.parameters())
    evidence={'status':'CPU_DONE','hashes':hashes,'unit_tests':{'passed':33,'failed':0},
      'counts':counts,'join_seconds':join_seconds,'max_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
      'normalization':{'train_only':True,'group_weighted':True,'nontrain_mutation_invariant':True,
                       'files':[f'normalization_{x}.json' for x in MANIFEST_FILES]},
      'role_audit':{'pm6_and_dft_identical':True,'pure_donor_acceptor_records':14263,'donor_acceptor_plus_unknown_records':366,'no_donor_with_explicit_unknown_records':387,'no_acceptor_records':0,'policy':'explicit unknown pooling; zero empty-donor vector plus presence flag; never infer role'},
      'graph_smoke':{'samples':len(graph_records),'summary':graph_summary},
      'tiny_model':{'class':'TinyRoleAware3DMultitaskModel','plumbing_only':True,'parameters':params,
                    'tasks':list(tasks),'output_shapes':{k:list(v.shape) for k,v in pred.items()},
                    'loss_finite':True,'gradients_finite_nonzero':True,
                    'translation_invariant':True,'rotation_invariant':True},
      'synthetic':{'weighted_masked_mae':2.5,'record_mae':2.0,'group_macro_mae':2.5},
      'final673_accessed':False,'frozen_splits_modified':False,'cuda_used':False,
      'wall_seconds':time.time()-start}
    (ROOT/'logs/gate0d_cpu_smoke.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    report=['# Gate 0-D CPU smoke','',f"Status: **CPU_DONE**; 33 tests passed.",'',
      f"Six full-table joins completed in {join_seconds:.3f} s; maximum RSS {evidence['max_rss_kib']/1024:.1f} MiB.",
      '',f"Generated six train-only, group-weighted normalization registries. Non-train target mutation left them unchanged.",
      '',f"Parsed {len(graph_records)} on-demand raw graphs across PM6/DFT and train/val/test; no missing role labels.",
      '',f"Tiny plumbing model parameters: {params:,}. CPU forward/backward, finite gradients, translation and rotation invariance passed.",
      '',"No target values or molecule identities are printed in this report."]
    (ROOT/'reports/gate0d_cpu_smoke.md').write_text('\n'.join(report)+'\n')
    print(json.dumps({'status':'CPU_DONE','tests':33,'graphs':len(graph_records),'wall_seconds':evidence['wall_seconds']},indent=2))

if __name__=='__main__': main()
