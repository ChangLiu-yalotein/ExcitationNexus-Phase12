#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import torch

ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT/'src'))
from excitationnexus_phase12.contracts import TaskGraph,verify_frozen_inputs
from excitationnexus_phase12.dataset import Phase12Dataset,load_bound_table
from excitationnexus_phase12.losses import smoke_task_weights,weighted_masked_multitask_loss
from excitationnexus_phase12.models import TinyRoleAware3DMultitaskModel
from excitationnexus_phase12.samplers import make_dataloader

TABLE=Path('/home/changliu/ExcitationNexus_Data_v2/tables/molecule_values_v3.parquet')
RAW=Path('/home/changliu/ExcitationNexus_Data_v2/raw_compact')


def main():
    start=time.time(); (ROOT/'runs/gate0d_gpu_smoke').mkdir(parents=True,exist_ok=True)
    cpu=json.loads((ROOT/'logs/gate0d_cpu_smoke.json').read_text())
    if cpu['status']!='CPU_DONE': raise RuntimeError('BLOCKED_CPU')
    physical=os.environ.get('CUDA_VISIBLE_DEVICES','')
    if not physical or ',' in physical: raise RuntimeError('single physical GPU binding required')
    if not torch.cuda.is_available() or torch.cuda.device_count()!=1: raise RuntimeError('BLOCKED_GPU: binding unavailable')
    verify_frozen_inputs(TABLE,ROOT/'manifests'); device=torch.device('cuda:0')
    torch.manual_seed(42); torch.cuda.set_device(0); torch.empty(1, device=device); torch.cuda.manual_seed_all(42); torch.cuda.reset_peak_memory_stats(0)
    task_graph=TaskGraph.load(); frame=load_bound_table(TABLE,ROOT/'manifests/split_iid_group_seed42_v1.csv')
    smallest=frame.loc[frame.partition.eq('train')].sort_values(['num_atoms_total','molecule_id']).head(8)
    norm=json.loads((ROOT/'data_registry/normalization_iid_group_seed42_v1.json').read_text())
    ds=Phase12Dataset(smallest,partition='train',view='tier1_pm6_3d',raw_root=RAW,task_graph=task_graph,
        pm6_dipole_enabled=False,target_stats=norm['targets'])
    loader=make_dataloader(ds,batch_size=4,shuffle=False,seed=42,num_workers=0)
    tasks=task_graph.optimization_tasks
    model=TinyRoleAware3DMultitaskModel(tasks,len(ds.scalar_fields),32,16,2).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-4)
    weights=smoke_task_weights(task_graph.primary,task_graph.secondary,task_graph.auxiliary)
    before={n:p.detach().clone() for n,p in model.named_parameters()}
    losses=[]; shared_nonzero=False; primary_nonzero=False; batches=0; first_batch=None
    for batch in loader:
        if batches>=2: break
        batch=batch.to(device); first_batch=batch.clone() if first_batch is None else first_batch
        optimizer.zero_grad(set_to_none=True); pred=model(batch); B=batch.num_graphs
        targets=batch.targets.reshape(B,-1); masks=batch.target_mask.reshape(B,-1)
        loss,per=weighted_masked_multitask_loss(pred,targets,masks,batch.group_weight.reshape(B),
            task_graph.all_reportable_tasks,weights,report_only=task_graph.report_only,base_loss='smooth_l1')
        if not torch.isfinite(loss): raise RuntimeError('non-finite GPU loss')
        loss.backward()
        for n,p in model.named_parameters():
            if p.grad is not None and not torch.isfinite(p.grad).all(): raise RuntimeError(f'non-finite gradient:{n}')
            nz=p.grad is not None and float(p.grad.abs().sum())>0
            shared_nonzero |= bool(nz and n.startswith('shared_trunk'))
            primary_nonzero |= bool(nz and n.startswith('heads.'+task_graph.primary[0]))
        optimizer.step(); losses.append(float(loss.detach().cpu())); batches+=1
        assert all(list(pred[x].shape)==[B] for x in tasks)
    changed=any(not torch.equal(before[n],p.detach()) for n,p in model.named_parameters())
    if batches!=2 or not shared_nonzero or not primary_nonzero or not changed:
        raise RuntimeError('GPU gradient/parameter update contract failed')
    model.eval()
    with torch.no_grad():
        ref=model(first_batch)[task_graph.primary[0]]
        translated=first_batch.clone(); translated.pos=translated.pos+torch.tensor([1.1,-2.2,3.3],device=device)
        pt=model(translated)[task_graph.primary[0]]
        torch.manual_seed(11); q,_=torch.linalg.qr(torch.randn(3,3,device=device)); rotated=first_batch.clone(); rotated.pos=rotated.pos@q
        pr=model(rotated)[task_graph.primary[0]]
    trans_ok=bool(torch.allclose(ref,pt,atol=2e-5,rtol=2e-5)); rot_ok=bool(torch.allclose(ref,pr,atol=2e-5,rtol=2e-5))
    if not trans_ok or not rot_ok: raise RuntimeError('GPU invariance failed')
    torch.cuda.synchronize(); elapsed=time.time()-start
    allocated=int(torch.cuda.max_memory_allocated(0)); reserved=int(torch.cuda.max_memory_reserved(0))
    evidence={'status':'GPU_DONE','physical_gpu_id':int(physical),'process_device':'cuda:0',
      'gpu_name':torch.cuda.get_device_name(0),'batches':batches,'batch_size':4,'dtype':'float32',
      'losses':losses,'loss_finite':all(torch.isfinite(torch.tensor(losses))),
      'gradients_finite':True,'shared_trunk_gradient_nonzero':shared_nonzero,
      'primary_head_gradient_nonzero':primary_nonzero,'parameter_changed':changed,
      'output_task_count':len(tasks),'output_shape_per_task':[4],
      'translation_invariant':trans_ok,'rotation_invariant':rot_ok,
      'peak_allocated_bytes':allocated,'peak_reserved_bytes':reserved,
      'peak_allocated_gib':allocated/2**30,'peak_reserved_gib':reserved/2**30,
      'wall_seconds':elapsed,'checkpoint_saved':False,'epoch_run':False,'ddp_used':False}
    (ROOT/'logs/gate0d_gpu_smoke.json').write_text(json.dumps(evidence,indent=2,sort_keys=True)+'\n')
    report=['# Gate 0-D GPU smoke','',f"Status: **GPU_DONE** on physical GPU {physical} (`cuda:0`).",'',
      f"Two FP32 batches of size 4 completed forward, masked loss, backward, and AdamW step in {elapsed:.3f} s.",
      f"Peak allocated/reserved memory: {allocated/2**20:.1f}/{reserved/2**20:.1f} MiB.",
      f"Finite loss/gradients, nonzero shared/primary gradients, parameter change, and GPU translation/rotation invariance all passed.",
      '',"No epoch, DDP, checkpoint, model comparison, or selection was run."]
    (ROOT/'reports/gate0d_gpu_smoke.md').write_text('\n'.join(report)+'\n')
    print(json.dumps(evidence,indent=2))

if __name__=='__main__': main()
