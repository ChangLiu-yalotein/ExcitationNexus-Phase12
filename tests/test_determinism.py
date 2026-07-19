import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Batch
from excitationnexus_phase12.graph_builder import build_graph_from_json,raw_paths
from excitationnexus_phase12.models import TinyRoleAware3DMultitaskModel
from excitationnexus_phase12.samplers import make_dataloader

def test_same_input_graph_exact(paths):
    a,b=raw_paths(paths['raw'],'D1_A31','tier1_pm6_3d'); x=build_graph_from_json(a,b); y=build_graph_from_json(a,b)
    assert torch.equal(x.z,y.z) and torch.equal(x.role,y.role) and torch.equal(x.edge_index,y.edge_index) and torch.equal(x.pos,y.pos)

def test_model_translation_rotation_permutation_invariance(paths):
    a,b=raw_paths(paths['raw'],'D100_A14','tier1_pm6_3d'); g=build_graph_from_json(a,b)
    g.scalar_inputs=torch.zeros(3); model=TinyRoleAware3DMultitaskModel(['p'],3).eval()
    def run(x): return model(Batch.from_data_list([x]))['p'].detach()
    ref=run(g.clone()); t=g.clone(); t.pos=t.pos+torch.tensor([2.,-3.,1.]); assert torch.allclose(ref,run(t),atol=1e-5)
    q,_=torch.linalg.qr(torch.randn(3,3)); r=g.clone(); r.pos=r.pos@q; assert torch.allclose(ref,run(r),atol=1e-5)
    perm=torch.randperm(g.num_nodes); inv=torch.empty_like(perm); inv[perm]=torch.arange(g.num_nodes)
    z=g.clone(); z.z=g.z[perm]; z.pos=g.pos[perm]; z.role=g.role[perm]; z.edge_index=inv[g.edge_index]; z.scalar_inputs=g.scalar_inputs
    assert torch.allclose(ref,run(z),atol=1e-5)

def test_seeded_loader_order_repeats(paths):
    class D(torch.utils.data.Dataset):
        def __len__(self): return 20
        def __getitem__(self,i): return {'x':torch.tensor(i)}
    def order(): return torch.cat([b['x'] for b in make_dataloader(D(),batch_size=4,shuffle=True,seed=9)]).tolist()
    assert order()==order()
