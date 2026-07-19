from __future__ import annotations

import torch
from torch import nn


class TinyRoleAware3DMultitaskModel(nn.Module):
    """Plumbing-only invariant distance model; not a paper architecture."""
    def __init__(self, tasks, scalar_dim: int, hidden_dim: int = 32, num_rbf: int = 16,
                 message_layers: int = 2, cutoff: float = 5.0):
        super().__init__(); self.tasks=tuple(tasks); self.cutoff=cutoff
        self.z_embedding=nn.Embedding(100,hidden_dim); self.role_embedding=nn.Embedding(3,hidden_dim)
        self.register_buffer("rbf_centers",torch.linspace(0,cutoff,num_rbf)); self.rbf_gamma=10.0/cutoff**2
        self.messages=nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim+num_rbf,hidden_dim),nn.SiLU(),
                                                    nn.Linear(hidden_dim,hidden_dim)) for _ in range(message_layers)])
        self.updates=nn.ModuleList([nn.Sequential(nn.Linear(2*hidden_dim,hidden_dim),nn.SiLU())
                                    for _ in range(message_layers)])
        interaction_dim=5*hidden_dim+scalar_dim+2
        self.shared_trunk=nn.Sequential(nn.Linear(interaction_dim,2*hidden_dim),nn.SiLU(),
                                        nn.Linear(2*hidden_dim,hidden_dim),nn.SiLU())
        self.heads=nn.ModuleDict({t:nn.Linear(hidden_dim,1) for t in self.tasks})

    @staticmethod
    def _pool(x,batch,mask,n_graphs):
        out=x.new_zeros((n_graphs,x.shape[1])); count=x.new_zeros((n_graphs,1))
        index=batch[mask]; out.index_add_(0,index,x[mask]); count.index_add_(0,index,x.new_ones((int(mask.sum()),1)))
        return out/count.clamp_min(1.0), count.squeeze(-1).gt(0)

    def forward(self,batch):
        x=self.z_embedding(batch.z)+self.role_embedding(batch.role)
        src,dst=batch.edge_index; distance=(batch.pos[src]-batch.pos[dst]).norm(dim=-1)
        rbf=torch.exp(-self.rbf_gamma*(distance[:,None]-self.rbf_centers[None,:])**2)
        for message,update in zip(self.messages,self.updates):
            msg=message(torch.cat([x[src],rbf],dim=-1)); agg=x.new_zeros(x.shape); agg.index_add_(0,dst,msg)
            x=x+update(torch.cat([x,agg],dim=-1))
        n_graphs=int(batch.num_graphs); hd,has_d=self._pool(x,batch.batch,batch.role.eq(0),n_graphs)
        ha,has_a=self._pool(x,batch.batch,batch.role.eq(1),n_graphs)
        hu,_=self._pool(x,batch.batch,batch.role.eq(2),n_graphs)
        scalars=batch.scalar_inputs.reshape(n_graphs,-1)
        presence=torch.stack([has_d,has_a],dim=-1).to(x.dtype)
        h=self.shared_trunk(torch.cat([hd,ha,(hd-ha).abs(),hd*ha,hu,presence,scalars],dim=-1))
        return {task:self.heads[task](h).squeeze(-1) for task in self.tasks}
