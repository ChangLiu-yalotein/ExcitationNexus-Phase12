import json
import torch
import pytest
from excitationnexus_phase12.graph_builder import build_graph_from_json,raw_paths

def test_pm6_and_dft_parse_real_roles(paths):
    for view in ('tier1_pm6_3d','tier2_dft_3d'):
        a,b=raw_paths(paths['raw'],'D1_A31',view); g=build_graph_from_json(a,b)
        assert g.z.shape[0]==g.pos.shape[0] and g.donor_mask.any() and g.acceptor_mask.any()
        assert g.edge_index.shape[0]==2 and torch.isfinite(g.pos).all()

def test_missing_role_fails(tmp_path):
    s={'atoms':[{'index':1,'atomic_num':6,'element':'C','coords':[0,0,0],'type':'donor'},
                {'index':2,'atomic_num':6,'element':'C','coords':[1,0,0],'type':'acceptor'}]}
    a=tmp_path/'a.json'; b=tmp_path/'b.json'; a.write_text(json.dumps(s)); b.write_text(json.dumps({'atom_origins':['donor']}))
    with pytest.raises(ValueError): build_graph_from_json(a,b)
