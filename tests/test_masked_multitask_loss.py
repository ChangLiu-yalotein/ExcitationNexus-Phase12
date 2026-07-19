import torch
from excitationnexus_phase12.losses import weighted_masked_multitask_loss

def test_partial_mask_hand_calculation():
    pred={'p':torch.tensor([1.,5.]),'a':torch.tensor([9.,9.])}; target=torch.tensor([[0.,0.],[1.,0.]])
    mask=torch.tensor([[1,0],[1,0]],dtype=torch.bool); w=torch.tensor([.5,.5])
    total,per=weighted_masked_multitask_loss(pred,target,mask,w,['p','a'],{'p':1,'a':1},base_loss='mae')
    assert torch.isclose(total,torch.tensor(2.5)) and set(per)=={'p'}

def test_report_only_never_changes_total():
    pred={'p':torch.tensor([1.]),'report':torch.tensor([999.])}; target=torch.tensor([[0.,-999.]])
    mask=torch.ones_like(target,dtype=torch.bool); w=torch.ones(1)
    a,_=weighted_masked_multitask_loss(pred,target,mask,w,['p','report'],{'p':1,'report':100},report_only=['report'],base_loss='mae')
    assert a.item()==1

def test_all_empty_masks_safe_and_differentiable():
    p=torch.tensor([1.],requires_grad=True); total,per=weighted_masked_multitask_loss({'a':p},torch.zeros(1,1),torch.zeros(1,1,dtype=torch.bool),torch.ones(1),['a'],{'a':1})
    total.backward(); assert total.item()==0 and per=={} and torch.isfinite(p.grad).all()
