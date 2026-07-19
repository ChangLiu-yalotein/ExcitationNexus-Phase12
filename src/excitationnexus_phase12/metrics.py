from __future__ import annotations

import numpy as np
from scipy.stats import kendalltau, spearmanr


def _safe_r2(y, p, weights=None):
    y=np.asarray(y,float); p=np.asarray(p,float)
    w=np.ones_like(y) if weights is None else np.asarray(weights,float)
    mean=np.sum(w*y)/np.sum(w); denom=np.sum(w*(y-mean)**2)
    if denom <= 1e-15:
        return {"value": float("nan"), "reason": "CONSTANT_TARGET"}
    return {"value": float(1-np.sum(w*(y-p)**2)/denom), "reason": None}


def regression_metrics(y_true, y_pred, structure_groups, donor_groups=None, acceptor_groups=None):
    y=np.asarray(y_true,float); p=np.asarray(y_pred,float); groups=np.asarray(structure_groups)
    valid=np.isfinite(y)&np.isfinite(p); y,p,groups=y[valid],p[valid],groups[valid]
    if not len(y): return {"valid_count":0}
    err=p-y; unique=np.unique(groups)
    group_abs=np.array([np.mean(np.abs(err[groups==g])) for g in unique])
    group_sq=np.array([np.mean(err[groups==g]**2) for g in unique])
    group_weights=np.array([1.0/np.sum(groups==g) for g in groups])
    out={
      "valid_count":int(len(y)), "record_mae":float(np.mean(np.abs(err))),
      "record_rmse":float(np.sqrt(np.mean(err**2))), "record_r2":_safe_r2(y,p),
      "group_macro_mae":float(np.mean(group_abs)),
      "group_macro_rmse":float(np.sqrt(np.mean(group_sq))),
      "group_macro_r2":_safe_r2(y,p,group_weights),
      "spearman":float(spearmanr(y,p).statistic) if len(y)>1 else float("nan"),
      "kendall":float(kendalltau(y,p).statistic) if len(y)>1 else float("nan"),
      "worst_decile_error":float(np.mean(np.sort(np.abs(err))[-max(1,int(np.ceil(.1*len(err)))):])),
      "worst_group_mae":float(np.max(group_abs)),
    }
    for label, source in (("donor",donor_groups),("acceptor",acceptor_groups)):
        if source is not None:
            arr=np.asarray(source)[valid]
            vals={str(g):float(np.mean(np.abs(err[arr==g]))) for g in np.unique(arr)}
            out[f"per_{label}_group_mae"]=vals
    return out


def multitask_metrics(targets, predictions, masks, task_order, structure_groups,
                      donor_groups=None, acceptor_groups=None):
    result={}
    for j,task in enumerate(task_order):
        m=np.asarray(masks)[:,j].astype(bool)
        result[task]=regression_metrics(np.asarray(targets)[m,j],np.asarray(predictions)[m,j],
            np.asarray(structure_groups)[m], None if donor_groups is None else np.asarray(donor_groups)[m],
            None if acceptor_groups is None else np.asarray(acceptor_groups)[m])
    return result
