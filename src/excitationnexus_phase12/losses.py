from __future__ import annotations

import torch
import torch.nn.functional as F


def _base_loss(pred, target, kind):
    if kind == "mse": return (pred - target).pow(2)
    if kind == "mae": return (pred - target).abs()
    if kind in {"smooth_l1", "smoothl1"}: return F.smooth_l1_loss(pred, target, reduction="none")
    raise ValueError(kind)


def weighted_masked_multitask_loss(predictions: dict[str, torch.Tensor],
                                   targets: torch.Tensor, target_mask: torch.Tensor,
                                   group_weight: torch.Tensor, task_order,
                                   task_weights: dict[str, float], *,
                                   report_only=(), base_loss="smooth_l1"):
    report_only = set(report_only)
    if targets.shape != target_mask.shape or targets.ndim != 2:
        raise ValueError("targets/mask must have matching [B,T] shape")
    if len(task_order) != targets.shape[1]:
        raise ValueError("task order does not match target tensor")
    w = group_weight.reshape(-1).to(targets)
    per_task, active = {}, []
    for j, task in enumerate(task_order):
        if task in report_only or task not in task_weights:
            continue
        pred = predictions[task].reshape(-1)
        mask = target_mask[:, j].bool()
        denom = (w * mask).sum()
        if not bool(denom > 0):
            continue
        value = (_base_loss(pred, targets[:, j], base_loss) * w * mask).sum() / denom
        per_task[task] = value
        active.append(value * float(task_weights[task]))
    if active:
        total = torch.stack(active).sum()
    else:
        anchor = next(iter(predictions.values()))
        total = anchor.sum() * 0.0
    return total, per_task


def smoke_task_weights(primary, secondary, auxiliary):
    weights = {t: 1.0 / max(len(primary), 1) for t in primary}
    weights.update({t: 0.5 / max(len(secondary), 1) for t in secondary})
    weights.update({t: 0.1 / max(len(auxiliary), 1) for t in auxiliary})
    return weights
