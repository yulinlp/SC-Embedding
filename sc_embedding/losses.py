"""Within-pool retrieval and selective cross-view self-distillation."""
import math

import torch
from torch.nn import functional as F


def ranking_loss(scores, positives, temperature=.05):
    positives = sorted(set(positives))
    if scores.ndim != 1 or temperature <= 0 or not positives:
        raise ValueError("Expected scores, positive indices, and a positive temperature")
    if any(type(i) is not int or not 0 <= i < len(scores) for i in positives):
        raise ValueError("Positive indices are out of range")
    negatives = [i for i in range(len(scores)) if i not in positives]
    if not negatives:
        raise ValueError("At least one negative is required")
    logits = scores.float() / temperature
    return F.softplus(torch.logsumexp(logits[negatives], 0) - logits[positives]).mean() * math.log(len(negatives) + 1)


def retrieval_loss(documents, query, positives, temperature=.05):
    return ranking_loss(F.normalize(documents.float(), dim=-1) @
                        F.normalize(query.float(), dim=-1), positives, temperature)


def selective_loss(documents, query, positives, view_scores, *, step,
                   temperature=.05, kd_temperature=.05, ramp_steps=200,
                   reliability_gap=.01, improvement_gap=.01,
                   gradient_ratio=.2, alpha_cap=1., epsilon=1e-12):
    """Calibrate at embedding leaves; targets and weights are detached.

    Every target view is evaluated with the current model without gradients.
    The complete ordered candidate pool is shared by all views.
    """
    if not documents.is_leaf or not documents.requires_grad:
        raise ValueError("Gradient calibration requires document embedding leaves")
    if not {"full", "bare"} <= set(view_scores):
        raise ValueError("Full and empty history views are required")
    base = retrieval_loss(documents, query.detach(), positives, temperature)
    scores = F.normalize(documents.float(), dim=-1) @ F.normalize(query.detach().float(), dim=-1)
    with torch.no_grad():
        fixed = {name: value.detach().float().to(scores.device) for name, value in view_scores.items()}
        if any(value.shape != scores.shape for value in fixed.values()):
            raise ValueError("History views must retain the same candidate pool")
        losses = {name: ranking_loss(value, positives, temperature) for name, value in fixed.items()}
        ordered = sorted(losses, key=lambda name: (float(losses[name]), name))
        best, runner_up = ordered[:2]
        gap = float(losses[runner_up] - losses[best])
        improvement = float(base.detach() - losses[best])
        target = torch.softmax(fixed[best] / kd_temperature, dim=-1)
    kl = F.kl_div(F.log_softmax(scores / kd_temperature, dim=-1), target, reduction="sum")
    g_ret = torch.autograd.grad(base, documents, retain_graph=True)[0]
    g_kd = torch.autograd.grad(kl, documents, retain_graph=True)[0]
    ret_norm, kd_norm = float(g_ret.norm()), float(g_kd.norm())
    ramp = min(1., step / max(1, ramp_steps))
    active = gap > reliability_gap and improvement > improvement_gap and ret_norm > 0 and kd_norm > 0
    alpha = ramp * min(alpha_cap, gradient_ratio * ret_norm / (kd_norm + epsilon)) if active else 0.
    total = base + alpha * kl if alpha else base
    if not bool(torch.isfinite(total)):
        raise FloatingPointError("Nonfinite training loss")
    return total, dict(loss=float(total.detach()), retrieval_loss=float(base.detach()),
                      kl=float(kl.detach()), best_view=best, alpha=alpha,
                      reliability_gap=gap, improvement=improvement,
                      kd_to_retrieval_gradient_ratio=alpha * kd_norm / ret_norm if ret_norm else 0.)
