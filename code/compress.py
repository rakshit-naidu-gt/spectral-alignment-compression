"""Four compression mechanisms, each producing a new state_dict (theta_compressed)
from a trained model's state_dict, at a tunable severity. For the first three
(unstructured/channel/svd), severity is a fraction in [0, 1): the global
fraction of components removed. For quantize, severity is instead a
quantization step size in weight units (not a fraction), since a step size is
what the calibration grid search in run_gate.py actually needs to sweep to
reach a target relative perturbation norm."""
import copy
import torch


def _compressible_keys(state_dict):
    """conv/linear weight tensors only (skip biases, BN params)."""
    return [k for k in state_dict if k.endswith("weight") and state_dict[k].dim() >= 2]


def unstructured_prune(state_dict, fraction):
    sd = copy.deepcopy(state_dict)
    keys = _compressible_keys(sd)
    all_vals = torch.cat([sd[k].abs().flatten() for k in keys])
    if fraction <= 0:
        return sd
    k = int(fraction * all_vals.numel())
    if k == 0:
        return sd
    thresh = torch.kthvalue(all_vals, k).values
    for key in keys:
        mask = sd[key].abs() > thresh
        sd[key] = sd[key] * mask
    return sd


def channel_prune(state_dict, fraction):
    """Structured: zero whole output channels (rows of the weight tensor) with
    smallest L2 norm, ranked GLOBALLY across all layers (not independently per
    layer). A fixed per-layer fraction would round down to zero prune-count on
    any layer with fewer channels than roughly 1/fraction, collapsing severity
    onto whichever single layer has the most channels; a global threshold
    avoids that."""
    sd = copy.deepcopy(state_dict)
    keys = _compressible_keys(sd)
    channel_norms = {}
    all_norms = []
    for key in keys:
        w = sd[key]
        out_ch = w.shape[0]
        if out_ch <= 1:
            continue
        norms = w.view(out_ch, -1).norm(dim=1)
        channel_norms[key] = norms
        all_norms.append(norms)
    if fraction <= 0 or not all_norms:
        return sd
    all_norms_flat = torch.cat(all_norms)
    k = int(fraction * all_norms_flat.numel())
    if k == 0:
        return sd
    thresh = torch.kthvalue(all_norms_flat, k).values
    for key, norms in channel_norms.items():
        mask_1d = (norms > thresh).float()
        mask = mask_1d.view(-1, *([1] * (sd[key].dim() - 1)))
        sd[key] = sd[key] * mask
    return sd


def svd_truncate(state_dict, fraction):
    """Low-rank truncation: reshape each weight to 2D, discard the GLOBALLY
    smallest singular values across all layers (not a fixed per-layer rank
    fraction, for the same reason as channel_prune above)."""
    sd = copy.deepcopy(state_dict)
    keys = _compressible_keys(sd)
    svd_cache = {}
    all_sv = []
    for key in keys:
        w = sd[key]
        orig_shape = w.shape
        w2d = w.reshape(orig_shape[0], -1).float()
        if min(w2d.shape) <= 1:
            continue
        try:
            U, S, Vh = torch.linalg.svd(w2d, full_matrices=False)
        except Exception:
            continue
        svd_cache[key] = (U, S, Vh, orig_shape, w.dtype)
        all_sv.append(S)
    if fraction <= 0 or not all_sv:
        return sd
    all_sv_flat = torch.cat(all_sv)
    k = int(fraction * all_sv_flat.numel())
    if k == 0:
        return sd
    thresh = torch.kthvalue(all_sv_flat, k).values
    for key, (U, S, Vh, orig_shape, dtype) in svd_cache.items():
        S_trunc = S.clone()
        S_trunc[S_trunc <= thresh] = 0.0
        if torch.all(S_trunc == 0):
            # never fully zero a layer's spectrum -- keep its top singular value
            top_idx = torch.argmax(S)
            S_trunc[top_idx] = S[top_idx]
        w2d_approx = (U * S_trunc.unsqueeze(0)) @ Vh
        sd[key] = w2d_approx.reshape(orig_shape).to(dtype)
    return sd


def quantize(state_dict, step):
    """Uniform scalar quantization: round every compressible weight to the
    nearest multiple of `step`, applied globally with the same step size
    across all layers (no layer-wise scaling), matching the "no
    mechanism-specific tuning beyond the single severity knob" treatment of
    the other three mechanisms. For weights whose values are not aligned to
    the quantization grid, the rounding error on each weight is close to
    uniform on [-step/2, step/2] and approximately independent across weights
    (the standard quantization-noise heuristic; it is not exact, since
    deterministic rounding -- unlike dithered quantization -- correlates
    weakly with the signal itself, but holds well when, as here, weight
    magnitudes span many multiples of the step size). This is the mechanism
    Section~\\ref{sec:isotropic-special-case} (see app:quantization) predicts
    should be well-described by the isotropic baseline, unlike pruning or
    low-rank truncation, because the resulting delta is close to isotropic
    noise rather than a structured, direction-specific perturbation."""
    sd = copy.deepcopy(state_dict)
    keys = _compressible_keys(sd)
    if step <= 0:
        return sd
    for key in keys:
        w = sd[key]
        sd[key] = torch.round(w / step) * step
    return sd


MECHANISMS = {
    "unstructured": unstructured_prune,
    "channel": channel_prune,
    "svd": svd_truncate,
    "quantize": quantize,
}
