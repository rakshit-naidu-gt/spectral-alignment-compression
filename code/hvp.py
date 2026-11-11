"""Exact per-group gradient and Hessian-vector product via double backward.

Theory: R(a) ~= g_a^T delta + 0.5 * delta^T H_a delta, where g_a, H_a are the
group's gradient/Hessian of TEST loss at the pre-compression weights theta*,
and delta = theta_compressed - theta*. We need exact H_a @ delta (a vector),
not the full Hessian -- computed via double backward (Pearlmutter's trick),
restricted to the compressible (conv/linear weight) parameters only, to match
what the compression mechanisms actually perturb.
"""
import torch
import torch.nn.functional as F


def _compressible_params(model):
    names, params = [], []
    for name, p in model.named_parameters():
        if name.endswith("weight") and p.dim() >= 2:
            names.append(name)
            params.append(p)
    return names, params


def group_loss(model, params, names, x, y):
    """Functional forward pass using the given (possibly perturbed) params,
    via torch.func for a clean functional HVP."""
    param_dict = dict(zip(names, params))
    full_state = dict(model.state_dict())
    full_state.update(param_dict)
    logits = torch.func.functional_call(model, full_state, (x,))
    return F.cross_entropy(logits, y)


def flatten(params):
    return torch.cat([p.reshape(-1) for p in params])


def unflatten(flat, like_params):
    out, i = [], 0
    for p in like_params:
        n = p.numel()
        out.append(flat[i:i + n].reshape(p.shape))
        i += n
    return out


@torch.no_grad()
def _state_to_param_list(state_dict, names):
    return [state_dict[n].clone() for n in names]


def compute_g_and_Hv(model, names, theta_star_params, x, y, delta_flat):
    """Returns (g_flat, Hv_flat) -- gradient and Hessian-vector product of
    group test loss at theta_star, restricted to compressible params, using
    torch.func.grad + jvp (forward-over-reverse) for the HVP."""
    delta_list = unflatten(delta_flat, theta_star_params)

    def loss_fn(*params):
        return group_loss(model, list(params), names, x, y)

    argnums = tuple(range(len(theta_star_params)))
    grad_fn = torch.func.grad(loss_fn, argnums=argnums)
    g_tuple = grad_fn(*theta_star_params)
    g_flat = flatten(g_tuple)

    # Hessian-vector product via forward-mode-over-reverse-mode autodiff (jvp of grad)
    _, hv_tuple = torch.func.jvp(grad_fn, tuple(theta_star_params), tuple(delta_list))
    hv_flat = flatten(hv_tuple)
    return g_flat.detach(), hv_flat.detach()
