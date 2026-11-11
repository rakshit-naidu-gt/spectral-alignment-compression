"""
Strategist-flagged critical missing experiment: does the full directional
quadratic predictor (g_a^T delta + 0.5 delta^T H_a delta) actually beat the
isotropic/diagonal baselines already published, on the SAME 36 data points
already collected (reusing cached checkpoints -- no retraining)?

Baselines:
  1. Norm-bound "most-harmed" accuracy (the real baseline, not 0% -- within each
     flip set, how often does the norm bound's single (mechanism-independent)
     ranking happen to match the actual most-harmed group for that mechanism?)
  2. Liu et al. ICML 2025 isotropic predictor: ||g_a|| * ||delta|| + 0.5 * tr(H_a) * ||delta||^2 / d
     (i.e. assumes delta is "isotropic" and only the trace of H_a matters, not direction)
  3. Diagonal-Hessian saliency (FairPrune-style): g_a^T delta + 0.5 * sum_i H_a[i,i] * delta[i]^2
     (per-coordinate curvature x per-coordinate delta^2, ignores off-diagonal/direction coupling)
  4. Full directional predictor (already computed): g_a^T delta + 0.5 * delta^T H_a delta (exact HVP)

Compare all four on: (a) pooled Spearman vs actual per-group dloss, (b) accuracy
at calling the correct most-harmed group within each of the 9 flip cases.
"""
import copy
import json
import numpy as np
import torch
from scipy.stats import spearmanr

from model import ResNet18
from spurious_data import SpuriousCatsDogs, GROUP_NAMES
from compress import MECHANISMS
from hvp import _compressible_params, flatten, group_loss
import torch.func as func

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_GROUPS = 4
GROUP_EVAL_N = 50
ds_test = SpuriousCatsDogs(root="./data", train=False, seed=100)


def get_group_batch(ds, group_id, n):
    xs, ys = [], []
    idx = np.where(ds.group == group_id)[0]
    idx = idx[:n]
    for i in idx:
        x, y, g = ds[i]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs).to(DEVICE), torch.tensor(ys).to(DEVICE)


def diag_hessian_and_grad(model, names, theta_star_params, x, y, probe_seed):
    """Diagonal of the Hessian via Hutchinson-free exact diagonal (expensive in
    general, but feasible here since we only need it once per group per seed:
    use the standard trick diag(H) ~= HVP with each basis vector is too slow for
    11M params, so instead use the Hutchinson estimator with Rademacher probes
    for both the trace (for Liu's isotropic baseline) and the diagonal (for the
    FairPrune-style baseline), which is the standard scalable approach.
    probe_seed makes the Rademacher probes reproducible across reruns (otherwise
    tr(H)/diag(H) -- and anything derived from them -- differ run to run)."""
    def loss_fn(*params):
        return group_loss(model, list(params), names, x, y)

    argnums = tuple(range(len(theta_star_params)))
    grad_fn = func.grad(loss_fn, argnums=argnums)
    g_tuple = grad_fn(*theta_star_params)
    g_flat = flatten(g_tuple)

    n_probes = 100
    total = sum(p.numel() for p in theta_star_params)
    gen = torch.Generator(device="cpu").manual_seed(probe_seed)  # CPU generator: MPS backend
    trace_est = 0.0                                              # doesn't support generator=
    diag_est = torch.zeros(total, device=g_flat.device)
    for _ in range(n_probes):
        v_flat = (torch.randint(0, 2, (total,), generator=gen).float() * 2 - 1).to(g_flat.device)  # Rademacher
        v_list = []
        i = 0
        for p in theta_star_params:
            n = p.numel()
            v_list.append(v_flat[i:i + n].reshape(p.shape))
            i += n
        _, hv_tuple = func.jvp(grad_fn, tuple(theta_star_params), tuple(v_list))
        hv_flat = flatten(hv_tuple)
        trace_est += (v_flat * hv_flat).sum().item()
        diag_est += v_flat * hv_flat
    trace_est /= n_probes
    diag_est /= n_probes
    return g_flat.detach(), trace_est, diag_est.detach()


def main():
    results = json.load(open("gate_results.json"))
    rows = []

    for res in results:
        seed = res["seed"]
        print(f"=== seed {seed}: recomputing baselines from checkpoint ===", flush=True)
        model = ResNet18(num_classes=2)
        model.load_state_dict(torch.load(f"ckpt_seed{seed}_minfrac15.pt", map_location="cpu", weights_only=True))
        model.to(DEVICE)
        model.eval()

        names, params = _compressible_params(model)
        theta_star = [p.detach().clone() for p in params]
        sd0 = copy.deepcopy(model.state_dict())
        d_total = sum(p.numel() for p in theta_star)

        group_batches = {g: get_group_batch(ds_test, g, GROUP_EVAL_N) for g in range(N_GROUPS)}

        # precompute g_a, trace(H_a), diag(H_a) per group ONCE per seed (model doesn't change)
        group_stats = {}
        for g in range(N_GROUPS):
            x, y = group_batches[g]
            g_flat, trace_est, diag_est = diag_hessian_and_grad(
                model, names, theta_star, x, y, probe_seed=1000 * seed + g)
            group_stats[g] = (g_flat, trace_est, diag_est)
            print(f"  group {g}: ||g||={g_flat.norm().item():.3f} tr(H)~={trace_est:.3f}", flush=True)

        # norm-bound's prediction is MECHANISM-INDEPENDENT: whichever group has
        # the largest ||g_a|| is predicted "most harmed" under every mechanism,
        # since the norm bound ||g_a||*||delta|| scales every group by the same
        # ||delta|| for a given mechanism.
        norm_bound_pred_group = max(range(N_GROUPS), key=lambda g: group_stats[g][0].norm().item())

        for mech, mres in res["mechanisms"].items():
            severity = mres["severity"]
            sd_c = MECHANISMS[mech](sd0, severity)
            delta_list = [(sd_c[n] - sd0[n]) for n in names]
            delta_flat = flatten(delta_list).to(DEVICE)
            delta_norm_sq = (delta_flat @ delta_flat).item()

            for g in range(N_GROUPS):
                g_flat, trace_est, diag_est = group_stats[g]
                actual = mres["groups"][str(g)]["actual_dloss"]
                pred_full = mres["groups"][str(g)]["predicted_dloss"]

                pred_isotropic = (g_flat.norm() * delta_flat.norm()).item() + 0.5 * trace_est * delta_norm_sq / d_total
                pred_isotropic_curv_only = 0.5 * trace_est * delta_norm_sq / d_total
                pred_diag = (g_flat @ delta_flat).item() + 0.5 * (diag_est * (delta_flat ** 2)).sum().item()
                pred_first = (g_flat @ delta_flat).item()

                rows.append({
                    "seed": seed, "mech": mech, "group": g,
                    "actual": actual, "pred_full": pred_full,
                    "pred_isotropic": pred_isotropic,
                    "pred_isotropic_curv_only": pred_isotropic_curv_only,
                    "pred_diag": pred_diag,
                    "pred_first": pred_first,
                    "norm_bound_pred_group": norm_bound_pred_group,
                })

    json.dump(rows, open("baseline_compare_results.json", "w"), indent=2)

    print("\n" + "=" * 78)
    print("BASELINE COMPARISON")
    print("=" * 78)
    actual = np.array([r["actual"] for r in rows])
    for key, label in [("pred_full", "Full directional (HVP)"),
                        ("pred_isotropic", "Liu-style isotropic (trace)"),
                        ("pred_isotropic_curv_only", "Isotropic curvature-ONLY"),
                        ("pred_diag", "FairPrune-style diagonal"),
                        ("pred_first", "First-order only")]:
        pred = np.array([r[key] for r in rows])
        rho, p = spearmanr(actual, pred)
        sign_agree = (np.sign(actual) == np.sign(pred)).mean()
        print(f"{label:30s}: Spearman={rho:+.3f} (p={p:.1e})  sign_agreement={sign_agree:.1%}")

    # flip-case accuracy for each predictor
    print("\nFlip-case most-harmed-group accuracy:")
    by_seed_mech = {}
    for r in rows:
        by_seed_mech.setdefault((r["seed"], r["mech"]), []).append(r)

    seeds = sorted(set(r["seed"] for r in rows))
    for key, label in [("pred_full", "Full directional"), ("pred_isotropic", "Liu isotropic"),
                        ("pred_isotropic_curv_only", "Isotropic curvature-ONLY"),
                        ("pred_diag", "FairPrune diagonal"), ("pred_first", "First-order only")]:
        correct, total = 0, 0
        for seed in seeds:
            mechs = sorted(set(m for s, m in by_seed_mech if s == seed))
            most_harmed_actual = {}
            most_harmed_pred = {}
            for mech in mechs:
                grp = by_seed_mech[(seed, mech)]
                most_harmed_actual[mech] = max(grp, key=lambda r: r["actual"])["group"]
                most_harmed_pred[mech] = max(grp, key=lambda r: r[key])["group"]
            if len(set(most_harmed_actual.values())) > 1:  # flip seed
                for mech in mechs:
                    total += 1
                    if most_harmed_pred[mech] == most_harmed_actual[mech]:
                        correct += 1
        if total > 0:
            print(f"  {label:25s}: {correct}/{total} = {correct/total:.1%}")

    # the REAL norm-bound baseline: a single, mechanism-independent prediction
    # (whichever group has max ||g_a||) checked against each mechanism's actual
    # most-harmed group, restricted to flip seeds.
    correct, total = 0, 0
    for seed in seeds:
        mechs = sorted(set(m for s, m in by_seed_mech if s == seed))
        most_harmed_actual = {}
        for mech in mechs:
            grp = by_seed_mech[(seed, mech)]
            most_harmed_actual[mech] = max(grp, key=lambda r: r["actual"])["group"]
        if len(set(most_harmed_actual.values())) > 1:
            nb_pred = by_seed_mech[(seed, mechs[0])][0]["norm_bound_pred_group"]
            for mech in mechs:
                total += 1
                if nb_pred == most_harmed_actual[mech]:
                    correct += 1
    if total > 0:
        print(f"  {'Norm bound (real)':25s}: {correct}/{total} = {correct/total:.1%}  "
              f"(single mechanism-independent guess per seed)")

    print(f"\n(n={len(rows)} total points)")


if __name__ == "__main__":
    main()
