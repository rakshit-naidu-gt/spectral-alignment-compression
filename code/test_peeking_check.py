"""
Reviewer-requested robustness check: the main comparison (baseline_compare.py)
computes each group's five predictors on a 50-example batch, and scores them
against "actual" loss measured as that group's FULL test-set average loss
(850 examples for majority groups, 150 for minority groups) -- so the
predictor's batch is a SUBSET of the examples used to measure actual loss
(5.9% overlap for majority groups, 33.3% for minority groups), not a disjoint
set.

This script redoes the comparison with a clean, fully disjoint split: for
every group, examples [0:50] are used only for the predictors (g_a, H_a,
trace, diagonal) and examples [50:100] are used only to measure actual loss
(before and after compression), with zero overlap for every group including
the minority ones. Severities are NOT recalibrated -- we reuse the exact
severities already calibrated in gate_results.json, so this isolates the
effect of the overlap alone, not a different compression target.
"""
import copy
import json
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from model import ResNet18
from spurious_data import SpuriousCatsDogs
from compress import MECHANISMS
from hvp import _compressible_params, flatten, group_loss, compute_g_and_Hv
import torch.func as func

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_GROUPS = 4
HALF_N = 50  # predictor half and eval half, each 50 examples, fully disjoint
ds_test = SpuriousCatsDogs(root="./data", train=False, seed=100)


def get_group_halves(ds, group_id, half_n):
    idx = np.where(ds.group == group_id)[0]
    pred_idx = idx[:half_n]
    eval_idx = idx[half_n:2 * half_n]
    assert len(set(pred_idx.tolist()) & set(eval_idx.tolist())) == 0
    def batch(indices):
        xs, ys = [], []
        for i in indices:
            x, y, g = ds[i]
            xs.append(x)
            ys.append(y)
        return torch.stack(xs).to(DEVICE), torch.tensor(ys).to(DEVICE)
    return batch(pred_idx), batch(eval_idx)


@torch.no_grad()
def eval_loss_on_batch(model, x, y):
    logits = model(x)
    return F.cross_entropy(logits, y).item()


def diag_hessian_and_grad(model, names, theta_star_params, x, y, probe_seed, n_probes=100):
    def loss_fn(*params):
        return group_loss(model, list(params), names, x, y)
    argnums = tuple(range(len(theta_star_params)))
    grad_fn = func.grad(loss_fn, argnums=argnums)
    g_tuple = grad_fn(*theta_star_params)
    g_flat = flatten(g_tuple)
    total = sum(p.numel() for p in theta_star_params)
    gen = torch.Generator(device="cpu").manual_seed(probe_seed)
    trace_est = 0.0
    diag_est = torch.zeros(total, device=g_flat.device)
    for _ in range(n_probes):
        v_flat = (torch.randint(0, 2, (total,), generator=gen).float() * 2 - 1).to(g_flat.device)
        v_list, i = [], 0
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
    gate_results = json.load(open("gate_results.json"))
    rows = []
    for res in gate_results:
        seed = res["seed"]
        print(f"=== seed {seed} ===", flush=True)
        model = ResNet18(num_classes=2)
        model.load_state_dict(torch.load(f"ckpt_seed{seed}_minfrac15.pt", map_location="cpu", weights_only=True))
        model.to(DEVICE)
        model.eval()

        names, params = _compressible_params(model)
        theta_star = [p.detach().clone() for p in params]
        sd0 = copy.deepcopy(model.state_dict())
        d_total = sum(p.numel() for p in theta_star)

        pred_batches, eval_batches = {}, {}
        for g in range(N_GROUPS):
            pred_batches[g], eval_batches[g] = get_group_halves(ds_test, g, HALF_N)

        group_stats = {}
        base_eval_loss = {}
        for g in range(N_GROUPS):
            x, y = pred_batches[g]
            g_flat, trace_est, diag_est = diag_hessian_and_grad(
                model, names, theta_star, x, y, probe_seed=1000 * seed + g)
            group_stats[g] = (g_flat, trace_est, diag_est)
            xe, ye = eval_batches[g]
            base_eval_loss[g] = eval_loss_on_batch(model, xe, ye)
            print(f"  group {g}: ||g||={g_flat.norm().item():.3f} tr(H)~={trace_est:.3f} "
                  f"base_eval_loss={base_eval_loss[g]:.4f}", flush=True)

        for mech, mres in res["mechanisms"].items():
            severity = mres["severity"]
            sd_c = MECHANISMS[mech](sd0, severity)
            delta_list = [(sd_c[n] - sd0[n]) for n in names]
            delta_flat = flatten(delta_list).to(DEVICE)
            delta_norm_sq = (delta_flat @ delta_flat).item()

            model.load_state_dict(sd_c)
            model.eval()
            for g in range(N_GROUPS):
                xe, ye = eval_batches[g]
                comp_eval_loss = eval_loss_on_batch(model, xe, ye)
                actual_disjoint = comp_eval_loss - base_eval_loss[g]

                g_flat, trace_est, diag_est = group_stats[g]
                g_flat_fresh, hv_flat = compute_g_and_Hv(model, names, theta_star, *pred_batches[g], delta_flat)
                pred_full = (g_flat_fresh @ delta_flat).item() + 0.5 * (delta_flat @ hv_flat).item()
                pred_first = (g_flat @ delta_flat).item()
                pred_diag = pred_first + 0.5 * (diag_est * (delta_flat ** 2)).sum().item()
                pred_isotropic = (g_flat.norm() * delta_flat.norm()).item() + 0.5 * trace_est * delta_norm_sq / d_total

                actual_overlapping = mres["groups"][str(g)]["actual_dloss"]
                pred_full_overlapping = mres["groups"][str(g)]["predicted_dloss"]

                rows.append({
                    "seed": seed, "mech": mech, "group": g,
                    "actual_disjoint": actual_disjoint,
                    "actual_overlapping": actual_overlapping,
                    "pred_full": pred_full, "pred_first": pred_first,
                    "pred_diag": pred_diag, "pred_isotropic": pred_isotropic,
                    "pred_full_overlapping": pred_full_overlapping,
                })
            model.load_state_dict(sd0)
            model.eval()

    json.dump(rows, open("test_peeking_check_results.json", "w"), indent=2)

    print("\n" + "=" * 90)
    print("DISJOINT-SPLIT CHECK: predictor (50 examples) vs actual loss on a DIFFERENT 50 examples")
    print("=" * 90)
    actual = np.array([r["actual_disjoint"] for r in rows])
    for key, label in [("pred_full", "Full directional (HVP)"),
                        ("pred_isotropic", "Liu-style isotropic (trace)"),
                        ("pred_diag", "FairPrune-style diagonal"),
                        ("pred_first", "First-order only")]:
        pred = np.array([r[key] for r in rows])
        rho, p = spearmanr(actual, pred)
        sign_agree = (np.sign(actual) == np.sign(pred)).mean()
        print(f"{label:30s}: Spearman={rho:+.3f} (p={p:.1e})  sign_agreement={sign_agree:.1%}")

    print("\nFlip-case most-harmed-group accuracy (disjoint actual loss):")
    by_seed_mech = {}
    for r in rows:
        by_seed_mech.setdefault((r["seed"], r["mech"]), []).append(r)
    seeds = sorted(set(r["seed"] for r in rows))
    for key, label in [("pred_full", "Full directional"), ("pred_isotropic", "Liu isotropic"),
                        ("pred_diag", "FairPrune diagonal"), ("pred_first", "First-order only")]:
        correct, total = 0, 0
        for seed in seeds:
            mechs = sorted(set(m for s, m in by_seed_mech if s == seed))
            most_harmed_actual = {}
            most_harmed_pred = {}
            for mech in mechs:
                grp = by_seed_mech[(seed, mech)]
                most_harmed_actual[mech] = max(grp, key=lambda r: r["actual_disjoint"])["group"]
                most_harmed_pred[mech] = max(grp, key=lambda r: r[key])["group"]
            if len(set(most_harmed_actual.values())) > 1:
                for mech in mechs:
                    total += 1
                    if most_harmed_pred[mech] == most_harmed_actual[mech]:
                        correct += 1
        if total > 0:
            print(f"  {label:25s}: {correct}/{total} = {correct/total:.1%}")

    # compare actual_disjoint vs actual_overlapping directly: how much does using
    # a different 50-example half change the measured actual loss itself?
    ao = np.array([r["actual_overlapping"] for r in rows])
    ad = np.array([r["actual_disjoint"] for r in rows])
    rho_ao_ad, p_ao_ad = spearmanr(ao, ad)
    print(f"\nActual-loss stability: Spearman(actual_overlapping [full group avg], actual_disjoint [other half]) "
          f"= {rho_ao_ad:+.3f} (p={p_ao_ad:.1e})")
    print(f"(n={len(rows)} total points)")


if __name__ == "__main__":
    main()
