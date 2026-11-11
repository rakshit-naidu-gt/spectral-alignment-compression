"""
Severity sweep: map the second-order theory's "validity frontier" directly,
by holding the model fixed (cached checkpoints, no overfitting confound) and
varying compression severity (target ||delta||/||theta*||) from very mild to
aggressive, at each level recomputing all four predictors (full HVP, diagonal,
first-order-only, isotropic) against actual per-group test-loss change.

This complements the Waterbirds early-stopping ablation (which varied model
generalization at FIXED ~3% severity) with the other axis the paper's theory
depends on: Taylor-expansion truncation error growing with ||delta|| itself,
at FIXED (well-generalizing) models.
"""
import copy
import gc
import json
import numpy as np
import torch
from scipy.stats import spearmanr

from model import ResNet18
from spurious_data import SpuriousCatsDogs, GROUP_NAMES
from train import eval_per_group, overall_eval
from compress import MECHANISMS
from hvp import _compressible_params, flatten, group_loss, compute_g_and_Hv
import torch.func as func

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEEDS = [0, 1, 2]
N_GROUPS = 4
GROUP_EVAL_N = 50
SEVERITY_TARGETS = [0.01, 0.03, 0.10, 0.25]
N_PROBES = 50  # trimmed from the main experiment's 100 for sweep speed; noted explicitly when reporting

ds_test = SpuriousCatsDogs(root="./data", train=False, seed=100)


def get_group_batch(ds, group_id, n):
    xs, ys = [], []
    idx = np.where(ds.group == group_id)[0][:n]
    for i in idx:
        x, y, g = ds[i]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs).to(DEVICE), torch.tensor(ys).to(DEVICE)


def weight_norm(sd, names):
    return flatten([sd[n] for n in names]).norm().item()


def delta_norm_for_severity(sd0, names, mechanism_fn, severity, w0_norm):
    sd_c = mechanism_fn(sd0, severity)
    d = flatten([sd_c[n] - sd0[n] for n in names]).norm().item()
    del sd_c
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    return d / w0_norm


def calibrate_severity_by_delta_norm(sd0, names, mechanism_fn, target_rel_delta, w0_norm):
    candidates = []
    grid = np.concatenate([np.geomspace(1e-4, 0.05, 12), np.linspace(0.06, 0.95, 12)])
    for sev in grid:
        rel_d = delta_norm_for_severity(sd0, names, mechanism_fn, float(sev), w0_norm)
        candidates.append((float(sev), rel_d))

    candidates.sort(key=lambda c: c[0])
    lo_c, hi_c = None, None
    for i in range(len(candidates) - 1):
        a, b = candidates[i], candidates[i + 1]
        if a[1] <= target_rel_delta <= b[1] or b[1] <= target_rel_delta <= a[1]:
            lo_c, hi_c = a, b
            break

    if lo_c is not None:
        lo, hi = lo_c[0], hi_c[0]
        for _ in range(12):
            mid = (lo + hi) / 2
            rel_d = delta_norm_for_severity(sd0, names, mechanism_fn, mid, w0_norm)
            candidates.append((mid, rel_d))
            if rel_d < target_rel_delta:
                lo = mid
            else:
                hi = mid

    best_sev = min(candidates, key=lambda c: abs(c[1] - target_rel_delta))[0]
    achieved = delta_norm_for_severity(sd0, names, mechanism_fn, best_sev, w0_norm)
    return best_sev, achieved


def diag_hessian_and_grad(model, names, theta_star_params, x, y, probe_seed, n_probes=N_PROBES):
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
    rows = []
    import os
    if os.path.exists("severity_sweep_results.json"):
        rows = json.load(open("severity_sweep_results.json"))
        print(f"resuming: loaded {len(rows)} rows from a prior checkpoint", flush=True)
    completed_seeds = set(r["seed"] for r in rows)

    for seed in SEEDS:
        if seed in completed_seeds:
            print(f"\n=== SEED {seed} already completed (checkpoint), skipping ===", flush=True)
            continue
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
        print(f"\n=== SEED {seed} ===", flush=True)
        model = ResNet18(num_classes=2)
        model.load_state_dict(torch.load(f"ckpt_seed{seed}_minfrac15.pt", map_location="cpu", weights_only=True))
        model.to(DEVICE)
        model.eval()

        names, params = _compressible_params(model)
        theta_star = [p.detach().clone() for p in params]
        sd0 = copy.deepcopy(model.state_dict())
        d_total = sum(p.numel() for p in theta_star)
        w0_norm = weight_norm(sd0, names)

        base_group = eval_per_group(model, ds_test, DEVICE, n_groups=N_GROUPS)
        group_batches = {g: get_group_batch(ds_test, g, GROUP_EVAL_N) for g in range(N_GROUPS)}

        group_stats = {}
        for g in range(N_GROUPS):
            x, y = group_batches[g]
            g_flat, trace_est, diag_est = diag_hessian_and_grad(
                model, names, theta_star, x, y, probe_seed=1000 * seed + g)
            group_stats[g] = (g_flat, trace_est, diag_est)

        for target in SEVERITY_TARGETS:
            for mech_name, mech_fn in MECHANISMS.items():
                severity, achieved_rel_delta = calibrate_severity_by_delta_norm(
                    sd0, names, mech_fn, target, w0_norm)
                sd_c = mech_fn(sd0, severity)
                model.load_state_dict(sd_c)
                model.eval()
                comp_group = eval_per_group(model, ds_test, DEVICE, n_groups=N_GROUPS)
                model.load_state_dict(sd0)
                model.eval()

                delta_list = [(sd_c[n] - sd0[n]) for n in names]
                delta_flat = flatten(delta_list).to(DEVICE)
                delta_norm_sq = (delta_flat @ delta_flat).item()

                for g in range(N_GROUPS):
                    g_flat, trace_est, diag_est = group_stats[g]
                    x, y = group_batches[g]
                    g_flat_fresh, hv_flat = compute_g_and_Hv(model, names, theta_star, x, y, delta_flat)
                    pred_full = (g_flat_fresh @ delta_flat).item() + 0.5 * (delta_flat @ hv_flat).item()
                    pred_first = (g_flat @ delta_flat).item()
                    pred_diag = pred_first + 0.5 * (diag_est * (delta_flat ** 2)).sum().item()
                    pred_isotropic = (g_flat.norm() * delta_flat.norm()).item() + 0.5 * trace_est * delta_norm_sq / d_total
                    actual = comp_group[g]["loss"] - base_group[g]["loss"]

                    rows.append({
                        "seed": seed, "target_rel_delta": target, "mech": mech_name,
                        "severity": severity, "achieved_rel_delta": achieved_rel_delta,
                        "group": g, "actual": actual,
                        "pred_full": pred_full, "pred_first": pred_first,
                        "pred_diag": pred_diag, "pred_isotropic": pred_isotropic,
                    })
                    del g_flat_fresh, hv_flat
                    if torch.backends.mps.is_available():
                        torch.mps.empty_cache()
                del sd_c, delta_list, delta_flat
                gc.collect()
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
            mean_ard = np.mean([r["achieved_rel_delta"] for r in rows
                                 if r["seed"] == seed and r["target_rel_delta"] == target])
            print(f"  target={target:.3f} seed={seed}: mean achieved_rel_delta={mean_ard:.4f}", flush=True)
        json.dump(rows, open("severity_sweep_results.json", "w"), indent=2)
        print(f"  [checkpoint] saved {len(rows)} rows after seed {seed}", flush=True)
        del model, theta_star, sd0, group_stats, group_batches, base_group
        gc.collect()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

    json.dump(rows, open("severity_sweep_results.json", "w"), indent=2)

    print("\n" + "=" * 90)
    print("SEVERITY SWEEP: predictor quality vs. compression severity")
    print("=" * 90)
    header = f"{'target':>7s} {'achiev.':>8s} {'n':>4s} {'flips':>6s} " \
             f"{'rho_full':>9s} {'rho_first':>10s} {'rho_diag':>9s} {'rho_iso':>8s} " \
             f"{'acc_full':>9s} {'acc_iso':>8s}"
    print(header)

    summary = []
    for target in SEVERITY_TARGETS:
        level_rows = [r for r in rows if r["target_rel_delta"] == target]
        achieved = np.mean([r["achieved_rel_delta"] for r in level_rows])
        actual = np.array([r["actual"] for r in level_rows])

        rhos = {}
        for key in ["pred_full", "pred_first", "pred_diag", "pred_isotropic"]:
            pred = np.array([r[key] for r in level_rows])
            rho, _ = spearmanr(actual, pred)
            rhos[key] = rho

        by_seed_mech = {}
        for r in level_rows:
            by_seed_mech.setdefault((r["seed"], r["mech"]), []).append(r)
        seeds_here = sorted(set(r["seed"] for r in level_rows))
        flip_seeds = []
        for seed in seeds_here:
            mechs = sorted(set(m for s, m in by_seed_mech if s == seed))
            most_harmed = {m: max(by_seed_mech[(seed, m)], key=lambda r: r["actual"])["group"] for m in mechs}
            if len(set(most_harmed.values())) > 1:
                flip_seeds.append((seed, most_harmed, mechs))

        accs = {}
        for key in ["pred_full", "pred_isotropic"]:
            correct, total = 0, 0
            for seed, most_harmed, mechs in flip_seeds:
                for mech in mechs:
                    grp = by_seed_mech[(seed, mech)]
                    pred_g = max(grp, key=lambda r: r[key])["group"]
                    total += 1
                    if pred_g == most_harmed[mech]:
                        correct += 1
            accs[key] = (correct / total) if total > 0 else float("nan")
            accs[key + "_n"] = total

        print(f"{target:7.3f} {achieved:8.4f} {len(level_rows):4d} {len(flip_seeds):6d} "
              f"{rhos['pred_full']:9.3f} {rhos['pred_first']:10.3f} {rhos['pred_diag']:9.3f} {rhos['pred_isotropic']:8.3f} "
              f"{accs['pred_full']:9.1%} {accs['pred_isotropic']:8.1%}")

        summary.append({
            "target_rel_delta": target, "achieved_rel_delta": achieved, "n": len(level_rows),
            "n_flip_seeds": len(flip_seeds), "rho": rhos, "flip_acc": accs,
        })

    json.dump(summary, open("severity_sweep_summary.json", "w"), indent=2)
    print("\nSaved severity_sweep_results.json (raw) and severity_sweep_summary.json (per-level).")


if __name__ == "__main__":
    main()
