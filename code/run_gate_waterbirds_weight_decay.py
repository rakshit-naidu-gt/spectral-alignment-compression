"""
Weight-decay ablation, Waterbirds counterpart to run_gate_weight_decay.py: does
the direction-vs-magnitude finding survive when theta_star is trained with
standard L2 weight decay rather than the unregularized training
Proposition 1 assumes? Same early-stopped (3-epoch) recipe as
run_gate_waterbirds_earlystop.py, same 3 original seeds (matching the CIFAR
ablation's scope), changing only weight_decay.
"""
import copy
import json
import os
import time
import numpy as np
import torch
from scipy.stats import spearmanr

from model import pretrained_resnet18
from waterbirds_data import WaterbirdsDataset, GROUP_NAMES
from train import train_model, eval_per_group, overall_eval
from compress import MECHANISMS
from hvp import _compressible_params, flatten, compute_g_and_Hv

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
SEEDS = [0, 1, 2]
EPOCHS = 3
WEIGHT_DECAY = 5e-4
TARGET_REL_DELTA = 0.03
N_GROUPS = 4
GROUP_EVAL_N = 50
DATA_ROOT = "./data_waterbirds"

print("loading Waterbirds train/test splits...", flush=True)
ds_train_full = WaterbirdsDataset(DATA_ROOT, "train")
ds_test = WaterbirdsDataset(DATA_ROOT, "test")


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
    return d / w0_norm


def calibrate_severity_by_delta_norm(model, names, mechanism_fn, target_rel_delta):
    sd0 = copy.deepcopy(model.state_dict())
    w0_norm = weight_norm(sd0, names)
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
        for _ in range(10):
            mid = (lo + hi) / 2
            rel_d = delta_norm_for_severity(sd0, names, mechanism_fn, mid, w0_norm)
            candidates.append((mid, rel_d))
            if rel_d < target_rel_delta:
                lo = mid
            else:
                hi = mid
    best_sev = min(candidates, key=lambda c: abs(c[1] - target_rel_delta))[0]
    sd_c = mechanism_fn(sd0, best_sev)
    model.load_state_dict(sd_c)
    loss, acc = overall_eval(model, ds_test, DEVICE)
    model.load_state_dict(sd0)
    achieved_rel_delta = delta_norm_for_severity(sd0, names, mechanism_fn, best_sev, w0_norm)
    return best_sev, loss, acc, achieved_rel_delta


def run_seed(seed):
    print(f"\n=== SEED {seed} (weight_decay={WEIGHT_DECAY}) ===", flush=True)
    model = pretrained_resnet18(num_classes=2)
    ckpt_path = f"ckpt_waterbirds_wd_seed{seed}.pt"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu", weights_only=True))
        model.to(DEVICE)
        print(f"  loaded cached checkpoint {ckpt_path}", flush=True)
    else:
        t0 = time.time()
        model = train_model(model, ds_train_full, DEVICE, epochs=EPOCHS, lr=0.003, batch_size=64,
                             seed=seed, weight_decay=WEIGHT_DECAY)
        print(f"  train took {time.time()-t0:.1f}s", flush=True)
        torch.save(model.state_dict(), ckpt_path)
    model.eval()

    base_loss, base_acc = overall_eval(model, ds_test, DEVICE)
    base_group = eval_per_group(model, ds_test, DEVICE, n_groups=N_GROUPS)
    print(f"  base overall loss={base_loss:.4f} acc={base_acc:.4f}")
    for g in range(N_GROUPS):
        print(f"    group {g} ({GROUP_NAMES[g]}): loss={base_group[g]['loss']:.4f} acc={base_group[g]['acc']:.4f} n={base_group[g]['n']}")

    names, params = _compressible_params(model)
    theta_star = [p.detach().clone() for p in params]
    sd0 = copy.deepcopy(model.state_dict())
    group_batches = {g: get_group_batch(ds_test, g, GROUP_EVAL_N) for g in range(N_GROUPS)}

    seed_result = {"seed": seed, "base_loss": base_loss, "base_acc": base_acc,
                   "base_group": base_group, "mechanisms": {}, "weight_decay": WEIGHT_DECAY}

    for mech_name, mech_fn in MECHANISMS.items():
        severity, comp_loss, comp_acc, rel_delta = calibrate_severity_by_delta_norm(
            model, names, mech_fn, TARGET_REL_DELTA)
        sd_compressed = mech_fn(sd0, severity)
        model.load_state_dict(sd_compressed)
        model.eval()
        comp_group = eval_per_group(model, ds_test, DEVICE, n_groups=N_GROUPS)
        model.load_state_dict(sd0)
        model.eval()

        delta_list = [(sd_compressed[n] - sd0[n]) for n in names]
        delta_flat = flatten(delta_list).to(DEVICE)

        mech_result = {"severity": severity, "overall_loss": comp_loss, "overall_acc": comp_acc,
                       "rel_delta": rel_delta, "groups": {}}
        for g in range(N_GROUPS):
            x, y = group_batches[g]
            g_flat, hv_flat = compute_g_and_Hv(model, names, theta_star, x, y, delta_flat)
            pred = (g_flat @ delta_flat).item() + 0.5 * (delta_flat @ hv_flat).item()
            actual = comp_group[g]["loss"] - base_group[g]["loss"]
            mech_result["groups"][g] = {
                "actual_dloss": actual,
                "predicted_dloss": pred,
                "comp_loss": comp_group[g]["loss"],
                "comp_acc": comp_group[g]["acc"],
            }
        seed_result["mechanisms"][mech_name] = mech_result
        print(f"  [{mech_name}] severity={severity:.5f} rel_delta={rel_delta:.4f} (target {TARGET_REL_DELTA}) "
              f"overall_loss={comp_loss:.4f} overall_acc={comp_acc:.4f}")
        for g in range(N_GROUPS):
            r = mech_result["groups"][g]
            print(f"      group {g}: actual_dloss={r['actual_dloss']:+.4f}  predicted={r['predicted_dloss']:+.4f}")

    return seed_result


def main():
    all_results = [run_seed(s) for s in SEEDS]
    with open("gate_results_waterbirds_weight_decay.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "=" * 78)
    print("ANALYSIS (Waterbirds weight-decay ablation)")
    print("=" * 78)
    actual_pooled, pred_pooled = [], []
    for res in all_results:
        for mech, mres in res["mechanisms"].items():
            for g, gres in mres["groups"].items():
                actual_pooled.append(gres["actual_dloss"])
                pred_pooled.append(gres["predicted_dloss"])
    rho, pval = spearmanr(actual_pooled, pred_pooled)
    print(f"\n(i) Pooled Spearman(actual, predicted) = {rho:.3f} (p={pval:.4f}, n={len(actual_pooled)})")
    print(f"    PASS threshold: >= 0.80  ->  {'PASS' if rho >= 0.8 else 'FAIL'}")

    print("\n(ii) Most-harmed group per mechanism per seed:")
    flips = []
    for res in all_results:
        seed = res["seed"]
        most_harmed = {}
        for mech, mres in res["mechanisms"].items():
            g_harms = {g: gres["actual_dloss"] for g, gres in mres["groups"].items()}
            worst_g = max(g_harms, key=g_harms.get)
            most_harmed[mech] = worst_g
        print(f"    seed {seed}: {most_harmed}")
        if len(set(most_harmed.values())) > 1:
            flips.append((seed, most_harmed))
    print(f"\n    Flip events: {len(flips)}/{len(all_results)} seeds")

    print("\n(iii) Directional predictor accuracy on flip cases:")
    correct, total = 0, 0
    for seed, most_harmed in flips:
        res = next(r for r in all_results if r["seed"] == seed)
        for mech, worst_g_actual in most_harmed.items():
            mres = res["mechanisms"][mech]
            pred_harms = {g: gres["predicted_dloss"] for g, gres in mres["groups"].items()}
            worst_g_pred = max(pred_harms, key=pred_harms.get)
            total += 1
            if worst_g_pred == worst_g_actual:
                correct += 1
    frac = correct / total if total > 0 else float("nan")
    print(f"    {correct}/{total} correct = {frac:.1%}  PASS threshold: >= 75%  ->  {'PASS' if frac >= 0.75 else 'FAIL/NA'}")
    print("\n" + "=" * 78)
    print(f"GATE VERDICT (weight decay={WEIGHT_DECAY}): (i) {'PASS' if rho>=0.8 else 'FAIL'}  "
          f"(ii) {'PASS' if len(flips)>0 else 'FAIL'} ({len(flips)} flip seeds)  "
          f"(iii) {'PASS' if total>0 and frac>=0.75 else ('FAIL' if total>0 else 'N/A (no flips)')}")
    print("=" * 78)


if __name__ == "__main__":
    main()
