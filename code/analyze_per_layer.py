"""
Reviewer Q2: "Given batch normalization scale invariances, do layer-wise
rescalings meaningfully change g_a^T delta signs? Could you report a per-layer
contribution analysis to confirm that the observed flips are not driven by a
few layers with unstable scalings?"

Every compressible layer here is a conv (or the final linear) weight that
(except the final linear) is immediately followed by BatchNorm, so conv
layers sit on a continuous rescaling symmetry (scale the conv weight by c,
compensate in BN's gamma) that leaves the function unchanged but changes the
RAW gradient magnitude in that layer by 1/c. If one or two such layers
happen to sit at an extreme point on this orbit, their raw contribution to
g_a^T delta could dominate the group ranking for reasons unrelated to the
group-sensitivity signal the theory is actually about.

This script, using ONLY cached checkpoints and the exact severities already
calibrated in gate_results.json (no retraining):
  1. Decomposes g_a^T delta into an exact per-layer sum (an identity, not an
     approximation: g^T delta = sum_l g_l . delta_l).
  2. For each cross-mechanism flip case, reports the top contributing layers
     and whether the model's predicted most-harmed group survives a
     leave-top-layer-out ablation (zero out the single most dominant layer's
     contribution and recompute the argmax).
  3. Correlates each layer's |weight|/|gradient|-norm ratio (a proxy for where
     a BN-adjacent conv layer sits on its scale-invariant orbit) against its
     share of |g_a^T delta|, contrasting conv layers (scale-symmetric, has a
     following BN) against the final linear layer (not scale-symmetric, no
     following BN) as a built-in control.
"""
import copy
import json
import numpy as np
import torch

from model import ResNet18
from spurious_data import SpuriousCatsDogs
from compress import MECHANISMS
from hvp import _compressible_params, flatten, unflatten, compute_g_and_Hv

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
N_GROUPS = 4
GROUP_EVAL_N = 50
ds_test = SpuriousCatsDogs(root="./data", train=False, seed=100)


def get_group_batch(ds, group_id, n):
    xs, ys = [], []
    idx = np.where(ds.group == group_id)[0][:n]
    for i in idx:
        x, y, g = ds[i]
        xs.append(x)
        ys.append(y)
    return torch.stack(xs).to(DEVICE), torch.tensor(ys).to(DEVICE)


def is_conv_layer(name):
    return not name.startswith("linear")  # everything but the final FC is conv-followed-by-BN


def main():
    gate_results = json.load(open("gate_results.json"))

    # Identify flip cases exactly as run_gate.py's analysis does.
    flip_cases = []  # (seed, mech, actual_most_harmed_group)
    for res in gate_results:
        seed = res["seed"]
        most_harmed = {}
        for mech, mres in res["mechanisms"].items():
            g_harms = {int(g): gres["actual_dloss"] for g, gres in mres["groups"].items()}
            most_harmed[mech] = max(g_harms, key=g_harms.get)
        if len(set(most_harmed.values())) > 1:
            for mech, g in most_harmed.items():
                flip_cases.append((seed, mech, g))

    print(f"Analyzing {len(flip_cases)} flip cases: {flip_cases}\n")

    ratio_vals_conv, share_vals_conv = [], []
    ratio_vals_linear, share_vals_linear = [], []
    n_sign_flips_from_ablation = 0
    n_cases_checked = 0
    all_case_records = []
    ablation_records = []

    seeds_needed = sorted(set(s for s, m, g in flip_cases))
    for seed in seeds_needed:
        model = ResNet18(num_classes=2)
        model.load_state_dict(torch.load(f"ckpt_seed{seed}_minfrac15.pt", map_location="cpu", weights_only=True))
        model.to(DEVICE)
        model.eval()

        names, params = _compressible_params(model)
        theta_star = [p.detach().clone() for p in params]
        sd0 = copy.deepcopy(model.state_dict())
        group_batches = {g: get_group_batch(ds_test, g, GROUP_EVAL_N) for g in range(N_GROUPS)}

        res = next(r for r in gate_results if r["seed"] == seed)
        mechs_this_seed = sorted(set(m for s, m, g in flip_cases if s == seed))

        for mech in mechs_this_seed:
            severity = res["mechanisms"][mech]["severity"]
            sd_c = MECHANISMS[mech](sd0, severity)
            delta_list_per_layer = [(sd_c[n] - sd0[n]).to(DEVICE) for n in names]
            delta_flat = flatten(delta_list_per_layer)

            # per-group per-layer first-order contribution, and per-layer weight norm
            layer_weight_norm = {n: sd0[n].norm().item() for n in names}
            group_layer_contrib = {}  # group -> list of per-layer g_l . delta_l
            group_layer_gradnorm = {}  # group -> list of per-layer ||g_l||
            actual_most_harmed = next(g for s, m, g in flip_cases if s == seed and m == mech)

            for g in range(N_GROUPS):
                x, y = group_batches[g]
                g_flat, hv_flat = compute_g_and_Hv(model, names, theta_star, x, y, delta_flat)
                g_per_layer = unflatten(g_flat, theta_star)
                delta_per_layer = delta_list_per_layer
                contribs = [(g_per_layer[i] * delta_per_layer[i]).sum().item() for i in range(len(names))]
                gradnorms = [g_per_layer[i].norm().item() for i in range(len(names))]
                group_layer_contrib[g] = contribs
                group_layer_gradnorm[g] = gradnorms

            # --- Report top-3 dominant layers for the actual most-harmed group ---
            contribs = group_layer_contrib[actual_most_harmed]
            total = sum(contribs)
            order = sorted(range(len(names)), key=lambda i: -abs(contribs[i]))
            cancellation_index = sum(abs(c) for c in contribs) / abs(total) if abs(total) > 1e-12 else float("inf")
            print(f"[seed {seed}, {mech}] actual most-harmed group={actual_most_harmed}  "
                  f"total g^T delta={total:+.6e}  cancellation_index(sum|c_l|/|total|)={cancellation_index:.2f}")
            for rank, i in enumerate(order[:3]):
                share = contribs[i] / total if abs(total) > 1e-12 else float("nan")
                print(f"    #{rank+1} layer={names[i]:30s} contrib={contribs[i]:+.3e}  share_of_total={share:+.1%}")
            all_case_records.append({
                "seed": seed, "mech": mech, "group": actual_most_harmed,
                "total": total, "cancellation_index": cancellation_index,
                "per_layer": {names[i]: contribs[i] for i in range(len(names))},
            })

            # --- Leave-top-layer-out robustness check on the predicted argmax group ---
            # original predicted group (first-order-only, all layers)
            group_totals_full = {g: sum(group_layer_contrib[g]) for g in range(N_GROUPS)}
            pred_full = max(group_totals_full, key=group_totals_full.get)

            # find the single globally most dominant layer (by |contribution|, summed abs across groups)
            layer_dominance = [sum(abs(group_layer_contrib[g][i]) for g in range(N_GROUPS)) for i in range(len(names))]
            top_layer_idx = int(np.argmax(layer_dominance))

            group_totals_ablated = {
                g: sum(c for i, c in enumerate(group_layer_contrib[g]) if i != top_layer_idx)
                for g in range(N_GROUPS)
            }
            pred_ablated = max(group_totals_ablated, key=group_totals_ablated.get)

            n_cases_checked += 1
            flipped = pred_full != pred_ablated
            n_sign_flips_from_ablation += int(flipped)
            print(f"    leave-top-layer-out ({names[top_layer_idx]}): "
                  f"pred(all layers)={pred_full}  pred(top layer excluded)={pred_ablated}  "
                  f"{'CHANGED' if flipped else 'unchanged'}\n")
            ablation_records.append({
                "seed": seed, "mech": mech, "top_layer": names[top_layer_idx],
                "pred_full": pred_full, "pred_ablated": pred_ablated, "flipped": flipped,
            })

            # --- weight/grad-norm ratio vs. contribution share, conv vs linear ---
            for i, n in enumerate(names):
                wnorm = layer_weight_norm[n]
                gnorm = group_layer_gradnorm[actual_most_harmed][i]
                if gnorm < 1e-12:
                    continue
                ratio = wnorm / gnorm
                share = abs(contribs[i]) / sum(abs(c) for c in contribs) if sum(abs(c) for c in contribs) > 0 else 0.0
                if is_conv_layer(n):
                    ratio_vals_conv.append(ratio)
                    share_vals_conv.append(share)
                else:
                    ratio_vals_linear.append(ratio)
                    share_vals_linear.append(share)

    print("=" * 78)
    print(f"SUMMARY: leave-top-layer-out changed the predicted most-harmed group in "
          f"{n_sign_flips_from_ablation}/{n_cases_checked} flip cases.")
    print("=" * 78)

    print("\nCancellation index by mechanism (sum_l |contrib_l| / |total| -- "
          "near 1 means one term truly dominates with little offsetting;")
    print("much greater than 1 means the total is a small residual of larger, "
          "partially-canceling per-layer terms, i.e. intrinsically fragile):")
    mechs_seen = sorted(set(r["mech"] for r in all_case_records))
    for mech in mechs_seen:
        idxs = [r["cancellation_index"] for r in all_case_records if r["mech"] == mech]
        flips = [r["flipped"] for r in ablation_records if r["mech"] == mech]
        print(f"  {mech:<14}: cancellation_index = {idxs}  "
              f"-> leave-top-layer-out flipped {sum(flips)}/{len(flips)}")

    json.dump({"per_case": all_case_records, "ablation": ablation_records},
              open("per_layer_analysis_results.json", "w"), indent=2)

    if len(ratio_vals_conv) > 2:
        corr_conv = np.corrcoef(ratio_vals_conv, share_vals_conv)[0, 1]
        print(f"\nConv layers (scale-symmetric, BN-adjacent), n={len(ratio_vals_conv)}:")
        print(f"  corr(||w||/||g||, |contribution share|) = {corr_conv:+.3f}")
    if len(ratio_vals_linear) > 2:
        corr_lin = np.corrcoef(ratio_vals_linear, share_vals_linear)[0, 1]
        print(f"\nFinal linear layer (NOT scale-symmetric, no following BN), n={len(ratio_vals_linear)}:")
        print(f"  corr(||w||/||g||, |contribution share|) = {corr_lin:+.3f}")
    print("\nInterpretation guide: a strong positive correlation for conv (BN-adjacent)")
    print("layers but not for the linear layer would support the reviewer's specific")
    print("concern (scale symmetry inflates certain layers' raw contribution).")


if __name__ == "__main__":
    main()
