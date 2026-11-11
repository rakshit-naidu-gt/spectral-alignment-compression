"""
Follow-up to analyze_per_layer.py: the weight/grad-norm "scale invariance"
hypothesis did NOT explain which layers dominate g_a^T delta (negative
correlation, no conv-vs-linear contrast). The observed pattern instead splits
cleanly by MECHANISM:
  - channel pruning & SVD truncation: dominant layer is almost always a 1x1
    shortcut conv (or the 3-channel stem conv1).
  - unstructured pruning & quantization: dominant layer is always a regular
    3x3 conv, never a shortcut.

Hypothesis: this is a STRUCTURAL artifact of how each mechanism ranks
components GLOBALLY, not a gradient/BN effect at all.
  - channel_prune ranks per-OUTPUT-CHANNEL L2 norm globally across layers,
    with no normalization for channel width. A 1x1 shortcut conv's output
    channel has only `in_planes` elements; a 3x3 conv's has `in_planes*9`
    elements. For similar per-weight magnitude, the 3x3 channel's norm is
    ~3x larger purely from having 9x more elements -- so a SHARED global
    threshold on raw norms will systematically prune shortcut channels
    first, regardless of any gradient or BN effect.
  - svd_truncate ranks singular values globally the same way: a (Cout, Cin)
    matrix (1x1 conv) has fundamentally different singular-value scale than
    a (Cout, Cin*9) matrix (3x3 conv) reshaped from the same per-weight
    magnitude distribution.
  - unstructured_prune and quantize operate on INDIVIDUAL weights (not
    channel/singular-value groups), so they have no such width-dependent
    scale mismatch -- perturbation should land roughly uniformly across
    layer types by weight count.

This script tests that directly, using ONLY the cached checkpoints and the
exact severities already in gate_results.json: NO model forward/backward
passes, no gradients, no retraining -- just the four compress.py mechanisms
applied to state dicts, which is what actually determines delta.
"""
import copy
import json
import numpy as np
import torch

from model import ResNet18
from compress import MECHANISMS, _compressible_keys

SEEDS = [0, 1, 2]


def layer_kind(name, w):
    if name.startswith("linear"):
        return "linear"
    if "shortcut" in name:
        return "shortcut_1x1"
    if name == "conv1.weight":
        return "stem_3x3"
    return "regular_3x3"


def fraction_zeroed(w_before, w_after):
    return (w_after == 0).float().mean().item() - (w_before == 0).float().mean().item()


def main():
    gate_results = json.load(open("gate_results.json"))
    rows = []

    for seed in SEEDS:
        model = ResNet18(num_classes=2)
        model.load_state_dict(torch.load(f"ckpt_seed{seed}_minfrac15.pt", map_location="cpu", weights_only=True))
        sd0 = copy.deepcopy(model.state_dict())
        keys = _compressible_keys(sd0)

        res = next(r for r in gate_results if r["seed"] == seed)
        for mech, mres in res["mechanisms"].items():
            severity = mres["severity"]
            sd_c = MECHANISMS[mech](sd0, severity)
            for k in keys:
                kind = layer_kind(k, sd0[k])
                w0, wc = sd0[k], sd_c[k]
                numel = w0.numel()
                delta = (wc - w0)
                delta_density = (delta.norm().item() ** 2) / numel  # mean squared perturbation per weight
                frac_zeroed_delta = fraction_zeroed(w0, wc)  # how much MORE sparse after compression
                rows.append({
                    "seed": seed, "mech": mech, "layer": k, "kind": kind,
                    "numel": numel, "delta_density": delta_density,
                    "extra_frac_zeroed": frac_zeroed_delta,
                })

    print(f"{'Mechanism':<14}{'Layer kind':<16}{'n_layers':>9}{'mean numel':>12}"
          f"{'mean Δ-density':>16}{'mean extra_zeroed':>19}")
    print("-" * 90)
    mechs = sorted(set(r["mech"] for r in rows))
    kinds = ["shortcut_1x1", "stem_3x3", "regular_3x3", "linear"]
    for mech in mechs:
        for kind in kinds:
            sub = [r for r in rows if r["mech"] == mech and r["kind"] == kind]
            if not sub:
                continue
            mean_numel = np.mean([r["numel"] for r in sub])
            mean_density = np.mean([r["delta_density"] for r in sub])
            mean_extra_zero = np.mean([r["extra_frac_zeroed"] for r in sub])
            print(f"{mech:<14}{kind:<16}{len(sub):>9}{mean_numel:>12.0f}"
                  f"{mean_density:>16.3e}{mean_extra_zero:>19.1%}")
        print()

    print("=" * 90)
    print("Ratio check: for channel/svd, is extra_frac_zeroed on shortcuts >> regular convs?")
    print("For unstructured/quantize, are they comparable (no structural width effect)?")
    print("=" * 90)
    for mech in mechs:
        shortcut = [r["extra_frac_zeroed"] for r in rows if r["mech"] == mech and r["kind"] == "shortcut_1x1"]
        regular = [r["extra_frac_zeroed"] for r in rows if r["mech"] == mech and r["kind"] == "regular_3x3"]
        if shortcut and regular:
            ratio = np.mean(shortcut) / np.mean(regular) if np.mean(regular) != 0 else float("inf")
            print(f"  {mech:<14}: mean extra_zeroed shortcut={np.mean(shortcut):.1%}  "
                  f"regular={np.mean(regular):.1%}  ratio={ratio:.2f}x")


if __name__ == "__main__":
    main()
