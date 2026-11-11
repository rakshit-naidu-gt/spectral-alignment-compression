"""
Reviewer Q7: "For Waterbirds, can you quantify the 'locality' gap directly
(e.g., measuring higher-order residuals or comparing delta^T H delta vs. the
observed second-order change) to more sharply locate the boundary at which the
second-order expansion breaks?"

gate_results_waterbirds.json (8-epoch, overfit) and
gate_results_waterbirds_earlystop.json (3-epoch, early-stopped) already store,
per (seed, mechanism, group), both "actual_dloss" and "predicted_dloss" (the
full second-order term g_a^T delta + 0.5 delta^T H_a delta). The residual
  residual = actual_dloss - predicted_dloss
IS the higher-order (O(||delta||^3)) term the Taylor expansion drops -- no new
computation needed, just reading what's already on disk.
"""
import json
import numpy as np

REGIMES = [
    ("8 epochs (overfit)", "gate_results_waterbirds.json"),
    ("3 epochs (early-stopped)", "gate_results_waterbirds_earlystop.json"),
]

for label, path in REGIMES:
    data = json.load(open(path))
    residuals, actuals, base_losses_of_group = [], [], []
    for res in data:
        seed = res["seed"]
        base_group = res["base_group"]
        for mech, mres in res["mechanisms"].items():
            for g_str, gres in mres["groups"].items():
                actual = gres["actual_dloss"]
                pred = gres["predicted_dloss"]
                residual = actual - pred
                residuals.append(residual)
                actuals.append(actual)
                base_losses_of_group.append(base_group[g_str]["loss"])

    residuals = np.array(residuals)
    actuals = np.array(actuals)
    base_losses_of_group = np.array(base_losses_of_group)

    abs_res = np.abs(residuals)
    # residual as a fraction of the actual change magnitude (only where actual is non-negligible)
    mask = np.abs(actuals) > 1e-4
    rel_res = abs_res[mask] / np.abs(actuals[mask])

    print("=" * 78)
    print(f"Regime: {label}  (n={len(residuals)} (seed, mechanism, group) points)")
    print("=" * 78)
    print(f"  mean |residual|          = {abs_res.mean():.5f}")
    print(f"  median |residual|        = {np.median(abs_res):.5f}")
    print(f"  max |residual|           = {abs_res.max():.5f}")
    print(f"  mean |actual dloss|      = {np.abs(actuals).mean():.5f}")
    print(f"  median relative |resid|/|actual|  = {np.median(rel_res):.2f}  "
          f"(n={mask.sum()} points with |actual|>1e-4)")
    corr = np.corrcoef(base_losses_of_group, abs_res)[0, 1]
    print(f"  corr(base group loss [pre-compression], |residual|) = {corr:+.3f}")
    print(f"  mean base group loss where |residual| is in top quartile: "
          f"{base_losses_of_group[abs_res >= np.percentile(abs_res, 75)].mean():.3f}")
    print(f"  mean base group loss where |residual| is in bottom quartile: "
          f"{base_losses_of_group[abs_res <= np.percentile(abs_res, 25)].mean():.3f}")
    print()

print("Interpretation guide: if |residual| grows with pre-compression base group")
print("loss (the 'how far the group already sits from theta*'s local neighborhood'")
print("proxy used in Appendix D.2's text), and is larger in the 8-epoch regime than")
print("the 3-epoch regime at matched ||delta||, that directly locates the breakdown")
print("in overfitting / pre-compression confident-wrongness, not in ||delta|| alone.")
