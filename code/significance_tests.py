"""
Reviewer comment #5 (experimental eval): "only 12 flips across seeds and
mechanisms; inferences could be fragile; bootstrapped CIs or permutation tests
would add robustness."

Three complementary tests on the SAME 12 flip cases already in
baseline_compare_results.json (no new experiments):

  (a) Exact binomial test: is each predictor's flip-case accuracy
      distinguishable from the 25% chance rate (4 equally-likely groups)?
  (b) McNemar's exact test (paired, since every predictor is scored on the
      SAME 12 cases): does the full directional predictor differ from each
      magnitude-only predictor on a per-case basis, not just in aggregate rate?
  (c) Seed-cluster bootstrap (resampling the 3 independent seeds with
      replacement, 10000 resamples): a CI on flip-case accuracy that respects
      the true independent unit (seed), at the honest cost that n=3 clusters
      gives a crude, discrete CI.
"""
import json
from itertools import product
import numpy as np
from scipy.stats import binomtest

rows = json.load(open("baseline_compare_results.json"))

by_seed_mech = {}
for r in rows:
    by_seed_mech.setdefault((r["seed"], r["mech"]), []).append(r)

seeds = sorted(set(r["seed"] for r in rows))
PREDICTORS = ["pred_full", "pred_isotropic", "pred_isotropic_curv_only", "pred_diag", "pred_first"]
LABELS = {
    "pred_full": "Full directional (HVP)",
    "pred_isotropic": "Liu-style isotropic (trace)",
    "pred_isotropic_curv_only": "Isotropic curvature-ONLY",
    "pred_diag": "FairPrune-style diagonal",
    "pred_first": "First-order only",
}

# Build, per seed, the flip cases (mechanisms disagree on actual most-harmed group)
# and per-predictor correctness on each.
case_correct = {k: [] for k in PREDICTORS}   # list of 0/1, one per (seed, mech) flip case
case_seed = []                                # parallel list: which seed each case belongs to

for seed in seeds:
    mechs = sorted(set(m for s, m in by_seed_mech if s == seed))
    most_harmed_actual = {}
    for mech in mechs:
        grp = by_seed_mech[(seed, mech)]
        most_harmed_actual[mech] = max(grp, key=lambda r: r["actual"])["group"]
    if len(set(most_harmed_actual.values())) <= 1:
        continue  # not a flip seed
    for mech in mechs:
        grp = by_seed_mech[(seed, mech)]
        case_seed.append(seed)
        for key in PREDICTORS:
            pred_g = max(grp, key=lambda r: r[key])["group"]
            case_correct[key].append(int(pred_g == most_harmed_actual[mech]))

n_cases = len(case_seed)
print(f"n flip cases = {n_cases} (across seeds {sorted(set(case_seed))})\n")

print("=" * 78)
print("(a) Exact binomial test vs. 25% chance (4 equally-likely groups)")
print("=" * 78)
for key in PREDICTORS:
    k = sum(case_correct[key])
    res = binomtest(k, n_cases, p=0.25, alternative="greater")
    print(f"  {LABELS[key]:30s}: {k}/{n_cases} = {k/n_cases:.1%}  "
          f"one-sided p(>chance) = {res.pvalue:.4f}")

print()
print("=" * 78)
print("(b) McNemar's exact test: full-directional vs. each other predictor")
print("    (paired on the same 12 cases; tests the discordant pairs only)")
print("=" * 78)
full = case_correct["pred_full"]
for key in PREDICTORS:
    if key == "pred_full":
        continue
    other = case_correct[key]
    # b = full right, other wrong; c = full wrong, other right
    b = sum(1 for i in range(n_cases) if full[i] == 1 and other[i] == 0)
    c = sum(1 for i in range(n_cases) if full[i] == 0 and other[i] == 1)
    n_disc = b + c
    if n_disc == 0:
        print(f"  full vs {LABELS[key]:28s}: no discordant pairs (identical on all {n_cases} cases)")
        continue
    res = binomtest(min(b, c), n_disc, p=0.5, alternative="two-sided")
    print(f"  full vs {LABELS[key]:28s}: b={b} c={c} (discordant n={n_disc})  "
          f"McNemar exact p = {res.pvalue:.4f}")

print()
print("=" * 78)
print("(c) Seed-cluster bootstrap CI on flip-case accuracy (resample the 3 seeds)")
print("    Honest caveat: only 3 independent clusters -> CI is coarse/discrete.")
print("=" * 78)
rng = np.random.default_rng(0)
unique_seeds = sorted(set(case_seed))
case_idx_by_seed = {s: [i for i, cs in enumerate(case_seed) if cs == s] for s in unique_seeds}

N_BOOT = 10000
for key in PREDICTORS:
    accs = []
    for _ in range(N_BOOT):
        resampled_seeds = rng.choice(unique_seeds, size=len(unique_seeds), replace=True)
        idxs = [i for s in resampled_seeds for i in case_idx_by_seed[s]]
        if not idxs:
            continue
        accs.append(np.mean([case_correct[key][i] for i in idxs]))
    accs = np.array(accs)
    lo, hi = np.percentile(accs, [2.5, 97.5])
    print(f"  {LABELS[key]:30s}: point={np.mean(case_correct[key]):.1%}  "
          f"bootstrap mean={accs.mean():.1%}  95% CI=[{lo:.1%}, {hi:.1%}]")
