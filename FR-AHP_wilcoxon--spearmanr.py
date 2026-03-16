"""
FR-AHP Statistical Validation
==============================
Compares F-AHP and FR-AHP priority vectors across two case studies.

Data sources (all from the article):
  Case 1 weights  : Table 7  (F-AHP)  and Table 9  (FR-AHP)
  Case 2 scores   : Table 13 (F-AHP)  and Table 16 (FR-AHP)
  Monte-Carlo stats: Table 20

Author: computed for article validation
"""

import numpy as np
from scipy.stats import wilcoxon, spearmanr

# ─────────────────────────────────────────────
# DATA — directly from article tables
# ─────────────────────────────────────────────

# Case 1: Traffic Accessibility
# Source: Table 7 (F-AHP, column 'Normalized') and Table 9 (FR-AHP, column 'Hi')
fahp_c1  = np.array([0.4467, 0.1100, 0.2435, 0.1999])
frahp_c1 = np.array([0.3873, 0.1618, 0.1768, 0.2740])
labels_c1 = ["A1", "A2", "A3", "A4"]
ranks_fahp_c1  = [1, 4, 2, 3]   # Table 7,  Rank column
ranks_frahp_c1 = [1, 4, 3, 2]   # Table 9,  Rank column

# Case 2: Online Food Delivery
# Source: Table 13 (F-AHP, row 'SUM') and Table 16 (FR-AHP, column 'Overall score Si')
fahp_c2  = np.array([0.4144, 0.4674, 0.1183])
frahp_c2 = np.array([0.4241, 0.4669, 0.1092])
labels_c2 = ["A1 (Grabfood)", "A2 (Foodpanda)", "A3 (OdaMakan)"]
ranks_fahp_c2  = [2, 1, 3]      # Table 13, RANKING row
ranks_frahp_c2 = [2, 1, 3]      # Table 16, inferred from scores

# Monte-Carlo statistics
# Source: Table 20 — 10,000 trials, ±10% noise on criterion weights
mc_fahp_mean,  mc_fahp_std  = 0.0530, 0.0018
mc_frahp_mean, mc_frahp_std = 0.0428, 0.0014

# ─────────────────────────────────────────────
# HELPER FUNCTIONS
# ─────────────────────────────────────────────

def entropy(w):
    """Shannon entropy of a weight vector (higher = more balanced)."""
    w = w / w.sum()
    return -np.sum(w * np.log(w))

def spread(w):
    """Max-min spread of a weight vector."""
    return w.max() - w.min()

def winner_runner_up_gap(w):
    """Gap between the top two scores."""
    s = np.sort(w)[::-1]
    return s[0] - s[1]

def rank_changes(r1, r2):
    """Count how many alternatives changed rank between two ranking lists."""
    return sum(1 for a, b in zip(r1, r2) if a != b)

def pct_change(old, new):
    return (new - old) / old * 100

# ─────────────────────────────────────────────
# CASE 1 ANALYSIS
# ─────────────────────────────────────────────

print("=" * 60)
print("CASE 1 — Traffic Accessibility")
print("Data: Table 7 (F-AHP) vs Table 9 (FR-AHP)")
print("=" * 60)

print("\nIndividual weight shifts:")
print(f"  {'Alt':<6} {'F-AHP':>8} {'FR-AHP':>8} {'Diff':>8} {'%Change':>10}")
for i, (f, r) in enumerate(zip(fahp_c1, frahp_c1)):
    print(f"  {labels_c1[i]:<6} {f:>8.4f} {r:>8.4f} {r-f:>+8.4f} {pct_change(f,r):>+9.1f}%")

print(f"\nDistributional metrics:")
print(f"  {'Metric':<22} {'F-AHP':>10} {'FR-AHP':>10} {'Change':>10}")
print(f"  {'Spread':<22} {spread(fahp_c1):>10.4f} {spread(frahp_c1):>10.4f} {pct_change(spread(fahp_c1), spread(frahp_c1)):>+9.1f}%")
print(f"  {'Variance':<22} {np.var(fahp_c1):>10.6f} {np.var(frahp_c1):>10.6f} {pct_change(np.var(fahp_c1), np.var(frahp_c1)):>+9.1f}%")
print(f"  {'Entropy':<22} {entropy(fahp_c1):>10.4f} {entropy(frahp_c1):>10.4f} {entropy(frahp_c1)-entropy(fahp_c1):>+9.4f}")

print(f"\nRank changes: {rank_changes(ranks_fahp_c1, ranks_frahp_c1)}/4")
print(f"  F-AHP  ranks: {ranks_fahp_c1}  (Table 7)")
print(f"  FR-AHP ranks: {ranks_frahp_c1}  (Table 9)")

rho1, p_rho1 = spearmanr(ranks_fahp_c1, ranks_frahp_c1)
print(f"  Spearman rho: {rho1:.4f}, p={p_rho1:.4f}")

stat1, p1 = wilcoxon(fahp_c1, frahp_c1)
print(f"\nWilcoxon signed-rank test:")
print(f"  W={stat1:.1f}, p={p1:.4f}")
print(f"  NOTE: n=4 — below minimum for adequate power. Non-significance")
print(f"        reflects sample size, not equivalence of methods.")

# ─────────────────────────────────────────────
# CASE 2 ANALYSIS
# ─────────────────────────────────────────────

print()
print("=" * 60)
print("CASE 2 — Online Food Delivery")
print("Data: Table 13 (F-AHP) vs Table 16 (FR-AHP)")
print("=" * 60)

print("\nIndividual score shifts:")
print(f"  {'Alt':<20} {'F-AHP':>8} {'FR-AHP':>8} {'Diff':>8} {'%Change':>10}")
for i, (f, r) in enumerate(zip(fahp_c2, frahp_c2)):
    print(f"  {labels_c2[i]:<20} {f:>8.4f} {r:>8.4f} {r-f:>+8.4f} {pct_change(f,r):>+9.1f}%")

print(f"\nDistributional metrics:")
print(f"  {'Metric':<28} {'F-AHP':>10} {'FR-AHP':>10} {'Change':>10}")
print(f"  {'Winner-runner-up gap':<28} {winner_runner_up_gap(fahp_c2):>10.4f} {winner_runner_up_gap(frahp_c2):>10.4f} {pct_change(winner_runner_up_gap(fahp_c2), winner_runner_up_gap(frahp_c2)):>+9.1f}%")
print(f"  {'Spread':<28} {spread(fahp_c2):>10.4f} {spread(frahp_c2):>10.4f} {pct_change(spread(fahp_c2), spread(frahp_c2)):>+9.1f}%")
print(f"  {'Variance':<28} {np.var(fahp_c2):>10.6f} {np.var(frahp_c2):>10.6f} {pct_change(np.var(fahp_c2), np.var(frahp_c2)):>+9.1f}%")
print(f"  {'Entropy':<28} {entropy(fahp_c2):>10.4f} {entropy(frahp_c2):>10.4f} {entropy(frahp_c2)-entropy(fahp_c2):>+9.4f}")

print(f"\nRank changes: {rank_changes(ranks_fahp_c2, ranks_frahp_c2)}/3")
print(f"  F-AHP  ranks: {ranks_fahp_c2}  (Table 13)")
print(f"  FR-AHP ranks: {ranks_frahp_c2}  (Table 16)")

rho2, p_rho2 = spearmanr(ranks_fahp_c2, ranks_frahp_c2)
print(f"  Spearman rho: {rho2:.4f}")

stat2, p2 = wilcoxon(fahp_c2, frahp_c2)
print(f"\nWilcoxon signed-rank test:")
print(f"  W={stat2:.1f}, p={p2:.4f}")
print(f"  NOTE: n=3 — well below minimum for adequate power.")

# ─────────────────────────────────────────────
# MONTE-CARLO ANALYSIS (Table 20)
# ─────────────────────────────────────────────

print()
print("=" * 60)
print("MONTE-CARLO SUMMARY — Table 20")
print("10,000 trials, ±10% noise on criterion weights")
print("=" * 60)

print(f"\n  {'Method':<12} {'Mean gap':>10} {'Std dev':>10}")
print(f"  {'F-AHP':<12} {mc_fahp_mean:>10.4f} {mc_fahp_std:>10.4f}")
print(f"  {'FR-AHP':<12} {mc_frahp_mean:>10.4f} {mc_frahp_std:>10.4f}")
print(f"\n  Gap mean reduction : {pct_change(mc_fahp_mean, mc_frahp_mean):+.1f}%")
print(f"  Gap std  reduction : {pct_change(mc_fahp_std,  mc_frahp_std):+.1f}%")

# ─────────────────────────────────────────────
# CONCLUSION
# ─────────────────────────────────────────────

print()
print("=" * 60)
print("CONCLUSION")
print("=" * 60)
print("""
  The rough-set layer adds value in a context-dependent manner:

  Case 1 (divergent experts):
    - Variance reduced by 46.5%  → more balanced weight distribution
    - Spread reduced by 33.0%    → less dominance by top alternative
    - Entropy increased by 0.05  → more equitable weight allocation
    - 2/4 alternatives changed rank → structural correction of outlier bias

  Case 2 (consensus experts):
    - Rank order fully preserved  → no distortion introduced
    - Winner-runner-up gap reduced by 19.2% → more conservative confidence
    - MC std reduced by 22.2%    → lower volatility under perturbation

  Key finding: the value of rough sets scales with expert disagreement.
  Under polarised panels it corrects rankings; under consensus panels
  it acts as a conservative damper — and never degrades the outcome.
""")
