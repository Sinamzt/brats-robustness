"""Patient-aligned paired comparisons; CIs do not establish equivalence."""
import numpy as np
from scipy.stats import wilcoxon


def paired_vectors(df, a_cond, b_cond):
    a = df[(df.condition == a_cond) & (df.region == "WT")].set_index("pid")["DSC"]
    b = df[(df.condition == b_cond) & (df.region == "WT")].set_index("pid")["DSC"]
    if not a.index.is_unique or not b.index.is_unique or set(a.index) != set(b.index):
        raise ValueError("Paired comparison requires identical unique patient IDs")
    ids = sorted(a.index)
    av, bv = a.loc[ids].to_numpy(), b.loc[ids].to_numpy()
    if not len(av) or not np.isfinite(av).all() or not np.isfinite(bv).all():
        raise ValueError("Paired DSCs must be nonempty and finite")
    return av, bv


def paired_comparisons(df, pairs, expected_n, seed=1337):
    rows = []
    rng = np.random.default_rng(seed)
    for label, ac, bc in pairs:
        a, b = paired_vectors(df, ac, bc)
        if len(a) != expected_n:
            raise ValueError(f"Missing cases in {label}: {len(a)} != {expected_n}")
        diff = a - b
        if np.all(diff == 0):
            stat, p = 0.0, 1.0
        else:
            stat, p = wilcoxon(a, b, zero_method="wilcox", alternative="two-sided")
        boot = diff[rng.integers(0, len(diff), (5000, len(diff)))].mean(axis=1)
        low, high = np.quantile(boot, [0.025, 0.975])
        rows.append({"comparison": label, "n": len(a), "median_A": float(np.median(a)),
                     "median_B": float(np.median(b)),
                     "median_diff_A_minus_B": float(np.median(diff)),
                     "mean_diff_A_minus_B": float(diff.mean()),
                     "mean_diff_bootstrap_ci_low": float(low),
                     "mean_diff_bootstrap_ci_high": float(high),
                     "wilcoxon_W": float(stat), "p_value_two_sided": float(p)})
    order = sorted(range(len(rows)), key=lambda i: rows[i]["p_value_two_sided"])
    last = 0.0
    for rank, i in enumerate(order):
        last = max(last, min(1.0, (len(rows) - rank) * rows[i]["p_value_two_sided"]))
        rows[i]["p_value_holm"] = last
    return rows
