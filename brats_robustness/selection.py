"""Validation-only severity choice independent of mitigation performance."""
import numpy as np


def select_severity(severities, clean_mean, corrupted_means, target_drop=0.08):
    s = np.asarray(severities, dtype=float)
    scores = np.asarray(corrupted_means, dtype=float)
    if not len(s) or s.shape != scores.shape or not np.isfinite(scores).all() or not np.isfinite(clean_mean):
        raise ValueError("Expected finite validation scores for every severity")
    if clean_mean < 0.85:
        raise ValueError("Clean validation WT-DSC below 0.85: verify inputs/model before severity selection")
    candidates = s[(clean_mean - scores) >= target_drop]
    if len(candidates):
        return float(candidates.min()), "smallest severity reaching target validation drop"
    return float(s[np.argmax(clean_mean - scores)]), "fallback: maximum observed validation drop; target unmet"
