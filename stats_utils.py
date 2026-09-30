"""Effect sizes and small helpers used in the report."""
import numpy as np
from scipy import stats


def cohens_d(a, b):
    """Standardised mean difference with pooled SD (b - a)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    pooled = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    return float((b.mean() - a.mean()) / pooled)


def rank_biserial(a, b):
    """Effect size for Mann-Whitney U: P(b > a) - P(b < a), in [-1, 1]."""
    u = stats.mannwhitneyu(b, a, alternative="two-sided").statistic
    return float(2 * u / (len(a) * len(b)) - 1)


def cramers_v(table):
    """Association strength for a contingency table (0 = none, 1 = perfect)."""
    table = np.asarray(table)
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.sum()
    return float(np.sqrt(chi2 / (n * (min(table.shape) - 1))))


def eta_squared(groups):
    """Share of variance explained by group membership (one-way ANOVA)."""
    all_ = np.concatenate(groups)
    grand = all_.mean()
    ss_between = sum(len(g) * (np.mean(g) - grand) ** 2 for g in groups)
    ss_total = ((all_ - grand) ** 2).sum()
    return float(ss_between / ss_total)


def pct_effect(log_coef):
    """Convert a coefficient from a log(y) model into a % change in y."""
    return float(np.expm1(log_coef) * 100)
