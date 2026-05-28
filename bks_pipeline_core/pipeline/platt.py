"""Platt scaling calibration for prop probability models.

Pure math module — no I/O, no Firestore, no external dependencies.
Implements Normal CDF survival functions, Platt (2000) logistic
calibration via Newton-Raphson, and American-odds no-vig conversion.

All functions are pure — no I/O or side effects.
"""

from __future__ import annotations

import logging
import math
from typing import Any

logger = logging.getLogger(__name__)

# Coefficient of variation per stat category — baseline for a league-average player.
_STAT_CV: dict[str, float] = {
    "pts": 0.35,
    "reb": 0.47,
    "ast": 0.52,
    "fg3m": 0.65,  # high variance: shooting streaks, shot-creation variance
    "stl": 0.80,  # very high variance: opportunistic, defense-scheme dependent
    "blk": 0.85,  # very high variance: matchup-driven
    "tov": 0.70,  # high variance: usage and game-script dependent
}

# Consistency scaling: how much consistency_score narrows/widens the CV.
# consistency=1.0 → CV × (1 - _CONSISTENCY_CV_SCALE) = CV × 0.70 (30% narrower)
# consistency=0.0 → CV × (1 + _CONSISTENCY_CV_SCALE) = CV × 1.30 (30% wider)
# consistency=0.5 → no change (league average)
_CONSISTENCY_CV_SCALE: float = 0.30

# Matchup sigma adjustment: soft matchup raises uncertainty (more upside variance),
# tough matchup lowers it. Bounded to ±15% of base sigma.
_MATCHUP_SIGMA_SCALE: float = 0.15

# Platt scaling optimizer limits.
_PLATT_MAX_ITER: int = 50
_PLATT_CONVERGENCE: float = 1e-7


def normal_cdf_sf(x: float, mu: float, sigma: float) -> float:
    """Survival function P(X > x) for a Normal distribution.

    Returns 0.5 if sigma <= 0 (degenerate distribution).
    """
    if sigma <= 0.0:
        return 0.5
    return 0.5 * math.erfc((x - mu) / (sigma * math.sqrt(2.0)))


def compute_stat_distributions(
    result: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    """Derive Normal distribution parameters for pts/reb/ast/fg3m/stl/blk/tov.

    µ priority per stat:
      1. projected_{stat} — model-adjusted forward projection (set by projection.py)
      2. season_{stat}pg   — season average counting stat

    Using the model's own forward projection as µ keeps the prop probability
    consistent with the signal pipeline rather than anchoring to a stale season average
    that the signal pipeline may have adjusted away from.

    Sigma is adjusted by two contextual signals:
    1. consistency_score — consistent players get a narrower sigma,
       volatile players get a wider one (±30% of base CV).
    2. matchup_multiplier — soft matchup raises variance (more upside),
       tough matchup lowers it (±15% of base sigma).
    """
    # projected_stats key → season fallback field.
    # projected_stats is a nested dict on the result: result["projected_stats"]["pts"].
    # fg3 is keyed as "fg3" in projected_stats but "fg3m" externally (The Odds API market).
    stat_season_keys: dict[str, str] = {
        "pts": "season_ppg",
        "reb": "season_rpg",
        "ast": "season_apg",
        "fg3m": "season_fg3_pct",  # proxy: used only when projected_stats absent
        "stl": "season_spg",
        "blk": "season_bpg",
        "tov": "season_topg",
    }
    # projected_stats uses "fg3" key for made 3-pointers; all others match stat name.
    _proj_key_map: dict[str, str] = {"fg3m": "fg3"}

    projected_stats: dict[str, Any] = result.get("projected_stats") or {}

    # Consistency adjustment: 0.5 = neutral, 1.0 = max narrow, 0.0 = max wide
    _raw_consistency = result.get("consistency_score")
    consistency: float = float(_raw_consistency) if _raw_consistency is not None else 0.5
    # Maps consistency [0, 1] → CV scale factor [1.30, 0.70]
    consistency_factor: float = 1.0 + _CONSISTENCY_CV_SCALE * (0.5 - consistency) * 2.0

    # Matchup adjustment: multiplier > 1.0 = soft (wider sigma), < 1.0 = tough (narrower)
    matchup_mult: float = float(result.get("matchup_multiplier") or 1.0)
    # Maps multiplier deviation from 1.0 → ±_MATCHUP_SIGMA_SCALE sigma adjustment
    matchup_factor: float = 1.0 + _MATCHUP_SIGMA_SCALE * (matchup_mult - 1.0)
    matchup_factor = max(1.0 - _MATCHUP_SIGMA_SCALE, min(1.0 + _MATCHUP_SIGMA_SCALE, matchup_factor))

    distributions: dict[str, dict[str, Any]] = {}
    for stat, season_key in stat_season_keys.items():
        mu: float
        # Priority 1: model-adjusted projection from projected_stats.
        # Treat 0 / None as absent — fall through to season average.
        proj_key = _proj_key_map.get(stat, stat)
        proj_raw = projected_stats.get(proj_key)
        season_raw = result.get(season_key)
        if proj_raw is not None and float(proj_raw) > 0.0:
            mu = float(proj_raw)
        elif season_raw is not None:
            mu = float(season_raw)
        else:
            continue

        if mu <= 0.0:
            continue

        cv = _STAT_CV.get(stat, 0.50)
        base_sigma = mu * cv
        sigma = round(base_sigma * consistency_factor * matchup_factor, 4)
        distributions[stat] = {"mu": mu, "sigma": sigma, "dist": "normal"}

    return distributions


def _ppf(p: float, mu: float, sigma: float) -> float:
    """Inverse Normal CDF (percent-point function).

    Returns the value *x* such that P(X <= x) = *p* for X ~ N(mu, sigma).
    Uses the Acklam rational approximation (accurate to ~1e-9).
    """
    p = max(1e-12, min(1.0 - 1e-12, p))

    # Acklam rational approximation for standard normal quantile
    _a = (
        -3.969683028665376e1,
        2.209460984245205e2,
        -2.759285104469687e2,
        1.383577518672690e2,
        -3.066479806614716e1,
        2.506628277459239e0,
    )
    _b = (
        -5.447609879822406e1,
        1.615858368580409e2,
        -1.556989798598866e2,
        6.680131188771972e1,
        -1.328068155288572e1,
    )
    _c = (
        -7.784894002430293e-3,
        -3.223964580411365e-1,
        -2.400758277161838e0,
        -2.549732539343734e0,
        4.374664141464968e0,
        2.938163982698783e0,
    )
    _d = (
        7.784695709041462e-3,
        3.224671290700398e-1,
        2.445134137142996e0,
        3.754408661907416e0,
    )

    p_low = 0.02425
    p_high = 1.0 - p_low

    if p < p_low:
        q = math.sqrt(-2.0 * math.log(p))
        z = (((((_c[0] * q + _c[1]) * q + _c[2]) * q + _c[3]) * q + _c[4]) * q + _c[5]) / ((((_d[0] * q + _d[1]) * q + _d[2]) * q + _d[3]) * q + 1.0)
    elif p <= p_high:
        q = p - 0.5
        r = q * q
        z = (
            (((((_a[0] * r + _a[1]) * r + _a[2]) * r + _a[3]) * r + _a[4]) * r + _a[5])
            * q
            / (((((_b[0] * r + _b[1]) * r + _b[2]) * r + _b[3]) * r + _b[4]) * r + 1.0)
        )
    else:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        z = -(((((_c[0] * q + _c[1]) * q + _c[2]) * q + _c[3]) * q + _c[4]) * q + _c[5]) / ((((_d[0] * q + _d[1]) * q + _d[2]) * q + _d[3]) * q + 1.0)

    return mu + sigma * z


def compute_confidence_intervals(
    result: dict[str, Any],
    fp_mu: float | None = None,
    fp_sigma: float | None = None,
) -> dict[str, Any]:
    """Compute percentile bands for fantasy points and per-stat categories.

    Args:
        result: Opportunity result dict (needs projected_stats or season_ppg/rpg/apg).
        fp_mu: Mean fantasy points from minutes-environment model.
        fp_sigma: Std dev of fantasy points from minutes-environment model.

    Returns:
        Dict with ``fp_percentiles`` (p10-p90 for total FP) and
        ``stat_percentiles`` (p10-p90 per stat: pts, reb, ast).
    """
    ci: dict[str, Any] = {}

    # FP percentiles from minutes-environment distribution
    if fp_mu is not None and fp_sigma is not None and fp_sigma > 0:
        ci["fp_percentiles"] = {
            "p10": round(max(0.0, _ppf(0.10, fp_mu, fp_sigma)), 1),
            "p25": round(max(0.0, _ppf(0.25, fp_mu, fp_sigma)), 1),
            "p50": round(max(0.0, _ppf(0.50, fp_mu, fp_sigma)), 1),
            "p75": round(_ppf(0.75, fp_mu, fp_sigma), 1),
            "p90": round(_ppf(0.90, fp_mu, fp_sigma), 1),
        }
        # Confidence that player beats their season baseline (opportunity_score)
        avg_opp = float(result.get("opportunity_score") or 0)
        if avg_opp > 0:
            ci["confidence_pct"] = round(normal_cdf_sf(avg_opp, fp_mu, fp_sigma), 3)

    # Per-stat percentiles from Normal(mu, sigma)
    dists = compute_stat_distributions(result)
    if dists:
        stat_pcts: dict[str, dict[str, float]] = {}
        for stat, info in dists.items():
            mu = info["mu"]
            sigma = info["sigma"]
            stat_pcts[stat] = {
                "p10": round(max(0.0, _ppf(0.10, mu, sigma)), 1),
                "p25": round(max(0.0, _ppf(0.25, mu, sigma)), 1),
                "p50": round(max(0.0, _ppf(0.50, mu, sigma)), 1),
                "p75": round(_ppf(0.75, mu, sigma), 1),
                "p90": round(_ppf(0.90, mu, sigma), 1),
            }
        ci["stat_percentiles"] = stat_pcts

    return ci


def prob_over_line(mu: float, sigma: float, line: float) -> float:
    """Raw model probability that a stat exceeds *line*."""
    return normal_cdf_sf(line, mu, sigma)


def _platt_objective(
    a: float,
    b: float,
    raw_probs: list[float],
    targets: list[float],
) -> float:
    """Negative log-likelihood for Platt calibration."""
    total = 0.0
    for i in range(len(raw_probs)):
        fval = a * raw_probs[i] + b
        fval = max(-500.0, min(500.0, fval))
        p_i = 1.0 / (1.0 + math.exp(fval))
        p_i = max(1e-15, min(1.0 - 1e-15, p_i))
        total += targets[i] * math.log(p_i) + (1.0 - targets[i]) * math.log(1.0 - p_i)
    return -total


def fit_platt(
    prob_outcome_pairs: list[tuple[float, bool]],
) -> dict[str, Any]:
    """Fit Platt scaling logistic via Newton-Raphson.

    Implements the Platt (2000) algorithm:
      calibrated_p = 1 / (1 + exp(A * raw_p + B))

    Returns ``{"A": float, "B": float, "brier": float, "samples": int}``.
    If optimisation fails to converge the fallback is identity-like
    (A=-1.0, B=0.0).
    """
    n = len(prob_outcome_pairs)
    if n == 0:
        return {"A": -1.0, "B": 0.0, "brier": 0.0, "samples": 0}

    raw_probs = [p for p, _ in prob_outcome_pairs]
    outcomes = [o for _, o in prob_outcome_pairs]

    n_pos = sum(1 for o in outcomes if o)
    n_neg = n - n_pos

    # Target labels per Platt (2000).
    t_pos = (n_pos + 1.0) / (n_pos + 2.0)
    t_neg = 1.0 / (n_neg + 2.0)
    targets = [t_pos if o else t_neg for o in outcomes]

    # Initial parameters.
    a = 0.0
    b = math.log((n_neg + 1.0) / (n_pos + 1.0)) if n_pos > 0 else 0.0

    converged = False
    old_err = _platt_objective(a, b, raw_probs, targets)

    for _ in range(_PLATT_MAX_ITER):
        # Gradient and Hessian accumulation.
        grad_a = 0.0
        grad_b = 0.0
        hess_aa = 1e-6
        hess_ab = 0.0
        hess_bb = 1e-6

        for i in range(n):
            fval = a * raw_probs[i] + b
            fval = max(-500.0, min(500.0, fval))
            p_i = 1.0 / (1.0 + math.exp(fval))
            d = p_i - targets[i]
            h = p_i * (1.0 - p_i)
            h = max(h, 1e-12)
            grad_a += d * raw_probs[i]
            grad_b += d
            hess_aa += h * raw_probs[i] * raw_probs[i]
            hess_ab += h * raw_probs[i]
            hess_bb += h

        # Solve 2x2 Newton system.
        det = hess_aa * hess_bb - hess_ab * hess_ab
        if abs(det) < 1e-30:
            logger.warning("platt fit: singular Hessian at det=%s, stopping early", det)
            break

        # grad_a/grad_b hold sum[(p-t)*x] which is the negative of the
        # NLL gradient.  Newton step = +H^{-1} * grad  (since grad =
        # -dL/dA).
        da = (hess_bb * grad_a - hess_ab * grad_b) / det
        db = (hess_aa * grad_b - hess_ab * grad_a) / det

        # Backtracking line search — halve the step until objective
        # improves (standard Platt 2000 safeguard).
        step = 1.0
        for _ls in range(10):
            new_a = a + step * da
            new_b = b + step * db
            new_err = _platt_objective(new_a, new_b, raw_probs, targets)
            if new_err < old_err + 1e-12:
                break
            step *= 0.5
        else:
            # Line search exhausted; accept smallest step anyway.
            new_a = a + step * da
            new_b = b + step * db
            new_err = _platt_objective(new_a, new_b, raw_probs, targets)

        a = new_a
        b = new_b
        old_err = new_err

        if abs(step * da) < _PLATT_CONVERGENCE and abs(step * db) < _PLATT_CONVERGENCE:
            converged = True
            break

    if not converged:
        logger.warning(
            "platt fit did not converge after %d iterations, using fallback",
            _PLATT_MAX_ITER,
        )
        a = -1.0
        b = 0.0

    # Brier score on training data.
    preds: list[float] = [apply_platt(rp, a, b) for rp in raw_probs]
    bs = brier_score(preds, outcomes)

    return {"A": a, "B": b, "brier": bs, "samples": n}


def apply_platt(raw_prob: float, A: float, B: float) -> float:  # noqa: N803
    """Apply fitted Platt scaling.

    ``calibrated = 1 / (1 + exp(A * raw_prob + B))``

    The exponent is clamped to [-500, 500] to prevent overflow.
    """
    z = A * raw_prob + B
    z = max(-500.0, min(500.0, z))
    return 1.0 / (1.0 + math.exp(z))


def brier_score(probabilities: list[float], outcomes: list[bool]) -> float:
    """Mean squared error between predicted probabilities and outcomes.

    ``(1/N) * sum((p_i - o_i)^2)``  where o_i is 1.0 if True else 0.0.
    Returns 0.0 for empty inputs.
    """
    n = len(probabilities)
    if n == 0:
        return 0.0
    total = 0.0
    for p, o in zip(probabilities, outcomes):
        outcome_val = 1.0 if o else 0.0
        total += (p - outcome_val) ** 2
    return total / n


def no_vig_probability(over_odds: int, under_odds: int) -> float:
    """Convert American odds to no-vig implied probability for the over.

    Returns 0.5 when either line is zero or the implied probabilities
    cannot be computed.
    """
    if over_odds == 0 or under_odds == 0:
        return 0.5

    implied_over = _american_to_implied(over_odds)
    implied_under = _american_to_implied(under_odds)

    total = implied_over + implied_under
    if total <= 0.0:
        return 0.5

    return implied_over / total


def _american_to_implied(odds: int) -> float:
    """Convert a single American odds value to raw implied probability."""
    if odds > 0:
        return 100.0 / (odds + 100.0)
    # odds < 0 (odds == 0 handled by caller)
    return abs(odds) / (abs(odds) + 100.0)
