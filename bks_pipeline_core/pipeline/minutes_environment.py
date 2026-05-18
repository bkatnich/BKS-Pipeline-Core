"""Minutes-environment interaction model (Priority 2).

Estimates blowout probability from Vegas spread and models the resulting
minutes distribution as a two-scenario mixture. Produces mode-aware
multipliers: cash penalizes blowout risk (floor), GPP rewards close
games (ceiling).

All functions are pure — no I/O or side effects.
"""

import logging
import math
from statistics import stdev
from typing import Any

logger = logging.getLogger(__name__)

# P(blowout) breakpoints: (abs_spread, probability)
# Piecewise-linear interpolation between breakpoints.
# Based on published NBA research (2015-2024 seasons aggregate).
# "Blowout" = starters losing 5+ minutes due to garbage time.
_BLOWOUT_BREAKPOINTS: list[tuple[float, float]] = [
    (0.0, 0.00),
    (3.0, 0.05),
    (6.0, 0.15),
    (9.0, 0.25),
    (12.0, 0.35),
    (15.0, 0.45),  # caps here
]

# Underdogs get this fraction of the favorite's blowout rate
_UNDERDOG_BLOWOUT_FACTOR = 0.40

# In a blowout, starters lose this fraction of their normal minutes
_BLOWOUT_MINUTES_REDUCTION = 0.20

# Minimum std for minutes distribution when sample is too small
_MIN_MINUTES_STD = 2.0

# Z-scores for floor/ceiling percentile computation.
# Asymmetric to account for NBA fantasy point fat tails: players have more
# extreme downside games (foul trouble, mid-game injuries, blowouts) than a
# normal distribution predicts.
# 2026-05-01 calibration (260 player-games, playoffs R1):
#   Was z_floor=-1.55, z_ceil=2.00 → actual 6.5% below floor, 14.2% above ceiling.
#   Lowered z_ceil to 1.28 (targets 10% above in normal → ~8-9% actual given fat tails).
#   Raised z_floor to -1.28 (loosens floor to bring below-floor rate up toward 10%).
# 2026-05-07 calibration (40 player-games, playoffs R2):
#   z_ceil=1.28 → actual 24% DK / 18% FD above ceiling — still too tight.
#   Raised z_ceil to 1.64 (targets 5% in normal → expect ~10-12% actual given fat tails).
#   Floor held at -1.28 (below-floor rate was 8% — acceptable).
_Z_FLOOR = -1.28
_Z_CEIL = 1.64

# Multiplier clamp range
_MULT_MIN = 0.90
_MULT_MAX = 1.12

# High-score skepticism: above this threshold, projections are dampened
# hyperbolically — the higher the score, the more it's pulled back toward
# the threshold. Asymptotes at 2× the threshold regardless of input.
_HIGH_SCORE_SKEPTICISM_THRESHOLD = 70.0
_FP_HARD_CAP = 100.0

# project_minutes() constants
_ROLE_CHANGE_EXPANSION_WEIGHT = 0.70  # trust new role 70%, historical anchor 30%
_MAX_PROJECTED_MINUTES = 42.0  # physical ceiling for any player
_MIN_PROJECTED_MINUTES = 5.0  # floor — prevents degenerate distribution inputs
_TREND_MIN_DEAD_ZONE = 0.10  # ignore slopes within ±0.10 (noise)
_TREND_MIN_INTENSITY = 0.08  # slope of 1.0 shifts minutes by 8%
_TREND_MIN_MAX_ADJ = 3.0  # cap slope adjustment at ±3 minutes
_B2B_STAR_THRESHOLD = 32.0  # avg_min threshold for star-tier B2B reduction
_B2B_STARTER_THRESHOLD = 24.0  # avg_min threshold for starter-tier B2B reduction
_B2B_STAR_REDUCTION = 2.5  # minutes lost on B2B for stars
_B2B_STARTER_REDUCTION = 1.5  # minutes lost on B2B for starters


def _dampen_high_score(fp: float, threshold: float | None = None) -> float:
    """Apply hyperbolic dampening to projections above threshold.

    Below the threshold: no change.
    Above the threshold: excess is compressed via fp = T + excess × T / (T + excess),
    which asymptotes at 2T regardless of input. The higher above T, the harder
    the pull-back. Capped at _FP_HARD_CAP after dampening.

    Examples (T=70):
        70  → 70.0  (no change)
        80  → 75.9  (−4.1)
        90  → 80.5  (−9.5)
        100 → 84.0  (−16.0)
    """
    T = threshold if threshold is not None else _HIGH_SCORE_SKEPTICISM_THRESHOLD
    if fp <= T:
        return fp
    excess = fp - T
    damped = T + excess * T / (T + excess)
    return min(_FP_HARD_CAP, damped)


def blowout_probability(spread: float, is_favorite: bool, *, is_playoffs: bool = False) -> float:
    """Estimate P(blowout) from the Vegas spread.

    Uses piecewise-linear interpolation within spread brackets.
    Favorites face full blowout risk; underdogs get a reduced rate.
    In playoffs, probability is dampened (closer games, elimination pressure).

    Args:
        spread: Team's spread (negative = favorite on most feeds,
                but we use abs value internally).
        is_favorite: Whether this team is favored.
        is_playoffs: Whether the game is a playoff game.

    Returns:
        Probability of a blowout scenario [0.0, 1.0].
    """
    from config import PLAYOFF_BLOWOUT_DAMPENING

    abs_spread = abs(spread)

    # Piecewise-linear interpolation between breakpoints
    bps = _BLOWOUT_BREAKPOINTS
    if abs_spread >= bps[-1][0]:
        prob = bps[-1][1]
    else:
        prob = 0.0
        for i in range(len(bps) - 1):
            s0, p0 = bps[i]
            s1, p1 = bps[i + 1]
            if abs_spread <= s1:
                frac = (abs_spread - s0) / (s1 - s0) if s1 > s0 else 0.0
                prob = p0 + frac * (p1 - p0)
                break

    if not is_favorite:
        prob *= _UNDERDOG_BLOWOUT_FACTOR

    if is_playoffs:
        prob *= PLAYOFF_BLOWOUT_DAMPENING

    return round(prob, 4)


def compute_minutes_distribution(
    avg_minutes: float,
    recent_game_minutes: list[float] | None,
    blowout_prob: float,
) -> dict[str, Any]:
    """Model minutes as a two-scenario mixture distribution.

    Scenario A (prob = 1 - blowout_prob): Normal game
        minutes ~ N(avg_minutes, observed_std)
    Scenario B (prob = blowout_prob): Blowout
        minutes ~ N(avg_minutes * (1 - reduction), observed_std)

    Args:
        avg_minutes: Player's rolling average minutes.
        recent_game_minutes: Last N per-game minutes (for std estimation).
        blowout_prob: P(blowout) from blowout_probability().

    Returns:
        Dict with mean_minutes, std_minutes, p10_minutes, p90_minutes.
    """
    # Estimate std from recent games
    if recent_game_minutes and len(recent_game_minutes) >= 2:
        obs_std = max(_MIN_MINUTES_STD, stdev(recent_game_minutes))
    else:
        obs_std = _MIN_MINUTES_STD

    p = blowout_prob
    mu_normal = avg_minutes
    mu_blowout = avg_minutes * (1.0 - _BLOWOUT_MINUTES_REDUCTION)

    # Mixture mean
    mixture_mean = (1.0 - p) * mu_normal + p * mu_blowout

    # Mixture variance (law of total variance)
    # Var = E[Var(X|S)] + Var(E[X|S])
    within_var = obs_std**2  # same within each scenario
    between_var = (1.0 - p) * p * (mu_normal - mu_blowout) ** 2
    mixture_var = within_var + between_var
    mixture_std = math.sqrt(mixture_var)

    # Percentiles via normal approximation
    p10 = max(0.0, mixture_mean + _Z_FLOOR * mixture_std)
    p90 = mixture_mean + _Z_CEIL * mixture_std

    return {
        "mean_minutes": round(mixture_mean, 2),
        "std_minutes": round(mixture_std, 2),
        "p10_minutes": round(p10, 2),
        "p90_minutes": round(p90, 2),
    }


def compute_fp_distribution(
    minutes_dist: dict[str, Any],
    avg_fantasy_score: float,
    avg_minutes: float,
    recent_game_scores: list[float] | None = None,
) -> dict[str, Any]:
    """Convert minutes distribution to fantasy points distribution.

    Combines minutes variance (from blowout model) with per-minute
    production variance (from recent game scores) for realistic
    p10/p90 bands.

    Args:
        minutes_dist: Output from compute_minutes_distribution().
        avg_fantasy_score: Player's rolling average fantasy points.
        avg_minutes: Player's rolling average minutes.
        recent_game_scores: Recent per-game fantasy scores for
            production variance estimation. Falls back to CV=0.25
            default if None or too few entries.

    Returns:
        Dict with mean_fp, p10_fp (floor), p90_fp (ceiling), fp_per_minute.
    """
    if avg_minutes <= 0:
        return {
            "mean_fp": 0.0,
            "p10_fp": 0.0,
            "p90_fp": 0.0,
            "fp_per_minute": 0.0,
        }

    fp_per_min = avg_fantasy_score / avg_minutes
    mean_fp = fp_per_min * minutes_dist["mean_minutes"]

    # --- Minutes-driven FP variance ---
    minutes_std = (minutes_dist["p90_minutes"] - minutes_dist["p10_minutes"]) / (_Z_CEIL - _Z_FLOOR)  # invert asymmetric z-score spread
    fp_var_from_minutes = (fp_per_min * minutes_std) ** 2

    # --- Production variance (shooting, usage, matchup effects) ---
    # Cap at CV=0.45 to prevent outlier games (e.g. Wembanyama 90-pt explosion)
    # from blowing up the std on a 5-game sample and producing impossible ceilings.
    _max_production_std = 0.45 * avg_fantasy_score
    if recent_game_scores and len(recent_game_scores) >= 3:
        fp_production_std = min(stdev(recent_game_scores), _max_production_std)
    else:
        fp_production_std = 0.25 * avg_fantasy_score  # conservative CV=0.25
    fp_var_from_production = fp_production_std**2

    # --- Combined variance ---
    total_fp_std = math.sqrt(fp_var_from_minutes + fp_var_from_production)
    p10_fp = max(0.0, mean_fp + _Z_FLOOR * total_fp_std)
    # Apply high-score skepticism: dampen the mean once. Ceiling is built on the dampened
    # mean without a second pass — double-dampening would over-compress elite ceilings.
    mean_fp = _dampen_high_score(mean_fp)
    p90_fp = mean_fp + _Z_CEIL * total_fp_std

    return {
        "mean_fp": round(mean_fp, 2),
        "total_fp_std": round(total_fp_std, 2),
        "p10_fp": round(p10_fp, 2),
        "p90_fp": round(p90_fp, 2),
        "fp_per_minute": round(fp_per_min, 4),
    }


def project_minutes(
    player: dict[str, Any],
    *,
    is_back_to_back: bool = False,
    is_playoffs: bool = False,
    playoff_games: int = 0,
) -> dict[str, Any]:
    """Synthesize a forward-looking projected_minutes anchor from player signals.

    Synthesizes four adjustments on top of avg_minutes (backward-looking base):
      1. Role change expansion — blends the new expanded role at 70%
      2. Minutes trend slope — forward nudge from the last-5g slope
      3. B2B reduction — starters lose 1.5-2.5 min on back-to-backs
      4. Playoff rotation prior — fades in/out over the first 5 playoff games

    All signals come from the player dict — no I/O.

    Returns:
        {
            "projected_minutes": float,
            "projection_components": dict,   # diagnostic breakdown
        }
    """
    avg_min: float = float(player.get("avg_minutes") or 0.0)
    if avg_min <= 0.0:
        return {"projected_minutes": 0.0, "projection_components": {}}

    base = avg_min
    components: dict[str, float] = {}

    # 1. Role change expansion
    is_rc: bool = bool(player.get("is_role_change"))
    rc_delta: float | None = player.get("role_change_minutes_delta")
    role_adj = 0.0
    if is_rc and rc_delta is not None and rc_delta > 0:
        expanded = min(_MAX_PROJECTED_MINUTES, avg_min + rc_delta)
        new_base = _ROLE_CHANGE_EXPANSION_WEIGHT * expanded + (1.0 - _ROLE_CHANGE_EXPANSION_WEIGHT) * avg_min
        role_adj = new_base - base
        base = new_base
    components["role_change_adj"] = round(role_adj, 2)

    # 2. Trend slope on minutes
    trend_min: float = float(player.get("trend_min") or 0.0)
    slope_adj = 0.0
    if abs(trend_min) > _TREND_MIN_DEAD_ZONE:
        effective_slope = trend_min - (_TREND_MIN_DEAD_ZONE if trend_min > 0 else -_TREND_MIN_DEAD_ZONE)
        raw_adj = base * effective_slope * _TREND_MIN_INTENSITY
        slope_adj = max(-_TREND_MIN_MAX_ADJ, min(_TREND_MIN_MAX_ADJ, raw_adj))
        base = base + slope_adj
    components["trend_slope_adj"] = round(slope_adj, 2)

    # 3. B2B minutes reduction (applied after role change so tier reflects projected role)
    b2b_adj = 0.0
    if is_back_to_back:
        if base >= _B2B_STAR_THRESHOLD:
            b2b_adj = -_B2B_STAR_REDUCTION
        elif base >= _B2B_STARTER_THRESHOLD:
            b2b_adj = -_B2B_STARTER_REDUCTION
        base = base + b2b_adj
    components["b2b_adj"] = round(b2b_adj, 2)

    # 4. Playoff rotation prior (fades over 3 games — matches playoff_adjustments.py)
    rotation_adj = 0.0
    if is_playoffs and playoff_games < 3:
        from config import PLAYOFF_ROTATION_TIERS

        raw_mult = 1.0
        for threshold, mult, _ in PLAYOFF_ROTATION_TIERS:
            if avg_min >= threshold:
                raw_mult = mult
                break
        fade = 1.0 - (playoff_games / 3.0)
        effective_mult = 1.0 + (raw_mult - 1.0) * fade
        new_base = base * effective_mult
        rotation_adj = new_base - base
        base = new_base
    components["rotation_adj"] = round(rotation_adj, 2)

    # 5. Calibration bias correction: accuracy report (2026-04-22) showed +3.09 min
    # systematic over-projection. Subtract the correction before clamping.
    from config import PROJECTED_MINUTES_BIAS_CORRECTION

    base = base - PROJECTED_MINUTES_BIAS_CORRECTION
    components["bias_correction"] = -PROJECTED_MINUTES_BIAS_CORRECTION

    projected = round(max(_MIN_PROJECTED_MINUTES, min(_MAX_PROJECTED_MINUTES, base)), 1)
    components["base_avg_minutes"] = avg_min
    return {"projected_minutes": projected, "projection_components": components}


def minutes_environment_multiplier(
    player: dict[str, Any],
    vegas_team: dict[str, Any] | None,
    mode: str = "balanced",
    *,
    is_playoffs: bool = False,
    is_back_to_back: bool = False,
    playoff_games: int = 0,
    avg_fantasy_score_override: float | None = None,
    recent_game_scores_override: list[float] | None = None,
) -> dict[str, Any]:
    """Compute the minutes-environment interaction multiplier.

    Main entry point. Combines blowout probability, minutes distribution,
    and FP distribution to produce a single mode-aware multiplier.

    Args:
        player: Player dict with avg_minutes, recent_game_minutes,
                avg_fantasy_score.
        vegas_team: Team's vegas signals dict {spread, is_favorite, ...}.
                    None or empty if no odds available.
        mode: "balanced", "cash", or "gpp".
        is_playoffs: Whether the game is a playoff game (dampens blowout prob).
        is_back_to_back: Whether the player's team is on a back-to-back.
        playoff_games: Number of playoff games played (for rotation prior fade).
        avg_fantasy_score_override: Use instead of player["avg_fantasy_score"]
                    when computing for a specific platform (e.g. FD).
        recent_game_scores_override: Use instead of player["recent_game_scores"]
                    for platform-specific production variance (e.g. FD scores).

    Returns:
        Dict with minutes_env_multiplier, blowout_prob, minutes_ceiling,
        minutes_floor, fp_ceiling, fp_floor, projected_minutes.
    """
    neutral = {
        "minutes_env_multiplier": 1.0,
        "blowout_prob": None,
        "minutes_ceiling": None,
        "minutes_floor": None,
        "fp_ceiling": None,
        "fp_floor": None,
        "fp_mu": None,
        "fp_sigma": None,
    }

    avg_min = player.get("avg_minutes")
    avg_fs = avg_fantasy_score_override if avg_fantasy_score_override is not None else player.get("avg_fantasy_score")
    if not avg_min or avg_min <= 0 or not avg_fs or avg_fs <= 0:
        return neutral

    # When no Vegas odds are available (e.g. future dates in projections),
    # assume a neutral/pick-em game (blowout_prob=0) so floor/ceiling can
    # still be computed from minutes variance and production variance alone.
    spread = (vegas_team or {}).get("spread")
    is_favorite = (vegas_team or {}).get("is_favorite")
    if spread is not None and is_favorite is not None:
        bp = blowout_probability(spread, is_favorite, is_playoffs=is_playoffs)
    else:
        bp = 0.0  # no odds → assume neutral game, no blowout risk

    # 2. Forward-looking minutes anchor (role change, B2B, slope, playoff rotation)
    pm = project_minutes(player, is_back_to_back=is_back_to_back, is_playoffs=is_playoffs, playoff_games=playoff_games)
    projected_anchor = pm["projected_minutes"] if pm["projected_minutes"] > 0 else float(avg_min)

    # 3. Minutes distribution — anchored to projected minutes, not raw avg
    recent_min = player.get("recent_game_minutes")
    min_dist = compute_minutes_distribution(projected_anchor, recent_min, bp)

    # 4. FP distribution — fp_per_min uses avg_min (historical rate, not projected)
    _recent_scores = recent_game_scores_override if recent_game_scores_override is not None else player.get("recent_game_scores")
    fp_dist = compute_fp_distribution(min_dist, avg_fs, float(avg_min), _recent_scores)

    # 5. Mode-aware multiplier
    mean_fp = fp_dist["mean_fp"]
    if mean_fp <= 0:
        return neutral

    if mode == "cash":
        raw_mult = fp_dist["p10_fp"] / mean_fp
    elif mode == "gpp":
        raw_mult = fp_dist["p90_fp"] / mean_fp
    else:
        # Balanced: blowout risk pulls expected FP below baseline.
        # mean_fp already accounts for the blowout mixture model, so the
        # ratio to avg_fantasy_score captures the risk penalty directly.
        raw_mult = mean_fp / avg_fs

    multiplier = max(_MULT_MIN, min(_MULT_MAX, raw_mult))

    logger.debug(
        "Minutes env: blowout_prob=%.3f, min=[%.1f, %.1f], fp=[%.1f, %.1f], mult=%.4f (mode=%s)",
        bp,
        min_dist["p10_minutes"],
        min_dist["p90_minutes"],
        fp_dist["p10_fp"],
        fp_dist["p90_fp"],
        multiplier,
        mode,
    )

    return {
        "minutes_env_multiplier": round(multiplier, 4),
        "blowout_prob": bp,
        "minutes_ceiling": min_dist["p90_minutes"],
        "minutes_floor": min_dist["p10_minutes"],
        "fp_ceiling": fp_dist["p90_fp"],
        "fp_floor": fp_dist["p10_fp"],
        "fp_mu": fp_dist["mean_fp"],
        "fp_sigma": fp_dist["total_fp_std"],
        "projected_minutes": pm["projected_minutes"],
    }
