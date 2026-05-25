"""Shared constants and utility functions for opportunity scoring pipelines.

Used by both pipeline.opportunities (regular season + playoffs mode)
and pipeline.playoffs_pipeline.opportunities (playoffs-only engine).
"""

import logging
from typing import Any

from bks_pipeline_core.pipeline.platforms import DEFAULT_PLATFORM  # noqa: F401 (re-exported for callers)
from bks_pipeline_core.sport_config import get_active_config

logger = logging.getLogger(__name__)

__all__ = [
    "DEFAULT_PLATFORM",
    "VALID_MODES",
    "DEFAULT_MODE",
    "_MODE_WEIGHTS",
    "_USAGE_DELTA_DEAD_ZONE",
    "_USAGE_DELTA_INTENSITY",
    "_USAGE_DELTA_MAX",
    "_USAGE_DELTA_MIN",
    "_assign_top_picks",
    "_cat_trend_multiplier",
    "_clamp",
    "_health_factor",
    "_position_bucket",
    "_position_group",
    "_shooting_luck_multiplier",
]

# ---------------------------------------------------------------------------
# Health & eligibility
# ---------------------------------------------------------------------------

# Softened 2026-04-10: by pipeline run-time (6:30 PM ET) truly sidelined
# players are already "out" and excluded.  Remaining Q/D designations are
# playing — light discount only for minutes-limit risk.
_HEALTH_FACTOR: dict[str, float] = {
    "doubtful": 0.85,
    "questionable": 0.95,
    "day-to-day": 0.95,
    "probable": 1.0,  # expected to play — no discount, but explicit mapping avoids unknown-status fallback
}
_EXCLUDED_STATUSES = {
    "out",
    "out for season",
    "out indefinitely",
    "suspension",
    "suspended",
    "inactive",
    "not with team",
    "g league",
    "g league - on assignment",
}

# ---------------------------------------------------------------------------
# Position grouping & top picks
# ---------------------------------------------------------------------------

_POSITION_GROUP: dict[str, str] = {
    "PG": "G",
    "SG": "G",
    "G": "G",
    "G-F": "G",
    "SF": "F",
    "PF": "F",
    "F": "F",
    "F-C": "F",
    "C": "C",
    "C-F": "C",
}
_TOP_PICK_TARGET = 12  # exactly 3 positions × 4 tiers, one per slot

# ---------------------------------------------------------------------------
# Scoring mode profiles
# ---------------------------------------------------------------------------

# Shift weights for cash (floor) vs GPP (ceiling) contests.
# w_production raised from 0.40 on 2026-04-21: hot streak was over-ranking
# low-volume players.
_MODE_WEIGHTS: dict[str, dict[str, float]] = {
    "balanced": {
        "w_production": 0.50,
        "w_confidence": 0.20,
        "w_hot_streak": 0.10,
        "w_consistency": 0.20,
        "accel_cap": 0.05,
        "vegas_sensitivity": 1.0,
    },
    "cash": {
        "w_production": 0.45,
        "w_confidence": 0.20,
        "w_hot_streak": 0.10,
        "w_consistency": 0.25,
        "accel_cap": 0.03,
        "vegas_sensitivity": 0.5,
    },
    "gpp": {
        "w_production": 0.30,
        "w_confidence": 0.25,
        "w_hot_streak": 0.25,
        "w_consistency": 0.10,
        "accel_cap": 0.08,
        "vegas_sensitivity": 1.5,
    },
}

VALID_MODES = set(_MODE_WEIGHTS.keys())
DEFAULT_MODE = "gpp"

# ---------------------------------------------------------------------------
# Signal tuning constants
# ---------------------------------------------------------------------------

# Usage efficiency — derived from minutes vs. points trend relationship
_EFFICIENCY_MULT: dict[str, float] = {
    "expanding_efficiently": 1.02,
    "efficient_usage": 1.01,
    "volume_inflation": 0.98,
    "neutral": 1.0,
}

# Cat trend: points-only, dead-zone gated, tight clamp
_CAT_TREND_DEAD_ZONE = 0.10
_CAT_TREND_INTENSITY = 0.15
_CAT_TREND_MIN = 0.98
_CAT_TREND_MAX = 1.02

# Shooting luck regression: compares recent 5-game 3PT% to season 3PT%.
# Only fires for players averaging 3+ 3PA per game (filters non-shooters).
_SHOOTING_LUCK_DEAD_ZONE = 0.03
_SHOOTING_LUCK_REGRESSION = 0.35
_SHOOTING_LUCK_MIN_3PA = 3.0
_SHOOTING_LUCK_MIN = 0.97
_SHOOTING_LUCK_MAX = 1.03
_SHOOTING_LUCK_MIN_PLAYOFF_3PA = 10  # minimum total playoff 3PA before regression fires

# Usage delta: leading indicator of role expansion/contraction.
_USAGE_DELTA_DEAD_ZONE = 0.10
_USAGE_DELTA_INTENSITY = 0.40
_USAGE_DELTA_MIN = 0.98
_USAGE_DELTA_MAX = 1.04

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------


def _position_group(position: str | None) -> str:
    """Map a raw BDL position string to G, F, or C."""
    return _POSITION_GROUP.get((position or "").strip(), "F")


def _health_factor(injury_status: str | None) -> float | None:
    """Return health multiplier, or None if the player should be excluded entirely."""
    if not injury_status:
        return 1.0
    lower = injury_status.lower()
    if lower in _EXCLUDED_STATUSES:
        return None
    return _HEALTH_FACTOR.get(lower, 1.0)


def _position_bucket(position: str | None) -> str:
    """Map raw BDL position string to a defense bucket key."""
    cfg = get_active_config()
    if not position:
        return cfg.default_position_bucket
    return cfg.position_bucket_map.get(position.strip(), cfg.default_position_bucket)


def _clamp(
    val: float,
    signal_name: str,
    signal_clamps: dict[str, dict[str, float]] | None,
    default_min: float,
    default_max: float,
) -> float:
    """Clamp a multiplier value using tuned or default bounds."""
    if signal_clamps and signal_name in signal_clamps:
        sc = signal_clamps[signal_name]
        lo = sc.get("clamp_min", default_min)
        hi = sc.get("clamp_max", default_max)
    else:
        lo, hi = default_min, default_max
    return max(lo, min(hi, val))


def _cat_trend_multiplier(
    player: dict[str, Any],
    signal_clamps: dict[str, dict[str, float]] | None = None,
) -> float:
    """Disabled 2026-04-23: r=-0.156 DK / r=-0.042 FD, negative over 7-day window.
    Points-trend slope is negatively correlated with outcomes in playoffs.
    Returns 1.0 unconditionally.
    """
    return 1.0


def _shooting_luck_multiplier(
    player: dict[str, Any],
    signal_clamps: dict[str, dict[str, float]] | None = None,
    playoff_games: int = 0,
    vs_opponent_fg3_pct: float | None = None,
    vs_opponent_games: int = 0,
    min_opponent_games: int = 2,
) -> float:
    """Shooting luck regression multiplier.

    Compares recent 5-game 3PT% to season 3PT%. When a player is shooting
    well above their season rate, expect regression (downward adjustment);
    when shooting below, expect bounce-back (upward adjustment).

    The multiplier is INVERSE to the luck direction: positive luck → mult < 1.0.
    Only fires for players averaging 3+ 3PA per game (meaningful sample).

    In playoffs: also requires 10+ total playoff 3PA before firing. With fewer
    attempts, the recent_fg3_pct is a 4-7 game sample against one defense —
    regressing it toward the season rate adds noise rather than signal.

    Vs-opponent baseline (playoffs, NBA_STATS_ENABLED): when the player has
    min_opponent_games+ historical games vs this opponent, uses
    vs_opponent_fg3_pct as the expected rate instead of season_fg3_pct.
    Regular-season callers pass min_opponent_games=2; playoff-only callers
    pass min_opponent_games=3 (larger required sample for a single-series
    defense profile to be trusted).
    """
    recent_pct = player.get("recent_fg3_pct")
    season_pct = player.get("season_fg3_pct")
    attempts_pg = player.get("recent_fg3_attempts_pg") or 0.0
    if recent_pct is None or season_pct is None:
        return 1.0
    if attempts_pg < _SHOOTING_LUCK_MIN_3PA:
        return 1.0
    if playoff_games > 0:
        total_playoff_3pa = attempts_pg * min(playoff_games, 5)
        if total_playoff_3pa < _SHOOTING_LUCK_MIN_PLAYOFF_3PA:
            return 1.0
    if playoff_games > 0 and vs_opponent_fg3_pct is not None and vs_opponent_games >= min_opponent_games:
        adjusted_luck = recent_pct - vs_opponent_fg3_pct
        if abs(adjusted_luck) < _SHOOTING_LUCK_DEAD_ZONE:
            return 1.0
        luck = adjusted_luck
    else:
        luck = recent_pct - season_pct
        if abs(luck) < _SHOOTING_LUCK_DEAD_ZONE:
            return 1.0
    raw = 1.0 - luck * _SHOOTING_LUCK_REGRESSION
    return float(round(_clamp(raw, "shooting_luck_multiplier", signal_clamps, _SHOOTING_LUCK_MIN, _SHOOTING_LUCK_MAX), 4))


def _top_pick_reasons(
    player: dict[str, Any],
    tier_players: list[dict[str, Any]],
    role_change_playoff_gate: bool = False,
) -> list[str]:
    """Generate up to 3 signal-based reason tags for a Top Pick within its tier.

    role_change_playoff_gate: when True, the "Role change" reason is only
    emitted for players with fewer than 5 playoff games played (playoffs-only
    engine behavior).
    """
    reasons: list[str] = []

    max_matchup = max((p.get("matchup_multiplier") or 1.0 for p in tier_players), default=1.0)
    if (player.get("matchup_multiplier") or 1.0) >= max_matchup and max_matchup > 1.0:
        reasons.append("Best matchup in tier")

    max_vegas = max((p.get("vegas_multiplier") or 1.0 for p in tier_players), default=1.0)
    if (player.get("vegas_multiplier") or 1.0) >= max_vegas and max_vegas > 1.0:
        reasons.append("Top Vegas spot in tier")

    if (player.get("hot_streak") or 0) >= 3:
        reasons.append("Hot streak")

    role_change_ok = True
    if role_change_playoff_gate:
        role_change_ok = (player.get("playoff_games_played") or 0) < 5
    if player.get("is_role_change") and role_change_ok:
        reasons.append("Role change — expanded role")

    if (player.get("shooting_luck_multiplier") or 1.0) > 1.01:
        reasons.append("Shooting bounceback")

    if (player.get("opp_rest_multiplier") or 1.0) > 1.0:
        reasons.append("Rested vs B2B opponent")

    if (player.get("playoff_rotation_multiplier") or 1.0) > 1.03:
        reasons.append("Strong rotation tier")

    tier_cons = sorted((p.get("consistency_score") or 0.0 for p in tier_players), reverse=True)
    if tier_cons:
        threshold = tier_cons[max(0, len(tier_cons) // 4 - 1)]
        if threshold > 0 and (player.get("consistency_score") or 0.0) >= threshold:
            reasons.append("High consistency")

    if (player.get("def_ratio_at_position") or 1.0) > 1.05:
        reasons.append("Soft defense")

    if not reasons:
        reasons.append("Top projected score in tier")

    return reasons[:3]


def _assign_top_picks(
    results: list[dict[str, Any]],
    all_results: list[dict[str, Any]] | None = None,
    role_change_playoff_gate: bool = False,
) -> None:
    """Flag top picks: exactly one G, F, and C per bucket (12 total).

    Two-phase selection (operates in-place on the sorted results list):
      Phase 1 — one pick per position group (G/F/C) per bucket, highest score first.
      Phase 2 — per-bucket guarantee: every bucket gets all 3 position groups; missing
                 groups are filled from the full pool (all_results) by score.

    Within each bucket, top_pick_rank is assigned 1-indexed by opportunity_score desc.
    all_results: the full unsliced scored list; falls back to results when not provided.
    role_change_playoff_gate: passed through to _top_pick_reasons (see its docstring).
    """
    _BUCKETS = ["elite_opp", "good_opp", "solid_opp", "low_opp"]
    _GROUPS = ("G", "F", "C")
    pool = all_results if all_results is not None else results

    bucket_groups: dict[str, list[dict[str, Any]]] = {b: [] for b in _BUCKETS}
    for r in results:
        bucket_groups[r.get("_opp_bucket", "low_opp")].append(r)

    selected: list[dict[str, Any]] = []
    selected_ids: set = set()

    # Phase 1: one per position group per bucket (highest score wins within group)
    for bucket in _BUCKETS:
        seen_groups: set[str] = set()
        for player in bucket_groups[bucket]:
            grp = _position_group(player.get("position"))
            if grp not in seen_groups:
                selected.append(player)
                selected_ids.add(player["id"])
                seen_groups.add(grp)

    # Phase 2: per-bucket position guarantee — every bucket must have all 3 groups.
    gaps: list[tuple[str, str]] = []
    for bucket in _BUCKETS:
        bucket_pos_groups = {_position_group(p.get("position")) for p in selected if p.get("_opp_bucket") == bucket}
        for grp in _GROUPS:
            if grp not in bucket_pos_groups:
                gaps.append((bucket, grp))

    for bucket, grp in gaps:
        for player in pool:
            if player["id"] in selected_ids:
                continue
            if _position_group(player.get("position")) == grp:
                player["_opp_bucket"] = bucket
                selected.append(player)
                selected_ids.add(player["id"])
                bucket_groups[bucket].append(player)
                break

    # Reset all pick fields first
    for r in results:
        r["is_top_pick"] = False
        r["top_pick_rank"] = None
        r["top_pick_reasons"] = []
        r["is_top_ceiling"] = False
        r["top_ceiling_rank"] = None
        r["is_top_value"] = False
        r["top_value_rank"] = None

    # Assign is_top_pick and per-bucket rank (score-ordered within bucket)
    bucket_rank_counter: dict[str, int] = {b: 0 for b in _BUCKETS}
    for player in sorted(selected, key=lambda p: p["opportunity_score"], reverse=True):
        bucket = player.get("_opp_bucket", "low_opp")
        bucket_rank_counter[bucket] += 1
        player["is_top_pick"] = True
        player["top_pick_rank"] = bucket_rank_counter[bucket]
        player["top_pick_reasons"] = _top_pick_reasons(
            player, bucket_groups[bucket], role_change_playoff_gate=role_change_playoff_gate
        )

    # Top 3 ceiling: highest opportunity_score on the slate (results already sorted desc)
    for i, player in enumerate(results[:3], start=1):
        player["is_top_ceiling"] = True
        player["top_ceiling_rank"] = i

    # Top 3 value: best sal_val_mult among solid_opp + good_opp players
    value_pool = [r for r in results if r.get("_opp_bucket") in ("solid_opp", "good_opp")]
    value_pool.sort(key=lambda p: p.get("sal_val_mult", 1.0), reverse=True)
    for i, player in enumerate(value_pool[:3], start=1):
        player["is_top_value"] = True
        player["top_value_rank"] = i
