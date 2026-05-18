"""Playoff-mode algorithmic adjustments.

Three pure functions that modify the scoring chain when season_mode == "playoffs":
1. Cold-start anchor — meaningful projections with 0 playoff games
2. Trend trust phase-in — dampens trend signals until playoff sample is meaningful
3. Rotation tightening — structural prior for playoff minutes redistribution
"""

from __future__ import annotations

from typing import Any

from config import (
    PLAYOFF_COLD_START_ROLLING_WEIGHT,
    PLAYOFF_ELIMINATION_MULT,
    PLAYOFF_REST_GAME_MINUTES_THRESHOLD,
    PLAYOFF_ROTATION_TIERS,
)


def playoff_trend_trust(playoff_games: int, player_trust_score: float | None = None) -> float:
    """Return a [0, 1] trust factor for trend-derived signals during playoffs.

    With 0-1 playoff games, trends are pure noise (no slope possible).
    The static ramp serves as a floor when no empirical trust score exists.

    When player_trust_score is provided (from grade_playoff_trust grading),
    it takes precedence over the ramp — the ramp only applies as a minimum
    so a well-projecting player isn't held to 0 trust by the game-count gate.

    Args:
        playoff_games:      Number of playoff games played by this player.
        player_trust_score: Per-player MAE-derived trust score [0.25, 1.0],
                            or None when no games have been graded yet.
    """
    _RAMP: dict[int, float] = {0: 0.0, 1: 0.0, 2: 0.15, 3: 0.50, 4: 0.80}
    ramp_floor = _RAMP.get(playoff_games, 1.0)

    if player_trust_score is None:
        return ramp_floor

    # Use empirical score but never below the ramp floor — ensures cold-start
    # protection (games 0-1 stay at 0 regardless of historical grading).
    return max(ramp_floor, player_trust_score)


def playoff_cold_start_anchor(
    player: dict[str, Any],
    avg_fs_field: str = "avg_fantasy_score",
) -> dict[str, Any]:
    """Compute a playoff-appropriate FP anchor for the predicted_fp calculation.

    With 0 playoff games: blend the 5-game rolling avg (late regular-season form)
    with the full season fantasy score, filtering rest games from the rolling avg.

    As playoff games accumulate (1-4), gradually shift trust toward the standard
    anchor logic. At 5+ games the caller should use the normal anchor path.

    When series_fp_avg is present on the player dict (populated by load_series_stats
    in load_prediction_context), it is used as the primary rolling component instead
    of the heuristic _filter_rest_games() estimate. series_fp_avg is pre-filtered
    for rest games using actual minutes at write time, and is scoped to the current
    opponent — more accurate than the 30-day window that mixes pre-playoff games.

    Returns:
        {anchor: float, playoff_games: int, method: str}
    """
    playoff_games: int = player.get("playoff_games_played") or 0
    rolling_avg: float = player.get(avg_fs_field) or 0.0
    season_fs = compute_season_fantasy_score(player)

    # Prefer series_fp_avg (opponent-scoped, rest-filtered at write time) when
    # available; fall back to the heuristic for game 0 (no series data yet).
    series_avg: float | None = player.get("series_fp_avg")
    series_games: int = player.get("series_games") or 0
    if series_avg is not None and series_games > 0:
        filtered_rolling = series_avg
    else:
        filtered_rolling = _filter_rest_games(player, rolling_avg)

    anchor: float
    method: str

    if playoff_games == 0:
        # Pure cold start: trust recent regular-season form + season baseline
        if season_fs is not None and season_fs > 0 and filtered_rolling > 0:
            w = PLAYOFF_COLD_START_ROLLING_WEIGHT
            anchor = w * filtered_rolling + (1.0 - w) * season_fs
            method = "cold_start_blend"
        elif filtered_rolling > 0:
            anchor = filtered_rolling
            method = "cold_start_rolling_only"
        elif season_fs is not None and season_fs > 0:
            anchor = season_fs
            method = "cold_start_season_only"
        else:
            anchor = 0.0
            method = "cold_start_no_data"
    elif playoff_games < 5:
        # Transition: blend playoff series avg with cold-start anchor.
        # playoff_games=1 → 20% playoff, 80% cold-start
        # playoff_games=4 → 80% playoff, 20% cold-start
        # Use series_avg as the playoff component when available — it is
        # opponent-scoped and rest-filtered, unlike the 30d rolling_avg which
        # still mixes pre-playoff games in the first few games of a series.
        playoff_weight = playoff_games / 5.0
        series_component = series_avg if (series_avg is not None and series_games > 0) else rolling_avg
        cold_anchor: float = (
            (PLAYOFF_COLD_START_ROLLING_WEIGHT * filtered_rolling + (1.0 - PLAYOFF_COLD_START_ROLLING_WEIGHT) * (season_fs or filtered_rolling))
            if season_fs is not None and season_fs > 0
            else filtered_rolling
        )
        anchor = playoff_weight * series_component + (1.0 - playoff_weight) * cold_anchor
        method = f"transition_game_{playoff_games}"
    else:
        # 5+ games: caller should use normal anchor, but provide rolling as fallback
        anchor = rolling_avg
        method = "standard"

    return {
        "anchor": round(anchor, 2),
        "playoff_games": playoff_games,
        "method": method,
    }


def playoff_rotation_multiplier(player: dict[str, Any], playoff_games: int) -> dict[str, Any]:
    """Return a minutes-redistribution multiplier based on regular-season role.

    Playoff rotations shrink: starters play 36-42 min, deep bench goes to DNP.
    This is a structural prior that fades out over 5 games as the minutes_regime
    signal accumulates real playoff data.

    Returns:
        {playoff_rotation_multiplier: float, rotation_tier: str}
    """
    avg_min: float = player.get("avg_minutes") or 0.0

    # Determine raw multiplier from rotation tier
    raw_mult = 1.0
    tier = "bench"
    for threshold, mult, tier_name in PLAYOFF_ROTATION_TIERS:
        if avg_min >= threshold:
            raw_mult = mult
            tier = tier_name
            break

    # Fade out over 3 games: full strength at game 0, neutral at game 3+
    if playoff_games >= 3:
        effective_mult = 1.0
    else:
        fade = 1.0 - (playoff_games / 3.0)
        effective_mult = 1.0 + (raw_mult - 1.0) * fade

    return {
        "playoff_rotation_multiplier": round(effective_mult, 4),
        "rotation_tier": tier,
    }


def compute_season_fantasy_score(player: dict[str, Any]) -> float | None:
    """Compute DraftKings fantasy score from season averages stored on the player doc."""
    season_ppg: float | None = player.get("season_ppg")
    if season_ppg is None:
        return None
    # Use season_ppg availability as the quality gate — season_games now tracks
    # playoff game count, so it cannot be used as a minimum-sample guard here.
    if player.get("season_ppg") is None:
        return None
    result: float = (
        season_ppg * 1.0
        + float(player.get("season_rpg") or 0) * 1.2
        + float(player.get("season_apg") or 0) * 1.5
        + float(player.get("season_spg") or 0) * 3.0
        + float(player.get("season_bpg") or 0) * 3.0
        + float(player.get("season_ftmpg") or 0) * 1.0
        - float(player.get("season_topg") or 0) * 0.5
        - float(player.get("season_pfpg") or 0) * 0.25
    )
    return result


def _filter_rest_games(player: dict[str, Any], rolling_avg: float) -> float:
    """Filter rest games from the rolling average for high-minute players.

    If a player averages 25+ min but has recent games with < 15 min (rest games),
    those drag down the rolling average. Recompute excluding them.
    """
    avg_min: float = player.get("avg_minutes") or 0.0
    if avg_min < 25.0:
        return rolling_avg

    recent_scores: list[float] = player.get("recent_game_scores") or []
    recent_minutes: list[float] = player.get("recent_game_minutes") or []

    if not recent_scores or not recent_minutes:
        return rolling_avg

    # Take last 3 games to check for rest games
    check_scores = recent_scores[-3:]
    check_minutes = recent_minutes[-3:]

    filtered_scores = [float(s) for s, m in zip(check_scores, check_minutes) if float(m) >= PLAYOFF_REST_GAME_MINUTES_THRESHOLD]

    if not filtered_scores or len(filtered_scores) == len(check_scores):
        # No rest games found or all filtered out — use original
        return rolling_avg

    # Recompute with non-rest games only
    return round(sum(filtered_scores) / len(filtered_scores), 2)


def elimination_game_multiplier(player: dict[str, Any], rotation_tier: str | None) -> float:
    """Return a multiplier for elimination game pressure based on rotation tier.

    Only fires when is_elimination_game is True on the player dict (set by
    load_prediction_context when the player's series has elimination_game_next=True).
    Stars and starters get a small boost; fringe and bench players get penalized
    as coaches narrow their rotations in must-win games.

    Returns 1.0 (neutral) when not an elimination game or tier is unknown.
    """
    if not player.get("is_elimination_game"):
        return 1.0
    tier = rotation_tier or "bench"
    return PLAYOFF_ELIMINATION_MULT.get(tier, 1.0)
