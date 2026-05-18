"""Correlated stacking optimization (Priority 5).

Computes a slate-aware stacking multiplier that boosts players in
high-correlation environments (high O/U, fast pace, multiple eligible
teammates) for GPP, and slightly penalizes high-variance stacking
situations in cash.

This is a request-time signal — computed from the game slate context,
not stored in Firestore.

All functions are pure — no I/O or side effects.
"""

import logging
from collections import defaultdict
from statistics import stdev
from typing import Any

logger = logging.getLogger(__name__)

# Multiplier clamp range
_STACK_MULT_MIN = 0.92
_STACK_MULT_MAX = 1.12

# How much raw_stack deviates from center (0.5) before mode scaling
_STACK_BASE_SCALE = 0.08

_MODE_SENSITIVITY: dict[str, float] = {
    "gpp": 2.0,
    "balanced": 0.5,
    "cash": -0.3,
}

# Environment score component weights
_OU_WEIGHT = 0.3
_PACE_WEIGHT = 0.2

# Composite weights: environment vs density
_ENV_COMPONENT_WEIGHT = 0.6
_DENSITY_COMPONENT_WEIGHT = 0.4

# Density scaling: min(1.0, (density - 1) / _DENSITY_DIVISOR)
_DENSITY_DIVISOR = 4

# Signal classification thresholds
_HIGH_STACK_THRESHOLD = 0.7
_MODERATE_STACK_THRESHOLD = 0.4
_NEUTRAL_STACK_THRESHOLD = 0.2


def _game_key(team_a: str, team_b: str) -> str:
    """Canonical game key: alphabetically sorted team pair."""
    return "_".join(sorted([team_a, team_b]))


def build_stacking_context(
    eligible_players: list[dict[str, Any]],
    games_doc: dict[str, Any],
    vegas_signals: dict[str, Any] | None,
    pace_map: dict[str, Any] | None,
) -> dict[str, Any]:
    """Build slate-level stacking context from the game environment.

    Called once before the scoring loop. Analyzes the full slate to
    compute per-game environment scores and player density counts.

    Args:
        eligible_players: List of player dicts with '_team_abbr' injected.
        games_doc: Today's games document with 'games' list.
        vegas_signals: Per-team Vegas signals (may be None).
        pace_map: Per-team pace dict (may be None).

    Returns:
        Stacking context dict with game_environments, team_density,
        game_density, team_game_key, slate_avg_over_under.
    """
    empty_ctx: dict[str, object] = {
        "game_environments": {},
        "team_density": {},
        "game_density": {},
        "team_game_key": {},
        "slate_avg_over_under": None,
    }

    games = games_doc.get("games", [])
    if not games:
        return empty_ctx

    # Build team → game key mapping and collect per-game data
    team_game_map: dict[str, str] = {}
    game_data: dict[str, dict[str, str]] = {}

    for g in games:
        home = g.get("home_team_abbr")
        away = g.get("visitor_team_abbr")
        if not home or not away:
            continue
        gk = _game_key(home, away)
        team_game_map[home] = gk
        team_game_map[away] = gk
        game_data[gk] = {"home": home, "away": away}

    if not game_data:
        return empty_ctx

    # Collect O/U per game from vegas signals
    game_ou: dict[str, float] = {}
    if vegas_signals:
        for gk, gd in game_data.items():
            home_sig = vegas_signals.get(gd["home"], {})
            ou = home_sig.get("over_under")
            if ou is not None:
                game_ou[gk] = ou

    # Collect pace per game (average of both teams' pace)
    game_pace: dict[str, float] = {}
    if pace_map:
        for gk, gd in game_data.items():
            home_pace = pace_map.get(gd["home"])
            away_pace = pace_map.get(gd["away"])
            paces = [p for p in [home_pace, away_pace] if p is not None]
            if paces:
                game_pace[gk] = sum(paces) / len(paces)

    # Compute z-scores for O/U and pace across the slate
    ou_values = list(game_ou.values())
    pace_values = list(game_pace.values())

    slate_avg_ou = sum(ou_values) / len(ou_values) if ou_values else None
    slate_std_ou = stdev(ou_values) if len(ou_values) >= 2 else None
    slate_avg_pace = sum(pace_values) / len(pace_values) if pace_values else None
    slate_std_pace = stdev(pace_values) if len(pace_values) >= 2 else None

    # Compute environment score per game
    game_environments: dict[str, dict[str, Any]] = {}
    for gk in game_data:
        ou_z = 0.0
        if gk in game_ou and slate_avg_ou is not None and slate_std_ou and slate_std_ou > 0:
            ou_z = max(-2.0, min(2.0, (game_ou[gk] - slate_avg_ou) / slate_std_ou))

        pace_z = 0.0
        if gk in game_pace and slate_avg_pace is not None and slate_std_pace and slate_std_pace > 0:
            pace_z = max(-2.0, min(2.0, (game_pace[gk] - slate_avg_pace) / slate_std_pace))

        env_score = max(0.0, min(1.0, 0.5 + _OU_WEIGHT * ou_z + _PACE_WEIGHT * pace_z))

        game_environments[gk] = {
            "over_under": game_ou.get(gk),
            "pace": game_pace.get(gk),
            "environment_score": round(env_score, 4),
        }

    # Count team density (eligible players per team) and game density
    team_density: dict[str, int] = defaultdict(int)
    for p in eligible_players:
        t = p.get("_team_abbr") or p.get("team")
        if t:
            team_density[t] += 1

    game_density: dict[str, int] = defaultdict(int)
    for team, gk in team_game_map.items():
        game_density[gk] += team_density.get(team, 0)

    high_env_count = sum(1 for ge in game_environments.values() if ge["environment_score"] > 0.7)
    logger.debug(
        "Stacking context: %d games, avg_ou=%s, avg_pace=%s, high_env_games=%d",
        len(game_data),
        f"{slate_avg_ou:.1f}" if slate_avg_ou else "N/A",
        f"{slate_avg_pace:.1f}" if slate_avg_pace else "N/A",
        high_env_count,
    )

    return {
        "game_environments": game_environments,
        "team_density": dict(team_density),
        "game_density": dict(game_density),
        "team_game_key": team_game_map,
        "slate_avg_over_under": round(slate_avg_ou, 1) if slate_avg_ou else None,
    }


def stacking_multiplier(
    player_team: str,
    opponent_team: str,
    stacking_context: dict[str, Any],
    mode: str = "balanced",
) -> dict[str, Any]:
    """Compute the stacking multiplier for a player based on slate context.

    Args:
        player_team: Player's team abbreviation.
        opponent_team: Opponent's team abbreviation.
        stacking_context: Pre-computed context from build_stacking_context().
        mode: Scoring mode — "balanced", "cash", or "gpp".

    Returns:
        Dict with stacking_multiplier, stacking_environment,
        stacking_team_density, stacking_game_density, stacking_signal.
    """
    neutral = {
        "stacking_multiplier": 1.0,
        "stacking_environment": None,
        "stacking_team_density": 0,
        "stacking_game_density": 0,
        "stacking_signal": "neutral",
    }

    if not stacking_context or not stacking_context.get("game_environments"):
        return neutral

    # Look up game key for this player
    team_game_map = stacking_context.get("team_game_key", {})
    gk = team_game_map.get(player_team)
    if not gk:
        # Try constructing it from player_team + opponent_team
        gk = _game_key(player_team, opponent_team)

    game_env = stacking_context.get("game_environments", {}).get(gk)
    if not game_env:
        return neutral

    env_score = game_env.get("environment_score", 0.5)
    team_dens = stacking_context.get("team_density", {}).get(player_team, 0)
    game_dens = stacking_context.get("game_density", {}).get(gk, 0)

    # Density component: 1 player = 0.0, 5+ = 1.0
    density_component = min(1.0, max(0.0, (team_dens - 1) / _DENSITY_DIVISOR))

    # Composite raw stacking score
    raw_stack = _ENV_COMPONENT_WEIGHT * env_score + _DENSITY_COMPONENT_WEIGHT * density_component

    # Classify signal
    if raw_stack >= _HIGH_STACK_THRESHOLD:
        signal = "high_stack"
    elif raw_stack >= _MODERATE_STACK_THRESHOLD:
        signal = "moderate_stack"
    elif raw_stack >= _NEUTRAL_STACK_THRESHOLD:
        signal = "neutral"
    else:
        signal = "anti_stack"

    # Dead zone: near-neutral stacking scores add noise, not signal.
    # Only apply a multiplier when the deviation is meaningful.
    _DEAD_ZONE = 0.15
    if abs(raw_stack - 0.5) < _DEAD_ZONE:
        return {
            "stacking_multiplier": 1.0,
            "stacking_environment": round(env_score, 4),
            "stacking_team_density": team_dens,
            "stacking_game_density": game_dens,
            "stacking_signal": signal,
        }

    # Mode-aware multiplier
    sensitivity = _MODE_SENSITIVITY.get(mode, 0.0)
    base_boost = (raw_stack - 0.5) * _STACK_BASE_SCALE
    multiplier = 1.0 + (base_boost * sensitivity)
    multiplier = max(_STACK_MULT_MIN, min(_STACK_MULT_MAX, multiplier))
    multiplier = round(multiplier, 4)

    logger.debug(
        "Stacking mult: %s vs %s, env=%.3f, density=%d, raw=%.3f, mult=%.4f (mode=%s, signal=%s)",
        player_team,
        opponent_team,
        env_score,
        team_dens,
        raw_stack,
        multiplier,
        mode,
        signal,
    )

    return {
        "stacking_multiplier": multiplier,
        "stacking_environment": round(env_score, 4),
        "stacking_team_density": team_dens,
        "stacking_game_density": game_dens,
        "stacking_signal": signal,
    }
