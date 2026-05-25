"""Vegas lines signal computation — pure functions, no I/O."""

from typing import Any

from bks_pipeline_core.sport_config import get_active_config

_VEGAS_MULT_MIN = 0.90
_VEGAS_MULT_MAX = 1.10


def compute_vegas_signals(odds: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Convert raw odds into per-team signals with implied team totals.

    Args:
        odds: List of game dicts from fetch_nba_odds(), each containing:
              home_team_abbr, away_team_abbr, over_under, home_spread

    Returns:
        {team_abbr: {"implied_team_total": float, "over_under": float,
                     "spread": float, "is_favorite": bool}}
    """
    signals: dict[str, dict[str, Any]] = {}
    for game in odds:
        home = game.get("home_team_abbr")
        away = game.get("away_team_abbr")
        ou = game.get("over_under")
        home_spread = game.get("home_spread")
        if not home or not away or ou is None or home_spread is None:
            continue

        # home_spread is negative when home is favorite (e.g. -3.5).
        # Subtracting it gives the favorite the higher implied total.
        home_itt = (ou / 2) - (home_spread / 2)
        away_itt = (ou / 2) + (home_spread / 2)

        signals[home] = {
            "implied_team_total": round(home_itt, 1),
            "over_under": ou,
            "spread": home_spread,
            "is_favorite": home_spread < 0,
        }
        signals[away] = {
            "implied_team_total": round(away_itt, 1),
            "over_under": ou,
            "spread": -home_spread,  # away spread is inverse of home
            "is_favorite": home_spread > 0,
        }

    return signals


def vegas_multiplier(
    signals: dict[str, dict[str, Any]] | None,
    team_abbr: str,
    vegas_sensitivity: float = 1.0,
    slate_avg_itt: float | None = None,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
    playoff_dead_zone: bool = False,
) -> float:
    """Compute the Vegas multiplier for a player's team.

    Returns 1.0 (neutral) if no odds data is available.

    Args:
        signals:            Per-team vegas signals from compute_vegas_signals().
        team_abbr:          The player's team abbreviation.
        vegas_sensitivity:  Mode-aware dampening/amplification factor
                            (balanced=1.0, cash=0.5, gpp=1.5).
        slate_avg_itt:      Dynamic baseline from today's slate average ITT.
                            Falls back to LEAGUE_AVG_IMPLIED_TEAM_TOTAL when None.
        clamp_min:          Override for lower clamp bound (default _VEGAS_MULT_MIN).
        clamp_max:          Override for upper clamp bound (default _VEGAS_MULT_MAX).
        playoff_dead_zone:  When True, widen dead zone to ±5% (vs ±2% regular season).
                            Playoff totals are less discriminating (2026-04-21: 7d r=-0.031).
    """
    if not signals or team_abbr not in signals:
        return 1.0

    team = signals[team_abbr]
    itt = team.get("implied_team_total")
    if itt is None:
        return 1.0

    lo = clamp_min if clamp_min is not None else _VEGAS_MULT_MIN
    hi = clamp_max if clamp_max is not None else _VEGAS_MULT_MAX

    # Normalize ITT relative to slate average (dynamic) or league average (static fallback)
    baseline = slate_avg_itt if slate_avg_itt is not None else get_active_config().league_avg_team_total
    vegas_ratio = itt / baseline
    vegas_raw = max(lo, min(hi, vegas_ratio))

    # Dead zone: suppress marginal ITT noise.
    # Regular season: ±2% (7d r=+0.013, 100% fire rate)
    # Playoffs: widened to ±5% — totals less discriminating in series context (2026-04-21)
    dead_zone_lo, dead_zone_hi = (0.95, 1.05) if playoff_dead_zone else (0.98, 1.02)
    if dead_zone_lo <= vegas_raw <= dead_zone_hi:
        return 1.0

    # Apply mode sensitivity — dampen or amplify the deviation from 1.0
    effective = 1.0 + (vegas_raw - 1.0) * vegas_sensitivity

    return float(round(effective, 4))


_LINE_MOVE_DEAD_ZONE = 1.0  # ±1.0 ITT pts = no signal
_LINE_MOVE_SCALE = 0.01  # 1% per point of movement past dead zone
_LINE_MOVE_MAX_MOVE = 5.0  # cap before scaling


def line_movement_multiplier(
    signals: dict[str, dict[str, Any]] | None,
    signals_open: dict[str, dict[str, Any]] | None,
    team_abbr: str,
    clamp_min: float | None = None,
    clamp_max: float | None = None,
) -> tuple[float, float | None, float | None, float | None]:
    """Multiplier based on ITT movement since opening lines.

    Returns (multiplier, itt_movement, ou_movement, spread_movement).
    Returns (1.0, None, None, None) when open lines are unavailable.
    """
    if not signals or not signals_open or team_abbr not in signals or team_abbr not in signals_open:
        return 1.0, None, None, None

    curr = signals[team_abbr]
    open_ = signals_open[team_abbr]

    curr_itt = curr.get("implied_team_total")
    open_itt = open_.get("implied_team_total")
    if curr_itt is None or open_itt is None:
        return 1.0, None, None, None

    itt_move = round(curr_itt - open_itt, 2)

    curr_ou = curr.get("over_under")
    open_ou = open_.get("over_under")
    ou_move = round(curr_ou - open_ou, 2) if curr_ou is not None and open_ou is not None else None

    curr_spd = curr.get("spread")
    open_spd = open_.get("spread")
    spd_move = round(curr_spd - open_spd, 2) if curr_spd is not None and open_spd is not None else None

    abs_move = abs(itt_move)
    if abs_move <= _LINE_MOVE_DEAD_ZONE:
        mult = 1.0
    else:
        direction = 1.0 if itt_move > 0 else -1.0
        capped = min(abs_move - _LINE_MOVE_DEAD_ZONE, _LINE_MOVE_MAX_MOVE)
        mult = 1.0 + direction * capped * _LINE_MOVE_SCALE

    lo = clamp_min if clamp_min is not None else 0.98
    hi = clamp_max if clamp_max is not None else 1.03
    mult = float(round(max(lo, min(hi, mult)), 4))

    return mult, itt_move, ou_move, spd_move
