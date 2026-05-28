from __future__ import annotations

from typing import Any


def parse_minutes(min_str: str | None) -> float:
    """Parse a minutes string like '35' or '35:42' into a float."""
    if not min_str:
        return 0.0
    parts = str(min_str).split(":")
    try:
        return float(parts[0]) + (float(parts[1]) / 60 if len(parts) > 1 else 0)
    except (ValueError, IndexError):
        return 0.0


def _compute_score(
    stat: dict[str, Any],
    weights: dict[str, float],
    minutes: float,
    per_minute_base: float,
    *,
    normalize: bool,
) -> float:
    """Compute a weighted fantasy score from a stat dict.

    Multiplies each key in weights by the matching value in stat, then sums.
    Stat keys not present in the stat dict contribute 0. This is sport-agnostic:
    the sport config owns all stat key names and their multipliers.

    Args:
        stat:            Box-score / result stat dict.
        weights:         Platform scoring weights — {stat_key: multiplier}.
        minutes:         Minutes/rounds played (used only when normalize=True).
        per_minute_base: Normalization base (e.g. 36.0 for NBA, 1.0 for golf).
        normalize:       If True, return per-unit-normalized score.
                         If False, return raw (game-total) score.

    Returns:
        0.0 if normalize=True and minutes <= 0.
    """
    if normalize and minutes <= 0:
        return 0.0

    raw = sum(float(stat.get(k) or 0) * v for k, v in weights.items())

    if normalize:
        return raw * (per_minute_base / minutes)
    return raw


def compute_season_averages(rows: list[dict[str, Any]], min_game_minutes: float = 8.0) -> dict[str, Any]:
    """Compute season averages from raw BDL stat rows.

    Filters out DNP/cameo games (< min_game_minutes). Returns a dict with:
      season_ppg, season_rpg, season_apg, season_spg, season_bpg,
      season_ftmpg, season_topg, season_pfpg,
      season_fg_pct, season_fg3_pct, season_ft_pct, season_games

    All rate stats (fg_pct etc.) are computed from cumulative makes/attempts,
    not averaged from per-game rates, to match standard calculation convention.
    Returns all None if no qualifying games.
    Pure function — no I/O.
    """
    pts = reb = ast = stl = blk = 0.0
    fgm = fga = fgm_3pt = fga_3pt = ftm = fta = 0.0
    to = pf = 0.0
    games = 0

    for row in rows:
        minutes = parse_minutes(row.get("min"))
        if minutes < min_game_minutes:
            continue
        pts += row.get("pts") or 0
        reb += row.get("reb") or 0
        ast += row.get("ast") or 0
        stl += row.get("stl") or 0
        blk += row.get("blk") or 0
        fgm += row.get("fgm") or 0
        fga += row.get("fga") or 0
        fgm_3pt += row.get("fgm_3pt") or row.get("fg3m") or 0
        fga_3pt += row.get("fga_3pt") or row.get("fg3a") or 0
        ftm += row.get("ftm") or 0
        fta += row.get("fta") or 0
        to += row.get("turnover") or 0
        pf += row.get("pf") or 0
        games += 1

    if games == 0:
        return {
            "season_ppg": None,
            "season_rpg": None,
            "season_apg": None,
            "season_spg": None,
            "season_bpg": None,
            "season_ftmpg": None,
            "season_topg": None,
            "season_pfpg": None,
            "season_fg_pct": None,
            "season_fg3_pct": None,
            "season_ft_pct": None,
            "season_games": 0,
        }

    return {
        "season_ppg": round(pts / games, 1),
        "season_rpg": round(reb / games, 1),
        "season_apg": round(ast / games, 1),
        "season_spg": round(stl / games, 1),
        "season_bpg": round(blk / games, 1),
        "season_ftmpg": round(ftm / games, 1),
        "season_topg": round(to / games, 1),
        "season_pfpg": round(pf / games, 1),
        "season_fg_pct": round(fgm / fga, 3) if fga > 0 else None,
        "season_fg3_pct": round(fgm_3pt / fga_3pt, 3) if fga_3pt > 0 else None,
        "season_ft_pct": round(ftm / fta, 3) if fta > 0 else None,
        "season_games": games,
    }


def compute_slope(scores: list[float]) -> float:
    """Exponentially-weighted least-squares slope, oldest → newest.

    Recent games receive more weight (weight = 1.5^i where i=0 is oldest).
    Base-1.5 gives effective sample size ~4.1 (vs ~2.8 with base-2), so all 5
    games meaningfully contribute while still prioritising recent performance.
    """
    n = len(scores)
    weights = [1.5**i for i in range(n)]
    w_total = sum(weights)
    x_vals = list(range(1, n + 1))
    x_wmean = sum(w * x for w, x in zip(weights, x_vals)) / w_total
    y_wmean = sum(w * y for w, y in zip(weights, scores)) / w_total
    numerator = sum(w * (x - x_wmean) * (y - y_wmean) for w, x, y in zip(weights, x_vals, scores))
    denominator = sum(w * (x - x_wmean) ** 2 for w, x in zip(weights, x_vals))
    return numerator / denominator if denominator else 0.0


def compute_trend_acceleration(scores: list[float], mean_score: float) -> float | None:
    """Compute the slope-of-slope (second derivative) of the fantasy score series.

    Splits the series into two halves and returns the difference between the second
    half's slope and the first half's slope, normalized by mean_score.
    Returns None if the series has fewer than 4 points or if mean_score is 0.
    """
    if len(scores) < 4 or mean_score == 0:
        return None
    mid = len(scores) // 2
    slope_first = compute_slope(scores[:mid])
    slope_second = compute_slope(scores[mid:])
    return round((slope_second - slope_first) / mean_score, 4)


def compute_streak(raw_scores: list[float]) -> tuple[int, int]:
    """Count consecutive above/below-average games from most recent backward.

    Returns (streak, streak_length) where streak is positive for a hot streak
    (consecutive above-average games) and negative for a cold streak.
    streak_length is the absolute value of streak.
    """
    if not raw_scores:
        return (0, 0)
    mean_raw = sum(raw_scores) / len(raw_scores)
    count = 0
    direction: int | None = None
    for score in reversed(raw_scores):
        if score > mean_raw:
            game_dir = 1
        elif score < mean_raw:
            game_dir = -1
        else:
            break
        if direction is None:
            direction = game_dir
        if game_dir != direction:
            break
        count += 1
    if direction is None or count == 0:
        return (0, 0)
    return (direction * count, count)
