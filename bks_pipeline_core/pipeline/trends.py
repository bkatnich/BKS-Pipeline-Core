"""Generic trend computation for multi-sport Firebase pipelines.

All sport-specific behaviour is injected via SportTrendHooks. Runtime
configuration values (batch sizes, thresholds, provider) are passed via
TrendsConfig so this module has zero imports from any sport project's config.py.

Usage in a sport project's pipeline/trends.py:

    from bks_pipeline_core.pipeline.trends import (
        SportTrendHooks,
        TrendsConfig,
        fetch_trends,
        fetch_season_stats,
    )
"""

from __future__ import annotations

import logging
import math
import time
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Protocol

from bks_pipeline_core.pipeline.scoring import (
    compute_season_averages,
    compute_slope,
    compute_streak,
    compute_trend_acceleration,
    fantasy_score,
    parse_minutes,
)
from bks_pipeline_core.sport_config import get_active_config
from bks_pipeline_core.utils.exceptions import TrendFetchAbortedError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Provider protocol stub — mirrors StatsProvider but scoped to what trends needs
# ---------------------------------------------------------------------------


class _TrendsStatsProviderProto(Protocol):
    def fetch_player_stats_batch(
        self,
        person_ids: list[int],
        api_key: str,
        start_date: str,
        logger: logging.Logger,
    ) -> tuple[list[dict[str, Any]], bool]: ...

    def fetch_player_season_stats(
        self,
        person_ids: list[int],
        api_key: str,
        logger: logging.Logger,
    ) -> dict[int, dict[str, Any]]: ...


# ---------------------------------------------------------------------------
# Runtime config bundle
# ---------------------------------------------------------------------------


@dataclass
class TrendsConfig:
    """Runtime configuration values for fetch_trends and fetch_season_stats.

    Build one from your sport project's config constants and pass it to the
    fetch_* functions. All values that differ by sport or environment belong
    here; generic math lives in the functions themselves.
    """

    stats_provider: Any  # _TrendsStatsProviderProto
    trend_window: int
    player_batch_size: int
    min_game_minutes: float
    min_recent_minutes: float
    stats_request_delay: float
    circuit_breaker_threshold: int
    current_season_start: str


# ---------------------------------------------------------------------------
# SportTrendHooks — sport-specific overrides
# ---------------------------------------------------------------------------


@dataclass
class SportTrendHooks:
    """Sport-specific dependency injection for fetch_trends.

    All callables have safe defaults so fetch_trends(player_ids, api_key, cfg)
    works without any hooks — backward-compatible with sports that don't need
    customisation.
    """


# ---------------------------------------------------------------------------
# Generic helper functions
# ---------------------------------------------------------------------------


def _normalized_slope(series: list[float]) -> float | None:
    """Return normalized slope for a series, or None if mean is 0."""
    mean = sum(series) / len(series)
    if mean == 0:
        return None
    return round(compute_slope(series) / mean, 4)


def _consistency_score(series: list[float]) -> float | None:
    """Coefficient of variation inverted to a [0,1] reliability score.

    Returns 1 - (std_dev / mean), clamped to [0, 1].
    None if mean is 0 or fewer than 2 data points.
    """
    n = len(series)
    if n < 2:
        return None
    mean = sum(series) / n
    if mean == 0:
        return None
    variance = sum((x - mean) ** 2 for x in series) / n
    std_dev = variance**0.5
    return float(round(max(0.0, min(1.0, 1.0 - (std_dev / mean))), 4))


def _role_change(all_games: list[dict[str, Any]], trend_window: int = 5) -> tuple[bool, float | None]:
    """Detect a minutes-based role change by comparing recent vs. baseline minutes.

    Recent  = last 3 games (index [-3:])
    Baseline = up to 7 games before the trend window

    Returns (is_role_change, role_change_minutes_delta).
    is_role_change is True when delta >= 6 min, recent avg >= 15 min, and either
    the absolute delta is >= 8 or the relative jump is >= 30%.
    Returns (False, None) when there are fewer than 4 total games.
    """
    recent_games = all_games[-3:] if len(all_games) >= 3 else all_games
    baseline_games = all_games[-(trend_window + 7) : -trend_window]

    if not baseline_games or not recent_games:
        return False, None

    recent_avg = sum(parse_minutes(g.get("min")) for g in recent_games) / len(recent_games)
    baseline_avg = sum(parse_minutes(g.get("min")) for g in baseline_games) / len(baseline_games)
    delta = round(recent_avg - baseline_avg, 1)

    is_rc = delta >= 6.0 and recent_avg >= 15.0 and (delta >= 8.0 or (baseline_avg > 0 and delta / baseline_avg >= 0.30))
    return is_rc, delta


def _is_home_game(stat_row: dict[str, Any]) -> bool:
    """Return True if the player was on the home team for this game."""
    team = stat_row.get("team")
    game = stat_row.get("game")
    if not isinstance(team, dict) or not isinstance(game, dict):
        return False
    return team.get("id") == game.get("home_team_id")


def _venue_avg(
    games: list[dict[str, Any]],
    score_fn: Callable[[dict[str, Any], float], float],
) -> float | None:
    """Compute average raw fantasy score for a list of game stat rows."""
    if not games:
        return None
    per_minute_base = get_active_config().per_minute_base
    raw = []
    for s in games:
        m = parse_minutes(s.get("min"))
        if m > 0:
            raw.append(score_fn(s, m) * (m / per_minute_base))
    return round(sum(raw) / len(raw), 2) if raw else None


def _empty_trend(
    stat_categories: list[tuple[str, str, bool]],
    now_iso: str,
    is_rc: bool,
    game_count: int,
    avg_minutes: float | None = None,
    avg_fantasy_score: float | None = None,
    trend_score: float | None = None,
    avg_fantasy_score_home: float | None = None,
    avg_fantasy_score_away: float | None = None,
) -> dict[str, Any]:
    """Return a zeroed-out trend field dict for players with insufficient data."""
    cat_trends = {key: None for key, _, _ in stat_categories}
    return {
        "trend_direction": "neutral",
        "trend_score": trend_score,
        "trend_games": game_count,
        "avg_minutes": avg_minutes,
        "avg_fantasy_score": avg_fantasy_score,
        "consistency_score": None,
        "trend_updated_at": now_iso,
        "trend_acceleration": None,
        "confidence_score": None,
        "hot_streak": 0,
        "cold_streak": 0,
        "streak_length": 0,
        "days_since_return": None,
        "is_return_game_window": False,
        "is_role_change": is_rc,
        "avg_fantasy_score_home": avg_fantasy_score_home,
        "avg_fantasy_score_away": avg_fantasy_score_away,
        "recent_game_scores": [],
        "recent_game_minutes": [],
        **cat_trends,
    }


def _compute_trend_fields(
    stats_by_player: dict[int, list[dict[str, Any]]],
    player_ids: list[int],
    trend_window: int = 5,
) -> dict[int, dict[str, Any]]:
    """Pure computation: given pre-fetched, DNP-filtered stats, return trend field dicts."""
    cfg = get_active_config()
    stat_categories = cfg.stat_categories
    per_minute_base = cfg.per_minute_base
    now_iso = datetime.now(timezone.utc).isoformat()
    trends: dict[int, dict[str, Any]] = {}

    for pid in player_ids:
        all_player_rows = stats_by_player.get(pid, [])
        recent = all_player_rows[-trend_window:]
        game_count = len(recent)

        playoff_games_played = sum(1 for r in all_player_rows if isinstance(r.get("game"), dict) and r["game"].get("postseason") is True)

        if game_count < 3:
            is_rc, _rc_delta = _role_change(all_player_rows, trend_window)
            t = _empty_trend(stat_categories, now_iso, is_rc, game_count)
            t["playoff_games_played"] = playoff_games_played
            trends[pid] = t
            continue

        minutes_series = [parse_minutes(s.get("min")) for s in recent]
        avg_minutes = round(sum(minutes_series) / len(minutes_series), 1)
        scores = [fantasy_score(s, m) for s, m in zip(recent, minutes_series)]
        mean_score = sum(scores) / len(scores)

        raw_scores = [score * (m / per_minute_base) for score, m in zip(scores, minutes_series)]
        avg_fantasy_score = round(sum(raw_scores) / len(raw_scores), 2)

        all_games = all_player_rows
        home_games = [s for s in all_games if _is_home_game(s)]
        away_games = [s for s in all_games if not _is_home_game(s)]
        avg_fantasy_score_home = _venue_avg(home_games, fantasy_score)
        avg_fantasy_score_away = _venue_avg(away_games, fantasy_score)

        if mean_score == 0:
            is_rc, _rc_delta = _role_change(all_player_rows, trend_window)
            t = _empty_trend(
                stat_categories,
                now_iso,
                is_rc,
                game_count,
                avg_minutes=avg_minutes,
                avg_fantasy_score=0.0,
                trend_score=0.0,
                avg_fantasy_score_home=avg_fantasy_score_home,
                avg_fantasy_score_away=avg_fantasy_score_away,
            )
            t["playoff_games_played"] = playoff_games_played
            trends[pid] = t
            continue

        mean_raw = sum(raw_scores) / len(raw_scores)
        slope = compute_slope(raw_scores)
        normalised = slope / mean_raw if mean_raw > 0 else 0.0

        if normalised > 0.05:
            direction = "up"
        elif normalised < -0.05:
            direction = "down"
        else:
            direction = "neutral"

        cat_trends: dict[str, float | None] = {}
        for field_key, stat_key, per36 in stat_categories:
            if stat_key == "min":
                series = minutes_series
            elif per36:
                series = [(stat.get(stat_key) or 0) * (per_minute_base / m) if m > 0 else 0.0 for stat, m in zip(recent, minutes_series)]
            else:
                series = [(stat.get(stat_key) or 0) for stat in recent]
            cat_trends[field_key] = _normalized_slope(series)

        consistency = _consistency_score(scores)

        trend_acceleration = compute_trend_acceleration(raw_scores, mean_raw)

        if consistency is not None:
            _raw_conf = 0.6 * normalised + 0.4 * (consistency - 0.5)
            confidence_score = round(1.0 / (1.0 + math.exp(-8.0 * _raw_conf)), 4)
        else:
            confidence_score = None

        streak, streak_length = compute_streak(raw_scores)
        hot_streak = streak if streak > 0 else 0
        cold_streak = streak if streak < 0 else 0

        is_rc, _rc_delta = _role_change(all_player_rows, trend_window)

        trends[pid] = {
            "trend_direction": direction,
            "trend_score": round(normalised, 4),
            "trend_games": game_count,
            "avg_minutes": avg_minutes,
            "avg_fantasy_score": avg_fantasy_score,
            "consistency_score": consistency,
            "trend_updated_at": now_iso,
            "trend_acceleration": trend_acceleration,
            "confidence_score": confidence_score,
            "hot_streak": hot_streak,
            "cold_streak": cold_streak,
            "streak_length": streak_length,
            "days_since_return": None,
            "is_return_game_window": False,
            "is_role_change": is_rc,
            "avg_fantasy_score_home": avg_fantasy_score_home,
            "avg_fantasy_score_away": avg_fantasy_score_away,
            "recent_game_scores": [round(s, 2) for s in raw_scores],
            "recent_game_minutes": [round(m, 1) for m in minutes_series],
            "playoff_games_played": playoff_games_played,
            **cat_trends,
        }

    return trends


# ---------------------------------------------------------------------------
# Public fetch functions
# ---------------------------------------------------------------------------


def fetch_trends(
    player_ids: list[int],
    api_key: str,
    cfg: TrendsConfig,
    hooks: SportTrendHooks | None = None,
) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]:
    """Fetch last trend_window game stats for each player and compute trend fields.

    Batches players player_batch_size at a time. Skips players with fewer than
    min_recent_minutes total in the last 30 days. Applies a rate-limit delay
    between requests. Raises TrendFetchAbortedError if the circuit breaker trips.

    Returns (trend_dict, raw_rows) where:
      - trend_dict maps player_id -> trend fields dict
      - raw_rows is the full list of all stat rows fetched (used by defense computation)
    """
    _hooks = hooks or SportTrendHooks()
    stats_by_player: dict[int, list[dict[str, Any]]] = defaultdict(list)
    all_raw_rows: list[dict[str, Any]] = []
    total = len(player_ids)
    skipped = 0
    consecutive_failures = 0

    start_date = (datetime.now(timezone.utc) - timedelta(days=30)).strftime("%Y-%m-%d")
    batches = [player_ids[i : i + cfg.player_batch_size] for i in range(0, len(player_ids), cfg.player_batch_size)]
    fetched_count = 0

    for batch_ids in batches:
        rows, success = cfg.stats_provider.fetch_player_stats_batch(batch_ids, api_key, start_date, logger)

        if success:
            consecutive_failures = 0
            all_raw_rows.extend(rows)
        else:
            consecutive_failures += 1
            if consecutive_failures >= cfg.circuit_breaker_threshold:
                logger.error(
                    "Circuit breaker tripped: %d consecutive batch failures (last batch %s, processed %d/%d). Aborting trend fetch.",
                    consecutive_failures,
                    batch_ids,
                    fetched_count,
                    total,
                )
                raise TrendFetchAbortedError(
                    f"Aborted after {consecutive_failures} consecutive API failures at batch {batch_ids}",
                    consecutive_failures=consecutive_failures,
                )

        rows_by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            pid = r.get("person_id")
            if pid and parse_minutes(r.get("min")) >= cfg.min_game_minutes:
                rows_by_pid[pid].append(r)

        for pid in batch_ids:
            played_rows = rows_by_pid.get(pid, [])
            total_minutes = sum(parse_minutes(r.get("min")) for r in played_rows)
            if total_minutes < cfg.min_recent_minutes:
                skipped += 1
            else:
                stats_by_player[pid] = played_rows

        fetched_count += len(batch_ids)
        if fetched_count % 50 == 0 or fetched_count == total:
            logger.info(
                "Trends progress: %d/%d players fetched (%d skipped)",
                fetched_count,
                total,
                skipped,
            )

        time.sleep(cfg.stats_request_delay)

    logger.info(
        "Trends fetch complete: %d/%d players had sufficient minutes",
        total - skipped,
        total,
    )
    return _compute_trend_fields(stats_by_player, player_ids, cfg.trend_window), all_raw_rows


def fetch_season_stats(
    player_ids: list[int],
    api_key: str,
    cfg: TrendsConfig,
) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    """Fetch full-season stats for each player and compute season averages.

    Uses cfg.current_season_start as the start_date to cover the entire season.
    Applies the same batching, rate-limiting, and circuit-breaker pattern as
    fetch_trends(). After computing per-game averages from raw rows, merges
    pre-computed rate stats from the provider's fetch_player_season_stats endpoint.

    Returns:
      - averages_by_player: player_id -> season average fields
      - raw_rows_by_player: player_id -> list of raw game row dicts
    """
    season_data: dict[int, dict[str, Any]] = {}
    raw_rows_by_player: dict[int, list[dict[str, Any]]] = {}
    total = len(player_ids)
    consecutive_failures = 0
    fetched_count = 0

    batches = [player_ids[i : i + cfg.player_batch_size] for i in range(0, total, cfg.player_batch_size)]

    for batch_ids in batches:
        rows, success = cfg.stats_provider.fetch_player_stats_batch(batch_ids, api_key, cfg.current_season_start, logger)

        if success:
            consecutive_failures = 0
        else:
            consecutive_failures += 1
            if consecutive_failures >= cfg.circuit_breaker_threshold:
                logger.error(
                    "Season stats circuit breaker tripped: %d consecutive failures (last batch %s, processed %d/%d). Aborting.",
                    consecutive_failures,
                    batch_ids,
                    fetched_count,
                    total,
                )
                raise TrendFetchAbortedError(
                    f"Season stats aborted after {consecutive_failures} consecutive API failures at batch {batch_ids}",
                    consecutive_failures=consecutive_failures,
                )
            logger.warning(
                "Season stats batch failed (failure %d/%d) — skipping %d players %s, no season data written for them",
                consecutive_failures,
                cfg.circuit_breaker_threshold,
                len(batch_ids),
                batch_ids,
            )
            fetched_count += len(batch_ids)
            time.sleep(cfg.stats_request_delay)
            continue

        rows_by_pid: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for r in rows:
            pid = r.get("person_id")
            if pid:
                rows_by_pid[pid].append(r)

        for pid in batch_ids:
            player_rows = rows_by_pid.get(pid, [])
            season_data[pid] = compute_season_averages(player_rows, cfg.min_game_minutes)
            raw_rows_by_player[pid] = player_rows

        fetched_count += len(batch_ids)
        if fetched_count % 50 == 0 or fetched_count == total:
            logger.info("Season stats progress: %d/%d players fetched", fetched_count, total)

        time.sleep(cfg.stats_request_delay)

    logger.info("Season stats fetch complete: %d players", len(season_data))

    bdl_season = cfg.stats_provider.fetch_player_season_stats(player_ids, api_key, logger)
    for pid, bdl in bdl_season.items():
        if pid in season_data:
            season_data[pid].update(bdl)
        else:
            season_data[pid] = bdl

    return season_data, raw_rows_by_player
