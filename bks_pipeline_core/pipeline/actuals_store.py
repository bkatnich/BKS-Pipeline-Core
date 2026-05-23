"""Fetch and store actual fantasy-point results for a given game date.

Extracted from ``handlers.capture_actuals`` so both the scheduler and
on-demand HTTP endpoint can share the same logic.  The snapshot-existence
gate has been **removed** — actuals are now independent of whether a
prediction snapshot exists.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)


def fetch_and_store_actuals(
    db: Any,
    date: str,
    api_key: str,
    log: logging.Logger | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Fetch box scores for *date*, compute fantasy points, and store.

    Idempotent: if ``actuals/{date}`` already exists the stored data is
    returned without re-fetching from the API.

    Returns:
        The actuals dict keyed by player ID, or ``None`` when the API
        returns no stats for the date.
    """
    from api.sport_provider import fetch_stats_by_date  # sport-specific lazy import
    from bks_pipeline_core.pipeline.scoring import (
        fantasy_score_raw_dk,
        fantasy_score_raw_fd,
        parse_minutes,
    )

    _log = log or logger
    now_et = datetime.now(ZoneInfo("America/New_York"))

    # Idempotent: return existing actuals if already stored
    existing = db.collection("actuals").document(date).get()
    if existing.exists:
        _log.info("fetch_and_store_actuals: actuals already exist for %s", date)
        return (existing.to_dict() or {}).get("results")

    # Fetch box scores from the sport-specific stats provider
    stats = fetch_stats_by_date(date, api_key, _log)
    if not stats:
        _log.warning("fetch_and_store_actuals: no stats returned for %s", date)
        return None

    # Build actuals keyed by player ID
    actuals: dict[str, dict[str, Any]] = {}
    for row in stats:
        player = row.get("player", {})
        pid = str(player.get("id", ""))
        if not pid:
            continue

        minutes = parse_minutes(row.get("min"))
        team = row.get("team", {})
        team_abbr = team.get("abbreviation") if isinstance(team, dict) else None

        # Determine opponent from game data
        game = row.get("game", {})
        home_team = game.get("home_team", {})
        visitor_team = game.get("visitor_team", {})
        home_abbr = home_team.get("abbreviation") if isinstance(home_team, dict) else None
        visitor_abbr = visitor_team.get("abbreviation") if isinstance(visitor_team, dict) else None
        opp_abbr = visitor_abbr if team_abbr == home_abbr else home_abbr

        # Detect double-double and triple-double for DK scoring
        cats_10 = sum(1 for cat in ["pts", "reb", "ast", "stl", "blk"] if (row.get(cat) or 0) >= 10)
        stat_with_bonuses = {**row, "dd": cats_10 >= 2, "td": cats_10 >= 3}

        actuals[pid] = {
            "actual_fp_dk": round(fantasy_score_raw_dk(stat_with_bonuses), 2),
            "actual_fp_fd": round(fantasy_score_raw_fd(row), 2),
            "actual_minutes": round(minutes, 1),
            "actual_pts": row.get("pts") or 0,
            "actual_reb": row.get("reb") or 0,
            "actual_ast": row.get("ast") or 0,
            "actual_stl": row.get("stl") or 0,
            "actual_blk": row.get("blk") or 0,
            "actual_fgm": row.get("fgm") or 0,
            "actual_fga": row.get("fga") or 0,
            "actual_fg_pct": row.get("fg_pct"),
            "actual_fg3m": row.get("fg3m") or 0,
            "actual_fg3a": row.get("fg3a") or 0,
            "actual_fg3_pct": row.get("fg3_pct"),
            "actual_ftm": row.get("ftm") or 0,
            "actual_fta": row.get("fta") or 0,
            "actual_ft_pct": row.get("ft_pct"),
            "actual_oreb": row.get("oreb") or 0,
            "actual_dreb": row.get("dreb") or 0,
            "actual_turnover": row.get("turnover") or 0,
            "actual_pf": row.get("pf") or 0,
            "actual_plus_minus": row.get("plus_minus"),
            "dnp": minutes < 1.0,
            "team": team_abbr,
            "opponent_abbr": opp_abbr,
        }

    ttl = now_et + timedelta(days=30)
    db.collection("actuals").document(date).set(
        {
            "date": date,
            "fetched_at": now_et.isoformat(),
            "player_count": len(actuals),
            "results": actuals,
            "ttl": ttl,
        }
    )

    _log.info("fetch_and_store_actuals complete: %d players for %s", len(actuals), date)
    return actuals


def get_actuals(
    db: Any,
    date: str,
) -> dict[str, dict[str, Any]] | None:
    """Load previously stored actuals for *date*.

    Returns the ``results`` dict keyed by player ID, or ``None`` if no
    actuals exist for the date.
    """
    doc = db.collection("actuals").document(date).get()
    if not doc.exists:
        return None
    return (doc.to_dict() or {}).get("results")
