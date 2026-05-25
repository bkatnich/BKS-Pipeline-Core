"""Shared data-loading logic for the prediction pipeline.

Extracts the duplicated Firestore reads from handlers.py into a single
reusable function. Both the live ``get_opportunities`` endpoint and the
``snapshot_predictions`` scheduler call this instead of inlining their
own (divergent) data-loading blocks.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from bks_pipeline_core.pipeline.games import compute_series_history, compute_team_rest_days, load_games_doc
from bks_pipeline_core.pipeline.league_state import get_league_state
from bks_pipeline_core.pipeline.platforms import PLATFORMS

logger = logging.getLogger(__name__)


@dataclass
class PredictionContext:
    """All data needed to call ``build_opportunity_results``."""

    date: str
    players: list[dict[str, Any]]
    defense_maps: dict[str, dict[str, dict[str, float]]]
    full_defense_maps: dict[str, dict[str, dict[str, float]]]
    games_doc: dict[str, Any]
    yesterday_teams: set[str]
    rest_days: dict[str, int]
    pace_map: dict[str, float]
    full_pace_map: dict[str, float]
    vegas_signals: dict[str, dict[str, Any]] | None
    vegas_signals_open: dict[str, dict[str, Any]] | None
    season_mode: str
    league_state: dict[str, Any] = field(default_factory=dict)
    playing_abbrs: list[str] = field(default_factory=list)
    series_history: dict | None = None


def load_prediction_context(
    db: Any,
    date: str | None = None,
) -> PredictionContext | None:
    """Load everything needed to run ``build_opportunity_results``.

    Args:
        db:   Firestore client.
        date: Target date as ``YYYY-MM-DD``.  Defaults to today (ET).

    Returns:
        A populated ``PredictionContext``, or ``None`` when there are no
        games on the target date.
    """
    now_et = datetime.now(ZoneInfo("America/New_York"))

    if date is None:
        date = now_et.strftime("%Y-%m-%d")

    yesterday_et = (datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=ZoneInfo("America/New_York")) - timedelta(days=1)).strftime("%Y-%m-%d")

    # --- league state ---
    league_state = get_league_state(db)
    season_mode = league_state.get("mode", "regular_season")

    # --- games doc ---
    games_doc = load_games_doc(db, date)
    if games_doc is None:
        return None
    playing_abbrs: list[str] = games_doc.get("playing_team_abbrs", [])
    if not playing_abbrs:
        return None

    # --- players ---
    # Exclude players marked is_active=False by the sync pipeline (retired/released).
    # Docs without the field (all existing docs before Phase 7 first runs) pass through.
    players = [d for doc in db.collection("players").stream() if (d := doc.to_dict() or {}).get("is_active") is not False]

    # --- full league defense & pace (all 30 teams) ---
    full_defense_maps: dict[str, dict[str, dict[str, float]]] = {p: {} for p in PLATFORMS}
    full_pace_map: dict[str, float] = {}
    for doc in db.collection("team_defense").stream():
        if doc.exists:
            abbr = doc.id
            d = doc.to_dict() or {}
            for p_key, p_cfg in PLATFORMS.items():
                pts = d.get(p_cfg["def_pts_field"])
                if isinstance(pts, dict):
                    full_defense_maps[p_key][abbr] = pts
            pace = d.get("pace")
            if pace is not None:
                full_pace_map[abbr] = pace

    # --- playing-teams subset (backward compat) ---
    defense_maps: dict[str, dict[str, dict[str, float]]] = {p: {} for p in PLATFORMS}
    pace_map: dict[str, float] = {}
    for abbr in playing_abbrs:
        for p_key in PLATFORMS:
            if abbr in full_defense_maps[p_key]:
                defense_maps[p_key][abbr] = full_defense_maps[p_key][abbr]
        if abbr in full_pace_map:
            pace_map[abbr] = full_pace_map[abbr]

    # --- yesterday's teams (for B2B detection) ---
    yesterday_teams: set[str] = set()
    yesterday_snap = db.collection("games").document(yesterday_et).get()
    if yesterday_snap.exists:
        yesterday_teams = set((yesterday_snap.to_dict() or {}).get("playing_team_abbrs", []))

    # --- rest days ---
    rest_days = compute_team_rest_days(db, date, playing_abbrs)

    # --- Vegas odds ---
    vegas_signals: dict[str, dict[str, Any]] | None = games_doc.get("odds")
    vegas_signals_open: dict[str, dict[str, Any]] | None = games_doc.get("odds_open")

    # --- Series history and elimination context (playoffs only) ---
    series_history: dict | None = None
    if season_mode == "playoffs":
        matchups = games_doc.get("games") or []
        if matchups:
            series_history = compute_series_history(db, date, matchups)

        # Build set of team abbrs in an elimination game today.
        # Reads active series docs; elimination_game_next is written by record_game_result.
        from bks_pipeline_core.pipeline.playoffs import get_all_series

        _season = get_league_state(db).get("season", 2025)
        _all_series = get_all_series(db, _season)
        elimination_teams: set[str] = set()
        for s in _all_series:
            if s.get("status") == "active" and s.get("elimination_game_next"):
                for key in ("higher_seed_team", "lower_seed_team"):
                    abbr = s.get(key)
                    if abbr:
                        elimination_teams.add(abbr)

        # Merge per-player trust scores, series stats, and elimination flag onto
        # player dicts so downstream scoring can read them without new parameters.
        from bks_pipeline_core.pipeline.playoff_trust import load_playoff_trust, load_series_stats

        trust_map = load_playoff_trust(db)
        series_stats_map = load_series_stats(db)
        for player in players:
            pid = str(player.get("id", ""))
            team = player.get("team")
            team_abbr = team if isinstance(team, str) else (team or {}).get("abbreviation", "")
            if pid in trust_map:
                player["playoff_trend_trust_score"] = trust_map[pid]
            player["is_elimination_game"] = team_abbr in elimination_teams if team_abbr else False
            if pid in series_stats_map:
                ss = series_stats_map[pid]
                player["series_fp_avg"] = ss.get("series_fp_avg")
                player["series_fp_avg_all"] = ss.get("series_fp_avg_all")
                player["series_minutes_avg"] = ss.get("series_minutes_avg")
                player["series_games"] = ss.get("series_games", 0)
                player["series_fg3_pct"] = ss.get("series_fg3_pct")
                player["series_fg3m_per_game"] = ss.get("series_fg3m_per_game")
                player["series_game_log"] = ss.get("game_log", [])

    return PredictionContext(
        date=date,
        players=players,
        defense_maps=defense_maps,
        full_defense_maps=full_defense_maps,
        games_doc=games_doc,
        yesterday_teams=yesterday_teams,
        rest_days=rest_days,
        pace_map=pace_map,
        full_pace_map=full_pace_map,
        vegas_signals=vegas_signals,
        vegas_signals_open=vegas_signals_open,
        season_mode=season_mode,
        league_state=league_state,
        playing_abbrs=playing_abbrs,
        series_history=series_history,
    )
