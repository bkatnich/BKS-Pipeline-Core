"""Per-game data freshness checks and targeted pre-tip syncs.

For each individual game on a given date, this module:
  1. Computes a per-game Cloud Task schedule time (tip − N minutes).
  2. Checks whether injuries, odds, and trends are fresh for the two
     teams in that game.
  3. Triggers targeted syncs for any stale data.
  4. Writes a ``game_freshness/{date}_{game_id}`` Firestore doc that
     records the audit trail and serves as a client signal.
  5. Handles dedup so back-to-back games don't cause redundant API calls.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any  # noqa: F401 — used in FreshnessResult dataclass fields

from bks_pipeline_core.sport_config import get_active_config

logger = logging.getLogger(__name__)

__all__ = [
    "FRESHNESS_COLLECTION",
    "GameScheduleItem",
    "FreshnessResult",
    "compute_per_game_schedule_times",
    "check_game_data_freshness",
    "refresh_stale_data",
    "refresh_team_trends",
    "write_freshness_signal",
    "should_skip_dedup",
    "_is_stale",
]

FRESHNESS_COLLECTION = "game_freshness"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class GameScheduleItem:
    """A single game to be scheduled for a freshness check."""

    game_id: str
    tip_off_utc: datetime
    home_team: str
    away_team: str
    schedule_time: datetime  # when the Cloud Task should fire


@dataclass
class FreshnessResult:
    """Per-data-type freshness assessment for a single game."""

    injuries_fresh: bool = True
    injuries_synced_at: str = ""
    injuries_triggered_sync: bool = False

    odds_fresh: bool = True
    odds_synced_at: str = ""
    odds_triggered_sync: bool = False

    trends_fresh: bool = True
    trends_synced_at: str = ""
    trends_triggered_sync: bool = False

    lineup_fresh: bool = True
    lineup_synced_at: str = ""
    lineup_triggered_sync: bool = False
    lineup_changes: list[dict[str, Any]] = field(default_factory=list)

    injury_changes: list[dict[str, str]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def all_fresh(self) -> bool:
        return self.injuries_fresh and self.odds_fresh and self.trends_fresh


# ---------------------------------------------------------------------------
# Scheduling
# ---------------------------------------------------------------------------


def compute_per_game_schedule_times(
    matchups: list[tuple[str, datetime, str, str]],
    now: datetime,
    lead_minutes: int = 15,
) -> list[GameScheduleItem]:
    """Return a :class:`GameScheduleItem` for each game whose task time is still in the future.

    *matchups* is a list of ``(game_id, tip_off_utc, home_abbr, away_abbr)`` tuples.
    """
    items: list[GameScheduleItem] = []
    for game_id, tip_off, home, away in matchups:
        schedule_time = tip_off - timedelta(minutes=lead_minutes)
        if schedule_time > now:
            items.append(
                GameScheduleItem(
                    game_id=game_id,
                    tip_off_utc=tip_off,
                    home_team=home,
                    away_team=away,
                    schedule_time=schedule_time,
                )
            )
    return items


# ---------------------------------------------------------------------------
# Freshness checks
# ---------------------------------------------------------------------------


def _max_timestamp_for_teams(
    db: Any,
    team_abbrs: list[str],
    field_name: str,
) -> str | None:
    """Return the most-recent *field_name* ISO timestamp across players on *team_abbrs*.

    Queries the ``players`` collection for each team, reads the target field,
    and returns the maximum value.  Returns ``None`` if no timestamp is found.
    """
    max_ts: str | None = None
    for abbr in team_abbrs:
        docs = db.collection("players").where("team", "==", abbr).select([field_name]).stream()
        for doc in docs:
            val = (doc.to_dict() or {}).get(field_name)
            if val and (max_ts is None or str(val) > str(max_ts)):
                max_ts = str(val)
    return max_ts


def _is_stale(ts_iso: str | None, now: datetime, stale_minutes: int) -> bool:
    """Return ``True`` if *ts_iso* is older than *stale_minutes* from *now*."""
    if not ts_iso:
        return True
    try:
        ts = datetime.fromisoformat(ts_iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        return (now - ts) > timedelta(minutes=stale_minutes)
    except (ValueError, TypeError):
        return True


def check_game_data_freshness(
    db: Any,
    date_str: str,
    home_team: str,
    away_team: str,
    now: datetime,
) -> FreshnessResult:
    """Assess whether injuries, odds, and trends are fresh for the two teams."""
    result = FreshnessResult()
    teams = [home_team, away_team]

    # --- Injuries ---
    injury_ts = _max_timestamp_for_teams(db, teams, "injury_updated_at")
    result.injuries_synced_at = injury_ts or ""
    _cfg = get_active_config()
    result.injuries_fresh = not _is_stale(injury_ts, now, _cfg.pregame_injury_stale_minutes)

    # --- Odds ---
    games_doc = db.collection("games").document(date_str).get()
    odds_ts = (games_doc.to_dict() or {}).get("odds_synced_at") if games_doc.exists else None
    result.odds_synced_at = str(odds_ts) if odds_ts else ""
    result.odds_fresh = not _is_stale(str(odds_ts) if odds_ts else None, now, _cfg.pregame_odds_stale_minutes)

    # --- Trends ---
    trend_ts = _max_timestamp_for_teams(db, teams, "trend_updated_at")
    result.trends_synced_at = trend_ts or ""
    result.trends_fresh = not _is_stale(trend_ts, now, _cfg.pregame_trends_stale_minutes)

    return result


# ---------------------------------------------------------------------------
# Targeted syncs
# ---------------------------------------------------------------------------


def refresh_stale_data(
    db: Any,
    result: FreshnessResult,
    home_team: str,
    away_team: str,
    date_str: str = "",
) -> FreshnessResult:
    """Trigger syncs for any stale data types, mutating *result* in place."""

    # --- Injuries (global sync — fast, ~5s for all players) ---
    if not result.injuries_fresh:
        try:
            from bks_pipeline_core.pipeline.injuries import fetch_and_store_injuries

            api_key = get_active_config().stats_api_key.value
            count, _ = fetch_and_store_injuries(db.collection("players"), db, api_key)
            result.injuries_triggered_sync = True
            result.injuries_synced_at = datetime.now(timezone.utc).isoformat()
            result.injuries_fresh = True
            logger.info("pregame_freshness: injury sync complete, %d injured", count)
        except Exception as exc:
            result.errors.append(f"injury_sync: {exc}")
            logger.exception("pregame_freshness: injury sync failed")

    # --- Odds (global sync — single API call for all games) ---
    if not result.odds_fresh:
        try:
            from pipeline.orchestrator import fetch_and_store_odds

            count = fetch_and_store_odds()
            result.odds_triggered_sync = True
            result.odds_synced_at = datetime.now(timezone.utc).isoformat()
            result.odds_fresh = True
            logger.info("pregame_freshness: odds sync complete, %d games", count)
        except Exception as exc:
            result.errors.append(f"odds_sync: {exc}")
            logger.exception("pregame_freshness: odds sync failed")

    # --- Trends (targeted — only the two teams' players) ---
    if not result.trends_fresh:
        try:
            refreshed = refresh_team_trends(db, [home_team, away_team])
            result.trends_triggered_sync = True
            result.trends_synced_at = datetime.now(timezone.utc).isoformat()
            result.trends_fresh = True
            logger.info("pregame_freshness: trend sync complete, %d players", refreshed)
        except Exception as exc:
            result.errors.append(f"trend_sync: {exc}")
            logger.exception("pregame_freshness: trend sync failed")

    # --- Lineups (ESPN summary — only within the lead window) ---
    if date_str:
        try:
            # Check if lineups are already confirmed for this game
            matchup_confirmed = False
            matchups = db.collection("games").document(date_str).collection("matchups").stream()
            lineup_check_ts: str | None = None
            for mdoc in matchups:
                md = mdoc.to_dict() or {}
                home = md.get("home_team_abbr", "")
                visitor = md.get("visitor_team_abbr", "")
                if home == home_team or visitor == home_team:
                    if md.get("lineups_confirmed"):
                        matchup_confirmed = True
                    lineup_check_ts = md.get("lineups_confirmed_at") or md.get("lineup_checked_at")
                    break

            if matchup_confirmed:
                result.lineup_fresh = True
            elif _is_stale(lineup_check_ts, datetime.now(timezone.utc), get_active_config().lineup_stale_minutes):
                from pipeline.orchestrator import fetch_and_store_lineup_status

                changes = fetch_and_store_lineup_status(db, date_str)
                result.lineup_triggered_sync = True
                result.lineup_synced_at = datetime.now(timezone.utc).isoformat()
                result.lineup_fresh = True
                result.lineup_changes = changes
                logger.info("pregame_freshness: lineup sync complete, %d changes", len(changes))
            else:
                result.lineup_fresh = True
        except Exception as exc:
            result.errors.append(f"lineup_sync: {exc}")
            logger.exception("pregame_freshness: lineup sync failed")

    # --- Detect injury changes for FCM signal ---
    result.injury_changes = _detect_injury_changes(db, [home_team, away_team])

    return result


def refresh_team_trends(
    db: Any,
    team_abbrs: list[str],
) -> int:
    """Fetch and update trends for players on *team_abbrs* whose trends are stale.

    Reuses :func:`pipeline.trends.fetch_trends` with a targeted player list.
    Returns the number of players refreshed.
    """
    from pipeline.trends import fetch_trends

    from bks_pipeline_core.pipeline.league_state import get_league_state

    api_key = get_active_config().stats_api_key.value
    if not api_key:
        raise RuntimeError("stats_api_key not set")

    now = datetime.now(timezone.utc)

    # Gather stale player IDs for the two teams.
    stale_ids: list[int] = []
    for abbr in team_abbrs:
        docs = db.collection("players").where("team", "==", abbr).select(["id", "trend_updated_at"]).stream()
        for doc in docs:
            d = doc.to_dict() or {}
            pid = d.get("id")
            if pid is None:
                continue
            updated = d.get("trend_updated_at")
            if not updated or _is_stale(str(updated), now, get_active_config().pregame_trends_stale_minutes):
                stale_ids.append(int(pid))

    if not stale_ids:
        logger.info("refresh_team_trends: all %s players already fresh", team_abbrs)
        return 0

    league_state = get_league_state(db)
    is_playoffs = league_state.get("mode") == "playoffs"

    trend_data, _ = fetch_trends(stale_ids, api_key, is_playoffs=is_playoffs)

    # Write updated trend fields back to Firestore.
    updated_at = now.isoformat()
    batch = db.batch()
    count = 0
    for pid, fields in trend_data.items():
        ref = db.collection("players").document(str(pid))
        fields["trend_updated_at"] = updated_at
        batch.update(ref, fields)
        count += 1
        # Firestore batch limit is 500; commit and reset if needed.
        if count % 400 == 0:
            batch.commit()
            batch = db.batch()
    if count % 400 != 0:
        batch.commit()

    logger.info(
        "refresh_team_trends: updated %d/%d stale players for %s",
        count,
        len(stale_ids),
        team_abbrs,
    )
    return count


def _detect_injury_changes(
    db: Any,
    team_abbrs: list[str],
) -> list[dict[str, str]]:
    """Return players on *team_abbrs* whose injury status recently changed.

    Compares ``injury_status`` to ``previous_injury_status`` (written by the
    injury sync pipeline on each status flip).
    """
    changes: list[dict[str, str]] = []
    for abbr in team_abbrs:
        docs = (
            db.collection("players")
            .where("team", "==", abbr)
            .select(
                [
                    "id",
                    "first_name",
                    "last_name",
                    "team",
                    "injury_status",
                    "previous_injury_status",
                    "injury_status_changed_at",
                ]
            )
            .stream()
        )
        for doc in docs:
            d = doc.to_dict() or {}
            current = d.get("injury_status") or ""
            previous = d.get("previous_injury_status") or ""
            if current != previous and d.get("injury_status_changed_at"):
                changes.append(
                    {
                        "player_id": str(d.get("id", "")),
                        "name": f"{d.get('first_name', '')} {d.get('last_name', '')}".strip(),
                        "team": d.get("team", ""),
                        "old_status": previous,
                        "new_status": current,
                    }
                )
    return changes


# ---------------------------------------------------------------------------
# Firestore signal doc
# ---------------------------------------------------------------------------


def write_freshness_signal(
    db: Any,
    date_str: str,
    game_id: str,
    home_team: str,
    away_team: str,
    tip_off_utc: str,
    result: FreshnessResult,
) -> dict[str, Any]:
    """Write (or update) the ``game_freshness/{date}_{game_id}`` doc.

    Returns the document data that was written.
    """
    doc_id = f"{date_str}_{game_id}"
    ref = db.collection(FRESHNESS_COLLECTION).document(doc_id)

    now_iso = datetime.now(timezone.utc).isoformat()

    # Increment data_version for change detection.
    existing = ref.get()
    prev_version = (existing.to_dict() or {}).get("data_version", 0) if existing.exists else 0

    status = "verified" if result.all_fresh else ("error" if result.errors else "stale")

    doc_data: dict[str, Any] = {
        "date": date_str,
        "game_id": game_id,
        "home_team": home_team,
        "away_team": away_team,
        "tip_off_utc": tip_off_utc,
        "task_executed_at": now_iso,
        "injuries_fresh": result.injuries_fresh,
        "injuries_synced_at": result.injuries_synced_at,
        "injuries_triggered_sync": result.injuries_triggered_sync,
        "odds_fresh": result.odds_fresh,
        "odds_synced_at": result.odds_synced_at,
        "odds_triggered_sync": result.odds_triggered_sync,
        "trends_fresh": result.trends_fresh,
        "trends_synced_at": result.trends_synced_at,
        "trends_triggered_sync": result.trends_triggered_sync,
        "lineup_fresh": result.lineup_fresh,
        "lineup_synced_at": result.lineup_synced_at,
        "lineup_triggered_sync": result.lineup_triggered_sync,
        "status": status,
        "verified_at": now_iso,
        "data_version": prev_version + 1,
        "errors": result.errors or [],
    }

    ref.set(doc_data, merge=True)
    logger.info(
        "write_freshness_signal: %s status=%s version=%d",
        doc_id,
        status,
        prev_version + 1,
    )
    return doc_data


# ---------------------------------------------------------------------------
# Dedup
# ---------------------------------------------------------------------------


def should_skip_dedup(
    db: Any,
    date_str: str,
    game_id: str,
    home_team: str,
    away_team: str,
    now: datetime,
) -> bool:
    """Return ``True`` if a recent per-game task already refreshed overlapping teams.

    Checks all ``game_freshness/{date}_*`` docs for a ``task_executed_at``
    within :data:`PREGAME_DEDUP_WINDOW_MINUTES` that covers at least one of
    the same teams.
    """
    cutoff = now - timedelta(minutes=get_active_config().pregame_dedup_window_minutes)
    game_teams = {home_team, away_team}

    docs = db.collection(FRESHNESS_COLLECTION).where("date", "==", date_str).where("status", "in", ["verified", "stale"]).stream()
    for doc in docs:
        d = doc.to_dict() or {}
        if d.get("game_id") == game_id:
            continue  # don't dedup against ourselves
        executed = d.get("task_executed_at")
        if not executed:
            continue
        try:
            exec_dt = datetime.fromisoformat(executed)
            if exec_dt.tzinfo is None:
                exec_dt = exec_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue
        if exec_dt < cutoff:
            continue
        doc_teams = {d.get("home_team", ""), d.get("away_team", "")}
        if game_teams & doc_teams:
            logger.info(
                "should_skip_dedup: game %s skipped — overlapping teams with %s (executed %s)",
                game_id,
                d.get("game_id"),
                executed,
            )
            return True

    return False
