import logging
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from bks_pipeline_core.sport_config import get_active_config

logger = logging.getLogger(__name__)


def compute_return_fields(return_date_str: str | None, today: Any) -> dict[str, Any]:
    """Compute days_since_return and is_return_game_window from an ISO date string.

    Args:
        return_date_str: ISO date "YYYY-MM-DD" or None.
        today: datetime.date representing today's date.
    """
    if not return_date_str:
        return {"days_since_return": None, "is_return_game_window": False}
    try:
        return_date = datetime.strptime(return_date_str, "%Y-%m-%d").date()
        delta = (today - return_date).days
        if delta >= 0:
            return {
                "days_since_return": delta,
                "is_return_game_window": delta <= get_active_config().injury_return_window_days,
            }
    except ValueError:
        pass
    return {"days_since_return": None, "is_return_game_window": False}


class InjuryFetchError(Exception):
    """Raised when the injury API fails mid-pagination.

    Callers must catch this and skip both Firestore write passes to avoid
    incorrectly clearing all players' injury status to null.
    """


def fetch_and_store_injuries(
    players_ref: Any,
    db: Any,
    api_key: str,
    date_str: str | None = None,
    fetch_page_fn: Callable[[int | None, str, logging.Logger], dict[str, Any] | None] | None = None,
) -> tuple[int, list[dict[str, Any]]]:
    """Fetch current injury records and update player docs.

    Paginates the injury API (cursor-based), writes injury_status,
    injury_return_date, injury_comment, injury_updated_at, days_since_return,
    and is_return_game_window to each affected player doc. Players not on the
    injury report get all fields set to null/False.

    Computing days_since_return here (every 20 min) ensures it stays fresh
    rather than going stale for up to 12h if computed during trend sync.

    Args:
        fetch_page_fn: Injectable page-fetch callable matching the
            fetch_injuries_page(cursor, api_key, logger) signature.
            Pass STATS_PROVIDER.fetch_injuries_page from the generated
            project to route through the active provider. Defaults to
            a direct import of api.sport_provider.fetch_injuries_page
            for backward compatibility with callers that cannot access
            STATS_PROVIDER (e.g. pregame_freshness).

    Raises InjuryFetchError if any page fetch fails, so the caller can skip
    all Firestore writes and leave existing injury data untouched.

    Returns the count of injured players found.
    """
    injury_map: dict[int, dict[str, Any]] = {}
    cursor: int | None = None
    page = 0

    if fetch_page_fn is None:
        from api.sport_provider import fetch_injuries_page  # sport-specific lazy import

        fetch_page_fn = fetch_injuries_page

    cfg = get_active_config()
    while page < cfg.max_pages:
        body = fetch_page_fn(cursor, api_key, logger)
        if body is None:
            logger.error(
                "Failed to fetch injuries page %d after 3 attempts — aborting injury sync",
                page + 1,
            )
            raise InjuryFetchError(f"Injury API failed on page {page + 1}")

        for record in body.get("data", []):
            player = record.get("player") or {}
            pid = player.get("id")
            if pid:
                injury_map[pid] = {
                    "injury_status": record.get("status"),
                    "injury_return_date": record.get("return_date"),
                    "injury_comment": record.get("description"),
                }

        cursor = body.get("meta", {}).get("next_cursor")
        page += 1
        if cursor is None:
            break

    now_iso = datetime.now(timezone.utc).isoformat()
    today = datetime.now(timezone.utc).date()
    injured_count = len(injury_map)
    logger.info("Injury sync: %d injured players found across %d pages", injured_count, page)

    # Collect all known player docs with existing injury fields for change detection
    existing_status: dict[str, dict[str, Any]] = {}
    _select_fields = [
        "id",
        "first_name",
        "last_name",
        "team",
        "injury_status",
        "injury_status_changed_at",
        "previous_injury_status",
        "injury_return_date",
        "is_playoff_active",
    ]
    for doc in players_ref.select(_select_fields).stream():
        d = doc.to_dict() or {}
        first = d.get("first_name") or ""
        last = d.get("last_name") or ""
        # team is stored as a nested object; abbreviation lives at team.abbreviation
        team_obj = d.get("team") or {}
        team_abbr = (team_obj.get("abbreviation") if isinstance(team_obj, dict) else "") or ""
        existing_status[doc.id] = {
            "injury_status": d.get("injury_status"),
            "injury_status_changed_at": d.get("injury_status_changed_at"),
            "previous_injury_status": d.get("previous_injury_status"),
            "injury_return_date": d.get("injury_return_date"),
            "is_playoff_active": bool(d.get("is_playoff_active")),
            "name": f"{first} {last}".strip(),
            "team": team_abbr,
            "has_profile": bool(first),  # False for ghost players not yet synced
        }

    known_doc_ids = set(existing_status.keys())
    injured_ids = {str(pid) for pid in injury_map}
    healthy_ids = known_doc_ids - injured_ids

    _OUT_FOR_SEASON_DATE = "2026-10-01"

    # Pass 1: write injury data for injured players (includes return window + change tracking)
    # Skip players whose return date is the out-for-season sentinel — their status will
    # never change this year and syncing them every 20 minutes is pure noise.
    injured_items = list(injury_map.items())
    status_changes = 0
    changed_players: list[dict[str, str]] = []
    skipped_out_for_season = 0
    for chunk_start in range(0, len(injured_items), 500):
        batch = db.batch()
        wrote_any = False
        for pid, fields in injured_items[chunk_start : chunk_start + 500]:
            existing = existing_status.get(str(pid), {})

            # Skip players whose profile hasn't been synced yet (no first_name).
            # They appear on the BallDontLie injury report but have no player doc
            # data — writing them produces nameless, teamless notification entries.
            if not existing.get("has_profile"):
                continue

            # Skip players who are out for the season — return date won't change.
            # Use the existing stored return date; fall back to the incoming value
            # so newly out-for-season players are written once before being skipped.
            stored_return = existing.get("injury_return_date")
            incoming_return = fields.get("injury_return_date")
            if stored_return == _OUT_FOR_SEASON_DATE and incoming_return == _OUT_FOR_SEASON_DATE:
                skipped_out_for_season += 1
                continue

            doc_ref = players_ref.document(str(pid))
            old_status = existing.get("injury_status")
            new_status = fields.get("injury_status")
            status_changed = old_status != new_status
            if status_changed:
                status_changes += 1
                changed_players.append(
                    {
                        "player_id": str(pid),
                        "old_status": str(old_status or ""),
                        "new_status": str(new_status or ""),
                        "name": existing.get("name") or "",
                        "team": existing.get("team") or "",
                        "is_playoff_active": existing.get("is_playoff_active", False),
                    }
                )
            change_fields = {
                "previous_injury_status": old_status if status_changed else existing.get("previous_injury_status"),
                "injury_status_changed_at": now_iso if status_changed else existing.get("injury_status_changed_at"),
            }
            batch.set(
                doc_ref,
                {
                    **fields,
                    **change_fields,
                    "injury_updated_at": now_iso,
                    **compute_return_fields(fields.get("injury_return_date"), today),
                },
                merge=True,
            )
            wrote_any = True
        if wrote_any:
            batch.commit()

    # Pass 2: null-clear players not on the injury report (includes change tracking)
    # Skip out-for-season players here too — their cleared status is irrelevant.
    healthy_list = list(healthy_ids)
    for chunk_start in range(0, len(healthy_list), 500):
        batch = db.batch()
        wrote_any = False
        for doc_id in healthy_list[chunk_start : chunk_start + 500]:
            existing = existing_status.get(doc_id, {})
            old_status = existing.get("injury_status")

            # If their stored return date was the out-for-season sentinel they were
            # already being skipped in Pass 1, so no Firestore write is needed here.
            if existing.get("injury_return_date") == _OUT_FOR_SEASON_DATE:
                skipped_out_for_season += 1
                continue

            doc_ref = players_ref.document(doc_id)
            # Transitioning from injured → healthy is a status change
            status_changed = old_status is not None
            if status_changed:
                status_changes += 1
                changed_players.append(
                    {
                        "player_id": doc_id,
                        "old_status": str(old_status or ""),
                        "new_status": "",
                        "name": existing.get("name") or "",
                        "team": existing.get("team") or "",
                        "is_playoff_active": existing.get("is_playoff_active", False),
                    }
                )
            batch.set(
                doc_ref,
                {
                    "injury_status": None,
                    "injury_return_date": None,
                    "injury_comment": None,
                    "injury_updated_at": now_iso,
                    "days_since_return": None,
                    "is_return_game_window": False,
                    "previous_injury_status": old_status if status_changed else existing.get("previous_injury_status"),
                    "injury_status_changed_at": now_iso if status_changed else existing.get("injury_status_changed_at"),
                },
                merge=True,
            )
            wrote_any = True
        if wrote_any:
            batch.commit()

    logger.info(
        "Injury sync complete: %d injured, %d cleared, %d status changes, %d out-for-season skipped",
        injured_count,
        len(healthy_ids),
        status_changes,
        skipped_out_for_season,
    )

    if changed_players and date_str:
        from pipeline.notifications import send_injury_change_signal

        from bks_pipeline_core.pipeline.activity_log import write_activity_event

        # Always log all changes to the activity feed for audit purposes.
        write_activity_event(db, "injury_update", date_str, changed_players, source="scheduled_sync")

        # Only notify for players on playoff-active teams — eliminated team injuries
        # are irrelevant for DFS purposes.
        playoff_changes = [p for p in changed_players if p.get("is_playoff_active")]
        if playoff_changes:
            send_injury_change_signal(date_str, playoff_changes)
        else:
            logger.info("Injury sync: %d change(s) suppressed — no playoff-active players affected", len(changed_players))

    return injured_count, changed_players
