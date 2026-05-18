"""Persistent activity log for player change events.

Writes one Firestore document per event batch to activity_feed/{auto_id}.
The write is non-fatal — exceptions are swallowed so a Firestore failure
never blocks an FCM send or scheduler return.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

logger = logging.getLogger(__name__)

_ACTIVITY_FEED_TTL_DAYS = 30


def has_activity_event_today(db: Any, event_type: str, date_str: str) -> bool:
    """Return True if an activity_feed doc with this event_type and date already exists."""
    try:
        docs = db.collection("activity_feed").where("event_type", "==", event_type).where("date", "==", date_str).limit(1).stream()
        return any(True for _ in docs)
    except Exception:
        logger.warning("has_activity_event_today: query failed for %s/%s", event_type, date_str, exc_info=True)
        return False


def write_activity_event(
    db: Any,
    event_type: str,
    date_str: str,
    changes: list[dict[str, Any]],
    source: str,
    game_id: str = "",
) -> str:
    """Persist one activity_feed document. Returns auto-generated event_id or '' on failure."""
    try:
        now_utc = datetime.now(timezone.utc)
        ttl = datetime.now(ZoneInfo("America/New_York")) + timedelta(days=_ACTIVITY_FEED_TTL_DAYS)
        doc_data: dict[str, Any] = {
            "event_type": event_type,
            "occurred_at": now_utc.isoformat(),
            "date": date_str,
            "source": source,
            "game_id": game_id,
            "changes": changes,
            "change_count": len(changes),
            "ttl": ttl,
        }
        _write_result, doc_ref = db.collection("activity_feed").add(doc_data)
        event_id: str = doc_ref.id
        doc_ref.update({"event_id": event_id})
        return event_id
    except Exception:
        logger.warning(
            "write_activity_event: failed to persist %s event for date=%s",
            event_type,
            date_str,
            exc_info=True,
        )
        return ""
