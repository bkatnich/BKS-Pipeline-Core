"""Fetch and store DraftKings winning lineup data for a given game date.

Data is sourced from RotoGrinders via Scrapfly (JS rendering + bot bypass).
Stored in dk_winning_lineups/{YYYY-MM-DD} with a 90-day TTL.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

_COLLECTION = "dk_winning_lineups"
_TTL_DAYS = 90


def fetch_and_store_dk_winning_lineups(
    db: Any,
    date_str: str,
    scrapfly_api_key: str,
    log: logging.Logger | None = None,
) -> dict[str, Any]:
    """Fetch DK top-50 winning lineups for date_str and store to Firestore.

    Idempotent: returns existing doc if already stored for the date.
    Returns the stored dict, or {} on off-nights or API failure.
    """
    from api.draftkings import fetch_dk_winning_lineups

    _log = log or logger

    existing = db.collection(_COLLECTION).document(date_str).get()
    if existing.exists:
        _log.info("fetch_and_store_dk_winning_lineups: already stored for %s", date_str)
        return existing.to_dict() or {}

    result = fetch_dk_winning_lineups(date_str, scrapfly_api_key, top_n=50, logger=_log)

    if not result:
        _log.info("fetch_and_store_dk_winning_lineups: no lineup data for %s (off-night?)", date_str)
        return {}

    now = datetime.now(timezone.utc)
    doc: dict[str, Any] = {
        "date": date_str,
        "fetched_at": now.isoformat(),
        "top_n_fetched": 50,
        "ttl": now + timedelta(days=_TTL_DAYS),
        **result,
    }

    try:
        db.collection(_COLLECTION).document(date_str).set(doc)
        _log.info(
            "fetch_and_store_dk_winning_lineups: stored %d lineups for %s",
            len(result.get("top_lineups", [])),
            date_str,
        )
    except Exception as exc:
        _log.warning("fetch_and_store_dk_winning_lineups: Firestore write failed — %s", exc)
        return {}

    return doc


def get_winning_lineups(db: Any, date_str: str) -> dict[str, Any] | None:
    """Load stored winning lineup doc for date_str, or None if not found."""
    doc = db.collection(_COLLECTION).document(date_str).get()
    if not doc.exists:
        return None
    return doc.to_dict() or None
