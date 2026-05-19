"""League state management — single source of truth for season mode.

The `system/league_state` document gates all mode-dependent behavior.
Possible modes: "regular_season", "playoffs", "offseason".
"""

import logging
from datetime import datetime, timezone
from typing import Any

from google.cloud.firestore_v1.base_document import DocumentSnapshot

logger = logging.getLogger(__name__)

__all__ = [
    "LEAGUE_STATE_DOC",
    "SYSTEM_COLLECTION",
    "VALID_MODES",
    "DEFAULT_MODE",
    "get_league_state",
    "set_league_state",
    "_default_state",
]

LEAGUE_STATE_DOC = "league_state"
SYSTEM_COLLECTION = "system"

VALID_MODES = {"regular_season", "playoffs", "offseason"}
DEFAULT_MODE = "regular_season"


def get_league_state(db: Any) -> dict[str, Any]:
    """Read the league_state document from Firestore.

    Returns the document dict if it exists, otherwise returns a default
    state and initializes the document in Firestore.
    """
    doc: DocumentSnapshot = db.collection(SYSTEM_COLLECTION).document(LEAGUE_STATE_DOC).get()
    if doc.exists:
        return doc.to_dict() or _default_state()

    # First access — seed the document with defaults
    state = _default_state()
    try:
        db.collection(SYSTEM_COLLECTION).document(LEAGUE_STATE_DOC).set(state)
        logger.info("Initialized system/league_state with defaults")
    except Exception as exc:
        logger.warning("Failed to initialize league_state: %s", exc)
    return state


_VALID_TRANSITIONS: dict[str, set[str]] = {
    "regular_season": {"playoffs", "offseason"},
    "playoffs": {"offseason"},
    "offseason": {"regular_season"},
}


def set_league_state(db: Any, updates: dict[str, Any], updated_by: str = "admin") -> dict[str, Any]:
    """Update the league_state document with validated fields.

    Only allows known fields. Returns the full updated state.
    Raises ValueError for invalid mode transitions.
    """
    current = get_league_state(db)
    new_mode = updates.get("mode")

    if new_mode and new_mode != current["mode"]:
        if new_mode not in VALID_MODES:
            raise ValueError(f"Invalid mode '{new_mode}'. Must be one of {VALID_MODES}")
        allowed = _VALID_TRANSITIONS.get(current["mode"], set())
        if new_mode not in allowed:
            raise ValueError(f"Cannot transition from '{current['mode']}' to '{new_mode}'. Allowed transitions: {allowed}")

    _ALLOWED_FIELDS = {
        "mode",
        "season",
        "playoff_start_date",
        "playoff_round",
        "play_in_active",
        "regular_season_end_date",
        "use_new_pipeline",
    }
    write_data: dict[str, Any] = {k: v for k, v in updates.items() if k in _ALLOWED_FIELDS}
    write_data["updated_at"] = datetime.now(timezone.utc).isoformat()
    write_data["updated_by"] = updated_by

    db.collection(SYSTEM_COLLECTION).document(LEAGUE_STATE_DOC).update(write_data)
    logger.info(
        "League state updated by %s: %s",
        updated_by,
        {k: v for k, v in write_data.items() if k != "updated_at"},
    )

    # Return the merged state
    current.update(write_data)
    return current


def _default_state() -> dict[str, Any]:
    """Return the default league state document."""
    return {
        "mode": DEFAULT_MODE,
        "season": 2025,
        "playoff_start_date": None,
        "playoff_round": None,
        "play_in_active": False,
        "regular_season_end_date": None,
        "use_new_pipeline": False,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "updated_by": "system",
    }
