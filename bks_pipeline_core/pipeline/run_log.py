"""Pipeline run log — lightweight per-stage status tracking in Firestore.

Each scheduled function calls record_stage() once it finishes (or on error).
The digest reader calls load_digest_data() to pull both dates' stages at once.

Firestore path: pipeline_runs/{YYYY-MM-DD}/stages/{stage_name}
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

_COLLECTION = "pipeline_runs"
_STAGES_SUB = "stages"

# Ordered list of all tracked stages, grouped for the digest report.
# (group, stage_name, display_label, expected_window_et)
STAGE_MANIFEST: list[tuple[str, str, str, str]] = [
    # Overnight processing — runs against yesterday's data
    ("overnight", "capture_actuals", "Capture Actuals", "3:00 AM"),
    ("overnight", "resolve_daily_props", "Resolve Prop Lines", "3:15 AM"),
    ("overnight", "compute_accuracy", "Compute Accuracy", "4:00 AM"),
    # Today's ingestion — fresh data for the current slate (ordered by run time)
    ("today", "sync_today_games", "Sync Today's Games", "12:05 AM"),
    ("today", "sync_active_players", "Sync Players / Trends", "~3:00 AM"),
    ("today", "snapshot_predictions_morning", "Morning Predictions Snapshot", "6:00 AM"),
    ("today", "snapshot_projections_daily", "Projections Snapshot", "6:30 AM"),
    ("today", "morning_analysis", "Morning Analysis", "9:30 AM"),
    ("today", "sync_odds", "Sync Vegas Odds", "9:00 AM"),
    ("today", "sync_prop_lines", "Sync Prop Lines", "12:00 PM"),
    ("today", "check_pipeline_morning", "Morning Health Check", "10:00 AM"),
    # Predictions — evening snapshot
    ("predictions", "snapshot_predictions_early", "Early Prediction Snapshot", "5:00 PM"),
    ("predictions", "snapshot_predictions", "Final Prediction Snapshot", "7:00 PM"),
    ("predictions", "snapshot_prop_predictions", "Prop Prediction Snapshot", "7:05 PM"),
]


def record_stage(
    db: Any,
    date_str: str,
    stage_name: str,
    *,
    status: str,
    count: int | None = None,
    detail: str | None = None,
    error_msg: str | None = None,
    warnings: list[str] | None = None,
    started_at: str | None = None,
) -> None:
    """Write a pipeline stage result to Firestore.

    Args:
        db:         Firestore client.
        date_str:   The date this stage ran for (YYYY-MM-DD).
        stage_name: Must match a name in STAGE_MANIFEST.
        status:     'success' | 'warning' | 'error' | 'skipped'
        count:      Records written/processed (optional).
        detail:     Short human-readable summary line (optional).
        error_msg:  Exception message on failure (optional).
        warnings:   List of warning strings (optional).
        started_at: ISO timestamp when stage began (optional).
    """
    now = datetime.now(timezone.utc).isoformat()
    doc: dict[str, Any] = {
        "stage": stage_name,
        "status": status,
        "completed_at": now,
    }
    if started_at is not None:
        doc["started_at"] = started_at
    if count is not None:
        doc["count"] = count
    if detail is not None:
        doc["detail"] = detail
    if error_msg is not None:
        doc["error_msg"] = error_msg
    if warnings:
        doc["warnings"] = warnings

    try:
        (db.collection(_COLLECTION).document(date_str).collection(_STAGES_SUB).document(stage_name).set(doc))
    except Exception as exc:
        logger.warning("record_stage: failed to write %s/%s — %s", date_str, stage_name, exc)


def load_digest_data(
    db: Any,
    overnight_date: str,
    today_date: str,
) -> dict[str, dict[str, Any]]:
    """Load all stage docs for both dates and return a flat name→doc mapping.

    overnight_date: yesterday (actuals, accuracy, prop grading run for this date)
    today_date:     today     (ingestion + prediction stages run for this date)
    """
    results: dict[str, dict[str, Any]] = {}

    def _load(date_str: str) -> None:
        try:
            docs = db.collection(_COLLECTION).document(date_str).collection(_STAGES_SUB).stream()
            for doc in docs:
                data = doc.to_dict() or {}
                results[doc.id] = data
        except Exception as exc:
            logger.warning("load_digest_data: failed to load %s — %s", date_str, exc)

    _load(overnight_date)
    _load(today_date)
    return results
