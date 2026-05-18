"""Versioned prediction storage for the backtesting pipeline.

Each prediction run is stored as a separate Firestore document in the
``prediction_runs`` collection with a unique auto-generated ID.  This
allows multiple algorithm iterations to be compared against the same
day's actuals.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from google.cloud.firestore_v1 import Query

logger = logging.getLogger(__name__)


def store_prediction_run(
    db: Any,
    date: str,
    version_label: str,
    platform: str,
    mode: str,
    predictions: dict[str, dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> str:
    """Write a prediction run to ``prediction_runs/{auto_id}``.

    Args:
        db:             Firestore client.
        date:           Game date (``YYYY-MM-DD``).
        version_label:  Human-readable version tag (e.g. ``"v2.1"``,
                        ``"scheduled"`` for the daily auto-run).
        platform:       Fantasy platform key (``"dk"`` or ``"fd"``).
        mode:           Scoring mode (``"balanced"``, ``"cash"``, ``"gpp"``).
        predictions:    Dict of player predictions keyed by player ID.
        metadata:       Optional arbitrary metadata to attach.

    Returns:
        The Firestore document ID of the newly created run.
    """
    now_et = datetime.now(ZoneInfo("America/New_York"))
    ttl = now_et + timedelta(days=30)

    doc_data: dict[str, Any] = {
        "date": date,
        "version_label": version_label,
        "created_at": now_et.isoformat(),
        "platform": platform,
        "mode": mode,
        "player_count": len(predictions),
        "predictions": predictions,
        "ttl": ttl,
    }
    if metadata:
        doc_data["metadata"] = metadata

    _, doc_ref = db.collection("prediction_runs").add(doc_data)
    run_id: str = doc_ref.id
    logger.info(
        "store_prediction_run: %s/%s — %d players → %s",
        date,
        version_label,
        len(predictions),
        run_id,
    )
    return run_id


def get_prediction_run(
    db: Any,
    run_id: str,
) -> dict[str, Any] | None:
    """Fetch a single prediction run by document ID."""
    doc = db.collection("prediction_runs").document(run_id).get()
    if not doc.exists:
        return None
    data = doc.to_dict() or {}
    data["id"] = doc.id
    return data


def get_prediction_run_by_version(
    db: Any,
    date: str,
    version_label: str,
) -> dict[str, Any] | None:
    """Find a prediction run by date + version label.

    Returns the most recent match if multiple exist, or ``None``.
    """
    docs = (
        db.collection("prediction_runs")
        .where("date", "==", date)
        .where("version_label", "==", version_label)
        .order_by("created_at", direction=Query.DESCENDING)
        .limit(1)
        .stream()
    )
    for doc in docs:
        data = doc.to_dict() or {}
        data["id"] = doc.id
        return data
    return None


def list_prediction_runs(
    db: Any,
    date: str,
) -> list[dict[str, Any]]:
    """List all prediction runs for a given date, newest first."""
    docs = db.collection("prediction_runs").where("date", "==", date).order_by("created_at", direction=Query.DESCENDING).stream()
    results: list[dict[str, Any]] = []
    for doc in docs:
        data = doc.to_dict() or {}
        data["id"] = doc.id
        # Omit the large predictions map from listing
        data.pop("predictions", None)
        results.append(data)
    return results
