"""On-demand comparison engine for prediction runs vs. actuals.

Reuses the pure-computation functions from ``backtesting.py`` and stores
each comparison result as a ``comparison_runs/{auto_id}`` Firestore
document so that algorithm iterations can be tracked over time.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from google.cloud.firestore_v1 import Query

from bks_pipeline_core.pipeline.backtesting import (
    compute_floor_ceiling_calibration,
    compute_overall_accuracy,
    compute_signal_accuracy,
    compute_tier_accuracy,
    join_predictions_actuals,
)
from bks_pipeline_core.pipeline.prediction_store import get_prediction_run

logger = logging.getLogger(__name__)


def run_comparison(
    db: Any,
    actuals_date: str,
    prediction_run_id: str,
    email_to: str | None = None,
) -> str | None:
    """Compare a prediction run against actuals and store the result.

    Args:
        db:                 Firestore client.
        actuals_date:       Date of the actuals document (``YYYY-MM-DD``).
        prediction_run_id:  Document ID of the prediction run.
        email_to:           If provided, queue an HTML accuracy report
                            email to this address via the ``mail`` collection.

    Returns:
        The Firestore document ID of the comparison run, or ``None``
        when either the prediction run or actuals are missing.
    """
    # Load prediction run
    pred_run = get_prediction_run(db, prediction_run_id)
    if pred_run is None:
        logger.warning("run_comparison: prediction run %s not found", prediction_run_id)
        return None

    predictions = pred_run.get("predictions", {})
    if not predictions:
        logger.warning("run_comparison: prediction run %s has no predictions", prediction_run_id)
        return None

    # Load actuals — fall back to previous day if target date has none
    actual_doc = db.collection("actuals").document(actuals_date).get()
    if not actual_doc.exists or not (actual_doc.to_dict() or {}).get("results"):
        yesterday = (datetime.strptime(actuals_date, "%Y-%m-%d") - timedelta(days=1)).strftime("%Y-%m-%d")
        logger.info(
            "run_comparison: no actuals for %s, falling back to %s",
            actuals_date,
            yesterday,
        )
        actual_doc = db.collection("actuals").document(yesterday).get()
        if actual_doc.exists and (actual_doc.to_dict() or {}).get("results"):
            actuals_date = yesterday
        else:
            logger.warning("run_comparison: no actuals for %s or %s", actuals_date, yesterday)
            return None
    actuals = (actual_doc.to_dict() or {}).get("results", {})
    if not actuals:
        logger.warning("run_comparison: empty actuals for %s", actuals_date)
        return None

    # Join and compute metrics (reuses backtesting.py pure functions)
    joined = join_predictions_actuals(predictions, actuals)
    if not joined:
        logger.warning(
            "run_comparison: no matched players for %s vs %s",
            prediction_run_id,
            actuals_date,
        )
        return None

    overall = compute_overall_accuracy(joined)
    tier_accuracy = compute_tier_accuracy(joined)
    signal_accuracy = compute_signal_accuracy(joined)
    floor_ceiling = compute_floor_ceiling_calibration(joined)

    # Build top picks summary (top 10 by predicted_fp)
    joined_sorted = sorted(
        joined,
        key=lambda r: float(r.get("predicted_fp") or r.get("avg_fantasy_score") or 0),
        reverse=True,
    )
    top_picks = [
        {
            "player_id": r.get("player_id"),
            "player_name": f"{r.get('first_name', '')} {r.get('last_name', '')}".strip(),
            "position": r.get("position"),
            "predicted_fp": round(float(r.get("predicted_fp") or r.get("avg_fantasy_score") or 0), 1),
            "actual_fp": round(float(r.get("actual_actual_fp_dk", 0)), 1),
            "delta": round(
                float(r.get("actual_actual_fp_dk", 0)) - float(r.get("predicted_fp") or r.get("avg_fantasy_score") or 0),
                1,
            ),
        }
        for r in joined_sorted[:10]
    ]

    now_et = datetime.now(ZoneInfo("America/New_York"))
    ttl = now_et + timedelta(days=30)

    comparison_doc: dict[str, Any] = {
        "actuals_date": actuals_date,
        "prediction_run_id": prediction_run_id,
        "version_label": pred_run.get("version_label", "unknown"),
        "created_at": now_et.isoformat(),
        "metrics": {
            "overall": overall,
            "tier_accuracy": tier_accuracy,
            "signal_accuracy": signal_accuracy,
            "floor_ceiling": floor_ceiling,
        },
        "summary": {
            "sample_size": overall.get("sample_size", 0),
            "predicted_fp_vs_actual_r": overall.get("predicted_fp_vs_actual_r"),
            "predicted_fp_mae": overall.get("predicted_fp_mae"),
            "added_value": overall.get("predicted_fp_added_value"),
        },
        "top_picks": top_picks,
        "ttl": ttl,
    }

    _, doc_ref = db.collection("comparison_runs").add(comparison_doc)
    comp_id: str = doc_ref.id
    logger.info(
        "run_comparison: %s vs %s → %s (r=%s, MAE=%s)",
        pred_run.get("version_label"),
        actuals_date,
        comp_id,
        overall.get("predicted_fp_vs_actual_r"),
        overall.get("predicted_fp_mae"),
    )

    # Send email report if requested
    if email_to:
        from bks_pipeline_core.pipeline.accuracy_report import generate_accuracy_report_html

        version_label = pred_run.get("version_label", "unknown")
        accuracy_doc: dict[str, Any] = {
            "date": actuals_date,
            "sample_size": overall.get("sample_size", 0),
            "overall": overall,
            "tier_accuracy": tier_accuracy,
            "signal_accuracy": signal_accuracy,
            "floor_ceiling": floor_ceiling,
        }
        html = generate_accuracy_report_html(accuracy_doc)
        mail_id = f"{now_et.strftime('%Y-%m-%d-%H:%M')}_comparison_{version_label}"
        db.collection("mail").document(mail_id).set(
            {
                "to": email_to,
                "message": {
                    "subject": (f"BKS Basketball Comparison — {actuals_date} [{version_label}]"),
                    "html": html,
                },
                "created_at": now_et.isoformat(),
            }
        )
        logger.info("run_comparison: email queued to %s (%s)", email_to, mail_id)

    return comp_id


def get_comparison(
    db: Any,
    comparison_id: str,
) -> dict[str, Any] | None:
    """Fetch a comparison run by document ID."""
    doc = db.collection("comparison_runs").document(comparison_id).get()
    if not doc.exists:
        return None
    data = doc.to_dict() or {}
    data["id"] = doc.id
    return data


def list_comparisons(
    db: Any,
    actuals_date: str | None = None,
    version_label: str | None = None,
) -> list[dict[str, Any]]:
    """List comparison runs, optionally filtered by date or version.

    Returns results newest first, without the full metrics payload.
    """
    query: Any = db.collection("comparison_runs")

    if actuals_date is not None:
        query = query.where("actuals_date", "==", actuals_date)
    if version_label is not None:
        query = query.where("version_label", "==", version_label)

    query = query.order_by("created_at", direction=Query.DESCENDING)

    results: list[dict[str, Any]] = []
    for doc in query.stream():
        data = doc.to_dict() or {}
        data["id"] = doc.id
        # Omit heavy nested fields from listing
        data.pop("metrics", None)
        data.pop("top_picks", None)
        results.append(data)
    return results
