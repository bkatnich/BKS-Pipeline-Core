"""Platt calibration coefficient persistence.

Reads/writes Platt scaling coefficients from ``system/calibration``
in Firestore.  Maintains a rolling history of fits for audit.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from google.cloud.firestore_v1.base_client import BaseClient

from bks_pipeline_core.pipeline.platt import brier_score, fit_platt
from bks_pipeline_core.sport_config import get_active_config

logger = logging.getLogger(__name__)


_MAX_HISTORY = 30  # keep last 30 daily fits


def load_platt_coefficients(
    db: BaseClient,
) -> dict[str, dict[str, Any]] | None:
    """Load current Platt coefficients from ``system/calibration``.

    Returns a dict keyed by stat type (e.g. ``"pts"``) with
    ``{"A": float, "B": float}`` values, or ``None`` if no
    calibration exists.
    """
    doc = db.collection("system").document("calibration").get()
    if not doc.exists:  # type: ignore[union-attr]
        return None
    data: dict[str, Any] = doc.to_dict() or {}  # type: ignore[union-attr]
    current: dict[str, Any] | None = data.get("current")
    if not current:
        return None
    return {k: v for k, v in current.items() if isinstance(v, dict)}


def store_platt_coefficients(
    db: BaseClient,
    coefficients: dict[str, dict[str, Any]],
) -> None:
    """Write updated Platt coefficients with history rotation."""
    now_iso = datetime.now(timezone.utc).isoformat()

    cal_ref = db.collection("system").document("calibration")
    existing = cal_ref.get()
    history: list[dict[str, Any]] = []
    if existing.exists:  # type: ignore[union-attr]
        data = existing.to_dict() or {}  # type: ignore[union-attr]
        history = data.get("history", [])

    # Add current fit to history.
    history.append(
        {
            "timestamp": now_iso,
            "window_days": get_active_config().platt_window_days,
            "coefficients": coefficients,
        }
    )

    # Trim to most recent entries.
    if len(history) > _MAX_HISTORY:
        history = history[-_MAX_HISTORY:]

    cal_ref.set(
        {
            "updated_at": now_iso,
            "current": coefficients,
            "history": history,
        }
    )
    logger.info("platt coefficients stored: %d stat types", len(coefficients))


def refit_platt_from_actuals(
    db: BaseClient,
    window_days: int | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Refit Platt coefficients from recent prop_actuals.

    Loads graded prop results from the last ``window_days`` days,
    groups by stat type, fits Platt scaling per stat, and stores
    updated coefficients.

    Returns the fitted coefficients dict, or ``None`` if insufficient data.
    """
    # Collect (raw_prob, outcome) pairs from prop_actuals across dates.
    # Scan the maximum window needed across all stats to avoid multiple passes.
    from datetime import timedelta

    _cfg = get_active_config()
    if window_days is None:
        window_days = _cfg.platt_window_days
    _platt_window_by_stat = _cfg.platt_window_days_by_stat or {}
    max_window = max([window_days] + list(_platt_window_by_stat.values()))
    today = datetime.now(timezone.utc).date()
    dates_to_check = [(today - timedelta(days=d)).isoformat() for d in range(1, max_window + 1)]

    # Store (date_index, raw_prob, outcome) so we can trim per-stat window later.
    pairs_by_stat: dict[str, list[tuple[int, float, bool]]] = {}

    for day_offset, date_str in enumerate(dates_to_check, start=1):
        sport_ref = db.collection("prop_actuals").document(date_str).collection(get_active_config().sport_collection_key)
        for player_doc in sport_ref.stream():
            player_data: dict[str, Any] = player_doc.to_dict() or {}
            lines: dict[str, Any] = player_data.get("lines", {})
            for _line_key, line_data in lines.items():
                stat = line_data.get("stat")
                raw_prob = line_data.get("model_prob_over")
                over_hit = line_data.get("over_hit")
                if stat is None or raw_prob is None or over_hit is None:
                    continue
                if stat not in pairs_by_stat:
                    pairs_by_stat[stat] = []
                pairs_by_stat[stat].append((day_offset, float(raw_prob), bool(over_hit)))

    # Fit per stat type, applying per-stat window cutoff.
    coefficients: dict[str, dict[str, Any]] = {}
    for stat, raw_pairs in pairs_by_stat.items():
        stat_window = _platt_window_by_stat.get(stat, window_days)
        pairs = [(p, o) for day_offset, p, o in raw_pairs if day_offset <= stat_window]
        if len(pairs) < _cfg.platt_min_samples:
            logger.info("platt refit: %s has %d samples in %d-day window (need %d), skipping", stat, len(pairs), stat_window, _cfg.platt_min_samples)
            continue

        result = fit_platt(pairs)
        coefficients[stat] = {
            "A": result["A"],
            "B": result["B"],
            "brier": result["brier"],
            "samples": result["samples"],
            "stat_type": stat,
        }
        logger.info("platt refit: %s A=%.4f B=%.4f brier=%.4f n=%d", stat, result["A"], result["B"], result["brier"], result["samples"])

    if not coefficients:
        logger.info("platt refit: no stat types had sufficient samples")
        return None

    # Also compute uncalibrated Brier for comparison (using same window as fit).
    for stat, raw_pairs in pairs_by_stat.items():
        if stat in coefficients:
            stat_window = _platt_window_by_stat.get(stat, window_days)
            pairs = [(p, o) for day_offset, p, o in raw_pairs if day_offset <= stat_window]
            uncal_brier = brier_score(
                [p for p, _ in pairs],
                [o for _, o in pairs],
            )
            coefficients[stat]["uncalibrated_brier"] = round(uncal_brier, 4)

    store_platt_coefficients(db, coefficients)
    return coefficients
