"""Signal weight auto-tuning persistence and nudge algorithm.

Reads/writes tuned signal clamp widths from ``system/signal_weights``
in Firestore.  Uses per-signal accuracy telemetry (residual_correlation,
hit_rate) to nudge clamp half-widths wider (signal is working) or
narrower (signal is noisy/harmful).

All tuning functions are pure — no I/O or side effects.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from google.cloud.firestore_v1.base_client import BaseClient

logger = logging.getLogger(__name__)

# --- Tuning hyperparameters ---
LEARNING_RATE = 0.3  # max ~15% width change per cycle
MIN_HALF_WIDTH = 0.005  # never fully zero a signal
MIN_SAMPLE_SIZE = 200  # player-games in rolling window
MIN_DAYS = 5  # days in rolling window
MIN_FIRE_RATE = 0.05  # 5% — signals that rarely fire lack data
ROLLBACK_THRESHOLD = -1.0  # predicted_fp_added_value below this triggers rollback
ROLLBACK_CONSECUTIVE_DAYS = 3  # consecutive days below threshold

_MAX_HISTORY = 30  # rolling history entries

# --- Default clamp ranges and max allowed half-widths ---
# Each signal: (default_min, default_max, max_half_width)
SIGNAL_DEFAULTS: dict[str, dict[str, float]] = {
    "matchup_multiplier": {"clamp_min": 0.97, "clamp_max": 1.03, "max_half": 0.15},
    "vegas_multiplier": {"clamp_min": 0.90, "clamp_max": 1.10, "max_half": 0.20},
    "venue_multiplier": {"clamp_min": 0.92, "clamp_max": 1.08, "max_half": 0.15},
    "cat_trend_multiplier": {"clamp_min": 0.98, "clamp_max": 1.02, "max_half": 0.08},
    "usage_delta_multiplier": {"clamp_min": 0.96, "clamp_max": 1.08, "max_half": 0.15},
    "shooting_luck_multiplier": {"clamp_min": 0.97, "clamp_max": 1.03, "max_half": 0.08},
    "game_env_cap": {"clamp_min": 0.65, "clamp_max": 1.25, "max_half": 0.40},
    "line_movement_multiplier": {"clamp_min": 0.98, "clamp_max": 1.03, "max_half": 0.05},
}


def compute_quality_score(
    residual_correlation: float | None,
    hit_rate: float | None,
) -> float:
    """Compute signal quality from accuracy metrics.

    Returns a value roughly in [-1, 1] where positive means the signal
    is directionally correct and negative means it's harmful.
    """
    corr = residual_correlation if residual_correlation is not None else 0.0
    hr = hit_rate if hit_rate is not None else 0.5
    return 0.6 * corr + 0.4 * (hr - 0.5)


def compute_tuned_weights(
    rolling_accuracy: dict[str, Any],
    current_weights: dict[str, dict[str, float]] | None = None,
) -> dict[str, dict[str, float]] | None:
    """Compute new clamp widths from rolling accuracy telemetry.

    Args:
        rolling_accuracy: Output of ``compute_rolling_accuracy()`` with
            ``signal_accuracy``, ``sample_size``, and ``days`` fields.
        current_weights: Current tuned weights (or None to start from defaults).

    Returns:
        Updated weights dict keyed by signal name with ``clamp_min``,
        ``clamp_max``, and ``quality_score`` per signal.  Returns None
        if insufficient data for tuning.
    """
    sample_size = rolling_accuracy.get("sample_size", 0)
    days = rolling_accuracy.get("days", 0)
    signal_accuracy = rolling_accuracy.get("signal_accuracy", {})

    if sample_size < MIN_SAMPLE_SIZE or days < MIN_DAYS:
        logger.info(
            "signal tuning: insufficient data (samples=%d, days=%d), skipping",
            sample_size,
            days,
        )
        return None

    base = current_weights if current_weights else {name: {"clamp_min": d["clamp_min"], "clamp_max": d["clamp_max"]} for name, d in SIGNAL_DEFAULTS.items()}

    result: dict[str, dict[str, float]] = {}

    for signal_name, defaults in SIGNAL_DEFAULTS.items():
        sig_data = signal_accuracy.get(signal_name, {})
        fire_rate = sig_data.get("fire_rate", 0.0)

        # Use current clamps as starting point
        current = base.get(signal_name, {})
        cur_min = current.get("clamp_min", defaults["clamp_min"])
        cur_max = current.get("clamp_max", defaults["clamp_max"])
        max_half = defaults["max_half"]

        if fire_rate is None or fire_rate < MIN_FIRE_RATE:
            # Signal fires too rarely — keep current, don't tune
            result[signal_name] = {
                "clamp_min": cur_min,
                "clamp_max": cur_max,
                "quality_score": 0.0,
            }
            continue

        residual_corr = sig_data.get("residual_correlation")
        hit_rate = sig_data.get("hit_rate")
        quality = compute_quality_score(residual_corr, hit_rate)

        # Nudge half-width proportionally to quality
        cur_half = (cur_max - cur_min) / 2.0
        target_ratio = max(0.5, min(1.5, 1.0 + quality * LEARNING_RATE))
        new_half = cur_half * target_ratio
        new_half = max(MIN_HALF_WIDTH, min(max_half, new_half))

        result[signal_name] = {
            "clamp_min": round(1.0 - new_half, 4),
            "clamp_max": round(1.0 + new_half, 4),
            "quality_score": round(quality, 4),
        }

    return result


def should_rollback(
    daily_results: list[dict[str, Any]],
    consecutive_days: int = ROLLBACK_CONSECUTIVE_DAYS,
) -> bool:
    """Check if tuned weights should be rolled back to defaults.

    Triggers when ``predicted_fp_added_value`` is below the threshold
    for N consecutive recent days.
    """
    if len(daily_results) < consecutive_days:
        return False

    recent = daily_results[-consecutive_days:]
    return all((d.get("overall", {}).get("predicted_fp_added_value") or 0.0) < ROLLBACK_THRESHOLD for d in recent)


def load_signal_weights(
    db: BaseClient,
) -> dict[str, dict[str, float]] | None:
    """Load current tuned signal weights from ``system/signal_weights``.

    Returns a dict keyed by signal name with ``clamp_min`` and
    ``clamp_max`` values, or ``None`` if no tuned weights exist.
    """
    doc = db.collection("system").document("signal_weights").get()
    if not doc.exists:  # type: ignore[union-attr]
        return None
    data: dict[str, Any] = doc.to_dict() or {}  # type: ignore[union-attr]
    current: dict[str, Any] | None = data.get("current")
    if not current:
        return None
    return {k: v for k, v in current.items() if isinstance(v, dict)}


def store_signal_weights(
    db: BaseClient,
    weights: dict[str, dict[str, float]],
    source_sample_size: int = 0,
) -> None:
    """Write updated signal weights with history rotation."""
    now_iso = datetime.now(timezone.utc).isoformat()

    ref = db.collection("system").document("signal_weights")
    existing = ref.get()
    history: list[dict[str, Any]] = []
    if existing.exists:  # type: ignore[union-attr]
        data = existing.to_dict() or {}  # type: ignore[union-attr]
        history = data.get("history", [])

    history.append(
        {
            "timestamp": now_iso,
            "coefficients": weights,
        }
    )

    if len(history) > _MAX_HISTORY:
        history = history[-_MAX_HISTORY:]

    # Store defaults for rollback reference
    defaults = {name: {"clamp_min": d["clamp_min"], "clamp_max": d["clamp_max"]} for name, d in SIGNAL_DEFAULTS.items()}

    ref.set(
        {
            "updated_at": now_iso,
            "source_sample_size": source_sample_size,
            "current": weights,
            "defaults": defaults,
            "history": history,
        }
    )
    logger.info("signal weights stored: %d signals tuned", len(weights))
