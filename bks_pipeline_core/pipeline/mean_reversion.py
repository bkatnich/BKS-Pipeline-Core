"""Post-disappointment mean-reversion signal (4A).

Identifies players whose most recent game was significantly below their
rolling average and estimates the likelihood of a bounce-back. The signal
is stronger when the underperformance was caused by low minutes (blowout /
coach's decision) rather than poor per-minute production.

All functions are pure — no I/O or side effects.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)

# Mode-aware multiplier thresholds.
# GPP amplifies the signal: post-disappointment players will be low-owned
# in tournaments, so the edge compounds (better projection + less competition).
# Cash barely uses it because cash rewards safety, not contrarian leverage.
_MODE_SENSITIVITY: dict[str, float] = {
    "gpp": 1.5,
    "balanced": 1.0,
    "cash": 0.4,
}

# Signal strength → base multiplier boost (before mode scaling)
_BOOST_STRONG = 0.06  # signal > 0.7
_BOOST_MODERATE = 0.04  # signal 0.4–0.7
# weak signal (< 0.4) → no boost

# Underperformance thresholds
_DISAPPOINTMENT_FLOOR = 0.20  # 20% below avg to activate
_DISAPPOINTMENT_CAP = 0.50  # signal maxes out at 50% below avg
_MINUTES_CAUSE_BONUS = 1.3  # signal multiplied when underperformance was minutes-driven
_MINUTES_CAUSE_THRESHOLD = 0.80  # per-minute production within 80% of avg → minutes-driven
# When the bad game was NOT minutes-driven (i.e. poor per-minute production),
# the player underperformed despite playing normal time — harder to predict
# reverting. Discount the signal to reduce false positives.
_PRODUCTION_CAUSE_DISCOUNT = 0.60  # multiply signal when production was the cause


def compute_mean_reversion_signal(
    player: dict[str, Any],
    mode: str = "balanced",
) -> dict[str, Any]:
    """Compute the post-disappointment mean-reversion signal for a player.

    Args:
        player: Player dict from Firestore, expected to contain:
            - avg_fantasy_score (float): rolling average FP
            - avg_minutes (float): rolling average minutes
            - recent_game_scores (list[float]): last N raw FP per game
            - recent_game_minutes (list[float]): last N minutes per game
        mode: Scoring mode — "balanced", "cash", or "gpp".

    Returns:
        Dict with:
            - mean_reversion_signal (float): 0.0–1.0 signal strength
            - is_minutes_driven (bool): was the bad game caused by low minutes?
            - disappointment_pct (float | None): how far below avg (0.0–1.0)
            - mean_reversion_multiplier (float): the multiplier for opportunity scoring
    """
    neutral = {
        "mean_reversion_signal": 0.0,
        "is_minutes_driven": False,
        "disappointment_pct": None,
        "mean_reversion_multiplier": 1.0,
    }

    avg_fs = player.get("avg_fantasy_score")
    avg_min = player.get("avg_minutes")
    recent_scores = player.get("recent_game_scores")
    recent_minutes = player.get("recent_game_minutes")

    # Guard: need all inputs with valid values
    if (
        not avg_fs
        or avg_fs <= 0
        or not avg_min
        or avg_min <= 0
        or not recent_scores
        or not recent_minutes
        or len(recent_scores) == 0
        or len(recent_minutes) == 0
    ):
        return neutral

    last_score = recent_scores[-1]
    last_minutes = recent_minutes[-1]

    if last_score is None or last_minutes is None:
        return neutral

    # 2-game lookback: if the player has 2+ recent games, average them for a
    # stronger signal. But skip reversion if the most recent game shows recovery
    # (i.e., last game > second-to-last game) — already bouncing back.
    if len(recent_scores) >= 2 and recent_scores[-2] is not None:
        if last_score > recent_scores[-2]:
            # Already recovering — no reversion boost needed
            return neutral
        last_score = (last_score + recent_scores[-2]) / 2
        if len(recent_minutes) >= 2 and recent_minutes[-2] is not None:
            last_minutes = (last_minutes + recent_minutes[-2]) / 2

    # Disappointment magnitude: how far below average was the last game?
    if last_score >= avg_fs:
        return neutral  # no disappointment — player met or exceeded average

    disappointment_pct = (avg_fs - last_score) / avg_fs

    if disappointment_pct < _DISAPPOINTMENT_FLOOR:
        return neutral  # not significant enough

    # Minutes decomposition: was the bad game caused by low minutes?
    is_minutes_driven = False
    if last_minutes > 0 and avg_min > 0:
        last_per_min = last_score / last_minutes
        avg_per_min = avg_fs / avg_min
        if avg_per_min > 0:
            is_minutes_driven = last_per_min >= avg_per_min * _MINUTES_CAUSE_THRESHOLD

    # Signal strength: linear scale from floor to cap, clamped [0, 1]
    raw_signal = (disappointment_pct - _DISAPPOINTMENT_FLOOR) / (_DISAPPOINTMENT_CAP - _DISAPPOINTMENT_FLOOR)
    signal = max(0.0, min(1.0, raw_signal))

    # Minutes-cause bonus: strengthen signal when it's a minutes issue (blowout/rest).
    # Production-cause discount: weaken signal when the player underperformed
    # despite normal minutes — poor per-minute production is less predictably
    # reverting than a context-driven (blowout/lineup) minutes reduction.
    if is_minutes_driven:
        signal = min(1.0, signal * _MINUTES_CAUSE_BONUS)
    else:
        signal *= _PRODUCTION_CAUSE_DISCOUNT

    # Convert signal to multiplier (mode-aware)
    sensitivity = _MODE_SENSITIVITY.get(mode, 1.0)
    if signal > 0.7:
        base_boost = _BOOST_STRONG
    elif signal >= 0.4:
        base_boost = _BOOST_MODERATE
    else:
        base_boost = 0.0

    multiplier = 1.0 + (base_boost * sensitivity)

    logger.debug(
        "Mean reversion: signal=%.3f, mult=%.3f, minutes_driven=%s, disappointment=%.1f%% (mode=%s)",
        signal,
        multiplier,
        is_minutes_driven,
        disappointment_pct * 100,
        mode,
    )

    return {
        "mean_reversion_signal": round(signal, 4),
        "is_minutes_driven": is_minutes_driven,
        "disappointment_pct": round(disappointment_pct, 4),
        "mean_reversion_multiplier": round(multiplier, 3),
    }
