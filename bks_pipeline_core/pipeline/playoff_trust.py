"""Per-player within-series projection accuracy tracker and series stats store.

Two collections written by grade_playoff_trust() after each playoff game:

1. ``playoff_player_trust/{player_id}`` — MAE-derived trust score that
   replaces the static ramp table in playoff_trend_trust().

   Trust score semantics:
     0 games graded → 0.0  (same as game 0 in old ramp — no data yet)
     MAE < 15% of avg_fs → 1.0  (projecting accurately)
     MAE 15–25%          → 0.75
     MAE 25–40%          → 0.50
     MAE > 40%           → 0.25

2. ``series_player_stats/{player_id}`` — per-game actuals scoped to the
   current series (rest-game-filtered aggregates + full game log). Used by
   playoff_cold_start_anchor() and the established anchor in opportunities.py
   to replace heuristics with real series data.

Both docs reset when a player's series_id changes (new playoff round).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from config import PLAYOFF_REST_GAME_MINUTES_THRESHOLD

logger = logging.getLogger(__name__)

# MAE-as-fraction-of-baseline → trust score mapping (upper bounds)
_TRUST_BREAKPOINTS: list[tuple[float, float]] = [
    (0.15, 1.00),
    (0.25, 0.75),
    (0.40, 0.50),
    (float("inf"), 0.25),
]

_COLLECTION = "playoff_player_trust"
_SERIES_STATS_COLLECTION = "series_player_stats"


def _mae_to_trust(mae_pct: float) -> float:
    """Convert MAE-as-fraction-of-baseline to a [0.25, 1.0] trust score."""
    for threshold, score in _TRUST_BREAKPOINTS:
        if mae_pct < threshold:
            return score
    return 0.25


def _compute_series_aggregates(game_log: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute rest-filtered and all-games aggregates from a series game log.

    Rest games (actual_minutes < PLAYOFF_REST_GAME_MINUTES_THRESHOLD) are
    excluded from averages so they don't drag down projections for stars who
    sat out a blowout. The raw all-games avg is also stored for diagnostics.
    """
    active = [g for g in game_log if not g.get("is_rest_game")]
    all_fps = [g["actual_fp_dk"] for g in game_log]
    active_fps = [g["actual_fp_dk"] for g in active]
    active_mins = [g["actual_minutes"] for g in active]
    active_fg3m = [g["actual_fg3m"] for g in active]
    active_fg3a = [g["actual_fg3a"] for g in active]
    n = len(active) or 1
    total_fg3a = sum(active_fg3a)
    return {
        "series_games": len(game_log),
        "series_fp_avg": round(sum(active_fps) / n, 2) if active_fps else 0.0,
        "series_fp_avg_all": round(sum(all_fps) / len(all_fps), 2) if all_fps else 0.0,
        "series_minutes_avg": round(sum(active_mins) / n, 2) if active_mins else 0.0,
        "series_fg3m_per_game": round(sum(active_fg3m) / n, 3) if active_fg3m else 0.0,
        "series_fg3a_per_game": round(sum(active_fg3a) / n, 3) if active_fg3a else 0.0,
        "series_fg3_pct": round(sum(active_fg3m) / total_fg3a, 3) if total_fg3a > 0 else None,
    }


def grade_playoff_trust(
    db: Any,
    date: str,
    predictions: dict[str, dict[str, Any]],
    actuals: dict[str, dict[str, Any]],
    active_series: list[dict[str, Any]],
) -> int:
    """Grade predicted_fp vs actual_fp for all active-series players and update trust docs.

    Called from compute_accuracy after daily actuals are confirmed. Idempotent
    per date — if a player's trust doc already has this date in graded_dates,
    the entry is skipped.

    Args:
        db:             Firestore client.
        date:           Game date just graded (YYYY-MM-DD).
        predictions:    Snapshot predictions dict keyed by player_id str.
                        Each entry must have ``predicted_fp`` and ``avg_fantasy_score``.
        actuals:        Actuals dict keyed by player_id str.
                        Each entry must have ``actual_fp_dk`` and ``dnp``.
        active_series:  List of series docs with status in ("scheduled", "active").
                        Used to build the active-team → series_id mapping.

    Returns:
        Number of player trust docs updated.
    """
    # Build team → series_id lookup from active series
    team_to_series: dict[str, str] = {}
    for s in active_series:
        sid = s.get("series_id")
        if not sid:
            continue
        for key in ("higher_seed_team", "lower_seed_team"):
            abbr = s.get(key)
            if abbr:
                team_to_series[abbr] = sid

    if not team_to_series:
        logger.info("grade_playoff_trust: no active series teams — skipping")
        return 0

    # Collect graded player rows: must have prediction, actual, and be playing
    now_iso = datetime.now(timezone.utc).isoformat()
    updated = 0
    batch = db.batch()
    batch_count = 0

    for pid_str, pred in predictions.items():
        act = actuals.get(pid_str)
        if act is None or act.get("dnp", False):
            continue

        actual_fp = act.get("actual_fp_dk")
        predicted_fp = pred.get("predicted_fp")
        avg_fs = pred.get("avg_fantasy_score")

        if actual_fp is None or predicted_fp is None or not avg_fs or avg_fs <= 0:
            continue

        # Only grade players on active-series teams
        team = pred.get("team") or act.get("team")
        series_id = team_to_series.get(team) if team else None
        if not series_id:
            continue

        abs_error = abs(float(predicted_fp) - float(actual_fp))
        error_pct = abs_error / float(avg_fs)

        trust_ref = db.collection(_COLLECTION).document(pid_str)
        series_ref = db.collection(_SERIES_STATS_COLLECTION).document(pid_str)
        trust_doc = trust_ref.get()
        series_doc = series_ref.get()
        existing: dict[str, Any] = trust_doc.to_dict() if trust_doc.exists else {}
        existing_stats: dict[str, Any] = series_doc.to_dict() if series_doc.exists else {}

        # Reset both docs if player moved to a new series (next round)
        if existing.get("series_id") and existing["series_id"] != series_id:
            logger.info(
                "grade_playoff_trust: player %s moved to new series %s → resetting trust + series stats",
                pid_str,
                series_id,
            )
            existing = {}
            existing_stats = {}

        # Idempotency: skip if this date already graded
        graded_dates: list[str] = existing.get("graded_dates", [])
        if date in graded_dates:
            continue

        # --- Trust doc update ---
        error_history: list[float] = existing.get("error_history", [])
        error_history.append(round(error_pct, 4))
        games_graded = len(error_history)
        mae_pct = sum(error_history) / games_graded
        trust_score = _mae_to_trust(mae_pct)
        graded_dates.append(date)

        batch.set(
            trust_ref,
            {
                "player_id": pid_str,
                "series_id": series_id,
                "games_graded": games_graded,
                "error_history": error_history,
                "mae_pct": round(mae_pct, 4),
                "trust_score": round(trust_score, 4),
                "graded_dates": graded_dates,
                "last_graded_at": now_iso,
            },
        )
        batch_count += 1
        updated += 1

        # --- Series stats doc update ---
        is_rest = float(act.get("actual_minutes") or 0.0) < PLAYOFF_REST_GAME_MINUTES_THRESHOLD
        new_game_entry: dict[str, Any] = {
            "date": date,
            "actual_fp_dk": round(float(actual_fp), 2),
            "actual_minutes": round(float(act.get("actual_minutes") or 0.0), 1),
            "actual_pts": int(act.get("actual_pts") or 0),
            "actual_reb": int(act.get("actual_reb") or 0),
            "actual_ast": int(act.get("actual_ast") or 0),
            "actual_fg3m": int(act.get("actual_fg3m") or 0),
            "actual_fg3a": int(act.get("actual_fg3a") or 0),
            "is_rest_game": is_rest,
        }
        game_log: list[dict[str, Any]] = existing_stats.get("game_log", [])
        game_log = [g for g in game_log if g.get("date") != date]  # idempotency guard
        game_log.append(new_game_entry)
        game_log.sort(key=lambda g: g.get("date", ""))

        batch.set(
            series_ref,
            {
                "player_id": pid_str,
                "series_id": series_id,
                "opponent_abbr": act.get("opponent_abbr"),
                "game_log": game_log,
                **_compute_series_aggregates(game_log),
                "last_updated_at": now_iso,
            },
        )
        batch_count += 1

        if batch_count >= 400:
            batch.commit()
            batch = db.batch()
            batch_count = 0

    if batch_count > 0:
        batch.commit()

    logger.info("grade_playoff_trust: updated %d player trust docs for %s", updated, date)
    return updated


def load_playoff_trust(db: Any) -> dict[str, float]:
    """Load all playoff trust scores into a player_id → trust_score dict.

    Returns an empty dict when not in playoffs or collection is empty.
    O(n) Firestore reads — called once per prediction context load.
    """
    docs = db.collection(_COLLECTION).stream()
    result: dict[str, float] = {}
    for doc in docs:
        d = doc.to_dict() or {}
        pid = d.get("player_id") or doc.id
        score = d.get("trust_score")
        if score is not None:
            result[str(pid)] = float(score)
    return result


def load_series_stats(db: Any) -> dict[str, dict[str, Any]]:
    """Load all series player stats into a player_id → stats dict.

    Returns an empty dict when not in playoffs or the collection is empty.
    O(n) Firestore reads — called once per prediction context load alongside
    load_playoff_trust().
    """
    docs = db.collection(_SERIES_STATS_COLLECTION).stream()
    result: dict[str, dict[str, Any]] = {}
    for doc in docs:
        d = doc.to_dict() or {}
        pid = d.get("player_id") or doc.id
        result[str(pid)] = d
    return result
