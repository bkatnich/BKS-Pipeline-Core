"""Pure computation functions for prediction-vs-actual accuracy analysis.

All functions are stateless and perform no I/O. They accept pre-loaded
prediction/actual dicts and return structured accuracy metrics.
"""

import math
from typing import Any

from bks_pipeline_core.pipeline.accuracy_report import DISABLED_SIGNAL_FIELDS as _ALWAYS_DISABLED_SIGNALS
from bks_pipeline_core.pipeline.accuracy_report import PLAYOFF_DISABLED_SIGNAL_FIELDS as _PLAYOFF_DISABLED_SIGNALS

__all__ = [
    "SIGNAL_MULTIPLIERS",
    "PLATFORMS",
    "join_predictions_actuals",
    "compute_match_coverage",
    "compute_overall_accuracy",
    "compute_signal_accuracy",
    "compute_floor_ceiling_calibration",
    "compute_stat_accuracy",
    "compute_prop_brier_scores",
    "compute_game_totals_accuracy",
    "compute_daily_accuracy",
    "compute_rolling_accuracy",
    "generate_insights",
    "_pearson_r",
]

# Multiplier fields present in the opportunity results dict.
# Each is decomposed independently for calibration analysis.
SIGNAL_MULTIPLIERS: list[str] = [
    "matchup_multiplier",
    "vegas_multiplier",
    "stacking_multiplier",
    "mean_reversion_multiplier",
    "minutes_env_multiplier",
    "b2b_penalty",
    "game_env_cap",
    "venue_multiplier",
    "role_change_multiplier",
    "health_factor",
    "pace_multiplier",
    "cat_trend_multiplier",
    "usage_delta_multiplier",
    "shooting_luck_multiplier",
    "playoff_rotation_multiplier",
    "elimination_game_multiplier",
]

# Supported scoring platforms. Each platform requires namespaced fields
# in snapshots (predicted_fp_{p}, avg_fantasy_score_{p}, fp_floor_{p},
# fp_ceiling_{p}) and actuals (actual_fp_{p}).
PLATFORMS: tuple[str, ...] = ("dk", "fd")

_PLATFORM_DISPLAY: dict[str, str] = {
    "dk": "DraftKings",
    "fd": "FanDuel",
}

def _pearson_r(xs: list[float], ys: list[float]) -> float | None:
    """Pearson correlation coefficient. Returns None if fewer than 3 pairs or zero variance."""
    n = len(xs)
    if n < 3 or n != len(ys):
        return None
    x_mean = sum(xs) / n
    y_mean = sum(ys) / n
    cov = sum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys))
    var_x = sum((x - x_mean) ** 2 for x in xs)
    var_y = sum((y - y_mean) ** 2 for y in ys)
    denom = math.sqrt(var_x * var_y)
    if denom == 0:
        return None
    return cov / denom


def _mae(predicted: list[float], actual: list[float]) -> float:
    """Mean absolute error."""
    if not predicted:
        return 0.0
    return sum(abs(p - a) for p, a in zip(predicted, actual)) / len(predicted)


def _derive_multiplier(pred: dict[str, Any], field: str) -> float:
    """Return a stored multiplier field, defaulting to 1.0 if absent."""
    val = pred.get(field)
    return float(val) if val is not None else 1.0


def join_predictions_actuals(
    predictions: dict[str, dict[str, Any]],
    actuals: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Join prediction and actual dicts by player ID.

    Returns a list of merged dicts containing both predicted and actual fields.
    Includes players who played (not DNP) and have actuals for at least one
    platform. Per-platform compute functions filter further for their platform.
    """
    joined: list[dict[str, Any]] = []
    for key, pred in predictions.items():
        # Prediction keys may be composite "pid_gameid" (doubleheader days) or plain "pid".
        # Always read the authoritative player id from the record itself.
        pid = str(pred.get("id", "")) or key
        act = actuals.get(pid)
        if act is None:
            continue
        if act.get("dnp", False):
            continue
        # Include if any platform actual is present
        has_any = any(act.get(f"actual_fp_{p}") is not None for p in PLATFORMS)
        if not has_any:
            continue
        joined.append({**pred, **{f"actual_{k}": v for k, v in act.items()}, "player_id": pid})
    return joined


def compute_match_coverage(
    predictions: dict[str, dict[str, Any]],
    actuals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute a breakdown of why box-score players were not matched to predictions.

    Returns a coverage dict with total counts and per-reason lists of player IDs,
    suitable for logging and storing in the accuracy doc.

    Reasons:
      matched          — player joined successfully
      dnp              — player had a box score but was marked DNP
      no_fp_actual     — box score exists but no platform FP recorded
      not_snapshotted  — player in actuals but absent from predictions entirely
    """
    matched: list[str] = []
    dnp: list[str] = []
    no_fp_actual: list[str] = []
    not_snapshotted: list[str] = []

    for pid, act in actuals.items():
        if pid not in predictions:
            not_snapshotted.append(pid)
            continue
        if act.get("dnp", False):
            dnp.append(pid)
            continue
        has_any = any(act.get(f"actual_fp_{p}") is not None for p in PLATFORMS)
        if not has_any:
            no_fp_actual.append(pid)
            continue
        matched.append(pid)

    total_actuals = len(actuals)
    coverage_rate = round(len(matched) / total_actuals, 4) if total_actuals else None

    return {
        "total_box_scores": total_actuals,
        "matched": len(matched),
        "coverage_rate": coverage_rate,
        "not_snapshotted": len(not_snapshotted),
        "dnp": len(dnp),
        "no_fp_actual": len(no_fp_actual),
        "not_snapshotted_ids": not_snapshotted,
        "dnp_ids": dnp,
        "no_fp_actual_ids": no_fp_actual,
    }


def compute_overall_accuracy(joined: list[dict[str, Any]], platform: str = "dk") -> dict[str, Any]:
    """Compute overall prediction accuracy metrics for a given platform.

    Returns correlation and MAE for both opportunity_score and baseline.
    """
    # Filter to rows that have this platform's actual and a scored opportunity_score.
    # Stub players (no trend data, DFS floor gate, playoff eligibility gate) have
    # opportunity_score=None and must be excluded from correlation/MAE computation.
    rows = [
        r for r in joined
        if r.get(f"actual_actual_fp_{platform}") is not None
        and r.get("opportunity_score") is not None
    ]

    if not rows:
        return {
            "sample_size": 0,
            "score_vs_actual_r": None,
            "baseline_vs_actual_r": None,
            "score_mae": 0.0,
            "baseline_mae": 0.0,
            "added_value": 0.0,
            "predicted_fp_vs_actual_r": None,
            "predicted_fp_mae": 0.0,
            "predicted_fp_added_value": 0.0,
        }

    scores = [r["opportunity_score"] for r in rows]
    baselines = [float(r.get(f"avg_fantasy_score_{platform}") or r.get("avg_fantasy_score") or 0) for r in rows]
    actuals = [float(r[f"actual_actual_fp_{platform}"]) for r in rows]
    predicted_fps = [
        float(r.get(f"predicted_fp_{platform}") or r.get("predicted_fp") or r.get(f"avg_fantasy_score_{platform}") or r.get("avg_fantasy_score") or 0)
        for r in rows
    ]

    score_r = _pearson_r(scores, actuals)
    baseline_r = _pearson_r(baselines, actuals)
    score_err = _mae(scores, actuals)
    baseline_err = _mae(baselines, actuals)

    predicted_fp_r = _pearson_r(predicted_fps, actuals)
    predicted_fp_err = _mae(predicted_fps, actuals)

    return {
        "sample_size": len(rows),
        "score_vs_actual_r": round(score_r, 4) if score_r is not None else None,
        "baseline_vs_actual_r": round(baseline_r, 4) if baseline_r is not None else None,
        "score_mae": round(score_err, 2),
        "baseline_mae": round(baseline_err, 2),
        "added_value": round(baseline_err - score_err, 2),
        "predicted_fp_vs_actual_r": round(predicted_fp_r, 4) if predicted_fp_r is not None else None,
        "predicted_fp_mae": round(predicted_fp_err, 2),
        "predicted_fp_added_value": round(baseline_err - predicted_fp_err, 2),
    }


def compute_signal_accuracy(
    joined: list[dict[str, Any]],
    platform: str = "dk",
    disabled_signals: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute per-signal calibration metrics for a given platform.

    For each multiplier signal, measures:
    - residual_correlation: Pearson r between (multiplier - 1.0) and (actual - baseline)
    - hit_rate: when signal boosts (>1.0), fraction of players who beat baseline
    - calibration_ratio: mean residual for boosted / mean signal deviation for boosted
    - fire_rate: fraction of players where signal deviates from 1.0

    Signals in ``disabled_signals`` are skipped entirely — they always return 1.0
    so their deviations are zero-variance and produce null correlations.
    """
    rows = [
        r for r in joined
        if r.get(f"actual_actual_fp_{platform}") is not None
        and r.get("opportunity_score") is not None
    ]
    all_signals = [s for s in SIGNAL_MULTIPLIERS if not (disabled_signals and s in disabled_signals)]

    result: dict[str, dict[str, Any]] = {}
    for sig in all_signals:
        deviations: list[float] = []
        residuals: list[float] = []
        boost_residuals: list[float] = []
        boost_deviations: list[float] = []
        boost_hits = 0
        boost_count = 0
        penalty_residuals: list[float] = []
        penalty_deviations: list[float] = []
        penalty_hits = 0
        penalty_count = 0
        fire_count = 0

        for r in rows:
            mult = _derive_multiplier(r, sig)
            baseline = float(r.get(f"avg_fantasy_score_{platform}") or r.get("avg_fantasy_score") or 0)
            actual_fp = float(r[f"actual_actual_fp_{platform}"])
            deviation = mult - 1.0
            residual = actual_fp - baseline

            deviations.append(deviation)
            residuals.append(residual)

            if abs(deviation) > 1e-6:
                fire_count += 1

            if deviation > 1e-6:
                boost_count += 1
                boost_residuals.append(residual)
                boost_deviations.append(deviation)
                if residual > 0:
                    boost_hits += 1
            elif deviation < -1e-6:
                penalty_count += 1
                penalty_residuals.append(residual)
                penalty_deviations.append(deviation)
                if residual < 0:
                    penalty_hits += 1

        corr = _pearson_r(deviations, residuals)
        hit_rate = (boost_hits / boost_count) if boost_count > 0 else None
        penalty_hit_rate = (penalty_hits / penalty_count) if penalty_count > 0 else None
        fire_rate = fire_count / len(rows) if rows else 0.0

        cal_ratio: float | None = None
        if boost_deviations and sum(boost_deviations) > 0:
            mean_res = sum(boost_residuals) / len(boost_residuals)
            mean_dev = sum(boost_deviations) / len(boost_deviations)
            if mean_dev > 0:
                cal_ratio = round(mean_res / mean_dev, 3)

        penalty_cal_ratio: float | None = None
        if penalty_deviations and sum(abs(d) for d in penalty_deviations) > 0:
            mean_res = sum(penalty_residuals) / len(penalty_residuals)
            mean_dev = sum(penalty_deviations) / len(penalty_deviations)
            if mean_dev < 0:
                penalty_cal_ratio = round(mean_res / mean_dev, 3)

        result[sig] = {
            "residual_correlation": round(corr, 4) if corr is not None else None,
            "hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
            "calibration_ratio": cal_ratio,
            "fire_rate": round(fire_rate, 3),
            "boost_count": boost_count,
            "penalty_count": penalty_count,
            "penalty_hit_rate": round(penalty_hit_rate, 3) if penalty_hit_rate is not None else None,
            "penalty_cal_ratio": penalty_cal_ratio,
            "total_count": len(rows),
        }

    return result


def compute_floor_ceiling_calibration(joined: list[dict[str, Any]], platform: str = "dk") -> dict[str, Any]:
    """Compute floor/ceiling bracket accuracy for a given platform.

    Checks what fraction of actual results fall within the predicted
    [fp_floor, fp_ceiling] range.
    """
    rows = [r for r in joined if r.get(f"actual_actual_fp_{platform}") is not None]

    if not rows:
        return {
            "below_floor_rate": 0.0,
            "above_ceiling_rate": 0.0,
            "within_range_rate": 0.0,
            "floor_mae": 0.0,
            "ceiling_mae": 0.0,
            "sample_size": 0,
        }

    below = 0
    above = 0
    within = 0
    floor_errors: list[float] = []
    ceiling_errors: list[float] = []
    counted = 0

    for r in rows:
        fp_floor = r.get(f"fp_floor_{platform}") or r.get("fp_floor")
        fp_ceiling = r.get(f"fp_ceiling_{platform}") or r.get("fp_ceiling")
        if fp_floor is None or fp_ceiling is None:
            continue
        actual_fp = float(r[f"actual_actual_fp_{platform}"])
        fp_floor = float(fp_floor)
        fp_ceiling = float(fp_ceiling)
        counted += 1

        floor_errors.append(abs(actual_fp - fp_floor))
        ceiling_errors.append(abs(actual_fp - fp_ceiling))

        if actual_fp < fp_floor:
            below += 1
        elif actual_fp > fp_ceiling:
            above += 1
        else:
            within += 1

    if counted == 0:
        return {
            "below_floor_rate": 0.0,
            "above_ceiling_rate": 0.0,
            "within_range_rate": 0.0,
            "floor_mae": 0.0,
            "ceiling_mae": 0.0,
            "sample_size": 0,
        }

    return {
        "below_floor_rate": round(below / counted, 3),
        "above_ceiling_rate": round(above / counted, 3),
        "within_range_rate": round(within / counted, 3),
        "floor_mae": round(sum(floor_errors) / len(floor_errors), 2),
        "ceiling_mae": round(sum(ceiling_errors) / len(ceiling_errors), 2),
        "sample_size": counted,
    }


_STAT_FIELDS: list[tuple[str, str, str]] = [
    # (predicted_field, actual_field, display_name)
    ("projected_pts", "actual_actual_pts", "PTS"),
    ("projected_reb", "actual_actual_reb", "REB"),
    ("projected_ast", "actual_actual_ast", "AST"),
    ("projected_stl", "actual_actual_stl", "STL"),
    ("projected_blk", "actual_actual_blk", "BLK"),
    ("projected_min", "actual_actual_minutes", "MIN"),
]


def compute_stat_accuracy(joined: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute per-stat prediction accuracy (MAE, bias, Pearson r).

    Field mapping is taken from the active SportConfig's stat_fields when set,
    falling back to _STAT_FIELDS (basketball defaults) otherwise.

    Returns a dict keyed by display_name with mae, bias, r, and sample_size.
    """
    from bks_pipeline_core.sport_config import get_active_config
    try:
        cfg_fields = get_active_config().stat_fields
    except RuntimeError:
        cfg_fields = None
    fields = cfg_fields if cfg_fields is not None else _STAT_FIELDS

    result: dict[str, dict[str, Any]] = {}
    for pred_field, act_field, label in fields:
        pairs: list[tuple[float, float]] = []
        for r in joined:
            pred_val = r.get(pred_field)
            act_val = r.get(act_field)
            if pred_val is None or act_val is None:
                continue
            pairs.append((float(pred_val), float(act_val)))

        if not pairs:
            result[label] = {"mae": None, "bias": None, "r": None, "sample_size": 0}
            continue

        preds = [p for p, _ in pairs]
        acts = [a for _, a in pairs]
        mae = round(_mae(preds, acts), 2)
        bias = round(sum(p - a for p, a in pairs) / len(pairs), 2)  # positive = over-projecting
        r = _pearson_r(preds, acts)

        result[label] = {
            "mae": mae,
            "bias": bias,
            "r": round(r, 4) if r is not None else None,
            "sample_size": len(pairs),
        }
    return result


def compute_prop_brier_scores(
    prop_actuals: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Compute Brier scores per stat type from graded prop actuals.

    Args:
        prop_actuals: List of prop_actuals documents, each with a ``lines``
            dict containing graded prop entries.

    Returns:
        Dict keyed by stat type with ``brier``, ``samples``, and
        ``uncalibrated_brier`` (if calibrated probs differ from raw).
    """
    from bks_pipeline_core.pipeline.platt import brier_score

    # Collect (model_prob, calibrated_prob, outcome) per stat
    stat_data: dict[str, list[tuple[float, float, bool]]] = {}
    for doc in prop_actuals:
        lines = doc.get("lines", {})
        for _key, line in lines.items():
            stat = line.get("stat")
            model_prob = line.get("model_prob_over")
            cal_prob = line.get("calibrated_prob_over")
            over_hit = line.get("over_hit")
            if stat is None or model_prob is None or over_hit is None:
                continue
            if stat not in stat_data:
                stat_data[stat] = []
            stat_data[stat].append(
                (
                    float(model_prob),
                    float(cal_prob) if cal_prob is not None else float(model_prob),
                    bool(over_hit),
                )
            )

    result: dict[str, dict[str, Any]] = {}
    for stat, triples in stat_data.items():
        raw_probs = [t[0] for t in triples]
        cal_probs = [t[1] for t in triples]
        outcomes = [t[2] for t in triples]

        cal_brier = brier_score(cal_probs, outcomes)
        raw_brier = brier_score(raw_probs, outcomes)

        result[stat] = {
            "brier": round(cal_brier, 4),
            "uncalibrated_brier": round(raw_brier, 4),
            "samples": len(triples),
        }
    return result


def compute_game_totals_accuracy(
    game_totals: list[dict[str, Any]],
    actuals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Grade game-level over/under projections against actual outcomes.

    Derives actual team totals by summing actual_pts per team from the
    player actuals dict. Returns per-game results and aggregate metrics.
    """
    team_actual_pts: dict[str, float] = {}
    for act in actuals.values():
        if act.get("dnp"):
            continue
        team = act.get("team")
        if team:
            team_actual_pts[team] = team_actual_pts.get(team, 0.0) + float(act.get("actual_pts") or 0)

    results = []
    for g in game_totals:
        home = g.get("home_team_abbr")
        visitor = g.get("visitor_team_abbr")
        proj = g.get("proj_total")
        vegas = g.get("vegas_over_under")
        home_actual = team_actual_pts.get(home) if home else None
        visitor_actual = team_actual_pts.get(visitor) if visitor else None
        if home_actual is None or visitor_actual is None or proj is None:
            continue

        actual_total = round(home_actual + visitor_actual, 1)
        proj_error = round(proj - actual_total, 1)
        vegas_error = round(vegas - actual_total, 1) if vegas is not None else None
        actual_over_vegas = (actual_total > vegas) if vegas is not None else None
        proj_over = (proj > vegas) if vegas is not None else None
        proj_correct_direction = (proj_over == actual_over_vegas) if (proj_over is not None and actual_over_vegas is not None) else None

        results.append(
            {
                "home_team_abbr": home,
                "visitor_team_abbr": visitor,
                "proj_total": proj,
                "home_proj_total": g.get("home_proj_total"),
                "visitor_proj_total": g.get("visitor_proj_total"),
                "vegas_over_under": vegas,
                "actual_total": actual_total,
                "proj_error": proj_error,
                "vegas_error": vegas_error,
                "proj_over": proj_over,
                "actual_over_vegas": actual_over_vegas,
                "proj_correct_direction": proj_correct_direction,
            }
        )

    if not results:
        return {"games": [], "sample_size": 0}

    errors = [r["proj_error"] for r in results]
    correct = [r for r in results if r["proj_correct_direction"] is True]

    return {
        "games": results,
        "sample_size": len(results),
        "proj_mae": round(sum(abs(e) for e in errors) / len(errors), 2),
        "proj_bias": round(sum(errors) / len(errors), 2),
        "directional_accuracy": round(len(correct) / len(results), 3),
    }


def compute_daily_accuracy(
    predictions: dict[str, dict[str, Any]],
    actuals: dict[str, dict[str, Any]],
    date: str,
    prop_actuals: list[dict[str, Any]] | None = None,
    platforms: tuple[str, ...] = PLATFORMS,
    game_totals: list[dict[str, Any]] | None = None,
    is_playoffs: bool = False,
) -> dict[str, Any]:
    """Orchestrate all accuracy computations for a single day.

    Computes metrics per platform (DK, FD, and any future platforms).
    Results are stored under doc["platforms"][platform_slug] as well as
    at top-level (DK values) for backward compatibility with rolling calcs
    and older stored documents.

    Returns the full accuracy document ready to write to Firestore.
    """
    joined = join_predictions_actuals(predictions, actuals)
    coverage = compute_match_coverage(predictions, actuals)

    disabled = _ALWAYS_DISABLED_SIGNALS | (_PLAYOFF_DISABLED_SIGNALS if is_playoffs else set())

    per_platform: dict[str, dict[str, Any]] = {}
    for p in platforms:
        per_platform[p] = {
            "overall": compute_overall_accuracy(joined, p),
            "signal_accuracy": compute_signal_accuracy(joined, p, disabled_signals=disabled),
            "floor_ceiling": compute_floor_ceiling_calibration(joined, p),
        }

    # Per-stat accuracy is platform-agnostic (raw stat counts, not FP)
    stat_accuracy = compute_stat_accuracy(joined)

    # Per-version breakdown: group joined rows by pipeline_version so accuracy
    # can be split after a flag flip (e.g. "legacy" vs "playoffs_v1").
    # Only written when more than one version appears in a single day's data.
    versions: dict[str, list[dict[str, Any]]] = {}
    for row in joined:
        v = row.get("pipeline_version") or "unknown"
        versions.setdefault(v, []).append(row)
    per_version: dict[str, Any] = {}
    if len(versions) > 1 or (len(versions) == 1 and "unknown" not in versions):
        for v, v_rows in versions.items():
            per_version[v] = {
                p: {"overall": compute_overall_accuracy(v_rows, p)}
                for p in platforms
            }
            per_version[v]["sample_size"] = len(v_rows)

    # Top-level keys mirror DK for backward compat
    dk = per_platform.get("dk", {})
    doc: dict[str, Any] = {
        "date": date,
        "sample_size": len(joined),
        "coverage": coverage,
        "platforms": per_platform,
        "stat_accuracy": stat_accuracy,
        # Backward-compat top-level keys (DK)
        "overall": dk.get("overall", {}),
        "signal_accuracy": dk.get("signal_accuracy", {}),
        "floor_ceiling": dk.get("floor_ceiling", {}),
    }
    if per_version:
        doc["per_version"] = per_version

    if prop_actuals:
        doc["prop_brier"] = compute_prop_brier_scores(prop_actuals)

    if game_totals:
        doc["game_totals_accuracy"] = compute_game_totals_accuracy(game_totals, actuals)

    return doc


def compute_rolling_accuracy(daily_docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple daily accuracy documents into rolling metrics.

    Computes weighted averages (by sample size) of all numeric metrics,
    both at top-level (DK, for backward compat) and per-platform.
    """
    if not daily_docs:
        return {"sample_size": 0, "days": 0}

    total_samples = sum(d.get("sample_size", 0) for d in daily_docs)
    if total_samples == 0:
        return {"sample_size": 0, "days": len(daily_docs)}

    overall_keys = [
        "score_vs_actual_r",
        "baseline_vs_actual_r",
        "score_mae",
        "baseline_mae",
        "added_value",
        "predicted_fp_vs_actual_r",
        "predicted_fp_mae",
        "predicted_fp_added_value",
    ]

    def _weighted_avg(docs: list[dict[str, Any]], get_val: Any, get_weight: Any) -> dict[str, Any]:
        """Compute weighted averages of overall_keys from a list of dicts."""
        result: dict[str, Any] = {}
        for key in overall_keys:
            values = [(get_val(d, key), get_weight(d)) for d in docs if get_val(d, key) is not None]
            if values:
                weighted_sum = sum(v * w for v, w in values)
                weight_sum = sum(w for _, w in values)
                result[key] = round(weighted_sum / weight_sum, 4) if weight_sum > 0 else None
            else:
                result[key] = None
        return result

    def _weighted_signal_avg(docs: list[dict[str, Any]], get_sigs: Any, get_weight: Any) -> dict[str, dict[str, Any]]:
        all_signal_keys: set[str] = set()
        for d in docs:
            all_signal_keys.update(get_sigs(d).keys())
        all_signal_keys &= set(SIGNAL_MULTIPLIERS)

        signal_accuracy: dict[str, dict[str, Any]] = {}
        for sig in all_signal_keys:
            sig_metrics: dict[str, Any] = {}
            for metric in ["residual_correlation", "hit_rate", "fire_rate"]:
                values = []
                for d in docs:
                    sig_data = get_sigs(d).get(sig, {})
                    val = sig_data.get(metric)
                    n = get_weight(d)
                    if val is not None and n > 0:
                        values.append((val, n))
                if values:
                    weighted_sum = sum(v * w for v, w in values)
                    weight_sum = sum(w for _, w in values)
                    sig_metrics[metric] = round(weighted_sum / weight_sum, 4) if weight_sum > 0 else None
                else:
                    sig_metrics[metric] = None
            signal_accuracy[sig] = sig_metrics
        return signal_accuracy

    def _weighted_fc_avg(docs: list[dict[str, Any]], get_fc: Any) -> dict[str, Any]:
        fc_keys = ["below_floor_rate", "above_ceiling_rate", "within_range_rate", "floor_mae", "ceiling_mae"]
        floor_ceiling: dict[str, Any] = {}
        for key in fc_keys:
            values = []
            for d in docs:
                fc = get_fc(d)
                val = fc.get(key)
                n = fc.get("sample_size", 0)
                if val is not None and n > 0:
                    values.append((val, n))
            if values:
                weighted_sum = sum(v * w for v, w in values)
                weight_sum = sum(w for _, w in values)
                floor_ceiling[key] = round(weighted_sum / weight_sum, 4) if weight_sum > 0 else None
            else:
                floor_ceiling[key] = None
        return floor_ceiling

    # Top-level (DK, backward compat)
    overall = _weighted_avg(
        daily_docs,
        get_val=lambda d, k: d.get("overall", {}).get(k),
        get_weight=lambda d: d.get("sample_size", 0),
    )
    overall["sample_size"] = total_samples

    signal_accuracy = _weighted_signal_avg(
        daily_docs,
        get_sigs=lambda d: d.get("signal_accuracy", {}),
        get_weight=lambda d: d.get("sample_size", 0),
    )

    floor_ceiling = _weighted_fc_avg(
        daily_docs,
        get_fc=lambda d: d.get("floor_ceiling", {}),
    )

    # Per-platform rolling aggregates
    all_platforms: set[str] = set()
    for d in daily_docs:
        all_platforms.update(d.get("platforms", {}).keys())

    per_platform_rolling: dict[str, dict[str, Any]] = {}
    for p in all_platforms:
        p_docs = [d for d in daily_docs if p in d.get("platforms", {})]
        if not p_docs:
            continue

        p_overall = _weighted_avg(
            p_docs,
            get_val=lambda d, k, _p=p: d.get("platforms", {}).get(_p, {}).get("overall", {}).get(k),
            get_weight=lambda d, _p=p: d.get("platforms", {}).get(_p, {}).get("overall", {}).get("sample_size", 0),
        )
        p_overall["sample_size"] = sum(d.get("platforms", {}).get(p, {}).get("overall", {}).get("sample_size", 0) for d in p_docs)

        p_signals = _weighted_signal_avg(
            p_docs,
            get_sigs=lambda d, _p=p: d.get("platforms", {}).get(_p, {}).get("signal_accuracy", {}),
            get_weight=lambda d, _p=p: d.get("platforms", {}).get(_p, {}).get("overall", {}).get("sample_size", 0),
        )

        p_fc = _weighted_fc_avg(
            p_docs,
            get_fc=lambda d, _p=p: d.get("platforms", {}).get(_p, {}).get("floor_ceiling", {}),
        )

        per_platform_rolling[p] = {
            "overall": p_overall,
            "signal_accuracy": p_signals,
            "floor_ceiling": p_fc,
        }

    # Rolling per-stat accuracy (simple weighted average of MAE/bias/r)
    all_stat_keys: set[str] = set()
    for d in daily_docs:
        all_stat_keys.update(d.get("stat_accuracy", {}).keys())

    rolling_stat_accuracy: dict[str, dict[str, Any]] = {}
    for stat in all_stat_keys:
        for metric in ("mae", "bias", "r"):
            values = []
            for d in daily_docs:
                s = d.get("stat_accuracy", {}).get(stat, {})
                val = s.get(metric)
                n = s.get("sample_size", 0)
                if val is not None and n > 0:
                    values.append((val, n))
            if values:
                weighted_sum = sum(v * w for v, w in values)
                weight_sum = sum(w for _, w in values)
                rolling_stat_accuracy.setdefault(stat, {})[metric] = round(weighted_sum / weight_sum, 4) if weight_sum > 0 else None
            else:
                rolling_stat_accuracy.setdefault(stat, {})[metric] = None
        rolling_stat_accuracy[stat]["sample_size"] = sum(d.get("stat_accuracy", {}).get(stat, {}).get("sample_size", 0) for d in daily_docs)

    return {
        "sample_size": total_samples,
        "days": len(daily_docs),
        "overall": overall,
        "signal_accuracy": signal_accuracy,
        "floor_ceiling": floor_ceiling,
        "platforms": per_platform_rolling,
        "stat_accuracy": rolling_stat_accuracy,
    }


def generate_insights(
    rolling_doc: dict[str, Any],
    daily_docs: list[dict[str, Any]] | None = None,
    disabled_signals: set[str] | None = None,
) -> list[dict[str, str]]:
    """Generate actionable insight recommendations from rolling accuracy data.

    Checks both top-level (DK) and per-platform metrics, deduplicating
    where both platforms show the same issue.

    Returns a list of insight dicts with severity, signal, and message fields.
    Suppresses recommendations when sample size is too small.
    """
    insights: list[dict[str, str]] = []
    total_samples = rolling_doc.get("sample_size", 0)
    days = rolling_doc.get("days", 0)

    if total_samples < 50:
        return insights  # not enough data to generate reliable insights

    def _check_overall(overall: dict[str, Any], label: str) -> None:
        added_value = overall.get("predicted_fp_added_value")
        if added_value is not None and added_value < 0:
            insights.append(
                {
                    "severity": "critical",
                    "signal": "overall",
                    "message": (
                        f"[{label}] Multiplier chain reduces accuracy vs raw baseline (predicted_fp_added_value={added_value:.2f}). Review signal stack."
                    ),
                }
            )

    def _check_signals(signal_accuracy: dict[str, Any], label: str) -> None:
        for sig, data in signal_accuracy.items():
            if disabled_signals and sig in disabled_signals:
                continue
            corr = data.get("residual_correlation")
            if corr is None:
                continue
            fire_rate = data.get("fire_rate", 0)
            if fire_rate < 0.05:
                continue
            if corr < -0.1 and days >= 7:
                insights.append(
                    {
                        "severity": "critical",
                        "signal": sig,
                        "message": (
                            f"[{label}] {sig} is inversely correlated with outcomes (r={corr:.3f} over {days} days). Consider reducing weight or inverting."
                        ),
                    }
                )
            elif corr < 0 and days >= 7:
                insights.append(
                    {
                        "severity": "warning",
                        "signal": sig,
                        "message": (f"[{label}] {sig} has negative correlation (r={corr:.3f}). Monitor for continued degradation."),
                    }
                )

    def _check_fc(fc: dict[str, Any], label: str) -> None:
        below_floor = fc.get("below_floor_rate")
        above_ceiling = fc.get("above_ceiling_rate")
        if below_floor is not None and below_floor > 0.20:
            insights.append(
                {
                    "severity": "critical",
                    "signal": "floor_ceiling",
                    "message": (f"[{label}] {below_floor:.0%} of players scoring below fp_floor (target ~10%). Widen minutes distribution model."),
                }
            )
        elif below_floor is not None and below_floor > 0.15:
            insights.append(
                {
                    "severity": "warning",
                    "signal": "floor_ceiling",
                    "message": (f"[{label}] {below_floor:.0%} of players scoring below fp_floor (target ~10%). Minutes variance may be too narrow."),
                }
            )
        if above_ceiling is not None and above_ceiling > 0.20:
            insights.append(
                {
                    "severity": "critical",
                    "signal": "floor_ceiling",
                    "message": (f"[{label}] {above_ceiling:.0%} of players scoring above fp_ceiling (target ~10%). Ceiling estimates too conservative."),
                }
            )
        elif above_ceiling is not None and above_ceiling > 0.15:
            insights.append(
                {
                    "severity": "warning",
                    "signal": "floor_ceiling",
                    "message": (f"[{label}] {above_ceiling:.0%} of players scoring above fp_ceiling (target ~10%). Consider widening ceiling estimates."),
                }
            )

    # Check per-platform if available; otherwise fall back to top-level
    per_platform = rolling_doc.get("platforms", {})
    if per_platform:
        for p, p_data in per_platform.items():
            label = _PLATFORM_DISPLAY.get(p, p.upper())
            _check_overall(p_data.get("overall", {}), label)
            _check_signals(p_data.get("signal_accuracy", {}), label)
            _check_fc(p_data.get("floor_ceiling", {}), label)
    else:
        # Backward compat: top-level only (old rolling docs)
        _check_overall(rolling_doc.get("overall", {}), "DK")
        _check_signals(rolling_doc.get("signal_accuracy", {}), "DK")
        _check_fc(rolling_doc.get("floor_ceiling", {}), "DK")

    return insights
