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
    "join_predictions_actuals",
    "compute_match_coverage",
    "compute_signal_accuracy",
    "compute_stat_accuracy",
    "compute_prop_brier_scores",
    "compute_game_totals_accuracy",
    "compute_daily_accuracy",
    "compute_rolling_accuracy",
    "generate_insights",
    "_pearson_r",
]

# Multiplier fields present in the opportunity results dict.
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
    Excludes DNP players and players with no actuals.
    """
    joined: list[dict[str, Any]] = []
    for key, pred in predictions.items():
        pid = str(pred.get("id", "")) or key
        act = actuals.get(pid)
        if act is None:
            continue
        if act.get("dnp", False):
            continue
        joined.append({**pred, **{f"actual_{k}": v for k, v in act.items()}, "player_id": pid})
    return joined


def compute_match_coverage(
    predictions: dict[str, dict[str, Any]],
    actuals: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Compute a breakdown of why box-score players were not matched to predictions."""
    matched: list[str] = []
    dnp: list[str] = []
    not_snapshotted: list[str] = []

    for pid, act in actuals.items():
        if pid not in predictions:
            not_snapshotted.append(pid)
            continue
        if act.get("dnp", False):
            dnp.append(pid)
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
        "not_snapshotted_ids": not_snapshotted,
        "dnp_ids": dnp,
    }


def compute_signal_accuracy(
    joined: list[dict[str, Any]],
    disabled_signals: set[str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Compute per-signal calibration metrics.

    For each multiplier signal, measures:
    - residual_correlation: Pearson r between (multiplier - 1.0) and opportunity_score
    - hit_rate: when signal boosts (>1.0), fraction of players with above-average opportunity_score
    - fire_rate: fraction of players where signal deviates from 1.0
    """
    rows = [r for r in joined if r.get("opportunity_score") is not None]
    all_signals = [s for s in SIGNAL_MULTIPLIERS if not (disabled_signals and s in disabled_signals)]

    result: dict[str, dict[str, Any]] = {}
    scores = [float(r["opportunity_score"]) for r in rows]
    mean_score = sum(scores) / len(scores) if scores else 0.0

    for sig in all_signals:
        deviations: list[float] = []
        residuals: list[float] = []
        boost_hits = 0
        boost_count = 0
        penalty_hits = 0
        penalty_count = 0
        fire_count = 0

        for r, score in zip(rows, scores):
            mult = _derive_multiplier(r, sig)
            deviation = mult - 1.0
            residual = score - mean_score

            deviations.append(deviation)
            residuals.append(residual)

            if abs(deviation) > 1e-6:
                fire_count += 1

            if deviation > 1e-6:
                boost_count += 1
                if residual > 0:
                    boost_hits += 1
            elif deviation < -1e-6:
                penalty_count += 1
                if residual < 0:
                    penalty_hits += 1

        corr = _pearson_r(deviations, residuals)
        hit_rate = (boost_hits / boost_count) if boost_count > 0 else None
        penalty_hit_rate = (penalty_hits / penalty_count) if penalty_count > 0 else None
        fire_rate = fire_count / len(rows) if rows else 0.0

        result[sig] = {
            "residual_correlation": round(corr, 4) if corr is not None else None,
            "hit_rate": round(hit_rate, 3) if hit_rate is not None else None,
            "fire_rate": round(fire_rate, 3),
            "boost_count": boost_count,
            "penalty_count": penalty_count,
            "penalty_hit_rate": round(penalty_hit_rate, 3) if penalty_hit_rate is not None else None,
            "total_count": len(rows),
        }

    return result


_STAT_FIELDS: list[tuple[str, str, str]] = [
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
        bias = round(sum(p - a for p, a in pairs) / len(pairs), 2)
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
    """Compute Brier scores per stat type from graded prop actuals."""
    from bks_pipeline_core.pipeline.platt import brier_score

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
    """Grade game-level over/under projections against actual outcomes."""
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
    game_totals: list[dict[str, Any]] | None = None,
    is_playoffs: bool = False,
) -> dict[str, Any]:
    """Orchestrate all accuracy computations for a single day.

    Returns the full accuracy document ready to write to Firestore.
    """
    joined = join_predictions_actuals(predictions, actuals)
    coverage = compute_match_coverage(predictions, actuals)

    disabled = _ALWAYS_DISABLED_SIGNALS | (_PLAYOFF_DISABLED_SIGNALS if is_playoffs else set())

    signal_accuracy = compute_signal_accuracy(joined, disabled_signals=disabled)
    stat_accuracy = compute_stat_accuracy(joined)

    versions: dict[str, list[dict[str, Any]]] = {}
    for row in joined:
        v = row.get("pipeline_version") or "unknown"
        versions.setdefault(v, []).append(row)
    per_version: dict[str, Any] = {}
    if len(versions) > 1 or (len(versions) == 1 and "unknown" not in versions):
        for v, v_rows in versions.items():
            per_version[v] = {"sample_size": len(v_rows)}

    doc: dict[str, Any] = {
        "date": date,
        "sample_size": len(joined),
        "coverage": coverage,
        "signal_accuracy": signal_accuracy,
        "stat_accuracy": stat_accuracy,
    }
    if per_version:
        doc["per_version"] = per_version

    if prop_actuals:
        doc["prop_brier"] = compute_prop_brier_scores(prop_actuals)

    if game_totals:
        doc["game_totals_accuracy"] = compute_game_totals_accuracy(game_totals, actuals)

    return doc


def compute_rolling_accuracy(daily_docs: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate multiple daily accuracy documents into rolling metrics."""
    if not daily_docs:
        return {"sample_size": 0, "days": 0}

    total_samples = sum(d.get("sample_size", 0) for d in daily_docs)
    if total_samples == 0:
        return {"sample_size": 0, "days": len(daily_docs)}

    def _weighted_signal_avg(docs: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        all_signal_keys: set[str] = set()
        for d in docs:
            all_signal_keys.update(d.get("signal_accuracy", {}).keys())
        all_signal_keys &= set(SIGNAL_MULTIPLIERS)

        signal_accuracy: dict[str, dict[str, Any]] = {}
        for sig in all_signal_keys:
            sig_metrics: dict[str, Any] = {}
            for metric in ["residual_correlation", "hit_rate", "fire_rate"]:
                values = []
                for d in docs:
                    sig_data = d.get("signal_accuracy", {}).get(sig, {})
                    val = sig_data.get(metric)
                    n = d.get("sample_size", 0)
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

    signal_accuracy = _weighted_signal_avg(daily_docs)

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
        "signal_accuracy": signal_accuracy,
        "stat_accuracy": rolling_stat_accuracy,
    }


def generate_insights(
    rolling_doc: dict[str, Any],
    daily_docs: list[dict[str, Any]] | None = None,
    disabled_signals: set[str] | None = None,
) -> list[dict[str, str]]:
    """Generate actionable insight recommendations from rolling accuracy data."""
    insights: list[dict[str, str]] = []
    total_samples = rolling_doc.get("sample_size", 0)
    days = rolling_doc.get("days", 0)

    if total_samples < 50:
        return insights

    signal_accuracy = rolling_doc.get("signal_accuracy", {})
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
                        f"{sig} is inversely correlated with outcomes (r={corr:.3f} over {days} days). Consider reducing weight or inverting."
                    ),
                }
            )
        elif corr < 0 and days >= 7:
            insights.append(
                {
                    "severity": "warning",
                    "signal": sig,
                    "message": (f"{sig} has negative correlation (r={corr:.3f}). Monitor for continued degradation."),
                }
            )

    return insights
