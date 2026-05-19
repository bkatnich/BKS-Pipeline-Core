"""Generate an HTML email report from daily accuracy metrics.

Pure function — takes an accuracy dict (and optional rolling data)
and returns an HTML string. No I/O.
"""

from html import escape
from typing import Any

from bks_pipeline_core.sport_config import get_active_config

_TIER_ORDER = ["elite_opp", "good_opp", "solid_opp", "low_opp"]

_SIGNAL_DISPLAY_NAMES: dict[str, str] = {
    "matchup_multiplier": "Matchup (Defense)",
    "vegas_multiplier": "Vegas ITT",
    "stacking_multiplier": "Stacking",
    "mean_reversion_multiplier": "Mean Reversion",
    "minutes_env_multiplier": "Minutes Environment",
    "b2b_penalty": "Back-to-Back",
    "game_env_cap": "Game Env Cap",
    "role_change_multiplier": "Role Change",
    "health_factor": "Health Factor",
    "pace_multiplier": "Pace",
    "cat_trend_multiplier": "Cat Trend",
    "venue_multiplier": "Venue (Home/Away)",
    "usage_delta_multiplier": "Usage Delta",
    "shooting_luck_multiplier": "Shooting Luck",
}

_PLATFORM_DISPLAY: dict[str, str] = {
    "dk": "DraftKings",
    "fd": "FanDuel",
}

# Signals hardcoded to 1.0 unconditionally — excluded from compute_signal_accuracy()
# to avoid zero-variance noise in the scorecard.
_DISABLED_SIGNAL_FIELDS: set[str] = {
    "pace_multiplier",  # disabled 2026-04-10, r=-0.164, 100% fire rate
    "cat_trend_multiplier",  # disabled 2026-04-20, r=-0.156, negative in playoffs
    "venue_multiplier",  # disabled 2026-04-20, r=-0.060, flat home/away premium
    "costar_multiplier",  # code removed 2026-04-27; guard for stale snapshot values
    "elimination_game_multiplier",  # disabled 2026-05-10, r=-0.031 DK / -0.033 FD, 77% fire rate
}
# Public alias so callers (e.g. backtesting.compute_signal_accuracy) can import it.
DISABLED_SIGNAL_FIELDS = _DISABLED_SIGNAL_FIELDS

# Signals hardcoded to 1.0 only during playoffs.  Active in regular season and
# should be evaluated then, but produce zero-variance deviations in playoff
# snapshots — computing r on them yields None or a stale rolling value.
PLAYOFF_DISABLED_SIGNAL_FIELDS: set[str] = {
    "vegas_multiplier",  # r=-0.237 DK, disabled 2026-04-28
    "stacking_multiplier",  # r=-0.120, disabled in playoffs
    "mean_reversion_multiplier",  # r=-0.093 FD, disabled in playoffs
    "matchup_multiplier",  # r=-0.128, disabled in playoffs
    "usage_delta_multiplier",  # r=-0.063, disabled in playoffs
    "role_change_multiplier",  # r=-0.061, disabled in playoffs
    "shooting_luck_multiplier",  # dampened to ~1.0 early playoffs
    "minutes_env_multiplier",  # r=-0.205, excluded from signal_product in playoffs
    "line_movement_multiplier",  # hardcoded 1.0 in playoffs (series lines ≠ per-game flow)
}


def _color_for_correlation(r: float | None) -> str:
    """Return a CSS color based on correlation strength."""
    if r is None:
        return "#999"
    if r > 0.05:
        return "#2d7d46"  # green
    if r < -0.05:
        return "#c0392b"  # red
    return "#d4a017"  # yellow


def _color_for_rate(rate: float, target: float, tolerance: float = 0.05) -> str:
    """Return CSS color based on how close a rate is to its target."""
    if abs(rate - target) <= tolerance:
        return "#2d7d46"
    if abs(rate - target) <= tolerance * 2:
        return "#d4a017"
    return "#c0392b"


def _severity_badge(severity: str) -> str:
    """Return an HTML badge for insight severity."""
    colors = {"critical": "#c0392b", "warning": "#d4a017", "info": "#3498db"}
    bg = colors.get(severity, "#999")
    return f'<span style="background:{bg};color:#fff;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold;">{severity.upper()}</span>'


def _fmt(val: Any, decimals: int = 3) -> str:
    """Format a numeric value for display."""
    if val is None:
        return "N/A"
    if isinstance(val, float):
        return f"{val:.{decimals}f}"
    return str(val)


def _pct(val: Any) -> str:
    """Format as percentage."""
    if val is None:
        return "N/A"
    return f"{val:.1%}"


_SUMMARY_STYLE = "margin:0 0 12px;padding:8px 12px;background:#f9f9fb;border-left:3px solid #ccd;font-size:13px;color:#555;line-height:1.5;"

_SUMMARY_STYLE_TIGHT = "margin:0 0 8px;padding:8px 12px;background:#f9f9fb;border-left:3px solid #ccd;font-size:13px;color:#555;line-height:1.5;"


def _verdict(val: float | None, good: float, bad: float, low_is_good: bool = False) -> str:
    """Return 'strong', 'marginal', 'poor', or 'unavailable'."""
    if val is None:
        return "unavailable"
    if low_is_good:
        return "strong" if val <= good else "marginal" if val <= bad else "poor"
    return "strong" if val >= good else "marginal" if val >= bad else "poor"


def _summary_overall(per_platform: dict[str, Any], platforms: list[str]) -> str:
    """Narrative summary for the Overall Performance section."""
    if not platforms:
        return ""

    def _ov(p: str) -> dict[str, Any]:
        return per_platform.get(p, {}).get("overall", {})

    def _r_sentence(r: float | None) -> str:
        v = _verdict(r, 0.60, 0.40)
        if v == "unavailable":
            return "Correlation data unavailable"
        if v == "strong":
            return f"Predictions correlate well with actual fantasy output (r={r:.3f})"
        if v == "marginal":
            return f"Predictions show moderate correlation with actuals (r={r:.3f})"
        return f"Prediction correlation is weak (r={r:.3f})"

    def _av_sentence(av: float | None) -> str:
        if av is None:
            return ""
        if av > 0:
            return f"The signal stack adds {av:+.2f} FP of value over the raw baseline — signals are helping."
        if av > -0.5:
            return f"Signals are marginally reducing accuracy vs the raw baseline ({av:+.2f} FP)."
        return f"Signals are hurting accuracy ({av:+.2f} FP); review the multiplier chain."

    parts: list[str] = []

    if len(platforms) >= 2:
        dk_ov = _ov(platforms[0])
        fd_ov = _ov(platforms[1])
        dk_r = dk_ov.get("predicted_fp_vs_actual_r")
        fd_r = fd_ov.get("predicted_fp_vs_actual_r")
        dk_v = _verdict(dk_r, 0.60, 0.40)
        fd_v = _verdict(fd_r, 0.60, 0.40)
        dk_label = _PLATFORM_DISPLAY.get(platforms[0], platforms[0].upper())
        fd_label = _PLATFORM_DISPLAY.get(platforms[1], platforms[1].upper())
        if dk_v == fd_v and dk_r is not None and fd_r is not None:
            parts.append(f"Both platforms show {dk_v} correlation ({dk_label} r={dk_r:.3f}, {fd_label} r={fd_r:.3f}).")
        else:
            parts.append(f"{dk_label}: {_r_sentence(dk_r)}. {fd_label}: {_r_sentence(fd_r)}.")
        dk_av = dk_ov.get("predicted_fp_added_value")
        fd_av = fd_ov.get("predicted_fp_added_value")
        if dk_av is not None:
            parts.append(f"{dk_label} — {_av_sentence(dk_av)}")
        if fd_av is not None:
            parts.append(f"{fd_label} — {_av_sentence(fd_av)}")
        dk_mae = dk_ov.get("predicted_fp_mae")
        dk_bsr = dk_ov.get("baseline_vs_actual_r")
        if dk_mae is not None and dk_bsr is not None:
            parts.append(f"{dk_label} MAE: {dk_mae:.2f} FP. Baseline r: {dk_bsr:.3f}.")
    else:
        p = platforms[0]
        ov = _ov(p)
        r = ov.get("predicted_fp_vs_actual_r")
        av = ov.get("predicted_fp_added_value")
        mae = ov.get("predicted_fp_mae")
        bsr = ov.get("baseline_vs_actual_r")
        parts.append(f"{_r_sentence(r)}.")
        av_s = _av_sentence(av)
        if av_s:
            parts.append(av_s)
        if mae is not None and bsr is not None:
            parts.append(f"MAE of {mae:.2f} FP. Baseline correlation: {bsr:.3f}.")

    text = " ".join(parts)
    return f'<p style="{_SUMMARY_STYLE}">{text}</p>' if text.strip() else ""


def _summary_tier(tiers: dict[str, Any], platform_label: str) -> str:
    """Narrative summary for one platform's tier block."""
    if not tiers:
        return ""
    tier_valid = tiers.get("_tier_ordering_valid", True)
    elite = tiers.get("elite_opp") or {}
    low = tiers.get("low_opp") or {}
    elite_fp = elite.get("mean_actual_fp")
    low_fp = low.get("mean_actual_fp")
    elite_hr = elite.get("hit_rate")

    parts: list[str] = []
    if not tier_valid:
        parts.append("&#9888; Tier ordering is INVERTED — lower tiers outperformed higher tiers. Investigate scoring or tier assignment.")
    elif elite_fp is not None and low_fp is not None:
        spread = elite_fp - low_fp
        if spread > 15:
            parts.append(
                f"Elite-tier players averaged {elite_fp:.1f} FP vs {low_fp:.1f} for low-tier"
                f" — a {spread:.1f}-pt spread confirms tier discrimination is working."
            )
        else:
            parts.append(f"Tier spread is modest ({spread:.1f} FP elite vs low) but ordering is intact.")

    if elite_hr is not None:
        if elite_hr >= 0.60:
            parts.append(f"Elite-tier hit rate of {_pct(elite_hr)} — top-rated players are reliably delivering.")
        elif elite_hr < 0.50:
            parts.append(f"Elite-tier hit rate of {_pct(elite_hr)} is below 50% — the tier may be mis-calibrated.")
        else:
            parts.append(f"Elite-tier hit rate: {_pct(elite_hr)}.")

    text = " ".join(parts)
    return f'<p style="{_SUMMARY_STYLE_TIGHT}">{text}</p>' if text.strip() else ""


def _summary_signal(all_sigs: dict[str, Any], platforms: list[str]) -> str:
    """Narrative summary for the Signal Scorecard section."""
    if not all_sigs or not platforms:
        return ""
    first_p = platforms[0]

    corrs: dict[str, float] = {}
    for sig, p_data in all_sigs.items():
        if sig in _DISABLED_SIGNAL_FIELDS:
            all_fire = [p_data.get(p, {}).get("fire_rate") or 0.0 for p in platforms]
            if all(fr == 0.0 for fr in all_fire):
                continue
        r = p_data.get(first_p, {}).get("residual_correlation")
        if r is not None:
            corrs[sig] = r

    if not corrs:
        return ""

    total = len(corrs)
    positive = sum(1 for r in corrs.values() if r > 0.05)
    negative = sum(1 for r in corrs.values() if r < -0.05)
    best_sig = max(corrs, key=corrs.get)  # type: ignore[arg-type]
    worst_sig = min(corrs, key=corrs.get)  # type: ignore[arg-type]
    best_r = corrs[best_sig]
    worst_r = corrs[worst_sig]
    best_name = _SIGNAL_DISPLAY_NAMES.get(best_sig, best_sig)
    worst_name = _SIGNAL_DISPLAY_NAMES.get(worst_sig, worst_sig)

    parts = [f"{positive} of {total} active signals correlate positively with outcomes."]
    parts.append(f"Strongest: {best_name} (r={best_r:+.3f}).")
    if negative > 0:
        parts.append(f"Watch: {negative} signal(s) showing negative correlation — {worst_name} (r={worst_r:+.3f}) is working against the model.")

    text = " ".join(parts)
    return f'<p style="{_SUMMARY_STYLE}">{text}</p>'


def _summary_floor_ceiling(per_platform: dict[str, Any], platforms: list[str]) -> str:
    """Narrative summary for the Floor/Ceiling Calibration section."""
    if not platforms:
        return ""
    fc = per_platform.get(platforms[0], {}).get("floor_ceiling", {})
    within = fc.get("within_range_rate")
    below = fc.get("below_floor_rate")
    above = fc.get("above_ceiling_rate")

    if within is None:
        return ""

    parts: list[str] = []
    if within >= 0.75:
        parts.append(f"{_pct(within)} of actuals landed within the projected range — brackets are well-calibrated.")
    elif within >= 0.65:
        parts.append(f"{_pct(within)} within projected range — slight over-confidence in bracket width.")
    else:
        parts.append(f"Only {_pct(within)} within projected range — floor/ceiling brackets are too narrow.")

    if below is not None and above is not None:
        if below > 0.15:
            parts.append(f"Floor too optimistic — {_pct(below)} missed their floor (target ~10%).")
        elif above > 0.15:
            parts.append(f"Ceiling too conservative — {_pct(above)} exceeded their ceiling (target ~10%).")
        elif below < 0.05:
            parts.append(f"Floor may be too pessimistic — only {_pct(below)} are missing it.")
        elif above < 0.05:
            parts.append(f"Ceiling may be too aggressive — only {_pct(above)} exceeded it.")
        else:
            parts.append("Floor and ceiling miss rates are both near the 10% target.")

    text = " ".join(parts)
    return f'<p style="{_SUMMARY_STYLE}">{text}</p>' if text.strip() else ""


def _summary_stat(stat_accuracy: dict[str, Any]) -> str:
    """Narrative summary for the Per-Stat Prediction Accuracy section."""
    if not stat_accuracy:
        return ""

    rs = {s: d["r"] for s, d in stat_accuracy.items() if d.get("r") is not None}
    biases = {s: d["bias"] for s, d in stat_accuracy.items() if d.get("bias") is not None}

    parts: list[str] = []

    if rs:
        best_stat = max(rs, key=rs.get)  # type: ignore[arg-type]
        best_r = rs[best_stat]
        parts.append(f"{best_stat} shows the strongest correlation (r={best_r:.3f}).")

    if biases:
        worst_stat = max(biases, key=lambda s: abs(biases[s]))
        worst_bias = biases[worst_stat]
        if abs(worst_bias) > 1.0:
            direction = "over-projecting" if worst_bias > 0 else "under-projecting"
            parts.append(f"The model is {direction} {worst_stat} by {abs(worst_bias):.1f} on average — consider a recalibration.")
        elif abs(worst_bias) <= 0.5:
            parts.append(f"Stat-level bias is minimal (largest: {worst_stat} at {worst_bias:+.2f}).")
        else:
            parts.append(f"{worst_stat} bias ({worst_bias:+.2f}) is within acceptable range.")

    text = " ".join(parts)
    return f'<p style="{_SUMMARY_STYLE}">{text}</p>' if text.strip() else ""


def _summary_prop_brier(prop_brier: dict[str, Any]) -> str:
    """Narrative summary for the Prop Calibration (Brier Score) section."""
    entries = {s: d for s, d in prop_brier.items() if d.get("brier") is not None}
    if not entries:
        return ""

    briers = {s: d["brier"] for s, d in entries.items()}
    total = len(briers)
    below_target = sum(1 for b in briers.values() if b < 0.22)
    best = min(briers, key=briers.get)  # type: ignore[arg-type]
    worst = max(briers, key=briers.get)  # type: ignore[arg-type]
    improvements = [d.get("uncalibrated_brier", 0) - d.get("brier", 0) for d in entries.values()]
    mean_imp = sum(improvements) / len(improvements) if improvements else 0.0

    if below_target == total:
        count_s = f"All {total} stats are below the 0.22 Brier target — prop calibration is solid"
    elif below_target > 0:
        count_s = f"{below_target} of {total} stats are below the 0.22 target"
    else:
        count_s = "No stats meet the 0.22 Brier target — prop calibration needs work"

    parts = [f"{count_s}. Best: {best.upper()} ({briers[best]:.4f}). Worst: {worst.upper()} ({briers[worst]:.4f})."]

    if mean_imp > 0:
        parts.append(f"Platt calibration is improving average Brier by {mean_imp:.4f} across stats.")
    else:
        parts.append("Calibration is not currently improving raw Brier scores — revisit Platt coefficients.")

    text = " ".join(parts)
    return f'<p style="{_SUMMARY_STYLE}">{text}</p>'


def _summary_rolling(rolling_7d: dict[str, Any], per_platform: dict[str, Any], platforms: list[str]) -> str:
    """Narrative summary for the 7-Day Rolling Trend section."""
    if not platforms:
        return ""
    p = platforms[0]
    p_label = _PLATFORM_DISPLAY.get(p, p.upper())
    r7_p = rolling_7d.get("platforms", {}).get(p, {})
    r7_ov = r7_p.get("overall", {})
    today_ov = per_platform.get(p, {}).get("overall", {})
    r7_r = r7_ov.get("predicted_fp_vs_actual_r")
    today_r = today_ov.get("predicted_fp_vs_actual_r")

    parts: list[str] = []

    if r7_r is not None and today_r is not None:
        delta = r7_r - today_r
        if delta > 0.02:
            parts.append(f"Rolling correlation ({r7_r:.3f}) is above today's ({today_r:.3f}) — today was a below-average day for the model.")
        elif delta < -0.02:
            parts.append(f"Rolling correlation ({r7_r:.3f}) is below today's ({today_r:.3f}) — today outperformed the recent trend.")
        else:
            parts.append(f"Rolling correlation ({r7_r:.3f}) is in line with today's ({today_r:.3f}).")
    elif r7_r is not None:
        parts.append(f"7-day rolling {p_label} correlation: {r7_r:.3f}.")
    else:
        parts.append("7-day rolling data available.")

    r7_sigs = r7_p.get("signal_accuracy", {})
    today_sigs = per_platform.get(p, {}).get("signal_accuracy", {})
    improved = sum(1 for s in r7_sigs if (r7_sigs[s].get("residual_correlation") or 0) > (today_sigs.get(s, {}).get("residual_correlation") or 0))
    degraded = sum(1 for s in r7_sigs if (r7_sigs[s].get("residual_correlation") or 0) < (today_sigs.get(s, {}).get("residual_correlation") or 0))
    if improved > degraded:
        parts.append(f"{improved} signals show better 7-day correlation than today; {degraded} have slipped.")
    elif degraded > improved:
        parts.append(f"{degraded} signals have degraded over the 7-day window relative to today.")
    else:
        parts.append("Signal performance is broadly stable vs the 7-day window.")

    text = " ".join(parts)
    return f'<p style="{_SUMMARY_STYLE}">{text}</p>' if text.strip() else ""


def _league_mode_label(league_state: dict[str, Any] | None) -> str:
    """Return a human-readable league mode string for email reports."""
    if not league_state:
        return "Unknown"
    mode = league_state.get("mode", "regular_season")
    season = league_state.get("season", "")
    if mode == "playoffs":
        rnd = league_state.get("playoff_round") or ""
        rnd_str = f" — Round {rnd}" if rnd else ""
        return f"Playoffs{rnd_str} ({season})"
    if mode == "offseason":
        return f"Offseason ({season})"
    return f"Regular Season ({season})"


def _platform_header_cells(platforms: list[str]) -> str:
    """Return <th> cells for each platform."""
    return "".join(f'<th style="text-align:center;padding:8px 12px;">{_PLATFORM_DISPLAY.get(p, p.upper())}</th>' for p in platforms)


def _overall_row(label: str, key: str, platforms: list[str], per_platform: dict[str, Any], fmt_fn: Any = None) -> str:
    """Render one row of the overall table with per-platform columns."""
    cells = ""
    for p in platforms:
        val = per_platform.get(p, {}).get("overall", {}).get(key)
        if fmt_fn:
            display = fmt_fn(val)
        elif key.endswith("_r"):
            color = _color_for_correlation(val)
            display = f'<span style="color:{color};font-weight:bold;">{_fmt(val)}</span>'
        elif key == "predicted_fp_added_value":
            color = "#2d7d46" if (val or 0) > 0 else "#c0392b" if (val or 0) < 0 else "#d4a017"
            prefix = "+" if (val or 0) > 0 else ""
            display = f'<span style="color:{color};font-weight:bold;font-size:15px;">{prefix}{_fmt(val, 2)}</span>'
        else:
            display = _fmt(val, 2)
        cells += f'<td style="text-align:center;padding:6px 12px;">{display}</td>'
    return f'<tr><td style="padding:6px 0;">{label}</td>{cells}</tr>'


def generate_accuracy_report_html(
    accuracy: dict[str, Any],
    rolling_7d: dict[str, Any] | None = None,
    insights: list[dict[str, str]] | None = None,
    league_state: dict[str, Any] | None = None,
) -> str:
    """Generate a complete HTML email report from accuracy data.

    Args:
        accuracy: Daily accuracy document (from compute_daily_accuracy).
        rolling_7d: Optional 7-day rolling accuracy for trend section.
        insights: Optional list of insight dicts from generate_insights.

    Returns:
        HTML string suitable for email body.
    """
    date = accuracy.get("date", "Unknown")
    sample_size = accuracy.get("sample_size", 0)

    # Determine which platforms are present
    per_platform: dict[str, Any] = accuracy.get("platforms", {})
    platforms = [p for p in ("dk", "fd") if p in per_platform]
    # Fall back to DK-only from top-level keys for old docs
    if not platforms:
        platforms = ["dk"]
        per_platform = {
            "dk": {
                "overall": accuracy.get("overall", {}),
                "tier_accuracy": accuracy.get("tier_accuracy", {}),
                "signal_accuracy": accuracy.get("signal_accuracy", {}),
                "floor_ceiling": accuracy.get("floor_ceiling", {}),
            }
        }

    sections: list[str] = []

    # --- Header ---
    sections.append(f"""
    <div style="background:#1a1a2e;color:#fff;padding:20px 24px;border-radius:8px 8px 0 0;">
        <h1 style="margin:0;font-size:22px;">BKS {get_active_config().sport_display_name} Accuracy Report</h1>
        <p style="margin:4px 0 0;color:#aab;font-size:14px;">{date} &middot; {sample_size} players analyzed</p>
    </div>
    """)

    # --- System Status (league mode context) ---
    if league_state is not None:
        mode_label = _league_mode_label(league_state)
        playoff_start = league_state.get("playoff_start_date")
        playoff_round = league_state.get("playoff_round")
        mode = league_state.get("mode", "regular_season")

        extra_rows = ""
        if mode == "playoffs":
            if playoff_round:
                extra_rows += (
                    f'<tr><td style="padding:4px 0;color:#555">Playoff Round</td><td style="text-align:right;font-weight:bold">{playoff_round}</td></tr>'
                )
            if playoff_start:
                extra_rows += f'<tr><td style="padding:4px 0;color:#555">Playoff Start</td><td style="text-align:right">{escape(str(playoff_start))}</td></tr>'

        sections.append(f"""
    <div style="padding:12px 24px;background:#f9f9f9;border-bottom:1px solid #eee;">
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tr>
                <td style="padding:4px 0;color:#555">League Mode</td>
                <td style="text-align:right;font-weight:bold">{escape(mode_label)}</td>
            </tr>
            {extra_rows}
        </table>
    </div>
    """)

    # --- Section 1: Overall Performance (per-platform columns) ---
    platform_headers = _platform_header_cells(platforms)

    sections.append(f"""
    <div style="padding:16px 24px;">
        <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;">
            Overall Performance
        </h2>
        {_summary_overall(per_platform, platforms)}
        <table style="width:100%;border-collapse:collapse;font-size:14px;">
            <tr style="background:#f5f5f5;">
                <th style="padding:8px;text-align:left;">Metric</th>
                {platform_headers}
            </tr>
            {_overall_row("Predicted FP vs Actual (Pearson r)", "predicted_fp_vs_actual_r", platforms, per_platform)}
            {_overall_row("Baseline vs Actual (r)", "baseline_vs_actual_r", platforms, per_platform)}
            {_overall_row("Predicted FP MAE", "predicted_fp_mae", platforms, per_platform)}
            {_overall_row("Baseline MAE", "baseline_mae", platforms, per_platform)}
            {_overall_row("Added Value (signals helping?)", "predicted_fp_added_value", platforms, per_platform)}
            {_overall_row("Opp Score vs Actual (r)", "score_vs_actual_r", platforms, per_platform)}
            {_overall_row("Opp Score MAE", "score_mae", platforms, per_platform)}
        </table>
    </div>
    """)

    # --- Section 2: Tier Report Card (per-platform sub-tables) ---
    tier_section_html = ""
    for p in platforms:
        tiers = per_platform.get(p, {}).get("tier_accuracy", {})
        if not tiers:
            continue
        label = _PLATFORM_DISPLAY.get(p, p.upper())
        tier_rows = ""
        for tier in _TIER_ORDER:
            t = tiers.get(tier, {})
            count = t.get("count", 0)
            if count == 0:
                continue
            tier_rows += f"""
            <tr>
                <td style="padding:5px 8px;">{tier.replace("_", " ").title()}</td>
                <td style="text-align:center;">{count}</td>
                <td style="text-align:center;font-weight:bold;">{_fmt(t.get("mean_actual_fp"), 1)}</td>
                <td style="text-align:center;">{_fmt(t.get("mean_predicted_baseline"), 1)}</td>
                <td style="text-align:center;">{_pct(t.get("hit_rate"))}</td>
            </tr>
            """
        tier_valid = tiers.get("_tier_ordering_valid", True)
        tier_status = (
            '<span style="color:#2d7d46;">Tiers properly ordered</span>' if tier_valid else '<span style="color:#c0392b;">Tier ordering INVERTED</span>'
        )
        tier_section_html += f"""
        <p style="font-size:13px;font-weight:bold;color:#555;margin:12px 0 4px;">{label}</p>
        {_summary_tier(tiers, label)}
        <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:4px;">
            <tr style="background:#f5f5f5;">
                <th style="padding:6px;text-align:left;">Tier</th>
                <th style="text-align:center;">Count</th>
                <th style="text-align:center;">Mean Actual FP</th>
                <th style="text-align:center;">Mean Predicted</th>
                <th style="text-align:center;">Hit Rate</th>
            </tr>
            {tier_rows}
        </table>
        <p style="font-size:12px;color:#666;margin:0 0 12px;">{tier_status}</p>
        """

    sections.append(f"""
    <div style="padding:16px 24px;">
        <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;">
            Tier Report Card
        </h2>
        {tier_section_html}
    </div>
    """)

    # --- Section 3: Signal Scorecard (per-platform r columns) ---
    # Collect all signals from all platforms
    all_sigs: dict[str, dict[str, dict[str, Any]]] = {}  # sig -> platform -> data
    for p in platforms:
        for sig, data in per_platform.get(p, {}).get("signal_accuracy", {}).items():
            all_sigs.setdefault(sig, {})[p] = data

    # Sort by first-platform correlation descending
    first_p = platforms[0]
    signal_rows_sorted = sorted(
        all_sigs.items(),
        key=lambda x: x[1].get(first_p, {}).get("residual_correlation") or -999,
        reverse=True,
    )

    r_headers = "".join(f'<th style="text-align:center;">{_PLATFORM_DISPLAY.get(p, p.upper())} r</th>' for p in platforms)
    hit_headers = "".join(f'<th style="text-align:center;">{_PLATFORM_DISPLAY.get(p, p.upper())} Hit%</th>' for p in platforms)

    _is_playoffs_mode = (league_state or {}).get("mode") == "playoffs"
    _all_disabled = _DISABLED_SIGNAL_FIELDS | (PLAYOFF_DISABLED_SIGNAL_FIELDS if _is_playoffs_mode else set())

    signal_rows_html = ""
    for sig, p_data in signal_rows_sorted:
        all_fire_rates = [p_data.get(p, {}).get("fire_rate") or 0.0 for p in platforms]
        is_zero_fire = all(fr == 0.0 for fr in all_fire_rates)

        # Always-disabled signals: silently skip (no row added).
        if is_zero_fire and sig in _DISABLED_SIGNAL_FIELDS:
            continue

        display_name = _SIGNAL_DISPLAY_NAMES.get(sig, sig)

        # Playoff-disabled signals: show a greyed "Disabled" row so readers
        # know the signal exists but is inactive — stale r values are not rendered.
        if is_zero_fire and sig in _all_disabled:
            n_data_cols = len(platforms) * 2  # r + hit% per platform
            disabled_cells = "".join('<td style="text-align:center;color:#bbb;">—</td>' for _ in range(n_data_cols))
            signal_rows_html += f"""
        <tr style="color:#bbb;">
            <td style="padding:6px 8px;">{display_name} <span style="font-size:10px;">(disabled)</span></td>
            {disabled_cells}
            <td style="text-align:center;">0.0%</td>
        </tr>
        """
            continue

        r_cells = ""
        hit_cells = ""
        fire_rate = None
        for p in platforms:
            data = p_data.get(p, {})
            corr = data.get("residual_correlation")
            color = _color_for_correlation(corr)
            r_cells += f'<td style="text-align:center;color:{color};font-weight:bold;">{_fmt(corr)}</td>'

            hit_rate = data.get("hit_rate")
            penalty_hit_rate = data.get("penalty_hit_rate")
            if hit_rate is not None:
                hit_cells += f'<td style="text-align:center;">{_pct(hit_rate)}</td>'
            elif penalty_hit_rate is not None:
                hit_cells += f'<td style="text-align:center;">{_pct(penalty_hit_rate)} <span style="font-size:10px;color:#999;">&#9660;</span></td>'
            else:
                hit_cells += '<td style="text-align:center;">N/A</td>'

            if fire_rate is None:
                fire_rate = data.get("fire_rate")

        signal_rows_html += f"""
        <tr>
            <td style="padding:6px 8px;">{display_name}</td>
            {r_cells}
            {hit_cells}
            <td style="text-align:center;">{_pct(fire_rate)}</td>
        </tr>
        """

    sections.append(f"""
    <div style="padding:16px 24px;">
        <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;">
            Signal Scorecard
        </h2>
        {_summary_signal(all_sigs, platforms)}
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tr style="background:#f5f5f5;">
                <th style="padding:8px;text-align:left;">Signal</th>
                {r_headers}
                {hit_headers}
                <th style="text-align:center;">Fire Rate</th>
            </tr>
            {signal_rows_html}
        </table>
        <p style="font-size:11px;color:#999;margin-top:6px;">
            Correlation: r&gt;0.05 = <span style="color:#2d7d46;">good</span>,
            |r|&lt;0.05 = <span style="color:#d4a017;">noise</span>,
            r&lt;-0.05 = <span style="color:#c0392b;">wrong direction</span>.
            &#9660; = penalty-only signal (hit rate = penalized players who underperformed).
            Signals disabled unconditionally are excluded. Signals disabled in playoffs are shown greyed as &#8220;disabled&#8221; with no r value.
        </p>
    </div>
    """)

    # --- Section 4: Floor/Ceiling Calibration (per-platform rows) ---
    fc_rows = ""
    fc_metric_labels = [
        ("below_floor_rate", "Below Floor (target ~10%)"),
        ("above_ceiling_rate", "Above Ceiling (target ~10%)"),
        ("within_range_rate", "Within Range (target ~80%)"),
        ("floor_mae", "Floor MAE"),
        ("ceiling_mae", "Ceiling MAE"),
    ]
    for key, label in fc_metric_labels:
        cells = ""
        for p in platforms:
            fc = per_platform.get(p, {}).get("floor_ceiling", {})
            val = fc.get(key)
            if val is None:
                cells += '<td style="text-align:center;">N/A</td>'
            elif key.endswith("_rate"):
                target = 0.10 if "floor" in key or "ceiling" in key else 0.80
                color = _color_for_rate(val, target)
                cells += f'<td style="text-align:center;font-weight:bold;color:{color};">{_pct(val)}</td>'
            else:
                cells += f'<td style="text-align:center;">{_fmt(val, 2)}</td>'
        fc_rows += f'<tr><td style="padding:6px 0;">{label}</td>{cells}</tr>'

    sections.append(f"""
    <div style="padding:16px 24px;">
        <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;">
            Floor/Ceiling Calibration
        </h2>
        {_summary_floor_ceiling(per_platform, platforms)}
        <table style="width:100%;border-collapse:collapse;font-size:14px;">
            <tr style="background:#f5f5f5;">
                <th style="padding:8px;text-align:left;">Metric</th>
                {platform_headers}
            </tr>
            {fc_rows}
        </table>
    </div>
    """)

    # --- Section 5: Per-Stat Prediction Accuracy ---
    stat_accuracy = accuracy.get("stat_accuracy", {})
    if stat_accuracy:
        stat_order = ["PTS", "REB", "AST", "STL", "BLK", "MIN"]
        stat_rows = ""
        for stat in stat_order:
            data = stat_accuracy.get(stat)
            if not data:
                continue
            mae = data.get("mae")
            bias = data.get("bias")
            r = data.get("r")
            n = data.get("sample_size", 0)
            r_color = _color_for_correlation(r)
            # Bias: green = near zero, red = large over/under projection
            bias_str = "N/A"
            if bias is not None:
                bias_color = "#2d7d46" if abs(bias) < 0.5 else "#d4a017" if abs(bias) < 1.5 else "#c0392b"
                prefix = "+" if bias > 0 else ""
                bias_str = f'<span style="color:{bias_color};">{prefix}{bias:.2f}</span>'
            stat_rows += f"""
            <tr>
                <td style="padding:6px 8px;font-weight:bold;">{stat}</td>
                <td style="text-align:center;">{_fmt(mae, 2) if mae is not None else "N/A"}</td>
                <td style="text-align:center;">{bias_str}</td>
                <td style="text-align:center;color:{r_color};font-weight:bold;">{_fmt(r)}</td>
                <td style="text-align:center;color:#999;">{n}</td>
            </tr>
            """

        sections.append(f"""
        <div style="padding:16px 24px;">
            <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;">
                Per-Stat Prediction Accuracy
            </h2>
            {_summary_stat(stat_accuracy)}
            <table style="width:100%;border-collapse:collapse;font-size:13px;">
                <tr style="background:#f5f5f5;">
                    <th style="padding:8px;text-align:left;">Stat</th>
                    <th style="text-align:center;">MAE</th>
                    <th style="text-align:center;">Bias</th>
                    <th style="text-align:center;">Pearson r</th>
                    <th style="text-align:center;">Samples</th>
                </tr>
                {stat_rows}
            </table>
            <p style="font-size:11px;color:#999;margin-top:6px;">
                MAE = mean absolute error. Bias = mean (predicted &minus; actual): positive = over-projecting, negative = under-projecting.
                r = correlation between projection and actual stat. Only players with snapshot projections included.
            </p>
        </div>
        """)

    # --- Section 6: Prop Brier Scores ---
    prop_brier = accuracy.get("prop_brier")
    if prop_brier:
        brier_rows = ""
        for stat in sorted(prop_brier.keys()):
            data = prop_brier[stat]
            cal_brier = data.get("brier", 0)
            raw_brier = data.get("uncalibrated_brier", 0)
            samples = data.get("samples", 0)
            color = "#2d7d46" if cal_brier < 0.22 else "#d4a017" if cal_brier < 0.25 else "#c0392b"
            improvement = raw_brier - cal_brier
            imp_str = f"{improvement:+.4f}" if improvement != 0 else "N/A"
            brier_rows += f"""
            <tr>
                <td style="padding:6px 8px;">{stat.upper()}</td>
                <td style="text-align:center;font-weight:bold;color:{color};">{cal_brier:.4f}</td>
                <td style="text-align:center;">{raw_brier:.4f}</td>
                <td style="text-align:center;">{imp_str}</td>
                <td style="text-align:center;">{samples}</td>
            </tr>
            """

        sections.append(f"""
        <div style="padding:16px 24px;">
            <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;">
                Prop Calibration (Brier Score)
            </h2>
            {_summary_prop_brier(prop_brier)}
            <table style="width:100%;border-collapse:collapse;font-size:13px;">
                <tr style="background:#f5f5f5;">
                    <th style="padding:8px;text-align:left;">Stat</th>
                    <th style="text-align:center;">Calibrated</th>
                    <th style="text-align:center;">Raw</th>
                    <th style="text-align:center;">Improvement</th>
                    <th style="text-align:center;">Samples</th>
                </tr>
                {brier_rows}
            </table>
            <p style="font-size:11px;color:#999;margin-top:6px;">
                Brier Score: 0 = perfect, 0.25 = coin flip baseline.
                Target &lt; 0.22. Green = good, Yellow = marginal, Red = needs work.
            </p>
        </div>
        """)

    # --- Section 6: Actionable Insights ---
    if insights:
        insight_items = ""
        for ins in insights:
            badge = _severity_badge(ins.get("severity", "info"))
            insight_items += f"""
            <div style="padding:8px 0;border-bottom:1px solid #f0f0f0;">
                {badge}
                <span style="margin-left:8px;font-size:13px;">{ins.get("message", "")}</span>
            </div>
            """
        sections.append(f"""
        <div style="padding:16px 24px;">
            <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;">
                Actionable Insights
            </h2>
            {insight_items}
        </div>
        """)

    # --- Section 7: 7-Day Rolling Trend (per-platform) ---
    if rolling_7d and rolling_7d.get("days", 0) >= 3:
        r7_per_platform: dict[str, Any] = rolling_7d.get("platforms", {})
        # Fall back to top-level for old rolling docs
        if not r7_per_platform:
            r7_per_platform = {
                "dk": {
                    "overall": rolling_7d.get("overall", {}),
                    "signal_accuracy": rolling_7d.get("signal_accuracy", {}),
                }
            }

        rolling_platform_sections = ""
        for p in platforms:
            r7_p = r7_per_platform.get(p, {})
            r7_overall = r7_p.get("overall", {})
            r7_signals = r7_p.get("signal_accuracy", {})
            if not r7_overall and not r7_signals:
                continue

            p_label = _PLATFORM_DISPLAY.get(p, p.upper())
            # Daily signal accuracy for delta
            daily_signals = per_platform.get(p, {}).get("signal_accuracy", {})

            trend_rows = ""
            for sig, data in sorted(
                r7_signals.items(),
                key=lambda x: x[1].get("residual_correlation") or -999,
                reverse=True,
            ):
                # Skip stale keys from old Firestore docs not in the current signal set
                if sig not in _SIGNAL_DISPLAY_NAMES:
                    continue
                corr_7d = data.get("residual_correlation")
                daily_corr = daily_signals.get(sig, {}).get("residual_correlation")
                if corr_7d is None:
                    continue
                # Skip disabled signals (0% fire rate)
                if sig in _DISABLED_SIGNAL_FIELDS and (data.get("fire_rate") or 0.0) == 0.0:
                    continue
                delta = (corr_7d - daily_corr) if daily_corr is not None else None
                delta_str = f"{delta:+.3f}" if delta is not None else "N/A"
                display_name = _SIGNAL_DISPLAY_NAMES.get(sig, sig)
                trend_rows += f"""
                <tr>
                    <td style="padding:4px 8px;">{display_name}</td>
                    <td style="text-align:center;">{_fmt(corr_7d)}</td>
                    <td style="text-align:center;">{delta_str}</td>
                </tr>
                """

            rolling_platform_sections += f"""
            <p style="font-size:13px;font-weight:bold;color:#555;margin:12px 0 4px;">{p_label}</p>
            <p style="font-size:12px;color:#666;margin:0 0 6px;">
                Rolling r: {_fmt(r7_overall.get("predicted_fp_vs_actual_r"))} &middot;
                Rolling MAE: {_fmt(r7_overall.get("predicted_fp_mae"), 2)} FP
            </p>
            <table style="width:100%;border-collapse:collapse;font-size:13px;margin-bottom:12px;">
                <tr style="background:#f5f5f5;">
                    <th style="padding:6px;text-align:left;">Signal</th>
                    <th style="text-align:center;">7d Correlation</th>
                    <th style="text-align:center;">vs Today</th>
                </tr>
                {trend_rows}
            </table>
            """

        sections.append(f"""
        <div style="padding:16px 24px;">
            <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;">
                7-Day Rolling Trend &middot; {rolling_7d.get("days", 0)} days
            </h2>
            {_summary_rolling(rolling_7d, per_platform, platforms)}
            {rolling_platform_sections}
        </div>
        """)

    # --- Game Total Projections ---
    gta = accuracy.get("game_totals_accuracy")
    if gta and gta.get("sample_size", 0) > 0:
        games = gta.get("games", [])
        proj_mae = gta.get("proj_mae")
        proj_bias = gta.get("proj_bias")
        dir_acc = gta.get("directional_accuracy")

        dir_color = "#2d7d46" if dir_acc is not None and dir_acc >= 0.55 else "#d4a017" if dir_acc is not None and dir_acc >= 0.45 else "#c0392b"
        dir_label = f"{dir_acc:.0%}" if dir_acc is not None else "N/A"
        bias_str = f"{proj_bias:+.1f}" if proj_bias is not None else "N/A"
        mae_str = f"{proj_mae:.1f}" if proj_mae is not None else "N/A"

        summary_line = f"MAE: {mae_str} pts &middot; Bias: {bias_str} pts &middot; Directional accuracy: <strong style='color:{dir_color}'>{dir_label}</strong>"

        rows_html = ""
        for g in games:
            home = g.get("home_team_abbr", "?")
            visitor = g.get("visitor_team_abbr", "?")
            proj = g.get("proj_total")
            vegas = g.get("vegas_over_under")
            actual = g.get("actual_total")
            err = g.get("proj_error")
            correct = g.get("proj_correct_direction")
            proj_over = g.get("proj_over")

            dir_cell_color = "#2d7d46" if correct else "#c0392b"
            if proj_over is None or correct is None:
                dir_text = "N/A"
            else:
                dir_text = ("✓ OVER" if proj_over else "✓ UNDER") if correct else ("✗ OVER" if proj_over else "✗ UNDER")
            err_str = f"{err:+.1f}" if err is not None else "N/A"

            rows_html += f"""
            <tr style="border-bottom:1px solid #eee;">
                <td style="padding:8px 12px;">{visitor} @ {home}</td>
                <td style="padding:8px 12px;text-align:right;">{vegas if vegas is not None else "N/A"}</td>
                <td style="padding:8px 12px;text-align:right;">{proj if proj is not None else "N/A"}</td>
                <td style="padding:8px 12px;text-align:right;">{actual if actual is not None else "N/A"}</td>
                <td style="padding:8px 12px;text-align:right;">{err_str}</td>
                <td style="padding:8px 12px;text-align:center;color:{dir_cell_color};font-weight:bold;">{dir_text}</td>
            </tr>"""

        sections.append(f"""
        <div style="padding:16px 24px;">
            <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;">Game Total Projections</h2>
            <p style="{_SUMMARY_STYLE_TIGHT}">{summary_line}</p>
            <table style="width:100%;border-collapse:collapse;font-size:13px;">
                <thead>
                    <tr style="background:#f5f5f5;font-weight:bold;">
                        <th style="padding:8px 12px;text-align:left;">Matchup</th>
                        <th style="padding:8px 12px;text-align:right;">Vegas O/U</th>
                        <th style="padding:8px 12px;text-align:right;">Our Proj</th>
                        <th style="padding:8px 12px;text-align:right;">Actual</th>
                        <th style="padding:8px 12px;text-align:right;">Error</th>
                        <th style="padding:8px 12px;text-align:center;">Direction</th>
                    </tr>
                </thead>
                <tbody>{rows_html}</tbody>
            </table>
        </div>
        """)

    # --- Footer ---
    sections.append("""
    <div style="background:#f5f5f5;padding:12px 24px;border-radius:0 0 8px 8px;
                font-size:11px;color:#999;text-align:center;">
        BKS {get_active_config().sport_display_name} Continuous Learning System &middot; Auto-generated report
    </div>
    """)

    body = "\n".join(sections)
    return f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"></head>
<body style="margin:0;padding:20px;background:#f0f0f5;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;">
<div style="max-width:680px;margin:0 auto;background:#fff;border-radius:8px;box-shadow:0 2px 8px rgba(0,0,0,0.08);">
{body}
</div>
</body>
</html>"""
