"""Generate an HTML email report from daily accuracy metrics.

Pure function — takes an accuracy dict (and optional rolling data)
and returns an HTML string. No I/O.
"""

from html import escape
from typing import Any

from bks_pipeline_core.sport_config import get_active_config

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

# Signals hardcoded to 1.0 unconditionally — excluded from compute_signal_accuracy()
# to avoid zero-variance noise in the scorecard.
_DISABLED_SIGNAL_FIELDS: set[str] = {
    "pace_multiplier",  # disabled 2026-04-10, r=-0.164, 100% fire rate
    "cat_trend_multiplier",  # disabled 2026-04-20, r=-0.156, negative in playoffs
    "venue_multiplier",  # disabled 2026-04-20, r=-0.060, flat home/away premium
    "costar_multiplier",  # code removed 2026-04-27; guard for stale snapshot values
    "elimination_game_multiplier",  # disabled 2026-05-10, r=-0.031, 77% fire rate
}
# Public alias so callers (e.g. backtesting.compute_signal_accuracy) can import it.
DISABLED_SIGNAL_FIELDS = _DISABLED_SIGNAL_FIELDS

# Signals hardcoded to 1.0 only during playoffs.  Active in regular season and
# should be evaluated then, but produce zero-variance deviations in playoff
# snapshots — computing r on them yields None or a stale rolling value.
PLAYOFF_DISABLED_SIGNAL_FIELDS: set[str] = {
    "vegas_multiplier",  # r=-0.237, disabled 2026-04-28
    "stacking_multiplier",  # r=-0.120, disabled in playoffs
    "mean_reversion_multiplier",  # r=-0.093, disabled in playoffs
    "matchup_multiplier",  # r=-0.128, disabled in playoffs
    "usage_delta_multiplier",  # r=-0.063, disabled in playoffs
    "role_change_multiplier",  # r=-0.061, disabled in playoffs
    "shooting_luck_multiplier",  # dampened to ~1.0 early playoffs
    "minutes_env_multiplier",  # r=-0.205, excluded from signal_product in playoffs
    "line_movement_multiplier",  # hardcoded 1.0 in playoffs (series lines ≠ per-game flow)
}


def _get_signal_display_names() -> dict[str, str]:
    """Return signal display names merged with any sport-config overrides."""
    try:
        overrides = get_active_config().signal_display_names or {}
    except RuntimeError:
        overrides = {}
    return {**_SIGNAL_DISPLAY_NAMES, **overrides}


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


def _summary_signal(signal_accuracy: dict[str, Any]) -> str:
    """Narrative summary for the Signal Scorecard section."""
    if not signal_accuracy:
        return ""

    corrs: dict[str, float] = {}
    for sig, data in signal_accuracy.items():
        if sig in _DISABLED_SIGNAL_FIELDS and (data.get("fire_rate") or 0.0) == 0.0:
            continue
        r = data.get("residual_correlation")
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
    _display = _get_signal_display_names()
    best_name = _display.get(best_sig, best_sig)
    worst_name = _display.get(worst_sig, worst_sig)

    parts = [f"{positive} of {total} active signals correlate positively with outcomes."]
    parts.append(f"Strongest: {best_name} (r={best_r:+.3f}).")
    if negative > 0:
        parts.append(f"Watch: {negative} signal(s) showing negative correlation — {worst_name} (r={worst_r:+.3f}) is working against the model.")

    text = " ".join(parts)
    return f'<p style="{_SUMMARY_STYLE}">{text}</p>'


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


def _summary_rolling(rolling_7d: dict[str, Any], today_signal_accuracy: dict[str, Any]) -> str:
    """Narrative summary for the 7-Day Rolling Trend section."""
    r7_sigs = rolling_7d.get("signal_accuracy", {})
    improved = sum(1 for s in r7_sigs if (r7_sigs[s].get("residual_correlation") or 0) > (today_signal_accuracy.get(s, {}).get("residual_correlation") or 0))
    degraded = sum(1 for s in r7_sigs if (r7_sigs[s].get("residual_correlation") or 0) < (today_signal_accuracy.get(s, {}).get("residual_correlation") or 0))

    parts: list[str] = ["7-day rolling signal accuracy:"]
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
    signal_accuracy: dict[str, Any] = accuracy.get("signal_accuracy", {})

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

    # --- Section 1: Signal Scorecard ---
    _is_playoffs_mode = (league_state or {}).get("mode") == "playoffs"
    _all_disabled = _DISABLED_SIGNAL_FIELDS | (PLAYOFF_DISABLED_SIGNAL_FIELDS if _is_playoffs_mode else set())

    signal_rows_sorted = sorted(
        signal_accuracy.items(),
        key=lambda x: x[1].get("residual_correlation") or -999,
        reverse=True,
    )

    signal_rows_html = ""
    for sig, data in signal_rows_sorted:
        fire_rate = data.get("fire_rate") or 0.0
        is_zero_fire = fire_rate == 0.0

        if is_zero_fire and sig in _DISABLED_SIGNAL_FIELDS:
            continue

        display_name = _get_signal_display_names().get(sig, sig)

        if is_zero_fire and sig in _all_disabled:
            signal_rows_html += f"""
        <tr style="color:#bbb;">
            <td style="padding:6px 8px;">{display_name} <span style="font-size:10px;">(disabled)</span></td>
            <td style="text-align:center;color:#bbb;">—</td>
            <td style="text-align:center;color:#bbb;">—</td>
            <td style="text-align:center;">0.0%</td>
        </tr>
        """
            continue

        corr = data.get("residual_correlation")
        color = _color_for_correlation(corr)
        hit_rate = data.get("hit_rate")
        penalty_hit_rate = data.get("penalty_hit_rate")
        if hit_rate is not None:
            hit_cell = f'<td style="text-align:center;">{_pct(hit_rate)}</td>'
        elif penalty_hit_rate is not None:
            hit_cell = f'<td style="text-align:center;">{_pct(penalty_hit_rate)} <span style="font-size:10px;color:#999;">&#9660;</span></td>'
        else:
            hit_cell = '<td style="text-align:center;">N/A</td>'

        signal_rows_html += f"""
        <tr>
            <td style="padding:6px 8px;">{display_name}</td>
            <td style="text-align:center;color:{color};font-weight:bold;">{_fmt(corr)}</td>
            {hit_cell}
            <td style="text-align:center;">{_pct(fire_rate)}</td>
        </tr>
        """

    sections.append(f"""
    <div style="padding:16px 24px;">
        <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;">
            Signal Scorecard
        </h2>
        {_summary_signal(signal_accuracy)}
        <table style="width:100%;border-collapse:collapse;font-size:13px;">
            <tr style="background:#f5f5f5;">
                <th style="padding:8px;text-align:left;">Signal</th>
                <th style="text-align:center;">Correlation r</th>
                <th style="text-align:center;">Hit%</th>
                <th style="text-align:center;">Fire Rate</th>
            </tr>
            {signal_rows_html}
        </table>
        <p style="font-size:11px;color:#999;margin-top:6px;">
            Correlation: r&gt;0.05 = <span style="color:#2d7d46;">good</span>,
            |r|&lt;0.05 = <span style="color:#d4a017;">noise</span>,
            r&lt;-0.05 = <span style="color:#c0392b;">wrong direction</span>.
            &#9660; = penalty-only signal (hit rate = penalized players who underperformed).
            Signals disabled unconditionally are excluded. Signals disabled in playoffs shown greyed as &#8220;disabled&#8221;.
        </p>
    </div>
    """)

    # --- Section 2: Per-Stat Prediction Accuracy ---
    stat_accuracy = accuracy.get("stat_accuracy", {})
    if stat_accuracy:
        stat_order = list(stat_accuracy.keys())
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

    # --- Section 3: 7-Day Rolling Trend ---
    if rolling_7d and rolling_7d.get("days", 0) >= 3:
        r7_signals: dict[str, Any] = rolling_7d.get("signal_accuracy", {})

        trend_rows = ""
        for sig, data in sorted(
            r7_signals.items(),
            key=lambda x: x[1].get("residual_correlation") or -999,
            reverse=True,
        ):
            _display_names = _get_signal_display_names()
            if sig not in _display_names:
                continue
            corr_7d = data.get("residual_correlation")
            daily_corr = signal_accuracy.get(sig, {}).get("residual_correlation")
            if corr_7d is None:
                continue
            if sig in _DISABLED_SIGNAL_FIELDS and (data.get("fire_rate") or 0.0) == 0.0:
                continue
            delta = (corr_7d - daily_corr) if daily_corr is not None else None
            delta_str = f"{delta:+.3f}" if delta is not None else "N/A"
            display_name = _display_names.get(sig, sig)
            trend_rows += f"""
            <tr>
                <td style="padding:4px 8px;">{display_name}</td>
                <td style="text-align:center;">{_fmt(corr_7d)}</td>
                <td style="text-align:center;">{delta_str}</td>
            </tr>
            """

        sections.append(f"""
        <div style="padding:16px 24px;">
            <h2 style="color:#1a1a2e;border-bottom:2px solid #eee;padding-bottom:8px;">
                7-Day Rolling Trend &middot; {rolling_7d.get("days", 0)} days
            </h2>
            {_summary_rolling(rolling_7d, signal_accuracy)}
            <table style="width:100%;border-collapse:collapse;font-size:13px;">
                <tr style="background:#f5f5f5;">
                    <th style="padding:6px;text-align:left;">Signal</th>
                    <th style="text-align:center;">7d Correlation</th>
                    <th style="text-align:center;">vs Today</th>
                </tr>
                {trend_rows}
            </table>
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
    sections.append(f"""
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
