from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from html import escape
from typing import Any

import requests

from bks_pipeline_core.sport_config import get_active_config


class CheckStatus(str, Enum):
    """Result status of a single health check."""

    PASS = "pass"
    FAIL = "fail"
    SKIP = "skip"


class PipelineStatus(str, Enum):
    """Overall pipeline health derived from individual checks."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    FAILED = "failed"


@dataclass
class CheckResult:
    """Outcome of one health-check probe."""

    name: str
    status: CheckStatus
    detail: str
    critical: bool = False


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO-8601 timestamp that may end with 'Z' or '+00:00'."""
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def _game_count(games_doc: dict[str, Any] | None) -> int:
    """Extract game_count from a games doc, defaulting to 0."""
    if games_doc is None:
        return 0
    return int(games_doc.get("game_count", 0))


def _is_stale(ts_iso: str | None, now: datetime, stale_hours: float) -> bool:
    """Return True when *ts_iso* is more than *stale_hours* before *now*."""
    if ts_iso is None:
        return True
    return (now - _parse_iso(ts_iso)) > timedelta(hours=stale_hours)


# ---------------------------------------------------------------------------
# API key reachability probe
# ---------------------------------------------------------------------------


def check_api_key_reachability(api_key: str, today_str: str) -> CheckResult:
    """Probe BallDontLie with a lightweight request to validate the API key.

    Uses GET /v1/games?dates[]=<today>&per_page=1 — cheapest authenticated call.

    Returns:
        PASS  — 200, key is valid and service is reachable
        FAIL  — 401, key rejected (invalid/expired on our side, or BDL auth service down)
        SKIP  — any other status or exception (network flakiness, don't block health)
    """
    from config import GAMES_URL

    try:
        resp = requests.get(
            GAMES_URL,
            headers={"Authorization": api_key},
            params={"dates[]": today_str, "per_page": "1"},
            timeout=10,
        )
        if resp.status_code == 200:
            return CheckResult("API key", CheckStatus.PASS, "BallDontLie reachable")
        if resp.status_code == 401:
            return CheckResult(
                "API key",
                CheckStatus.FAIL,
                "BallDontLie returned 401 — key invalid or auth service down",
                critical=True,
            )
        return CheckResult("API key", CheckStatus.SKIP, f"unexpected status {resp.status_code}")
    except Exception:
        return CheckResult("API key", CheckStatus.SKIP, "probe request failed")


# ---------------------------------------------------------------------------
# Morning checks
# ---------------------------------------------------------------------------


def evaluate_morning_checks(
    *,
    today: str,
    yesterday: str,
    games_doc: dict[str, Any] | None,
    games_yesterday_doc: dict[str, Any] | None,
    player_count: int,
    max_trend_updated_at: str | None,
    odds_synced_at: str | None,
    actuals_exists: bool,
    accuracy_exists: bool,
    circuit_breaker_doc: dict[str, Any] | None,
    now: datetime,
    stale_hours: float = 8.0,
    min_players: int = 300,
    api_key: str = "",
) -> list[CheckResult]:
    """Run all morning health checks and return a list of results."""
    checks: list[CheckResult] = []

    # 1. Games sync
    if games_doc and games_doc.get("synced_at"):
        checks.append(CheckResult("Games sync", CheckStatus.PASS, f"synced_at present for {today}"))
    else:
        checks.append(
            CheckResult(
                "Games sync",
                CheckStatus.FAIL,
                f"No games doc or synced_at for {today}",
                critical=True,
            )
        )

    # 2. Player sync
    players_ok = player_count > min_players
    trend_fresh = not _is_stale(max_trend_updated_at, now, stale_hours)
    if players_ok and trend_fresh:
        checks.append(CheckResult("Player sync", CheckStatus.PASS, f"{player_count} players, trends fresh"))
    else:
        parts = []
        if not players_ok:
            parts.append(f"only {player_count} players (need >{min_players})")
        if not trend_fresh:
            parts.append("trends stale" if max_trend_updated_at else "no trend timestamp")
        checks.append(CheckResult("Player sync", CheckStatus.FAIL, "; ".join(parts), critical=True))

    # 3. Odds sync
    gc = _game_count(games_doc)
    if odds_synced_at is not None:
        checks.append(CheckResult("Odds sync", CheckStatus.PASS, "odds synced"))
    elif gc == 0:
        checks.append(CheckResult("Odds sync", CheckStatus.SKIP, "no games today"))
    else:
        checks.append(CheckResult("Odds sync", CheckStatus.FAIL, f"{gc} games but odds not synced"))

    # 4. Actuals
    yesterday_gc = _game_count(games_yesterday_doc)
    if actuals_exists:
        checks.append(CheckResult("Actuals", CheckStatus.PASS, "actuals present"))
    elif games_yesterday_doc is None or yesterday_gc == 0:
        checks.append(CheckResult("Actuals", CheckStatus.SKIP, "no games yesterday"))
    else:
        checks.append(
            CheckResult(
                "Actuals",
                CheckStatus.FAIL,
                f"{yesterday_gc} games yesterday but no actuals",
            )
        )

    # 5. Accuracy
    no_actuals_expected = games_yesterday_doc is None or yesterday_gc == 0
    if accuracy_exists:
        checks.append(CheckResult("Accuracy", CheckStatus.PASS, "accuracy report present"))
    elif no_actuals_expected:
        checks.append(CheckResult("Accuracy", CheckStatus.SKIP, "no actuals expected"))
    else:
        checks.append(CheckResult("Accuracy", CheckStatus.FAIL, "actuals exist but no accuracy report"))

    # 6. Circuit breaker
    if circuit_breaker_doc is None:
        checks.append(CheckResult("Circuit breaker", CheckStatus.PASS, "not tripped"))
    else:
        reason = circuit_breaker_doc.get("reason", "unknown")
        checks.append(CheckResult("Circuit breaker", CheckStatus.FAIL, f"tripped: {reason}", critical=True))

    # 7. API key reachability (only when key is available)
    if api_key:
        checks.append(check_api_key_reachability(api_key, today))

    return checks


# ---------------------------------------------------------------------------
# Pre-game checks
# ---------------------------------------------------------------------------


def evaluate_pregame_checks(
    *,
    today: str,
    games_doc: dict[str, Any] | None,
    player_count: int,
    max_trend_updated_at: str | None,
    odds_synced_at: str | None,
    snapshot_exists: bool,
    earliest_tip: datetime | None,
    circuit_breaker_doc: dict[str, Any] | None,
    now: datetime,
    stale_hours: float = 4.0,
    min_players: int = 300,
) -> list[CheckResult]:
    """Run pre-game readiness checks and return a list of results."""
    checks: list[CheckResult] = []

    # 1. Games sync
    if games_doc and games_doc.get("synced_at"):
        checks.append(CheckResult("Games sync", CheckStatus.PASS, f"synced_at present for {today}"))
    else:
        checks.append(
            CheckResult(
                "Games sync",
                CheckStatus.FAIL,
                f"No games doc or synced_at for {today}",
                critical=True,
            )
        )

    # 2. Player sync (tighter staleness window)
    players_ok = player_count > min_players
    trend_fresh = not _is_stale(max_trend_updated_at, now, stale_hours)
    if players_ok and trend_fresh:
        checks.append(CheckResult("Player sync", CheckStatus.PASS, f"{player_count} players, trends fresh"))
    else:
        parts = []
        if not players_ok:
            parts.append(f"only {player_count} players (need >{min_players})")
        if not trend_fresh:
            parts.append("trends stale" if max_trend_updated_at else "no trend timestamp")
        checks.append(CheckResult("Player sync", CheckStatus.FAIL, "; ".join(parts), critical=True))

    # 3. Odds sync
    gc = _game_count(games_doc)
    if odds_synced_at is not None:
        checks.append(CheckResult("Odds sync", CheckStatus.PASS, "odds synced"))
    elif gc == 0:
        checks.append(CheckResult("Odds sync", CheckStatus.SKIP, "no games today"))
    else:
        checks.append(CheckResult("Odds sync", CheckStatus.FAIL, f"{gc} games but odds not synced"))

    # 4. Snapshot
    # Regular snapshot runs at 7 PM ET (23:00 UTC).  Early snapshot runs at 5 PM ET
    # (21:00 UTC) only for games tipping before 7:30 PM ET.  If the current time is
    # before 23:15 UTC (allowing 15 min for the regular snapshot to finish), the
    # snapshot is not yet expected to exist unless the early snapshot covered it.
    snapshot_expected_utc = now.replace(hour=23, minute=15, second=0, microsecond=0)
    if snapshot_exists:
        checks.append(CheckResult("Snapshot", CheckStatus.PASS, "snapshot present"))
    elif gc == 0:
        checks.append(CheckResult("Snapshot", CheckStatus.SKIP, "no games today"))
    elif now < snapshot_expected_utc:
        checks.append(CheckResult("Snapshot", CheckStatus.SKIP, "snapshot runs at 7 PM ET, not yet due"))
    else:
        checks.append(
            CheckResult(
                "Snapshot",
                CheckStatus.FAIL,
                "snapshot missing before tip-off",
                critical=True,
            )
        )

    # 5. Circuit breaker
    if circuit_breaker_doc is None:
        checks.append(CheckResult("Circuit breaker", CheckStatus.PASS, "not tripped"))
    else:
        reason = circuit_breaker_doc.get("reason", "unknown")
        checks.append(CheckResult("Circuit breaker", CheckStatus.FAIL, f"tripped: {reason}", critical=True))

    return checks


# ---------------------------------------------------------------------------
# Status rollup
# ---------------------------------------------------------------------------


def compute_pipeline_status(checks: list[CheckResult]) -> PipelineStatus:
    """Derive overall pipeline status from individual check results."""
    for c in checks:
        if c.status == CheckStatus.FAIL and c.critical:
            return PipelineStatus.FAILED
    for c in checks:
        if c.status == CheckStatus.FAIL:
            return PipelineStatus.DEGRADED
    return PipelineStatus.HEALTHY


# ---------------------------------------------------------------------------
# Scheduling helper
# ---------------------------------------------------------------------------


def compute_pregame_schedule_time(
    game_datetimes: list[datetime],
    now: datetime,
    lead_minutes: int = 60,
) -> datetime | None:
    """Return the UTC time to trigger the pre-game check.

    Finds the earliest future game and subtracts *lead_minutes*.
    Returns ``None`` when there are no future games.
    """
    future = [dt for dt in game_datetimes if dt > now]
    if not future:
        return None
    return min(future) - timedelta(minutes=lead_minutes)


# ---------------------------------------------------------------------------
# Alert email
# ---------------------------------------------------------------------------

_STATUS_COLORS: dict[PipelineStatus, str] = {
    PipelineStatus.FAILED: "#d32f2f",
    PipelineStatus.DEGRADED: "#ef6c00",
    PipelineStatus.HEALTHY: "#2e7d32",
}

_CHECK_COLORS: dict[CheckStatus, str] = {
    CheckStatus.PASS: "#2e7d32",
    CheckStatus.FAIL: "#d32f2f",
    CheckStatus.SKIP: "#9e9e9e",
}

_PHASE_LABELS: dict[str, str] = {
    "morning": "Morning Check",
    "pregame": "Pre-Game Check",
}

# Descriptions shown in the "Check Reference" box at the bottom of alert emails.
_CHECK_DESCRIPTIONS: dict[str, str] = {
    "Games sync": "Verifies the games document for today exists and has a synced_at timestamp from the schedule sync.",
    "Player sync": "Confirms enough players are loaded and trend data is fresh (not stale beyond the threshold).",
    "Odds sync": "Checks that Vegas odds have been synced for today's games (implied team totals, spreads, over/under).",
    "Snapshot": "Verifies a pre-game prediction snapshot exists for backtesting. Runs at 7 PM ET; early snapshot at 5 PM ET for early tips.",
    "Circuit breaker": "Checks whether the circuit breaker has been tripped by repeated pipeline failures. A tripped breaker halts automated runs.",
    "Yesterday actuals": "Confirms that actual box-score stats were captured for yesterday's games (needed for accuracy tracking).",
    "Accuracy report": "Checks that the daily accuracy report was generated comparing predictions against actuals.",
    "API reachability": "Probes the health endpoint to verify the API is responding to authenticated requests.",
}


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


def generate_alert_html(
    status: PipelineStatus,
    phase: str,
    checks: list[CheckResult],
    now: datetime,
    league_state: dict[str, Any] | None = None,
) -> str:
    """Build an HTML email body summarising pipeline health."""
    bg = _STATUS_COLORS.get(status, "#616161")
    label = _PHASE_LABELS.get(phase, phase)
    ts = now.strftime("%Y-%m-%d %H:%M UTC")

    rows = ""
    for c in checks:
        color = _CHECK_COLORS.get(c.status, "#000")
        bold = "font-weight:bold;" if c.status == CheckStatus.FAIL else ""
        rows += (
            f'<tr style="color:{color};{bold}">'
            f'<td style="padding:6px 12px">{escape(c.name)}</td>'
            f'<td style="padding:6px 12px">{c.status.value.upper()}</td>'
            f'<td style="padding:6px 12px">{escape(c.detail)}</td>'
            f"</tr>"
        )

    # Build system status block
    mode_label = _league_mode_label(league_state)
    cb_check = next((c for c in checks if "circuit" in c.name.lower()), None)
    if cb_check and cb_check.status == CheckStatus.FAIL:
        cb_color = "#e65100"
        cb_label = f"&#9888; Tripped — {escape(cb_check.detail)}"
    else:
        cb_color = "#2e7d32"
        cb_label = "Clear"

    system_status_html = (
        f'<table style="width:100%;border-collapse:collapse;margin-bottom:16px;font-size:14px">'
        f'<tr style="background:#f5f5f5"><td colspan="2" style="padding:6px 12px;font-weight:bold">System Status</td></tr>'
        f'<tr><td style="padding:4px 12px;color:#555">Mode</td>'
        f'<td style="padding:4px 12px;font-weight:bold">{escape(mode_label)}</td></tr>'
        f'<tr><td style="padding:4px 12px;color:#555">Circuit Breaker</td>'
        f'<td style="padding:4px 12px;font-weight:bold;color:{cb_color}">{cb_label}</td></tr>'
        f"</table>"
    )

    # Context reference — explains what each check does.
    check_names = {c.name for c in checks}
    context_rows = ""
    for name, desc in _CHECK_DESCRIPTIONS.items():
        if name in check_names:
            context_rows += (
                f'<tr><td style="padding:4px 12px;font-weight:bold;color:#333;white-space:nowrap">{escape(name)}</td>'
                f'<td style="padding:4px 12px;color:#555">{escape(desc)}</td></tr>'
            )
    context_html = (
        (
            f'<table style="width:100%;border-collapse:collapse;margin-top:16px;font-size:13px;'
            f'border:1px solid #e0e0e0;border-radius:4px">'
            f'<tr style="background:#f5f5f5"><td colspan="2" style="padding:6px 12px;font-weight:bold">'
            f"Check Reference</td></tr>"
            f"{context_rows}</table>"
        )
        if context_rows
        else ""
    )

    return (
        f'<div style="font-family:Arial,sans-serif;max-width:600px;margin:auto">'
        f'<div style="background:{bg};color:#fff;padding:16px 20px;border-radius:6px 6px 0 0">'
        f'<h2 style="margin:0">BKS {get_active_config().sport_display_name} Pipeline Alert</h2>'
        f'<span style="font-size:14px">{status.value.upper()}</span>'
        f"</div>"
        f'<div style="padding:16px 20px;border:1px solid #ddd;border-top:none">'
        f"<p><strong>Phase:</strong> {escape(label)}</p>"
        f"{system_status_html}"
        f'<table style="width:100%;border-collapse:collapse">'
        f'<tr style="background:#f5f5f5;font-weight:bold">'
        f'<td style="padding:6px 12px">Check</td>'
        f'<td style="padding:6px 12px">Status</td>'
        f'<td style="padding:6px 12px">Detail</td></tr>'
        f"{rows}</table>"
        f"{context_html}"
        f'<p style="color:#999;font-size:12px;margin-top:16px">{ts}</p>'
        f"</div></div>"
    )
