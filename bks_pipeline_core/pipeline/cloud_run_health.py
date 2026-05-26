"""Cloud Run service health check.

Calls the Cloud Run Admin REST API and returns any services that are not in
Ready state, plus any expected services that are missing entirely.

Pure function — no Firestore I/O.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# Canonical list of Cloud Run service names (hyphen-cased) that must exist.
# Firebase converts snake_case function names → hyphen-case service names.
EXPECTED_SERVICES: frozenset[str] = frozenset(
    [
        "apple-server-notification",
        "assign-insider-tier",
        "capture-actuals",
        "check-pipeline-morning",
        "check-subscription-expirations",
        "checkpipelinepregame",
        "compute-accuracy",
        "get-activity-feed",
        "get-daily-analysis",
        "get-league-state",
        "get-opportunities",
        "get-players",
        "get-playoff-bracket",
        "get-projections",
        "get-subscription-status",
        "get-user-preferences",
        "google-rtdn-notification",
        "health",
        "lineupcheck",
        "manage-promo-codes",
        "pregamefreshnesscheck",
        "prewarm-daily-analysis",
        "reconcile-pregame-tasks",
        "record-series-result",
        "redeem-promo-code",
        "refit-platt-calibration",
        "resolve-daily-props",
        "retrysyncodds",
        "retrytrendsync",
        "run-comparison",
        "run-fetch-actuals",
        "run-predictions",
        "snapshot-predictions",
        "morning-snapshot",
        "snapshot-predictions-early",
        "snapshot-prop-predictions",
        "sync-active-players",
        "sync-injury-status-weekday",
        "sync-injury-status-weekend",
        "sync-market-data",
        "sync-today-games",
        "transition-season-mode",
        "update-user-preferences",
        "validate-apple-receipt",
        "validate-google-receipt",
        "weekly-backtest-report",
    ]
)


@dataclass
class ServiceIssue:
    name: str
    issue: str  # "not_ready" | "missing"
    detail: str  # human-readable state or error message


def check_cloud_run_health(
    project: str | None = None,
    region: str = "us-central1",
) -> list[ServiceIssue]:
    """Return a list of unhealthy or missing Cloud Run services.

    Returns an empty list when all expected services are Ready.
    Returns a single-item error list if the API call itself fails.
    """
    import google.auth
    import google.auth.transport.requests
    import requests as http

    if not project:
        project = os.environ.get("GCLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    if not project:
        logger.warning("cloud_run_health: no project ID available, skipping")
        return [ServiceIssue(name="(unknown)", issue="not_ready", detail="Project ID not available in env")]

    try:
        credentials, _ = google.auth.default(scopes=["https://www.googleapis.com/auth/cloud-platform"])
        auth_req = google.auth.transport.requests.Request()
        credentials.refresh(auth_req)
        token = credentials.token
    except Exception as exc:
        logger.warning("cloud_run_health: auth error — %s", exc)
        return [ServiceIssue(name="(auth)", issue="not_ready", detail=f"Auth error: {exc}")]

    url = f"https://run.googleapis.com/v2/projects/{project}/locations/{region}/services"
    headers = {"Authorization": f"Bearer {token}"}

    try:
        resp = http.get(url, headers=headers, timeout=15)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
    except Exception as exc:
        logger.warning("cloud_run_health: API error — %s", exc)
        return [ServiceIssue(name="(api)", issue="not_ready", detail=f"API error: {exc}")]

    services: list[dict[str, Any]] = data.get("services", [])

    # name field is "projects/.../locations/.../services/<svc-name>"
    found: dict[str, str] = {}  # short-name → ready state
    for svc in services:
        short_name = svc.get("name", "").rsplit("/", 1)[-1]
        # v2 API: Ready condition lives in terminalCondition, not the conditions array.
        terminal = svc.get("terminalCondition", {})
        if terminal.get("type") == "Ready":
            state = terminal.get("state", "unknown")
        else:
            # Fallback: scan conditions array (future-proof)
            conditions: list[dict[str, Any]] = svc.get("conditions", [])
            ready_cond = next((c for c in conditions if c.get("type") == "Ready"), None)
            state = ready_cond.get("state", "unknown") if ready_cond else "unknown"
        found[short_name] = state

    issues: list[ServiceIssue] = []

    for svc_name in sorted(EXPECTED_SERVICES):
        if svc_name not in found:
            issues.append(ServiceIssue(name=svc_name, issue="missing", detail="Service not found in Cloud Run"))
        elif found[svc_name] != "CONDITION_SUCCEEDED":
            issues.append(ServiceIssue(name=svc_name, issue="not_ready", detail=f"State: {found[svc_name]}"))

    return issues
