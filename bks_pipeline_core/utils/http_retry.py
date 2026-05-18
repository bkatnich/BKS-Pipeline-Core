import logging
import random
import time
from typing import Any

import requests
import requests.exceptions

_default_logger = logging.getLogger(__name__)


class CredentialError(Exception):
    """Raised when an API call returns 401 after retry — indicates invalid/expired key."""


def request_with_backoff(
    url: str,
    *,
    headers: dict[str, str] | None = None,
    params: list[tuple[str, Any]] | dict[str, Any] | None = None,
    max_attempts: int = 3,
    max_wait: int | None = 60,
    max_wait_connection: int = 10,
    timeout: int = 30,
    logger: logging.Logger | None = None,
) -> requests.Response | None:
    """GET `url` with exponential backoff on 429, 5xx, and connection errors.

    Retries up to `max_attempts` times. On HTTP 429 or server errors (500-504),
    sleeps 2^(attempt+1) seconds capped at `max_wait`. On connection errors
    (dropped connections, SSL resets), caps at `max_wait_connection` instead —
    these resolve quickly and don't need long waits. Returns the Response on
    success, or None if all attempts are exhausted.

    Note: time.sleep() blocks the Cloud Function thread — budget accordingly
    against your function's timeout_sec.

    Callers are responsible for:
    - Calling .raise_for_status() on the returned response for non-retryable HTTP errors.
    - Deciding failure semantics (raise, return 0, increment circuit breaker, etc.).
    """
    _RETRYABLE_STATUSES = {429, 500, 502, 503, 504}
    log = logger or _default_logger
    for attempt in range(max_attempts):
        try:
            response = requests.get(url, headers=headers, params=params, timeout=timeout)
            if response.status_code in _RETRYABLE_STATUSES:
                wait = 2 ** (attempt + 1)
                wait += random.uniform(0, wait * 0.1)
                if max_wait is not None:
                    wait = min(wait, max_wait)
                log.warning(
                    "%d from %s, retrying in %.1fs (attempt %d/%d)",
                    response.status_code,
                    url,
                    wait,
                    attempt + 1,
                    max_attempts,
                )
                time.sleep(wait)
                continue
            # Single retry on 401 — catches transient auth service blips.
            # Only fires on the first attempt; a second 401 is returned as-is.
            if response.status_code == 401 and attempt == 0:
                log.warning("401 Unauthorized from %s, retrying once in 5s", url)
                time.sleep(5)
                continue
            return response
        except requests.exceptions.RequestException as exc:
            wait = 2 ** (attempt + 1)
            wait += random.uniform(0, wait * 0.1)
            # Connection resets resolve quickly — cap tighter than rate-limit waits.
            wait = min(wait, max_wait_connection)
            log.warning(
                "Request error from %s (%s), retrying in %.1fs (attempt %d/%d)",
                url,
                exc,
                wait,
                attempt + 1,
                max_attempts,
            )
            time.sleep(wait)
    return None
