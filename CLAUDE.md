# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Install package with dev dependencies (Python 3.13+ required)
pip install -e .[dev]

# Lint
ruff check .

# Run tests
pytest

# Run a single test
pytest tests/path/to/test_file.py::test_function_name
```

## Architecture

**BKS-Pipeline-Core** is a shared Python library consumed by BKS (BlackKatt Sports) backend services. It provides reusable components for multi-sport DFS (Daily Fantasy Sports) analytics pipelines, subscription gating, and Firebase/GCP integration.

### Module Map

- **`auth.py`** — Firebase auth decorators: `@require_auth`, `@require_admin`, `@require_tier()`. Auth failures return 401/403; subscription failures return 402/403. On Firestore errors, auth degrades gracefully to trial tier rather than hard-blocking.

- **`models/user.py`** — `UserDoc` dataclass representing a subscriber. Tier hierarchy: `expired < trial < basic < pro < premium < insider`. `effective_tier()` accounts for time-based expiration. Serializes to/from Firestore via `to_firestore()` / `from_firestore()`.

- **`sport_config/`** — Sport-agnostic runtime configuration. `SportConfig` holds all sport-specific constants (scoring weights per DFS platform, position taxonomy, stat categories, tier percentiles). Use `get_active_config()` / `set_active_config()` to swap sport context at startup without changing pipeline code.

- **`pipeline/`** — Core analytics modules: prediction vs actuals (`backtesting.py`, `accuracy_report.py`), Firestore persistence (`actuals_store.py`, `activity_log.py`), data models (`games.py`, `injuries.py`, `defense.py`), stat utilities (`mean_reversion.py`, `calibration_store.py`), and service health (`health_check.py`, `cloud_run_health.py`).

- **`iap.py`** — In-App Purchase receipt validation for Apple App Store and Google Play. Maps product IDs to subscription tiers. Server-to-server webhook handlers included. Full Apple API v2 (JWS) and Google Play service account validation are not yet implemented.

- **`entitlements.py`** — Controls which response fields each subscription tier can see.

- **`utils/http_retry.py`** — HTTP client with exponential backoff + jitter. Retries on 429, 5xx, and connection errors.

### Related Repos (local)

- **Basketball server**: `/Users/Britton/Documents/Repositories/BlackKatt/Basketball/BKS-Basketball-Server-Firebase`
- **Baseball server**: `/Users/Britton/Documents/Repositories/BlackKatt/Baseball/BKS-Baseball-Server-Firebase`

These are the Firebase/Cloud Functions backends that consume this package.

### Key Patterns

- **Sport abstraction**: All sport-specific constants live in `SportConfig`; pipeline logic stays generic and reads from `get_active_config()`.
- **Firestore serialization**: Data models use `to_firestore()` / `from_firestore()` for direct Firestore document mapping.
- **Firebase/GCP stack**: Firestore for persistence, Firebase Auth for identity, Cloud Functions framework, Google Cloud Tasks for queuing.
- **Claude API**: `anthropic` SDK is a dependency — AI features are part of the pipeline.
