"""Generic player/games/odds orchestration for multi-sport Firebase pipelines.

All sport-specific behaviour is injected via PlayerSyncHooks and GameSyncHooks
dataclasses.  The concrete provider objects (StatsProvider, OddsProvider,
LineupsProvider) and runtime configuration values are passed as explicit
parameters so this module has zero imports from any sport project's config.py.

Usage in a sport project's pipeline/orchestrator.py:

    from bks_pipeline_core.pipeline.orchestrator import (
        OrchestratorConfig,
        PlayerSyncHooks,
        GameSyncHooks,
        fetch_and_store_players as _core_fetch_and_store_players,
        fetch_and_store_odds as _core_fetch_and_store_odds,
        fetch_and_store_today_games,
        fetch_and_store_upcoming_games,
        fetch_and_store_espn_event_ids,
        fetch_and_store_lineup_status,
        enqueue_trend_retry,
    )
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Protocol

from google.cloud import tasks_v2
from google.protobuf import timestamp_pb2  # type: ignore[import-untyped]

from bks_pipeline_core.pipeline.defense import compute_team_defense
from bks_pipeline_core.pipeline.games import write_games
from bks_pipeline_core.pipeline.league_state import get_league_state
from bks_pipeline_core.pipeline.platforms import PLATFORMS
from bks_pipeline_core.pipeline.playoffs import get_all_series
from bks_pipeline_core.pipeline.tiers import assign_percentile_tiers
from bks_pipeline_core.pipeline.vegas import compute_vegas_signals
from bks_pipeline_core.sport_config import get_active_config
from bks_pipeline_core.utils.exceptions import TrendFetchAbortedError

logger = logging.getLogger(__name__)

_SYSTEM_COLLECTION = "system"


# ---------------------------------------------------------------------------
# Runtime config bundle — passed to core functions instead of config.py imports
# ---------------------------------------------------------------------------


@dataclass
class OrchestratorConfig:
    """Runtime configuration values for the orchestration functions.

    Decouples core pipeline logic from sport project config.py imports.
    Build one instance from your sport project's config constants and pass it
    to fetch_and_store_players / fetch_and_store_odds / etc.
    """

    stats_api_key: str
    the_odds_api_key: str
    firestore_database_id: str
    trend_stale_hours: float
    checkpoint_batch_size: int
    lineup_lead_minutes: int


# ---------------------------------------------------------------------------
# Provider protocol stubs — forward-declares the interface the core uses.
# Concrete implementations live in each sport project's api/ package.
# ---------------------------------------------------------------------------


class _StatsProviderProto(Protocol):
    def fetch_active_rosters(self, logger: logging.Logger) -> frozenset[int]: ...

    def fetch_player_details(self, person_ids: frozenset[int], logger: logging.Logger) -> list[dict[str, Any]]: ...

    def fetch_games_for_date(self, date_str: str, api_key: str, logger: logging.Logger) -> list[dict[str, Any]]: ...


class _OddsProviderProto(Protocol):
    def fetch_odds(self, api_key: str, logger: logging.Logger) -> list[dict[str, Any]]: ...


class _LineupsProviderProto(Protocol):
    def fetch_event_ids(self, logger: logging.Logger) -> list[dict[str, Any]]: ...

    def fetch_game_lineups(self, event_id: str, logger: logging.Logger) -> dict[str, Any]: ...


# ---------------------------------------------------------------------------
# Hook interfaces
# ---------------------------------------------------------------------------


class _EnrichPlayersHook(Protocol):
    """Enrich all_players in place (add external IDs, headshot URLs, etc.)."""

    def __call__(
        self,
        all_players: list[dict[str, Any]],
        existing_docs: dict[str, dict[str, Any]],
        db: Any,
        logger: logging.Logger,
    ) -> None: ...


class _FetchSeasonAndAdvancedHook(Protocol):
    """Fetch season averages and advanced metrics for a list of stale player IDs.

    Returns (season_data, advanced_metrics) where both are {player_id: {field: value}}.
    Return ({}, {}) to skip.
    """

    def __call__(
        self,
        stale_ids: list[int],
        api_key: str,
        logger: logging.Logger,
    ) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]: ...


class _FetchTrendsHook(Protocol):
    """Fetch trend data for a batch of player IDs.

    Returns (trends_by_player_id, raw_stat_rows).
    Raises TrendFetchAbortedError when the circuit breaker trips.
    """

    def __call__(
        self,
        player_ids: list[int],
        api_key: str,
    ) -> tuple[dict[int, dict[str, Any]], list[dict[str, Any]]]: ...


def _noop_enrich(
    all_players: list[dict[str, Any]],
    existing_docs: dict[str, dict[str, Any]],
    db: Any,
    logger: logging.Logger,
) -> None:
    pass


def _noop_fetch_season_and_advanced(
    stale_ids: list[int],
    api_key: str,
    logger: logging.Logger,
) -> tuple[dict[int, dict[str, Any]], dict[int, dict[str, Any]]]:
    return {}, {}


def _passthrough_game_totals(
    matchups: list[dict[str, Any]],
    signals: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    """Default game totals hook: return matchups unchanged."""
    return list(matchups)


class _GameTotalsHook(Protocol):
    """Enrich game objects with projected totals / sport-specific picks."""

    def __call__(
        self,
        matchups: list[dict[str, Any]],
        signals: dict[str, Any] | None,
    ) -> list[dict[str, Any]]: ...


@dataclass
class PlayerSyncHooks:
    """Sport-specific dependency injection for fetch_and_store_players.

    All callables are invoked during fetch_and_store_players and must be pure
    or I/O-only (no cross-sport imports at module level — use lazy imports inside
    the callable to avoid import errors in test environments).
    """

    enrich_players: _EnrichPlayersHook = field(default_factory=lambda: _noop_enrich)
    fetch_season_and_advanced: _FetchSeasonAndAdvancedHook = field(default_factory=lambda: _noop_fetch_season_and_advanced)
    fetch_trends: _FetchTrendsHook | None = None

    # Returns the frozenset of all trend field names written per player.
    # Used in Phase 6b to strip trend fields from fresh players.
    # Must be provided when fetching trends; None is only valid if fetch_trends is None.
    build_trend_fields: Any | None = None  # Callable[[], frozenset[str]] | None

    # Returns the tuple of field names included in the change-detection hash.
    # Must be provided when fetching trends; None is only valid if fetch_trends is None.
    build_hash_fields: Any | None = None  # Callable[[], tuple[str, ...]] | None

    # Returns the sport-specific default trend dict (for players with no trend data).
    # Callable[[SportConfig], dict[str, Any]]
    build_default_trend: Any | None = None  # Callable[[SportConfig], dict[str, Any]] | None


@dataclass
class GameSyncHooks:
    """Sport-specific dependency injection for fetch_and_store_odds."""

    compute_game_totals: _GameTotalsHook = field(default_factory=lambda: _passthrough_game_totals)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _write_circuit_breaker_state(db: Any, consecutive_failures: int, retry_enqueued_at: str) -> None:
    """Write circuit breaker trip state to system/circuit_breaker in Firestore."""
    try:
        db.collection(_SYSTEM_COLLECTION).document("circuit_breaker").set(
            {
                "tripped_at": datetime.now(timezone.utc).isoformat(),
                "consecutive_failures": consecutive_failures,
                "last_retry_enqueued_at": retry_enqueued_at,
            }
        )
        logger.info("Circuit breaker state written to Firestore")
    except Exception as exc:
        logger.warning("Failed to write circuit breaker state: %s", exc)


def enqueue_trend_retry(delay_seconds: int = 3600) -> None:
    """Enqueue a one-off Cloud Tasks task to retry the trend sync after a delay.

    Uses the Cloud Tasks queue that backs the `retryTrendSync` Firebase task
    queue function. Project ID and region are read from Cloud Run env vars.
    Failures are logged but not re-raised — a missed retry is non-fatal.
    """
    project = os.environ.get("GCLOUD_PROJECT") or os.environ.get("GOOGLE_CLOUD_PROJECT")
    region = os.environ.get("FUNCTION_REGION")
    if not region:
        firebase_config = os.environ.get("FIREBASE_CONFIG", "{}")
        try:
            region = json.loads(firebase_config).get("locationId", "us-central1")
        except (json.JSONDecodeError, AttributeError):
            region = "us-central1"

    if not project:
        logger.warning("Cannot enqueue trend retry: GCLOUD_PROJECT env var not set")
        return

    queue_name = f"projects/{project}/locations/{region}/queues/retryTrendSync"
    function_url = os.environ.get("RETRY_TREND_SYNC_URL")
    if not function_url:
        logger.warning("Cannot enqueue trend retry: RETRY_TREND_SYNC_URL env var not set")
        return

    schedule_time = datetime.now(timezone.utc) + timedelta(seconds=delay_seconds)
    ts = timestamp_pb2.Timestamp()
    ts.FromDatetime(schedule_time)

    task = tasks_v2.Task(
        http_request=tasks_v2.HttpRequest(
            http_method=tasks_v2.HttpMethod.POST,
            url=function_url,
            headers={"Content-Type": "application/json"},
            body=json.dumps({"data": {}}).encode(),
            oidc_token=tasks_v2.OidcToken(
                service_account_email=f"{project}@appspot.gserviceaccount.com",
            ),
        ),
        schedule_time=ts,
    )

    try:
        client = tasks_v2.CloudTasksClient()
        client.create_task(parent=queue_name, task=task)
        logger.info(
            "Enqueued trend retry task in queue %s, scheduled for %s",
            queue_name,
            schedule_time.isoformat(),
        )
    except Exception as exc:
        logger.warning("Failed to enqueue trend retry task: %s", exc)


def _trend_hash(trend_data: dict[str, Any], hash_fields: tuple[str, ...]) -> str:
    """Return a short SHA-256 hex digest of the trend fields for change detection."""
    payload = {k: trend_data.get(k) for k in hash_fields}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Core orchestration functions
# ---------------------------------------------------------------------------


def fetch_and_store_players(
    db: Any,
    cfg_obj: OrchestratorConfig,
    stats_provider: _StatsProviderProto,
    hooks: PlayerSyncHooks | None = None,
) -> int:
    """Fetch all active players from the stats API and upsert into Firestore.

    Orchestrates:
      1. Fetch active player list (cursor-paginated via stats_provider)
      2. Enrich with external IDs / headshots (via hooks.enrich_players)
      3. Identify stale vs. fresh players (cfg_obj.trend_stale_hours threshold)
      4. Fetch trends for stale players (via hooks.fetch_trends, with circuit breaker)
      4b. Compute team defense ratings from raw stat rows
      4c. Fetch season averages + advanced metrics (via hooks.fetch_season_and_advanced)
      5. Assign league-wide percentile tiers
      5b. Tag is_playoff_active per player
      6a. Write stale players (full data, skip unchanged via hash)
      6b. Write fresh players (base fields + updated player_tier only)
      7. Mark players no longer in the active set as is_active=False

    Returns the total number of players written (stale + fresh).
    """
    _hooks = hooks or PlayerSyncHooks()
    api_key = cfg_obj.stats_api_key
    checkpoint_batch_size = cfg_obj.checkpoint_batch_size
    trend_stale_hours = cfg_obj.trend_stale_hours

    run_start = datetime.now(timezone.utc)

    def _elapsed() -> float:
        return (datetime.now(timezone.utc) - run_start).total_seconds()

    players_ref = db.collection("players")

    # --- Phase 1: delta roster sync ---
    # Fetch the canonical set of active person IDs from the authoritative source.
    # Compare against the last-known set stored in system/roster_state to compute
    # adds and removes. Only fetch full player details for new arrivals.
    current_person_ids: frozenset[int] = stats_provider.fetch_active_rosters(logger)
    logger.info(
        "Phase 1a: %d active person IDs fetched from roster source (%.1fs)",
        len(current_person_ids),
        _elapsed(),
    )

    roster_ref = db.collection(_SYSTEM_COLLECTION).document("roster_state")
    roster_doc = roster_ref.get()
    prior_ids: frozenset[int] = frozenset(roster_doc.to_dict().get("person_ids", []) if roster_doc.exists else [])

    added_ids = current_person_ids - prior_ids
    removed_ids = prior_ids - current_person_ids

    logger.info(
        "Phase 1b: +%d added, -%d removed vs prior roster (%.1fs)",
        len(added_ids),
        len(removed_ids),
        _elapsed(),
    )

    # Fetch player details only for newly added players.
    new_player_details: list[dict[str, Any]] = []
    if added_ids:
        new_player_details = stats_provider.fetch_player_details(added_ids, logger)
        logger.info(
            "Phase 1c: %d/%d new player profiles fetched (%.1fs)",
            len(new_player_details),
            len(added_ids),
            _elapsed(),
        )

    # Persist updated roster state.
    roster_ref.set({"person_ids": sorted(current_person_ids), "updated_at": datetime.now(timezone.utc).isoformat()})

    # Load all currently-active player docs from Firestore for the trend/hash checks below.
    existing_docs: dict[str, dict[str, Any]] = {
        doc.id: (doc.to_dict() or {})
        for doc in players_ref.select(
            [
                "trend_updated_at",
                "avg_fantasy_score",
                "trend_games",
                "trend_hash",
                "is_active",
            ]
        ).stream()
    }

    # all_players is every player the pipeline should process this run:
    #   - new arrivals: use fetched profile dicts
    #   - returning players: reconstruct minimal dict from existing Firestore data
    returning_ids = current_person_ids - added_ids
    all_players: list[dict[str, Any]] = list(new_player_details)
    for pid in returning_ids:
        doc_data = existing_docs.get(str(pid), {})
        all_players.append({**doc_data, "person_id": pid})

    logger.info(
        "Phase 1 complete: %d total active players (%d new, %d returning) (%.1fs)",
        len(all_players),
        len(new_player_details),
        len(returning_ids),
        _elapsed(),
    )

    # --- Phase 2: sport-specific enrichment (headshots, external IDs) ---
    # enrich_players is a no-op for providers that supply profile data directly
    # (e.g. MLB Stats API). Kept for providers that need secondary ID matching.
    _hooks.enrich_players(all_players, existing_docs, db, logger)
    logger.info("Phase 2 complete: player enrichment done (%.1fs)", _elapsed())

    # --- Phase 3: stale guard ---
    stale_threshold = datetime.now(timezone.utc) - timedelta(hours=trend_stale_hours)
    existing_timestamps: dict[str, str] = {pid: d.get("trend_updated_at", "") for pid, d in existing_docs.items()}

    def _is_stale(person_id: int) -> bool:
        ts = existing_timestamps.get(str(person_id), "")
        if not ts:
            return True
        try:
            updated = datetime.fromisoformat(ts)
            if updated.tzinfo is None:
                updated = updated.replace(tzinfo=timezone.utc)
            return updated < stale_threshold
        except ValueError:
            return True

    stale_players = [p for p in all_players if _is_stale(p["person_id"])]
    fresh_count = len(all_players) - len(stale_players)
    logger.info(
        "Phase 3 complete: %d stale, %d fresh (%.1fs)",
        len(stale_players),
        fresh_count,
        _elapsed(),
    )

    _sport_cfg = get_active_config()

    # Build default trend from hooks (or use universal minimal defaults)
    if _hooks.build_default_trend is not None:
        default_trend: dict[str, Any] = _hooks.build_default_trend(_sport_cfg)
    else:
        default_trend = {
            "trend_direction": "neutral",
            "trend_score": None,
            "trend_games": 0,
            "avg_minutes": None,
            "avg_fantasy_score": None,
            "player_tier": "bottom_feeder",
            "consistency_score": None,
            "trend_updated_at": datetime.now(timezone.utc).isoformat(),
            **{key: None for key, _, _ in _sport_cfg.stat_categories},
            **{platform_cfg["tier_field"]: "bottom_feeder" for platform_cfg in PLATFORMS.values()},
        }

    stale_ids = {p["person_id"] for p in stale_players}

    # Resolve trend fields + hash fields (needed for Phase 6a hash check and Phase 6b strip)
    _trend_fields: frozenset[str] | None = None
    _hash_fields: tuple[str, ...] | None = None

    if _hooks.build_trend_fields is not None and _hooks.build_hash_fields is not None:
        _trend_fields = _hooks.build_trend_fields()
        _hash_fields = _hooks.build_hash_fields()
        assert set(_hash_fields) <= _trend_fields, (  # nosec B101
            f"_HASH_FIELDS contains fields not in _TREND_FIELDS: {set(_hash_fields) - _trend_fields}"
        )

    # --- Phase 4: fetch trends for stale players ---
    all_trend_data: dict[int, dict[str, Any]] = {}
    all_raw_rows: list[dict[str, Any]] = []

    if _hooks.fetch_trends is not None:
        for chunk_start in range(0, len(stale_players), checkpoint_batch_size):
            chunk = stale_players[chunk_start : chunk_start + checkpoint_batch_size]
            chunk_ids = [p["person_id"] for p in chunk]
            try:
                trends, chunk_raw_rows = _hooks.fetch_trends(chunk_ids, api_key)
                all_raw_rows.extend(chunk_raw_rows)
            except TrendFetchAbortedError as exc:
                skipped_ids = {p["person_id"] for p in stale_players} - set(all_trend_data.keys())
                logger.warning(
                    "Trend fetch aborted (%s). %d/%d stale players fetched; %d players skipped and flagged with trend_fetch_skipped_at.",
                    exc,
                    len(all_trend_data),
                    len(stale_players),
                    len(skipped_ids),
                )
                skipped_at = datetime.now(timezone.utc).isoformat()
                for batch_start in range(0, len(stale_players), 500):
                    skipped_chunk = [p for p in stale_players[batch_start : batch_start + 500] if p["person_id"] in skipped_ids]
                    if skipped_chunk:
                        batch = db.batch()
                        for player in skipped_chunk:
                            batch.set(
                                players_ref.document(str(player["person_id"])),
                                {"trend_fetch_skipped_at": skipped_at},
                                merge=True,
                            )
                        batch.commit()
                retry_enqueued_at = datetime.now(timezone.utc).isoformat()
                enqueue_trend_retry(delay_seconds=3600)
                _write_circuit_breaker_state(db, exc.consecutive_failures, retry_enqueued_at)
                break
            all_trend_data.update(trends)
            logger.info(
                "Phase 4 chunk: %d/%d stale players trend-fetched (%.1fs)",
                min(chunk_start + checkpoint_batch_size, len(stale_players)),
                len(stale_players),
                _elapsed(),
            )

    for player in stale_players:
        player.update(all_trend_data.get(player["person_id"], default_trend))
        updated_at = player.get("trend_updated_at")
        if updated_at:
            try:
                dt = datetime.fromisoformat(updated_at)
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                staleness = (datetime.now(timezone.utc) - dt).total_seconds() / 3600
                player["trend_staleness_hours"] = round(staleness, 1)
            except (ValueError, TypeError):
                player["trend_staleness_hours"] = None
        else:
            player["trend_staleness_hours"] = None

    # --- Phase 4c: fetch season averages + advanced metrics ---
    advanced_metrics_by_player: dict[int, dict[str, Any]] = {}
    stale_ids_list = [p["person_id"] for p in stale_players]
    if stale_ids_list:
        try:
            season_data, adv_data = _hooks.fetch_season_and_advanced(stale_ids_list, api_key, logger)
            advanced_metrics_by_player.update(adv_data)
            for player in stale_players:
                avgs = season_data.get(player["person_id"])
                if avgs:
                    player.update(avgs)
                adv = advanced_metrics_by_player.get(player["person_id"])
                if adv:
                    player.update(adv)
                    if player["person_id"] in all_trend_data:
                        all_trend_data[player["person_id"]].update(adv)
            logger.info(
                "Phase 4c complete: season stats merged for %d players (%.1fs)",
                len(season_data),
                _elapsed(),
            )
        except TrendFetchAbortedError as exc:
            logger.warning(
                "Phase 4c aborted (%s); season averages will be stale. (%.1fs)",
                exc,
                _elapsed(),
            )

    # --- Phase 4b: compute and persist team defensive ratings ---
    if all_raw_rows:
        defense_map, defense_map_fd, pace_map = compute_team_defense(all_raw_rows)
        defense_ref = db.collection("team_defense")
        now_iso = datetime.now(timezone.utc).isoformat()
        all_abbrs = set(defense_map) | set(defense_map_fd)
        all_abbrs_list = list(all_abbrs)
        for chunk_start in range(0, len(all_abbrs), 500):
            chunk_abbrs = all_abbrs_list[chunk_start : chunk_start + 500]
            batch = db.batch()
            for abbr in chunk_abbrs:
                empty_pos: dict[str, float] = {}
                batch.set(
                    defense_ref.document(abbr),
                    {
                        "team_abbr": abbr,
                        "pts_allowed_by_position": defense_map.get(abbr, empty_pos),
                        "pts_allowed_by_position_fd": defense_map_fd.get(abbr, empty_pos),
                        "pace": pace_map.get(abbr),
                        "games_sample": 15,
                        "updated_at": now_iso,
                    },
                )
            batch.commit()
        logger.info(
            "Phase 4b complete: wrote %d team_defense records (%.1fs)",
            len(all_abbrs),
            _elapsed(),
        )

    # --- Phase 5: assign percentile tiers ---
    assign_percentile_tiers(all_players, all_trend_data, existing_docs, advanced_metrics_by_player)
    logger.info("Phase 5 complete: percentile tiers assigned (%.1fs)", _elapsed())

    # --- Phase 5b: tag is_playoff_active ---
    league_state = get_league_state(db)
    if league_state.get("mode") == "playoffs":
        year = league_state.get("season", 2025)
        active_series = get_all_series(db, year)
        playoff_abbrs: set[str] = set()
        for s in active_series:
            if s.get("status") in ("scheduled", "active"):
                ht = s.get("higher_seed_team")
                lt = s.get("lower_seed_team")
                if ht:
                    playoff_abbrs.add(ht)
                if lt:
                    playoff_abbrs.add(lt)
        for p in all_players:
            team = p.get("team")
            team_abbr: str | None = (
                str(team.get("abbreviation")) if isinstance(team, dict) and team.get("abbreviation") is not None else (team if isinstance(team, str) else None)
            )
            p["is_playoff_active"] = team_abbr in playoff_abbrs if team_abbr else False
        logger.info(
            "Phase 5b: %d playoff-active teams, tagged %d players (%.1fs)",
            len(playoff_abbrs),
            sum(1 for p in all_players if p.get("is_playoff_active")),
            _elapsed(),
        )
    else:
        for p in all_players:
            p["is_playoff_active"] = False

    for p in all_players:
        p["is_active"] = True
        team = p.get("team")
        if isinstance(team, dict):
            p["team"] = team.get("abbreviation") or team.get("name") or ""

    # --- Phase 6a: write stale players (skip unchanged via hash) ---
    total_written = 0
    total_skipped_unchanged = 0
    for chunk_start in range(0, len(stale_players), 500):
        chunk = stale_players[chunk_start : chunk_start + 500]
        batch = db.batch()
        batch_count = 0
        for player in chunk:
            if _hash_fields is not None:
                new_hash = _trend_hash(player, _hash_fields)
                existing_hash = existing_docs.get(str(player["person_id"]), {}).get("trend_hash")
                existing_tier = existing_docs.get(str(player["person_id"]), {}).get("player_tier")
                if new_hash == existing_hash and player.get("player_tier") == existing_tier:
                    total_skipped_unchanged += 1
                    continue
                player["trend_hash"] = new_hash
            doc_ref = players_ref.document(str(player["person_id"]))
            batch.set(doc_ref, player, merge=True)
            batch_count += 1
        if batch_count:
            batch.commit()
        total_written += batch_count
        logger.info(
            "Phase 6a: committed %d/%d stale players (%d unchanged skipped) (%.1fs)",
            total_written,
            len(stale_players),
            total_skipped_unchanged,
            _elapsed(),
        )

    # --- Phase 6b: write fresh players (base fields + updated player_tier only) ---
    fresh_players = [p for p in all_players if p["person_id"] not in stale_ids]
    for chunk_start in range(0, len(fresh_players), 500):
        chunk = fresh_players[chunk_start : chunk_start + 500]
        batch = db.batch()
        for player in chunk:
            if _trend_fields is not None:
                for f in _trend_fields:
                    player.pop(f, None)
            player["trend_updated_at"] = datetime.now(timezone.utc).isoformat()
            doc_ref = players_ref.document(str(player["person_id"]))
            batch.set(doc_ref, player, merge=True)
        batch.commit()
        total_written += len(chunk)

    logger.info(
        "Phase 6b complete: %d fresh players written. Total: %d players stored (%.1fs elapsed)",
        len(all_players) - len(stale_players),
        len(all_players),
        _elapsed(),
    )

    # --- Phase 7: mark players no longer in active set as inactive ---
    active_ids = {str(p["person_id"]) for p in all_players}
    departed_ids = [pid for pid in existing_docs if pid not in active_ids and existing_docs[pid].get("is_active") is not False]
    if departed_ids:
        logger.info(
            "Phase 7: marking %d departed players is_active=False (%.1fs)",
            len(departed_ids),
            _elapsed(),
        )
        for chunk_start in range(0, len(departed_ids), 500):
            chunk = departed_ids[chunk_start : chunk_start + 500]
            batch = db.batch()
            for pid in chunk:
                batch.set(players_ref.document(pid), {"is_active": False}, merge=True)
            batch.commit()
    else:
        logger.info("Phase 7: no departed players found (%.1fs)", _elapsed())

    return len(all_players)


def fetch_and_store_today_games(
    db: Any,
    cfg_obj: OrchestratorConfig,
    stats_provider: _StatsProviderProto,
) -> int:
    """Fetch today's game schedule from the stats API and persist to Firestore.

    Writes a single document to games/{YYYY-MM-DD} containing the list of games,
    the set of playing team abbreviations, and a synced_at timestamp.

    Returns the number of games written (0 if no games today or fetch failed).
    """
    from zoneinfo import ZoneInfo

    today_et = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")
    games = stats_provider.fetch_games_for_date(today_et, cfg_obj.stats_api_key, logger)
    write_games(db, today_et, games)
    logger.info("fetch_and_store_today_games: %d games for %s", len(games), today_et)
    return len(games)


def fetch_and_store_upcoming_games(
    db: Any,
    cfg_obj: OrchestratorConfig,
    stats_provider: _StatsProviderProto,
    days_ahead: int = 7,
) -> dict[str, int]:
    """Fetch and store game schedules for the next ``days_ahead`` days (excluding today).

    Skips a date if the doc already exists and was synced within the last 6 hours.

    Returns a dict of {date_str: game_count} for each date processed.
    """
    from zoneinfo import ZoneInfo

    today_et = datetime.now(ZoneInfo("America/New_York")).date()
    results: dict[str, int] = {}

    for offset in range(1, days_ahead + 1):
        date_str = (today_et + timedelta(days=offset)).strftime("%Y-%m-%d")

        existing = db.collection("games").document(date_str).get()
        if existing.exists:
            synced_at_raw = (existing.to_dict() or {}).get("synced_at")
            if synced_at_raw:
                try:
                    synced_at = datetime.fromisoformat(synced_at_raw)
                    if synced_at.tzinfo is None:
                        synced_at = synced_at.replace(tzinfo=timezone.utc)
                    age_hours = (datetime.now(timezone.utc) - synced_at).total_seconds() / 3600
                    if age_hours < 6:
                        results[date_str] = (existing.to_dict() or {}).get("game_count", 0)
                        continue
                except (ValueError, TypeError):
                    pass

        games = stats_provider.fetch_games_for_date(date_str, cfg_obj.stats_api_key, logger)
        write_games(db, date_str, games)
        results[date_str] = len(games)
        logger.info("fetch_and_store_upcoming_games: %d games for %s", len(games), date_str)

    return results


def fetch_and_store_odds(
    db: Any,
    cfg_obj: OrchestratorConfig,
    odds_provider: _OddsProviderProto,
    lineups_provider: _LineupsProviderProto,
    game_hooks: GameSyncHooks | None = None,
) -> int:
    """Fetch Vegas odds from The Odds API and merge into today's games document.

    Computes implied team totals from spreads + totals markets and writes
    the signals into the existing games/{YYYY-MM-DD} doc under an 'odds' key.

    Returns the number of games with odds data (0 if fetch failed or no games doc).
    """
    _game_hooks = game_hooks or GameSyncHooks()
    from zoneinfo import ZoneInfo

    today_et = datetime.now(ZoneInfo("America/New_York")).strftime("%Y-%m-%d")

    games_ref = db.collection("games").document(today_et)
    games_doc = games_ref.get()
    if not games_doc.exists:
        logger.warning("fetch_and_store_odds: no games doc for %s — skipping", today_et)
        return 0

    raw_odds = odds_provider.fetch_odds(cfg_obj.the_odds_api_key, logger)
    if not raw_odds:
        logger.warning("fetch_and_store_odds: no odds data returned")
        return 0

    signals = compute_vegas_signals(raw_odds)

    games_ref.set(
        {
            "odds": signals,
            "odds_synced_at": datetime.now(timezone.utc).isoformat(),
        },
        merge=True,
    )

    logger.info("fetch_and_store_odds: %d teams with odds for %s", len(signals), today_et)

    try:
        games_data = games_ref.get().to_dict() or {}
        games_list: list[dict[str, Any]] = games_data.get("games_list") or []
        if games_list:
            enriched = _game_hooks.compute_game_totals(games_list, signals)
            games_ref.update({"games_list": enriched})
            logger.info("fetch_and_store_odds: enriched %d games with sport picks for %s", len(enriched), today_et)
    except Exception:
        logger.warning("fetch_and_store_odds: game totals enrichment failed (non-fatal)")

    try:
        fetch_and_store_espn_event_ids(db, today_et, lineups_provider)
    except Exception:
        logger.warning("fetch_and_store_odds: ESPN event ID capture failed (non-fatal)")

    return len(raw_odds)


def fetch_and_store_espn_event_ids(
    db: Any,
    date_str: str,
    lineups_provider: _LineupsProviderProto,
) -> int:
    """Populate event_id on matchup docs for date_str.

    Returns the number of matchup docs updated.
    """
    events = lineups_provider.fetch_event_ids(logger)
    if not events:
        return 0

    event_map: dict[tuple[str, str], str] = {(e["home_team_abbr"], e["visitor_team_abbr"]): e["event_id"] for e in events}

    matchups_ref = db.collection("games").document(date_str).collection("matchups")
    matchup_docs = list(matchups_ref.stream())
    if not matchup_docs:
        return 0

    batch = db.batch()
    count = 0
    for doc in matchup_docs:
        d = doc.to_dict() or {}
        home = d.get("home_team_abbr", "")
        visitor = d.get("visitor_team_abbr", "")
        eid = event_map.get((home, visitor))
        if eid:
            batch.set(matchups_ref.document(doc.id), {"espn_event_id": eid}, merge=True)
            count += 1

    if count:
        batch.commit()
    logger.info("fetch_and_store_espn_event_ids: updated %d matchup docs for %s", count, date_str)
    return count


def fetch_and_store_lineup_status(
    db: Any,
    date_str: str,
    lineups_provider: _LineupsProviderProto,
    lineup_lead_minutes: int,
) -> list[dict[str, Any]]:
    """Poll the lineups provider for confirmed starters for games on date_str.

    Returns a list of changed player dicts for FCM signaling:
        [{"player_name", "team", "is_starter", "previous_is_starter"}]
    """
    now = datetime.now(timezone.utc)
    window_start = now - timedelta(minutes=30)
    window_end = now + timedelta(minutes=lineup_lead_minutes)

    matchups_ref = db.collection("games").document(date_str).collection("matchups")
    matchup_docs = list(matchups_ref.stream())

    all_changes: list[dict[str, Any]] = []

    for doc in matchup_docs:
        d = doc.to_dict() or {}

        if d.get("lineups_confirmed"):
            continue

        espn_event_id = d.get("espn_event_id")
        if not espn_event_id:
            continue

        game_dt_raw = d.get("game_datetime")
        if not game_dt_raw:
            continue
        try:
            game_dt = datetime.fromisoformat(str(game_dt_raw).replace("Z", "+00:00"))
            if game_dt.tzinfo is None:
                game_dt = game_dt.replace(tzinfo=timezone.utc)
        except (ValueError, TypeError):
            continue

        if not (window_start <= game_dt <= window_end):
            continue

        lineup = lineups_provider.fetch_game_lineups(espn_event_id, logger)
        if not lineup.get("lineups_available"):
            continue

        home_abbr = d.get("home_team_abbr", "")
        visitor_abbr = d.get("visitor_team_abbr", "")
        confirmed_team_count = 0

        for team_entry in lineup.get("teams", []):
            team_abbr = team_entry["team_abbr"]
            starters = team_entry.get("starters", [])
            bench = team_entry.get("bench", [])

            if len(starters) >= 9:
                confirmed_team_count += 1

            starter_names: set[str] = {e["display_name"].lower() for e in starters}
            bench_names: set[str] = {e["display_name"].lower() for e in bench}
            all_names = starter_names | bench_names

            player_docs = db.collection("players").where("team", "==", team_abbr).select(["id", "first_name", "last_name", "is_confirmed_starter"]).stream()

            batch = db.batch()
            batch_count = 0
            now_iso = now.isoformat()

            for pdoc in player_docs:
                p = pdoc.to_dict() or {}
                first = p.get("first_name", "")
                last = p.get("last_name", "")
                full_name = f"{first} {last}".strip().lower()
                last_only = last.lower()

                if full_name in starter_names:
                    new_is_starter: bool | None = True
                elif full_name in bench_names:
                    new_is_starter = False
                elif last_only and any(last_only == n.split()[-1] for n in all_names):
                    matched = next(n for n in all_names if n.split()[-1] == last_only)
                    new_is_starter = matched in starter_names
                else:
                    continue

                existing = p.get("is_confirmed_starter")
                if existing == new_is_starter:
                    continue

                all_changes.append(
                    {
                        "player_name": f"{first} {last}".strip(),
                        "team": team_abbr,
                        "is_starter": new_is_starter,
                        "previous_is_starter": existing,
                    }
                )
                batch.set(
                    db.collection("players").document(pdoc.id),
                    {
                        "is_confirmed_starter": new_is_starter,
                        "lineup_confirmed_at": now_iso,
                        "lineup_source": "espn",
                    },
                    merge=True,
                )
                batch_count += 1

            if batch_count:
                batch.commit()

        if confirmed_team_count >= 2:
            matchups_ref.document(doc.id).set(
                {"lineups_confirmed": True, "lineups_confirmed_at": now.isoformat()},
                merge=True,
            )
            logger.info(
                "fetch_and_store_lineup_status: lineups confirmed for %s @ %s",
                visitor_abbr,
                home_abbr,
            )

    logger.info(
        "fetch_and_store_lineup_status: %d player changes for %s",
        len(all_changes),
        date_str,
    )
    return all_changes
