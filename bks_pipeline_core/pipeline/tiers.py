import bisect
import logging
from typing import Any

from bks_pipeline_core.pipeline.platforms import PLATFORMS
from bks_pipeline_core.sport_config import get_active_config

logger = logging.getLogger(__name__)

# Blended score component weights
_W_PER = 0.35
_W_TS = 0.25
_W_DFS = 0.20
_W_USAGE = 0.15
_W_MPG = 0.05

# Hard floor thresholds
_FLOOR_PER = 8.0
_FLOOR_MPG = 15.0
_FLOOR_DFS = 15.0  # players projecting below this are bottom_feeder regardless of efficiency


def _percentile_rank(sorted_vals: list[float], val: float | None) -> float | None:
    """Return the left-edge percentile rank of val in sorted_vals, in [0.0, 1.0].

    Returns None if val is None or sorted_vals is empty.
    Uses bisect_left so a player at the median of their distribution gets 0.5.
    """
    if val is None or not sorted_vals:
        return None
    n = len(sorted_vals)
    lo = bisect.bisect_left(sorted_vals, val)
    return lo / n


def _blend(components: list[tuple[float | None, float]]) -> float | None:
    """Compute a weighted blend of percentile components.

    Each element is (percentile_or_None, weight). Weights for None components
    are redistributed proportionally among non-None components. Returns None
    if every component is None (all weights redistributed away from DFS too,
    which cannot actually happen since DFS is always non-None when called).
    """
    valid = [(pct, w) for pct, w in components if pct is not None]
    if not valid:
        return None
    total_w = sum(w for _, w in valid)
    if total_w == 0.0:
        return None
    return sum(pct * (w / total_w) for pct, w in valid)


_PLAYOFF_ADV_MIN_GAMES = 3  # minimum playoff games before playoff metrics replace regular-season


def assign_percentile_tiers(
    all_players: list[dict[str, Any]],
    all_trend_data: dict[int, dict[str, Any]],
    existing_docs: dict[str, dict[str, Any]],
    advanced_metrics: dict[int, dict[str, Any]] | None = None,
    is_playoffs: bool = False,
) -> None:
    """Assign player_tier (and platform-specific equivalents) to every player.

    Iterates over all registered PLATFORMS and computes a blended percentile
    tier for each. The blend combines:
      - DFS avg fantasy score (platform-specific, 20 % weight)
      - simplified_per  (35 %)
      - true_shooting_pct (25 %)
      - usage_rate_proxy  (15 %)
      - season_mpg        (5 %)

    The five components are each normalized to [0,1] via percentile rank across
    qualified players, then blended. The blended score is itself ranked as a
    percentile among all qualified players' blended scores before mapping to a
    tier label. This preserves the relative-tier semantics of the original
    pure-DFS approach.

    In playoff mode (is_playoffs=True), players with >= _PLAYOFF_ADV_MIN_GAMES
    playoff games use playoff-specific advanced metrics (playoff_simplified_per,
    playoff_true_shooting_pct, playoff_usage_rate_proxy) instead of
    regular-season metrics. This prevents regular-season efficiency stats from
    high-efficiency bench players inflating their tier above playoff contributors.
    Players with fewer than _PLAYOFF_ADV_MIN_GAMES playoff games fall back to
    regular-season metrics (or DFS-only if no metrics are available).

    If advanced_metrics is None or a player has no advanced metric data, the
    missing component weights are redistributed proportionally to the non-None
    components (falling back to pure DFS when all advanced metrics are absent).

    Hard floor: if simplified_per < 8 AND season_mpg < 15 (both non-None),
    the player is forced to "bottom_feeder" regardless of blended score.

    Stale players use freshly computed values from all_trend_data; fresh
    players use existing_docs (Firestore pre-read). Modifies all_players and
    all_trend_data in-place.

    The DK tier is written to both "player_tier" (legacy field,
    backwards-compatible) and "player_tier_dk". All other platforms write to
    "player_tier_<platform>".
    """

    _cfg = get_active_config()
    TIER_MIN_GAMES = _cfg.tier_min_games
    TIER_ELITE_PCT = _cfg.tier_elite_pct
    TIER_GOOD_PCT = _cfg.tier_good_pct
    TIER_SOLID_PCT = _cfg.tier_solid_pct

    # ------------------------------------------------------------------
    # Pre-compute advanced metric percentile inputs (platform-agnostic)
    # ------------------------------------------------------------------
    # Build sorted lists of non-None metric values from qualified players.
    # "Qualified" = trend_games >= TIER_MIN_GAMES (same gate as DFS).
    #
    # In playoff mode: resolve each player's metrics from playoff fields when
    # they have sufficient playoff games, otherwise from regular-season fields.
    # Build ONE shared sorted pool per metric (all players contribute to the
    # same distribution regardless of which source they use) so percentile
    # ranks remain stable and comparable.

    sorted_per: list[float] = []
    sorted_ts: list[float] = []
    sorted_usage: list[float] = []
    sorted_mpg: list[float] = []

    # Determine qualified player IDs once (used for both adv metric pre-comp
    # and the per-platform DFS qualification gate).
    qualified_pids: set[int] = set()
    for player in all_players:
        pid = player["id"]
        if pid in all_trend_data:
            games = all_trend_data[pid].get("trend_games", 0) or 0
        else:
            games = existing_docs.get(str(pid), {}).get("trend_games", 0) or 0
        if games >= TIER_MIN_GAMES:
            qualified_pids.add(pid)

    def _resolve_adv_metrics(pid: int, player: dict[str, Any]) -> tuple[Any, Any, Any, Any]:
        """Return (per, ts, usage, mpg) for a player using the best available source.

        In playoff mode with >= _PLAYOFF_ADV_MIN_GAMES games: use playoff fields.
        Otherwise: use regular-season advanced_metrics or Firestore fields.
        """
        if is_playoffs:
            # Prefer Firestore playoff fields when the player has enough playoff games.
            # These are written by Phase 4 (compute_advanced_metrics on playoff rows).
            doc = existing_docs.get(str(pid), {})
            pg = (doc.get("playoff_games") or 0) or (player.get("playoff_games") or 0)
            if pg >= _PLAYOFF_ADV_MIN_GAMES:
                v_per = doc.get("playoff_simplified_per") or player.get("playoff_simplified_per")
                v_ts = doc.get("playoff_true_shooting_pct") or player.get("playoff_true_shooting_pct")
                v_usage = doc.get("playoff_usage_rate_proxy") or player.get("playoff_usage_rate_proxy")
                v_mpg = doc.get("playoff_mpg") or player.get("playoff_mpg")
                if v_per is not None:
                    return v_per, v_ts, v_usage, v_mpg
            # Fall through to regular-season below

        if advanced_metrics is not None:
            adv = advanced_metrics.get(pid, {})
            return (
                adv.get("simplified_per"),
                adv.get("true_shooting_pct"),
                adv.get("usage_rate_proxy"),
                adv.get("season_mpg"),
            )
        return None, None, None, None

    if advanced_metrics is not None or is_playoffs:
        per_vals: list[float] = []
        ts_vals: list[float] = []
        usage_vals: list[float] = []
        mpg_vals: list[float] = []

        for pid in qualified_pids:
            player = next((p for p in all_players if p["id"] == pid), {})
            v_per, v_ts, v_usage, v_mpg = _resolve_adv_metrics(pid, player)
            if v_per is not None:
                per_vals.append(float(v_per))
            if v_ts is not None:
                ts_vals.append(float(v_ts))
            if v_usage is not None:
                usage_vals.append(float(v_usage))
            if v_mpg is not None:
                mpg_vals.append(float(v_mpg))

        sorted_per = sorted(per_vals)
        sorted_ts = sorted(ts_vals)
        sorted_usage = sorted(usage_vals)
        sorted_mpg = sorted(mpg_vals)

    # ------------------------------------------------------------------
    # Per-platform tier assignment
    # ------------------------------------------------------------------

    def _tier_for_platform(platform_key: str, avg_fs_field: str) -> None:
        # Resolve DFS score and game count for every player.
        # Apply minutes-inflation correction: if avg_minutes > 1.4× season_mpg,
        # the rolling window contains fill-in games that inflate avg_fs. Scale
        # it down to the season_mpg baseline before percentile ranking so that
        # bench players with 1-2 anomalous starts don't receive inflated tiers.
        resolved: dict[int, tuple[float | None, int]] = {}
        for player in all_players:
            pid = player["id"]
            if pid in all_trend_data:
                t = all_trend_data[pid]
                dfs_score = t.get(avg_fs_field)
                games = t.get("trend_games", 0) or 0
                _avg_min = float(t.get("avg_minutes") or 0.0)
                _season_mpg = float(t.get("season_mpg") or 0.0)
            else:
                d = existing_docs.get(str(pid), {})
                dfs_score = d.get(avg_fs_field)
                games = d.get("trend_games", 0) or 0
                _avg_min = float(d.get("avg_minutes") or 0.0)
                _season_mpg = float(d.get("season_mpg") or 0.0)

            if dfs_score is not None and _season_mpg > 0 and _avg_min > _season_mpg * 1.4:
                dfs_score = round(float(dfs_score) * min(_season_mpg / _avg_min, 1.0), 2)

            resolved[pid] = (dfs_score, games)

        # Sorted DFS scores for qualified players (for DFS percentile input).
        qualified_dfs: list[float] = sorted(score for score, games in resolved.values() if score is not None and games >= TIER_MIN_GAMES)
        n_dfs = len(qualified_dfs)
        logger.info(
            "Percentile tier [%s]: %d qualified out of %d total players",
            platform_key,
            n_dfs,
            len(all_players),
        )

        # ------------------------------------------------------------------
        # Pass 1: compute raw blended scores and detect hard-floor players
        # ------------------------------------------------------------------
        # blended_scores[pid] = float score, or None if unqualified/hard-floor
        # hard_floored[pid] = True if forced to bottom_feeder
        blended_scores: dict[int, float | None] = {}
        hard_floored: dict[int, bool] = {}

        for player in all_players:
            pid = player["id"]
            dfs_score, games = resolved[pid]

            # Unqualified players → None (will become bottom_feeder)
            if dfs_score is None or games < TIER_MIN_GAMES or n_dfs == 0:
                blended_scores[pid] = None
                hard_floored[pid] = False
                continue

            # DFS percentile (platform-specific component)
            lo = bisect.bisect_left(qualified_dfs, dfs_score)
            dfs_pct: float = lo / n_dfs

            # Advanced metric percentiles (platform-agnostic)
            per_pct: float | None = None
            ts_pct: float | None = None
            usage_pct: float | None = None
            mpg_pct: float | None = None
            v_per: Any = None
            v_mpg: Any = None

            if advanced_metrics is not None or is_playoffs:
                v_per, v_ts, v_usage, v_mpg = _resolve_adv_metrics(pid, player)
                per_pct = _percentile_rank(sorted_per, v_per)
                ts_pct = _percentile_rank(sorted_ts, v_ts)
                usage_pct = _percentile_rank(sorted_usage, v_usage)
                mpg_pct = _percentile_rank(sorted_mpg, v_mpg)

            # Hard floor: low efficiency + low minutes
            if v_per is not None and v_mpg is not None and float(v_per) < _FLOOR_PER and float(v_mpg) < _FLOOR_MPG:
                blended_scores[pid] = None
                hard_floored[pid] = True
                continue

            # DFS floor: a player projecting < 15 DFS pts is bottom_feeder regardless of efficiency
            if dfs_score is not None and dfs_score < _FLOOR_DFS:
                blended_scores[pid] = None
                hard_floored[pid] = True
                continue

            hard_floored[pid] = False

            # Blend the five components (weight redistribution for None values)
            components: list[tuple[float | None, float]] = [
                (per_pct, _W_PER),
                (ts_pct, _W_TS),
                (dfs_pct, _W_DFS),
                (usage_pct, _W_USAGE),
                (mpg_pct, _W_MPG),
            ]
            blended_scores[pid] = _blend(components)

        # ------------------------------------------------------------------
        # Pass 2: rank blended scores → percentile → tier label
        # ------------------------------------------------------------------
        # Build sorted list of non-None blended scores from qualified players.
        sorted_blended: list[float] = sorted(s for s in blended_scores.values() if s is not None)
        n_blended = len(sorted_blended)

        def _label_from_blended(pid: int) -> str:
            """Map a player's blended score to a tier label.

            Uses bisect_right so the percentile counts all players with a
            blended score <= this player's, giving the inclusive rank. This
            prevents the single-player edge case where bisect_left always
            returns 0 (0th percentile → bottom_feeder for the sole survivor).
            """
            score = blended_scores[pid]
            if score is None:
                return "bottom_feeder"
            if n_blended == 0:
                return "bottom_feeder"
            lo = bisect.bisect_right(sorted_blended, score)
            pct = lo / n_blended
            if pct >= TIER_ELITE_PCT:
                return "elite"
            if pct >= TIER_GOOD_PCT:
                return "good"
            if pct >= TIER_SOLID_PCT:
                return "solid"
            return "bottom_feeder"

        tier_field = PLATFORMS[platform_key]["tier_field"]
        for player in all_players:
            pid = player["id"]
            tier = _label_from_blended(pid)
            player[tier_field] = tier
            if pid in all_trend_data:
                all_trend_data[pid][tier_field] = tier

        # DK also writes the legacy "player_tier" field for backwards compatibility
        if platform_key == "dk":
            for player in all_players:
                player["player_tier"] = player.get(tier_field, "bottom_feeder")
                pid = player["id"]
                if pid in all_trend_data:
                    all_trend_data[pid]["player_tier"] = all_trend_data[pid].get(tier_field, "bottom_feeder")

    for platform_key, platform_cfg in PLATFORMS.items():
        _tier_for_platform(platform_key, platform_cfg["avg_fs_field"])
