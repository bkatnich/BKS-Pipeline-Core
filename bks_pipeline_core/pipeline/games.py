"""Games data access layer — reads/writes games with subcollection + embedded list.

Storage layout:
  games/{YYYY-MM-DD}                     — metadata + games_list (denormalized array for single-read access)
  games/{YYYY-MM-DD}/matchups/{game_id}  — individual game docs (kept for writes/deletes)

load_games_doc reads from games_list when present (1 read), falling back to the
matchups subcollection stream (2 reads) for docs written before this change.
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)


def write_games(db: Any, date_str: str, games: list[dict[str, Any]]) -> None:
    """Write today's game schedule to Firestore using the subcollection layout.

    Creates:
      - games/{date_str} parent doc with metadata
      - games/{date_str}/matchups/{game_id} for each individual game
    """
    playing_abbrs = list({abbr for g in games for abbr in (g.get("home_team_abbr"), g.get("visitor_team_abbr")) if abbr})
    has_playoff_games = any(g.get("game_type") == "playoff" for g in games)

    games_ref = db.collection("games").document(date_str)

    # Write parent metadata doc — merge=True preserves odds written by sync_odds.
    # games_list is a denormalized copy so load_games_doc can read everything in one round trip.
    games_ref.set(
        {
            "date": date_str,
            "playing_team_abbrs": playing_abbrs,
            "has_playoff_games": has_playoff_games,
            "game_count": len(games),
            "games_list": games,
            "synced_at": datetime.now(timezone.utc).isoformat(),
        },
        merge=True,
    )

    # Write individual matchup documents in batches
    matchups_ref = games_ref.collection("matchups")
    incoming_ids = set()
    for chunk_start in range(0, len(games), 500):
        chunk = games[chunk_start : chunk_start + 500]
        batch = db.batch()
        for game in chunk:
            game_id = game.get("game_id")
            if game_id is None:
                logger.warning("write_games: skipping game with missing game_id: %s", game)
                continue
            game_id = str(game_id)
            incoming_ids.add(game_id)
            batch.set(matchups_ref.document(game_id), game)
        batch.commit()

    # Delete stale matchup docs no longer returned by BDL (e.g. cancelled series games)
    existing_ids = {doc.id for doc in matchups_ref.select([]).stream()}
    stale_ids = existing_ids - incoming_ids
    if stale_ids:
        batch = db.batch()
        for stale_id in stale_ids:
            batch.delete(matchups_ref.document(stale_id))
        batch.commit()
        logger.info("write_games: deleted %d stale matchup(s) for %s: %s", len(stale_ids), date_str, sorted(stale_ids))

    logger.info("write_games: %d matchups for %s", len(games), date_str)


def filter_completed_series_games(
    games: list[dict[str, Any]],
    db: Any,
    year: int,
) -> list[dict[str, Any]]:
    """Drop playoff games whose series is already completed.

    Prevents BDL's scheduling lag from writing phantom games for swept/closed series.
    Non-playoff games are always passed through unchanged.
    """
    from bks_pipeline_core.pipeline.playoffs import get_all_series

    completed_pairs: set[frozenset[str]] = {
        frozenset([s["higher_seed_team"], s["lower_seed_team"]])
        for s in get_all_series(db, year)
        if s.get("status") == "completed" and s.get("higher_seed_team") and s.get("lower_seed_team")
    }
    if not completed_pairs:
        return games

    filtered = []
    for g in games:
        if g.get("game_type") != "playoff":
            filtered.append(g)
            continue
        pair = frozenset([g.get("home_team_abbr", ""), g.get("visitor_team_abbr", "")])
        if pair in completed_pairs:
            logger.info(
                "filter_completed_series_games: dropping %s@%s — series completed",
                g.get("visitor_team_abbr"),
                g.get("home_team_abbr"),
            )
        else:
            filtered.append(g)
    return filtered


def load_games_doc(db: Any, date_str: str) -> dict[str, Any] | None:
    """Load games data into the dict shape expected by build_opportunity_results.

    Returns None if the parent doc doesn't exist.
    Returns a dict with keys: date, playing_team_abbrs, has_playoff_games,
    synced_at, odds, odds_synced_at, games (list of matchup dicts).

    Fast path: if the parent doc contains games_list (written by the current
    write_games), returns in a single Firestore read.  Falls back to streaming
    the matchups subcollection for docs written before this change.
    """
    parent_ref = db.collection("games").document(date_str)
    parent_snap = parent_ref.get()

    if not parent_snap.exists:
        return None

    parent = parent_snap.to_dict() or {}

    # Single-read fast path: games_list is a denormalized copy of the matchups.
    if "games_list" in parent:
        matchups = parent.pop("games_list") or []
        return {**parent, "games": matchups}

    # Legacy fallback: stream the subcollection (2 round trips total).
    matchups = [doc.to_dict() or {} for doc in parent_ref.collection("matchups").stream()]
    return {**parent, "games": matchups}


def compute_team_rest_days(db: Any, today_str: str, playing_abbrs: list[str], max_lookback: int = 4) -> dict[str, int]:
    """Compute days of rest for each team playing today.

    Checks the last max_lookback days of game documents to find each team's
    most recent game. Returns {team_abbr: days_since_last_game}.
    Teams not found in the lookback window are omitted (caller treats as neutral).

    Args:
        db: Firestore client.
        today_str: Today's date as YYYY-MM-DD.
        playing_abbrs: Teams playing today.
        max_lookback: Number of previous days to check.

    Returns:
        Dict mapping team abbreviation to rest days (0 = B2B, 1 = normal, 2+ = extended rest).
    """
    today = datetime.strptime(today_str, "%Y-%m-%d")
    playing_set = set(playing_abbrs)

    check_dates = [
        (days_ago, (today - timedelta(days=days_ago)).strftime("%Y-%m-%d"))
        for days_ago in range(1, max_lookback + 1)
    ]
    refs = [db.collection("games").document(date_str) for _, date_str in check_dates]
    docs = {date_str: doc for (_, date_str), doc in zip(check_dates, db.get_all(refs))}

    rest_days: dict[str, int] = {}
    for days_ago, date_str in check_dates:
        doc = docs.get(date_str)
        if doc is None or not doc.exists:
            continue
        teams_that_day = set((doc.to_dict() or {}).get("playing_team_abbrs", []))
        for team in playing_set - rest_days.keys():
            if team in teams_that_day:
                rest_days[team] = days_ago - 1  # 1 day ago = 0 rest days (B2B)
        if len(rest_days) >= len(playing_set):
            break

    return rest_days


_EXCLUDED_STATUSES = {"out", "suspension", "inactive", "suspended"}


_PACE_CLAMP_MIN = 0.93
_PACE_CLAMP_MAX = 1.07
_VEGAS_BLEND = 0.30  # weight on Vegas ITT when no series data (regular season)
_VEGAS_BLEND_WITH_SERIES = 0.20  # reduced Vegas weight when series history present (regular season)
_SERIES_WEIGHTS = {0: 0.0, 1: 0.15, 2: 0.25, 3: 0.35}  # 4+ games → _SERIES_WEIGHT_MAX
_SERIES_WEIGHT_MAX = 0.40
# Playoffs: Vegas lines incorporate schemes/injuries our PPG model misses entirely.
# HOU/LAL G1 2026: PPG model projected 237, Vegas 208.5, actual 192 — PPG was 45 pts wrong.
# Flip the blend: Vegas is majority signal, PPG is minority anchor.
_PLAYOFF_VEGAS_BLEND = 0.85
_PLAYOFF_VEGAS_BLEND_WITH_SERIES = 0.70


def _trimmed_series_avg(totals: list[float]) -> float:
    """Outlier-resistant average: drop the game furthest from median when n >= 4.

    Playoff series games within a ~10-day span have no meaningful recency gradient.
    Equal weighting with outlier trimming is more robust than exponential decay,
    which would amplify a single low-scoring game if it happened to be the most recent.
    """
    if not totals:
        return 0.0
    if len(totals) < 4:
        return sum(totals) / len(totals)
    median = sorted(totals)[len(totals) // 2]
    outlier_idx = max(range(len(totals)), key=lambda i: abs(totals[i] - median))
    trimmed = [t for i, t in enumerate(totals) if i != outlier_idx]
    return sum(trimmed) / len(trimmed)


def compute_series_history(
    db: Any,
    date: str,
    matchups: list[dict[str, Any]],
    lookback_days: int = 28,
) -> dict[frozenset, dict[str, Any]]:
    """Fetch prior playoff game totals for each matchup in the current series.

    Walks backward through Firestore games/{date} parent docs to find playoff dates
    where both teams played, loads actuals/{date} for scores, and reads the matchup
    subcollection to record which team was home. Returns a dict keyed by
    frozenset({home, visitor}).

    Each value:
      - games: total prior games found
      - avg_total: simple average of all games
      - trimmed_avg: outlier-resistant average (drops furthest-from-median when n>=4)
      - venue_avg: average for games played at tonight's venue (None if <2 matching games)
      - series_anchor: final blended signal used by compute_game_totals()
      - totals: all game totals, newest-first
      - venue_totals: game totals at tonight's venue only

    Returns games=0 for pairs with no history.
    """
    # Build pair → tonight's home team map so we can tag venue matches
    tonight_home: dict[frozenset, str] = {}
    for g in matchups:
        h, v = g.get("home_team_abbr"), g.get("visitor_team_abbr")
        if h and v:
            tonight_home[frozenset({h, v})] = h

    team_pairs: set[frozenset] = set(tonight_home.keys())
    if not team_pairs:
        return {}

    # history[pair] = list of (date_str, total_pts, home_team), newest appended first
    history: dict[frozenset, list[tuple[str, float, str]]] = {pair: [] for pair in team_pairs}

    today = datetime.strptime(date, "%Y-%m-%d")

    for days_ago in range(1, lookback_days + 1):
        check_date = (today - timedelta(days=days_ago)).strftime("%Y-%m-%d")

        parent_snap = db.collection("games").document(check_date).get()
        if not parent_snap.exists:
            continue
        parent = parent_snap.to_dict() or {}
        if not parent.get("has_playoff_games"):
            continue

        playing_set = set(parent.get("playing_team_abbrs", []))
        active_pairs = [pair for pair in team_pairs if pair.issubset(playing_set) and len(history[pair]) < 7]
        if not active_pairs:
            continue

        # Read matchup subcollection to get home team for each active pair
        matchup_docs = list(db.collection("games").document(check_date).collection("matchups").stream())
        date_home: dict[frozenset, str] = {}
        for doc in matchup_docs:
            d = doc.to_dict() or {}
            h, v = d.get("home_team_abbr"), d.get("visitor_team_abbr")
            if h and v:
                date_home[frozenset({h, v})] = h

        actuals_snap = db.collection("actuals").document(check_date).get()
        if not actuals_snap.exists:
            continue
        actuals_results = (actuals_snap.to_dict() or {}).get("results", {})

        team_pts: dict[str, float] = {}
        for player_data in actuals_results.values():
            if player_data.get("dnp"):
                continue
            team = player_data.get("team")
            pts = float(player_data.get("actual_pts") or 0)
            if team:
                team_pts[team] = team_pts.get(team, 0.0) + pts

        for pair in active_pairs:
            teams = list(pair)
            t1_pts = team_pts.get(teams[0])
            t2_pts = team_pts.get(teams[1])
            if t1_pts is not None and t2_pts is not None:
                home_team = date_home.get(pair, "")
                history[pair].append((check_date, round(t1_pts + t2_pts, 1), home_team))

        if all(len(v) >= 4 for v in history.values()):
            break

    result: dict[frozenset, dict[str, Any]] = {}
    for pair, entries in history.items():
        totals = [pts for _, pts, _ in entries]
        n = len(totals)

        tonight_h = tonight_home.get(pair, "")
        venue_totals = [pts for _, pts, h in entries if h == tonight_h]
        nv = len(venue_totals)

        trimmed = round(_trimmed_series_avg(totals), 1)
        avg = round(sum(totals) / n, 1) if n else 0.0
        venue_avg = round(sum(venue_totals) / nv, 1) if nv >= 2 else None

        # series_anchor: blend venue avg (60%) with trimmed avg (40%) when enough venue data,
        # otherwise fall back to trimmed avg alone
        if venue_avg is not None:
            anchor = round(0.6 * venue_avg + 0.4 * trimmed, 1)
        else:
            anchor = trimmed

        result[pair] = {
            "games": n,
            "avg_total": avg,
            "trimmed_avg": trimmed,
            "venue_avg": venue_avg,
            "venue_games": nv,
            "series_anchor": anchor,
            "totals": totals,
            "venue_totals": venue_totals,
        }
    return result


def compute_game_totals(
    matchups: list[dict[str, Any]],
    players: list[dict[str, Any]],
    pace_map: dict[str, float] | None = None,
    yesterday_teams: set[str] | None = None,
    vegas_signals: dict[str, dict[str, Any]] | None = None,
    series_history: dict | None = None,
    is_playoffs: bool = False,
) -> list[dict[str, Any]]:
    """Add home_proj_total, visitor_proj_total, proj_total to each matchup.

    Base formula: season_ppg × (avg_minutes / 36) per active player, summed by team.
    Enrichments (each optional, falls back gracefully when data absent):
      - Pace: game-level factor from both teams' pace vs league average.
      - B2B fatigue: minutes-weighted per-player penalty when team played yesterday.
      - Vegas anchor: regular season 70/30 PPG/Vegas blend; playoffs 40/60 (Vegas majority).
      - Series history: playoff series avg anchors the game total when ≥1 prior game
        exists. Vegas weight reduces from 0.30 to 0.20; series weight is 0.15-0.40
        depending on games played (recency-weighted, newest first).
    Fields are omitted when no qualifying players exist for a team.
    """
    # --- per-player raw totals and minutes registry (for B2B weighting) ---
    team_raw: dict[str, float] = {}
    # {team_abbr: [(avg_min, contrib), ...]} for B2B weighted penalty
    team_players: dict[str, list[tuple[float, float]]] = {}

    for p in players:
        team = p.get("team")
        team_abbr = team if isinstance(team, str) else (team or {}).get("abbreviation")
        if not team_abbr:
            continue
        if (p.get("injury_status") or "").lower() in _EXCLUDED_STATUSES:
            continue
        playoff_games = int(p.get("playoff_games") or 0)
        playoff_games_played = int(p.get("playoff_games_played") or 0)
        has_playoff_appearance = playoff_games >= 1 or playoff_games_played >= 1
        if is_playoffs:
            # In playoff mode: only include players who have appeared in at least
            # one playoff game this series. Bench warmers and two-way players with
            # zero playoff appearances inflate the raw team total with season stats
            # that don't reflect the tighter playoff rotation.
            if not has_playoff_appearance:
                continue
            # Always use playoff stats when available; fall back to season only as
            # a last resort (e.g. player just returned from injury in G1).
            ppg = float(p.get("playoff_ppg") or p.get("season_ppg") or 0.0)
            avg_min = float(p.get("playoff_mpg") or p.get("avg_minutes") or p.get("season_mpg") or 0.0)
        else:
            ppg = float(p.get("season_ppg") or 0.0)
            avg_min = float(p.get("avg_minutes") or p.get("season_mpg") or 0.0)
        if ppg <= 0 or avg_min <= 0:
            continue
        contrib = ppg * (avg_min / 36.0)
        team_raw[team_abbr] = team_raw.get(team_abbr, 0.0) + contrib
        team_players.setdefault(team_abbr, []).append((avg_min, contrib))

    # --- pace factor (game-level, symmetric) ---
    league_avg_pace: float | None = None
    if pace_map:
        paces = list(pace_map.values())
        if paces:
            league_avg_pace = sum(paces) / len(paces)

    def _pace_factor(home: str | None, visitor: str | None) -> float:
        if not pace_map or league_avg_pace is None or not league_avg_pace:
            return 1.0
        hp = pace_map.get(home or "")
        vp = pace_map.get(visitor or "")
        if hp is None or vp is None:
            return 1.0
        raw = (hp + vp) / (2.0 * league_avg_pace)
        return max(_PACE_CLAMP_MIN, min(_PACE_CLAMP_MAX, raw))

    # --- B2B factor (per-team, minutes-weighted) ---
    def _b2b_factor(team_abbr: str | None) -> float:
        if not yesterday_teams or not team_abbr or team_abbr not in yesterday_teams:
            return 1.0
        entries = team_players.get(team_abbr, [])
        if not entries:
            return 1.0
        total_contrib = sum(c for _, c in entries)
        if total_contrib <= 0:
            return 1.0
        weighted = sum((0.98 if m >= 32 else 0.99 if m >= 24 else 1.0) * c for m, c in entries)
        return weighted / total_contrib

    # --- Vegas blend ---
    def _apply_vegas(proj: float, team_abbr: str | None, vegas_weight: float) -> float:
        if not vegas_signals or not team_abbr:
            return proj
        itt = (vegas_signals.get(team_abbr) or {}).get("implied_team_total")
        if itt is None:
            return proj
        return (1.0 - vegas_weight) * proj + vegas_weight * float(itt)

    # --- Series history anchor ---
    def _series_factor(home: str | None, visitor: str | None) -> tuple[float, float]:
        """Return (series_weight, series_anchor). Both 0.0 when no data."""
        if not series_history or not home or not visitor:
            return 0.0, 0.0
        entry = series_history.get(frozenset({home, visitor}))
        if not entry:
            return 0.0, 0.0
        n = entry.get("games", 0)
        if n == 0:
            return 0.0, 0.0
        weight = _SERIES_WEIGHTS.get(n, _SERIES_WEIGHT_MAX)
        return weight, float(entry.get("series_anchor", 0.0))

    # --- assemble per-game ---
    pace_factor_cache: dict[tuple[str | None, str | None], float] = {}

    enriched = []
    for game in matchups:
        g = dict(game)
        home = g.get("home_team_abbr")
        visitor = g.get("visitor_team_abbr")

        key = (home, visitor)
        if key not in pace_factor_cache:
            pace_factor_cache[key] = _pace_factor(home, visitor)
        pf = pace_factor_cache[key]

        s_weight, s_weighted_avg = _series_factor(home, visitor)
        if is_playoffs:
            base_vw = _PLAYOFF_VEGAS_BLEND_WITH_SERIES if s_weight > 0.0 else _PLAYOFF_VEGAS_BLEND
        else:
            base_vw = _VEGAS_BLEND_WITH_SERIES if s_weight > 0.0 else _VEGAS_BLEND
        effective_vegas_weight = base_vw

        def _team_total(abbr: str | None) -> float | None:
            if not abbr or abbr not in team_raw:
                return None
            raw = team_raw[abbr]
            pace_adj = raw * pf
            b2b_adj = pace_adj * _b2b_factor(abbr)
            return _apply_vegas(b2b_adj, abbr, effective_vegas_weight)

        home_total = _team_total(home)
        visitor_total = _team_total(visitor)

        home_proj = round(home_total) if home_total is not None else None
        visitor_proj = round(visitor_total) if visitor_total is not None else None

        if home_proj is not None:
            g["home_proj_total"] = home_proj
        if visitor_proj is not None:
            g["visitor_proj_total"] = visitor_proj
        if home_proj is not None and visitor_proj is not None:
            our_combined = home_proj + visitor_proj
            if s_weight > 0.0 and s_weighted_avg > 0.0:
                entry = series_history[frozenset({home, visitor})]  # type: ignore[index]
                g["series_games"] = entry["games"]
                g["series_anchor"] = s_weighted_avg
                g["series_venue_avg"] = entry.get("venue_avg")

                # In playoff mode with series history, replace the player-derived
                # raw with Vegas ITTs as the non-series component. The player PPG
                # model is contaminated by Round 1 games against a different opponent
                # and rotation, making it unreliable mid-series. Vegas ITTs already
                # price in current defensive schemes, injuries, and pace — they are
                # a better foundation than playoff_ppg when a series history exists.
                home_itt = float((vegas_signals or {}).get(home or "", {}).get("implied_team_total") or 0)
                visitor_itt = float((vegas_signals or {}).get(visitor or "", {}).get("implied_team_total") or 0)
                vegas_combined = home_itt + visitor_itt

                if is_playoffs and vegas_combined > 0:
                    blended = round((1.0 - s_weight) * vegas_combined + s_weight * s_weighted_avg, 1)
                    ratio = home_itt / vegas_combined
                    home_proj = round(blended * ratio)
                    visitor_proj = blended - home_proj  # complement to avoid rounding drift
                    visitor_proj = round(visitor_proj)
                else:
                    # Regular season or no Vegas ITTs: blend series anchor onto player raw
                    blended = (1.0 - s_weight) * our_combined + s_weight * s_weighted_avg
                    blend_ratio = blended / our_combined if our_combined else 1.0
                    home_proj = round(home_proj * blend_ratio)
                    visitor_proj = round(visitor_proj * blend_ratio)

                g["home_proj_total"] = home_proj
                g["visitor_proj_total"] = visitor_proj
            g["proj_total"] = home_proj + visitor_proj

            # --- BK winner pick (straight up) ---
            our_margin = home_proj - visitor_proj
            bk_winner = home if our_margin >= 0 else visitor
            winner_confidence = round(min(1.0, abs(our_margin) / 20.0), 3)
            g["bk_winner"] = bk_winner
            g["bk_winner_confidence"] = winner_confidence

            # --- BK spread pick ---
            # home_spread: negative = home is favorite (e.g. -3.5 means home favored by 3.5)
            home_spread: float | None = (vegas_signals or {}).get(home or "", {}).get("spread")
            if home_spread is not None:
                # our margin vs what the spread requires the home team to beat
                # positive edge → home covers; negative edge → visitor covers
                spread_edge = our_margin - (-home_spread)  # edge = proj_margin - required_margin
                bk_spread_pick = home if spread_edge >= 0 else visitor
                bk_spread_covers = spread_edge >= 0
                spread_confidence = round(min(1.0, abs(spread_edge) / 10.0), 3)
                g["bk_spread_pick"] = bk_spread_pick
                g["bk_spread_pick_covers"] = bk_spread_covers
                g["bk_spread_confidence"] = spread_confidence

        enriched.append(g)
    return enriched


def enrich_matchups_with_odds(
    matchups: list[dict[str, Any]],
    odds: dict[str, dict[str, Any]] | None,
) -> list[dict[str, Any]]:
    """Merge team-keyed odds into per-game home_odds / visitor_odds dicts."""
    if not odds:
        return matchups

    enriched = []
    for game in matchups:
        g = dict(game)
        home = g.get("home_team_abbr")
        visitor = g.get("visitor_team_abbr")

        for side_key, abbr in (("home_odds", home), ("visitor_odds", visitor)):
            if not abbr:
                continue
            team_vegas = odds.get(abbr)
            if team_vegas is None:
                continue
            g[side_key] = dict(team_vegas)

        enriched.append(g)
    return enriched
