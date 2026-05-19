"""Playoff series management — first-class series entities in Firestore.

Collection path: playoffs/{year}/series/{series_id}

Series IDs are deterministic: {conference}_{round}_{higher_seed}_v_{lower_seed}
Example: "west_r1_1_v_8" for a Western Conference first-round 1-seed vs 8-seed.

The championship series uses the championship_conference_key from BracketConfig,
e.g. "nba_r4_east_v_west" for NBA.

All bracket-structure constants (conferences, seeding, round names, series length,
home-court pattern) are provided by SportConfig.playoff_bracket (a BracketConfig).
Functions that require bracket config will raise RuntimeError if playoff_bracket is None.
Read-only Firestore functions (get_all_series, get_series, update_series) are safe
to call regardless of whether playoff_bracket is configured.
"""

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from bks_pipeline_core.sport_config import get_active_config

if TYPE_CHECKING:
    from bks_pipeline_core.sport_config.base import BracketConfig

logger = logging.getLogger(__name__)

__all__ = [
    # Deprecated module-level constants — use get_active_config().playoff_bracket instead.
    "CONFERENCES",
    "ROUND_NAMES",
    "SERIES_WIN_THRESHOLD",
    "HOME_COURT_PATTERN",
    # Public API
    "series_id",
    "build_series_doc",
    "record_game_result",
    "generate_first_round_matchups",
    "determine_next_round_series",
    "advance_bracket",
    "write_series_to_firestore",
    "get_series",
    "get_all_series",
    "update_series",
    "_r1_matchup_index",
    "_winner_seed",
]

# ---------------------------------------------------------------------------
# Deprecated module-level constants (NBA literals kept for backward compat)
# Use get_active_config().playoff_bracket.<field> in new code.
# ---------------------------------------------------------------------------

# Deprecated: use get_active_config().playoff_bracket.conferences
CONFERENCES = ("east", "west")

# Deprecated: use get_active_config().playoff_bracket.round_names
ROUND_NAMES = {
    1: "First Round",
    2: "Conference Semifinals",
    3: "Conference Finals",
    4: "NBA Finals",
}

# Deprecated: use get_active_config().playoff_bracket.series_win_threshold
SERIES_WIN_THRESHOLD = 4

# Deprecated: use get_active_config().playoff_bracket.home_court_pattern
HOME_COURT_PATTERN = {"1": True, "2": True, "3": False, "4": False, "5": True, "6": False, "7": True}


# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

def _bracket() -> "BracketConfig":
    """Return the active sport's BracketConfig, raising clearly if absent."""
    bc = get_active_config().playoff_bracket
    if bc is None:
        raise RuntimeError(
            f"SportConfig.playoff_bracket is not configured for sport "
            f"'{get_active_config().sport_collection_key}'. "
            "Set playoff_bracket on SportConfig to use bracket functions."
        )
    return bc


# ---------------------------------------------------------------------------
# Series document construction
# ---------------------------------------------------------------------------

def series_id(conference: str, round_number: int, higher_seed: int | str, lower_seed: int | str) -> str:
    """Generate a deterministic series ID.

    Format: {conference}_r{round}_{higher_seed}_v_{lower_seed}
    """
    return f"{conference}_r{round_number}_{higher_seed}_v_{lower_seed}"


def build_series_doc(
    conference: str,
    round_number: int,
    higher_seed_team: str,
    lower_seed_team: str,
    higher_seed: int | str,
    lower_seed: int | str,
    year: int,
) -> dict[str, Any]:
    """Build a new series document ready for Firestore.

    Args:
        conference: e.g. "east", "west", or the championship_conference_key
        round_number: 1-based round number
        higher_seed_team: team abbreviation of the higher seed (e.g., "BOS")
        lower_seed_team: team abbreviation of the lower seed (e.g., "MIA")
        higher_seed: seed number or conference name for championship
        lower_seed: seed number or conference name for championship
        year: playoff year (e.g., 2026)
    """
    bc = _bracket()
    sid = series_id(conference, round_number, higher_seed, lower_seed)
    return {
        "series_id": sid,
        "year": year,
        "conference": conference,
        "round_number": round_number,
        "round_name": bc.round_names.get(round_number, f"Round {round_number}"),
        "higher_seed_team": higher_seed_team,
        "lower_seed_team": lower_seed_team,
        "higher_seed": higher_seed,
        "lower_seed": lower_seed,
        "wins_higher_seed": 0,
        "wins_lower_seed": 0,
        "status": "scheduled",  # scheduled | active | completed
        "winner": None,
        "loser": None,
        "games_played": 0,
        "elimination_game_next": False,
        "home_court_pattern": bc.home_court_pattern,
        "game_results": [],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }


def record_game_result(
    series_doc: dict[str, Any],
    game_number: int,
    winner_team: str,
    date: str,
    score: str | None = None,
    game_id: int | None = None,
) -> dict[str, Any]:
    """Record a game result and update series state.

    Returns the updated series document (caller is responsible for writing to Firestore).
    Does NOT mutate the input dict — returns a new copy.
    game_id is a stable game identifier stored for idempotency across runs.
    """
    threshold = _bracket().series_win_threshold
    doc = {**series_doc, "game_results": list(series_doc.get("game_results", []))}

    result = {
        "game_number": game_number,
        "date": date,
        "winner_team": winner_team,
        "score": score,
        "game_id": game_id,
    }
    doc["game_results"].append(result)
    doc["games_played"] = len(doc["game_results"])

    if winner_team == doc["higher_seed_team"]:
        doc["wins_higher_seed"] = doc.get("wins_higher_seed", 0) + 1
    elif winner_team == doc["lower_seed_team"]:
        doc["wins_lower_seed"] = doc.get("wins_lower_seed", 0) + 1

    doc["status"] = "active"

    # Safety guard: wins must not exceed games_played (catches double-recording)
    total_wins = doc["wins_higher_seed"] + doc["wins_lower_seed"]
    if total_wins > doc["games_played"]:
        logger.error(
            "record_game_result: wins (%d) exceed games_played (%d) for %s — reverting to active",
            total_wins,
            doc["games_played"],
            doc.get("series_id"),
        )
        doc["status"] = "active"
        return doc

    if doc["wins_higher_seed"] >= threshold:
        doc["status"] = "completed"
        doc["winner"] = doc["higher_seed_team"]
        doc["loser"] = doc["lower_seed_team"]
        doc["elimination_game_next"] = False
    elif doc["wins_lower_seed"] >= threshold:
        doc["status"] = "completed"
        doc["winner"] = doc["lower_seed_team"]
        doc["loser"] = doc["higher_seed_team"]
        doc["elimination_game_next"] = False
    else:
        doc["elimination_game_next"] = (
            doc["wins_higher_seed"] == threshold - 1
            or doc["wins_lower_seed"] == threshold - 1
        )

    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    return doc


# ---------------------------------------------------------------------------
# Bracket generation
# ---------------------------------------------------------------------------

def generate_first_round_matchups(
    seedings: dict[str, list[dict[str, Any]]],
    year: int,
) -> list[dict[str, Any]]:
    """Generate first-round series documents from conference seedings.

    Args:
        seedings: {conference_key: [...]} where each list contains dicts with
                  "team" (abbreviation) and "seed" (int), sorted by seed.
        year: playoff year

    Returns list of series documents (len = conferences × first_round_matchups).
    Requires SportConfig.playoff_bracket to be configured.
    """
    bc = _bracket()
    series_docs = []
    for conference in bc.conferences:
        teams = seedings.get(conference, [])
        seed_map = {t["seed"]: t["team"] for t in teams}

        for high, low in bc.first_round_matchups:
            high_team = seed_map.get(high)
            low_team = seed_map.get(low)
            if not high_team or not low_team:
                logger.warning(
                    "Missing seed %d or %d for %s conference — skipping matchup",
                    high, low, conference,
                )
                continue

            doc = build_series_doc(
                conference=conference,
                round_number=1,
                higher_seed_team=high_team,
                lower_seed_team=low_team,
                higher_seed=high,
                lower_seed=low,
                year=year,
            )
            series_docs.append(doc)

    return series_docs


# ---------------------------------------------------------------------------
# Bracket progression
# ---------------------------------------------------------------------------

def _r1_matchup_index(higher_seed: int, lower_seed: int) -> int | None:
    """Return the R1 matchup index for a seed pairing, or None if not found."""
    pair = (higher_seed, lower_seed)
    for idx, seeds in _bracket().r1_matchups_index.items():
        if seeds == pair:
            return idx
    return None


def determine_next_round_series(
    completed_series: dict[str, Any],
    all_series_for_year: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Determine if a completed series triggers creation of a next-round series.

    Returns a new series doc if both feeder series are complete, otherwise None.
    Pure function — does not touch Firestore.
    Requires SportConfig.playoff_bracket to be configured.
    """
    if completed_series.get("status") != "completed":
        return None

    bc = _bracket()
    round_num = completed_series["round_number"]
    conference = completed_series["conference"]
    year = completed_series["year"]
    winner = completed_series["winner"]

    if round_num >= bc.total_rounds:
        return None  # Championship complete — no next round

    conf_series = [
        s for s in all_series_for_year
        if s.get("conference") == conference and s.get("round_number") == round_num
    ]

    if round_num == 1:
        my_idx = _r1_matchup_index(completed_series["higher_seed"], completed_series["lower_seed"])
        if my_idx is None:
            logger.warning(
                "Cannot determine R1 matchup index for series %s",
                completed_series.get("series_id"),
            )
            return None

        paired_idx = None
        for r2_idx, (a, b) in bc.r2_feeders.items():
            if my_idx == a:
                paired_idx = b
                break
            if my_idx == b:
                paired_idx = a
                break

        if paired_idx is None:
            return None

        paired_seeds = bc.r1_matchups_index[paired_idx]
        paired_series = _find_series(conf_series, paired_seeds[0], paired_seeds[1])

        if paired_series is None or paired_series.get("status") != "completed":
            return None

        return _create_next_round_from_feeders(completed_series, paired_series, conference, 2, year)

    elif round_num == 2:
        other_r2 = [
            s for s in conf_series
            if s.get("series_id") != completed_series.get("series_id")
            and s.get("status") == "completed"
        ]
        if not other_r2:
            return None

        return _create_next_round_from_feeders(completed_series, other_r2[0], conference, 3, year)

    elif round_num == bc.total_rounds - 1:
        # Conference finals done — check if the other conference finals is done
        other_conf = next((c for c in bc.conferences if c != conference), None)
        if other_conf is None:
            return None

        other_conf_finals = [
            s for s in all_series_for_year
            if s.get("conference") == other_conf
            and s.get("round_number") == bc.total_rounds - 1
            and s.get("status") == "completed"
        ]
        if not other_conf_finals:
            return None

        other_winner = other_conf_finals[0]["winner"]

        # Championship: use conference names as seeds in the series ID
        first_conf_team = winner if conference == bc.conferences[0] else other_winner
        second_conf_team = winner if conference == bc.conferences[1] else other_winner

        return build_series_doc(
            conference=bc.championship_conference_key,
            round_number=bc.total_rounds,
            higher_seed_team=first_conf_team,
            lower_seed_team=second_conf_team,
            higher_seed=bc.conferences[0],
            lower_seed=bc.conferences[1],
            year=year,
        )

    return None


def _find_series(series_list: list[dict[str, Any]], higher_seed: int, lower_seed: int) -> dict[str, Any] | None:
    """Find a series by seed pairing in a list."""
    for s in series_list:
        if s.get("higher_seed") == higher_seed and s.get("lower_seed") == lower_seed:
            return s
    return None


def _create_next_round_from_feeders(
    feeder_a: dict[str, Any],
    feeder_b: dict[str, Any],
    conference: str,
    next_round: int,
    year: int,
) -> dict[str, Any]:
    """Create the next-round series from two completed feeder series."""
    winner_a = feeder_a["winner"]
    winner_b = feeder_b["winner"]

    seed_a = _winner_seed(feeder_a)
    seed_b = _winner_seed(feeder_b)

    if seed_a <= seed_b:
        higher_team, lower_team = winner_a, winner_b
        higher_seed, lower_seed = seed_a, seed_b
    else:
        higher_team, lower_team = winner_b, winner_a
        higher_seed, lower_seed = seed_b, seed_a

    return build_series_doc(
        conference=conference,
        round_number=next_round,
        higher_seed_team=higher_team,
        lower_seed_team=lower_team,
        higher_seed=higher_seed,
        lower_seed=lower_seed,
        year=year,
    )


def _winner_seed(series_doc: dict[str, Any]) -> int:
    """Return the original seed of the series winner."""
    if series_doc["winner"] == series_doc["higher_seed_team"]:
        return int(series_doc["higher_seed"])
    return int(series_doc["lower_seed"])


# ---------------------------------------------------------------------------
# Firestore I/O — safe to call without playoff_bracket configured
# ---------------------------------------------------------------------------

def advance_bracket(db: Any, year: int, completed_series: dict[str, Any]) -> dict[str, Any] | None:
    """Check if a completed series triggers bracket advancement and write the new series.

    Returns the new series doc if created, otherwise None.
    Idempotent: if the next-round series already exists, returns None without overwriting.
    Requires SportConfig.playoff_bracket to be configured.
    """
    all_series = get_all_series(db, year)
    next_series = determine_next_round_series(completed_series, all_series)

    if next_series is None:
        return None

    existing = get_series(db, year, next_series["series_id"])
    if existing is not None:
        logger.info(
            "Next-round series %s already exists — skipping creation",
            next_series["series_id"],
        )
        return None

    write_series_to_firestore(db, year, [next_series])
    logger.info(
        "Bracket advanced: created %s (round %d, %s)",
        next_series["series_id"],
        next_series["round_number"],
        next_series["conference"],
    )
    return next_series


def write_series_to_firestore(db: Any, year: int, series_docs: list[dict[str, Any]]) -> int:
    """Write series documents to Firestore at playoffs/{year}/series/{series_id}.

    Returns the number of documents written.
    """
    written = 0
    series_ref = db.collection("playoffs").document(str(year)).collection("series")

    for chunk_start in range(0, len(series_docs), 500):
        chunk = series_docs[chunk_start : chunk_start + 500]
        batch = db.batch()
        for doc in chunk:
            sid = doc["series_id"]
            batch.set(series_ref.document(sid), doc)
            written += 1
        batch.commit()

    logger.info("Wrote %d series documents for %d playoffs", written, year)
    return written


def get_series(db: Any, year: int, sid: str) -> dict[str, Any] | None:
    """Read a single series document from Firestore."""
    doc = db.collection("playoffs").document(str(year)).collection("series").document(sid).get()
    if doc.exists:
        result: dict[str, Any] | None = doc.to_dict()
        return result
    return None


def get_all_series(db: Any, year: int) -> list[dict[str, Any]]:
    """Read all series for a given playoff year."""
    docs = db.collection("playoffs").document(str(year)).collection("series").stream()
    result: list[dict[str, Any]] = [doc.to_dict() or {} for doc in docs]
    return result


def update_series(db: Any, year: int, series_doc: dict[str, Any]) -> None:
    """Write an updated series document back to Firestore."""
    sid = series_doc["series_id"]
    db.collection("playoffs").document(str(year)).collection("series").document(sid).set(series_doc)
