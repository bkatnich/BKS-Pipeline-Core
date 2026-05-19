"""Playoff series management — first-class series entities in Firestore.

Collection path: playoffs/{year}/series/{series_id}

Series IDs are deterministic: {conference}_{round}_{higher_seed}_v_{lower_seed}
Example: "west_r1_1_v_8" for the Western Conference first-round 1-seed vs 8-seed.

Finals series use conference="nba": "nba_r4_east_v_west"
"""

import logging
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

__all__ = [
    "CONFERENCES",
    "ROUND_NAMES",
    "SERIES_WIN_THRESHOLD",
    "HOME_COURT_PATTERN",
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

# NBA playoff bracket: first-round matchups by seed
_FIRST_ROUND_MATCHUPS = [(1, 8), (2, 7), (3, 6), (4, 5)]

CONFERENCES = ("east", "west")

ROUND_NAMES = {
    1: "First Round",
    2: "Conference Semifinals",
    3: "Conference Finals",
    4: "NBA Finals",
}

SERIES_WIN_THRESHOLD = 4  # best-of-7

# Home-court pattern for 2-2-1-1-1 format (standard NBA)
# True = higher seed has home court for that game number
# Keys must be strings — Firestore rejects integer dict keys (FieldPath validation)
HOME_COURT_PATTERN = {"1": True, "2": True, "3": False, "4": False, "5": True, "6": False, "7": True}


def series_id(conference: str, round_number: int, higher_seed: int | str, lower_seed: int | str) -> str:
    """Generate a deterministic series ID.

    For rounds 1-3: {conference}_r{round}_{higher_seed}_v_{lower_seed}
    For finals (round 4): nba_r4_{east_team}_v_{west_team}
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
        conference: "east", "west", or "nba" (finals)
        round_number: 1-4
        higher_seed_team: team abbreviation of the higher seed (e.g., "BOS")
        lower_seed_team: team abbreviation of the lower seed (e.g., "MIA")
        higher_seed: seed number of the higher seed (e.g., 1)
        lower_seed: seed number of the lower seed (e.g., 8)
        year: playoff year (e.g., 2026)
    """
    sid = series_id(conference, round_number, higher_seed, lower_seed)
    return {
        "series_id": sid,
        "year": year,
        "conference": conference,
        "round_number": round_number,
        "round_name": ROUND_NAMES.get(round_number, f"Round {round_number}"),
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
        "home_court_pattern": HOME_COURT_PATTERN,
        "game_results": [],  # list of {game_number, date, winner_team, score}
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
    game_id is the stable BDL game identifier, stored for idempotency across runs.
    """
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
        import logging as _logging
        _logging.getLogger(__name__).error(
            "record_game_result: wins (%d) exceed games_played (%d) for %s — reverting to active",
            total_wins,
            doc["games_played"],
            doc.get("series_id"),
        )
        doc["status"] = "active"
        return doc

    # Check for series completion
    if doc["wins_higher_seed"] >= SERIES_WIN_THRESHOLD:
        doc["status"] = "completed"
        doc["winner"] = doc["higher_seed_team"]
        doc["loser"] = doc["lower_seed_team"]
        doc["elimination_game_next"] = False
    elif doc["wins_lower_seed"] >= SERIES_WIN_THRESHOLD:
        doc["status"] = "completed"
        doc["winner"] = doc["lower_seed_team"]
        doc["loser"] = doc["higher_seed_team"]
        doc["elimination_game_next"] = False
    else:
        # Check if next game is an elimination game for either team
        doc["elimination_game_next"] = doc["wins_higher_seed"] == SERIES_WIN_THRESHOLD - 1 or doc["wins_lower_seed"] == SERIES_WIN_THRESHOLD - 1

    doc["updated_at"] = datetime.now(timezone.utc).isoformat()
    return doc


def generate_first_round_matchups(
    seedings: dict[str, list[dict[str, Any]]],
    year: int,
) -> list[dict[str, Any]]:
    """Generate first-round series documents from conference seedings.

    Args:
        seedings: {"east": [...], "west": [...]} where each list contains
                  dicts with "team" (abbreviation) and "seed" (1-8), sorted by seed.
        year: playoff year

    Returns list of 8 series documents (4 per conference).
    """
    series_docs = []
    for conference in CONFERENCES:
        teams = seedings.get(conference, [])
        seed_map = {t["seed"]: t["team"] for t in teams}

        for high, low in _FIRST_ROUND_MATCHUPS:
            high_team = seed_map.get(high)
            low_team = seed_map.get(low)
            if not high_team or not low_team:
                logger.warning(
                    "Missing seed %d or %d for %s conference — skipping matchup",
                    high,
                    low,
                    conference,
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
# Bracket progression — determines next-round matchups
# ---------------------------------------------------------------------------

# NBA bracket tree: maps (round, matchup_index) → feeder matchup indices from the previous round.
# Round 1 matchup indices (per conference): 0=(1v8), 1=(4v5), 2=(3v6), 3=(2v7)
# Round 2: winner of matchup 0 vs winner of matchup 1; winner of matchup 2 vs winner of matchup 3
# Round 3: winner of R2-matchup-0 vs winner of R2-matchup-1 (conference finals)
_R1_MATCHUPS = {0: (1, 8), 1: (4, 5), 2: (3, 6), 3: (2, 7)}

# R2 feeder: R2 matchup index → (R1 matchup index A, R1 matchup index B)
_R2_FEEDERS = {0: (0, 1), 1: (2, 3)}

# R3 (conf finals): single matchup fed by R2 matchup 0 and R2 matchup 1
_R3_FEEDERS = {0: (0, 1)}


def _r1_matchup_index(higher_seed: int, lower_seed: int) -> int | None:
    """Return the R1 matchup index for a seed pairing, or None if not found."""
    pair = (higher_seed, lower_seed)
    for idx, seeds in _R1_MATCHUPS.items():
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

    Logic:
      - Round 1 completion → check if paired R1 series is also done → create R2 series
      - Round 2 completion → check if paired R2 series is also done → create R3 (conf finals)
      - Round 3 completion → check if other conference finals is done → create R4 (NBA Finals)
      - Round 4 completion → no next round (champion determined)
    """
    if completed_series.get("status") != "completed":
        return None

    round_num = completed_series["round_number"]
    conference = completed_series["conference"]
    year = completed_series["year"]
    winner = completed_series["winner"]

    if round_num == 4:
        # NBA Finals complete — no next round
        return None

    # Build lookup of completed series for this conference and round
    conf_series = [s for s in all_series_for_year if s.get("conference") == conference and s.get("round_number") == round_num]

    if round_num == 1:
        # Find which R1 matchup index this series corresponds to
        my_idx = _r1_matchup_index(completed_series["higher_seed"], completed_series["lower_seed"])
        if my_idx is None:
            logger.warning(
                "Cannot determine R1 matchup index for series %s",
                completed_series.get("series_id"),
            )
            return None

        # Find the paired R1 series via R2 feeders
        paired_idx = None
        for r2_idx, (a, b) in _R2_FEEDERS.items():
            if my_idx == a:
                paired_idx = b
                break
            if my_idx == b:
                paired_idx = a
                break

        if paired_idx is None:
            return None

        paired_seeds = _R1_MATCHUPS[paired_idx]
        paired_series = _find_series(conf_series, paired_seeds[0], paired_seeds[1])

        if paired_series is None or paired_series.get("status") != "completed":
            return None  # Paired series not done yet

        # Both feeders complete — create R2 series
        return _create_next_round_from_feeders(completed_series, paired_series, conference, 2, year)

    elif round_num == 2:
        # Find paired R2 series in same conference
        other_r2 = [s for s in conf_series if s.get("series_id") != completed_series.get("series_id") and s.get("status") == "completed"]
        if not other_r2:
            return None  # Other R2 not done yet

        return _create_next_round_from_feeders(completed_series, other_r2[0], conference, 3, year)

    elif round_num == 3:
        # Conference finals done — check if other conference finals is done
        other_conf = "west" if conference == "east" else "east"
        other_conf_finals = [
            s for s in all_series_for_year if s.get("conference") == other_conf and s.get("round_number") == 3 and s.get("status") == "completed"
        ]
        if not other_conf_finals:
            return None  # Other conference finals not done yet

        other_winner = other_conf_finals[0]["winner"]

        # NBA Finals: use conference names as seeds for ID
        east_team = winner if conference == "east" else other_winner
        west_team = winner if conference == "west" else other_winner

        return build_series_doc(
            conference="nba",
            round_number=4,
            higher_seed_team=east_team,
            lower_seed_team=west_team,
            higher_seed="east",
            lower_seed="west",
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
    """Create the next-round series from two completed feeder series.

    The winner with the better (lower number) original seed gets higher_seed position.
    """
    winner_a = feeder_a["winner"]
    winner_b = feeder_b["winner"]

    # Determine original seeds for seeding the next round
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
# Bracket progression — Firestore orchestration
# ---------------------------------------------------------------------------


def advance_bracket(db: Any, year: int, completed_series: dict[str, Any]) -> dict[str, Any] | None:
    """Check if a completed series triggers bracket advancement and write the new series.

    Returns the new series doc if created, otherwise None.
    Idempotent: if the next-round series already exists, returns None without overwriting.
    """
    all_series = get_all_series(db, year)
    next_series = determine_next_round_series(completed_series, all_series)

    if next_series is None:
        return None

    # Idempotency check: don't overwrite an existing series
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

    # Batch in chunks of 500 (Firestore batch limit)
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
