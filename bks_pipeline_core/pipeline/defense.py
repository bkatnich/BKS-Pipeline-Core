import logging
from collections import defaultdict
from typing import Any

from bks_pipeline_core.pipeline.scoring import parse_minutes
from bks_pipeline_core.sport_config import get_active_config

logger = logging.getLogger(__name__)


def _bucket(position: str | None) -> str:
    cfg = get_active_config()
    if not position:
        return cfg.default_position_bucket
    return cfg.position_bucket_map.get(position.strip(), cfg.default_position_bucket)



def compute_team_defense(
    raw_rows: list[dict[str, Any]],
    postseason_only: bool = False,
) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    """Compute per-position stat points allowed and team pace for each team.

    Aggregates raw stats rows (as returned by fetch_player_stats_batch) to produce
    defensive ratings for each team at each position bucket. Only the most recent
    cfg.defense_game_window game-appearances per (team, position) pair are used.

    Also computes estimated team pace (possessions per game) using:
        Possessions ≈ FGA + 0.44×FTA - OREB + TOV

    Returns:
        (defense, pace_map) where defense is:
        {
          "LAL": {"PG": 52.3, "SG": 48.1, "SF": 44.7, "PF": 41.2, "C": 38.9},
          ...
        }
        and pace_map is: {"LAL": 98.3, "BOS": 101.2, ...}
    Pure function — no I/O.
    """
    cfg = get_active_config()

    team_id_to_abbr: dict[int, str] = {}
    for row in raw_rows:
        team_obj = row.get("team")
        if isinstance(team_obj, dict):
            tid = team_obj.get("id")
            abbr = team_obj.get("abbreviation")
            if tid is not None and abbr:
                team_id_to_abbr[tid] = abbr

    observations: dict[tuple[str, str], list[tuple[str, float]]] = defaultdict(list)
    game_poss: dict[tuple[str, str], dict[str, int]] = defaultdict(lambda: {"fga": 0, "fta": 0, "oreb": 0, "tov": 0})

    for row in raw_rows:
        if postseason_only and not (row.get("game") or {}).get("postseason"):
            continue

        player_team = row.get("team", {})
        if not isinstance(player_team, dict):
            continue
        player_team_id = player_team.get("id")
        player_abbr = player_team.get("abbreviation")

        game = row.get("game", {})
        if not isinstance(game, dict):
            continue
        home_team_id = game.get("home_team_id")
        visitor_team_id = game.get("visitor_team_id")
        game_date = game.get("date", "")

        if player_team_id == home_team_id:
            opp_id = visitor_team_id
        elif player_team_id == visitor_team_id:
            opp_id = home_team_id
        else:
            continue

        if opp_id is None:
            continue
        opp_abbr = team_id_to_abbr.get(opp_id)
        if not opp_abbr:
            continue

        player_obj = row.get("player", {})
        position = player_obj.get("position") if isinstance(player_obj, dict) else None
        pos_bucket = _bucket(position)

        minutes = parse_minutes(row.get("min"))
        observations[(opp_abbr, pos_bucket)].append((game_date, minutes))

        if player_abbr and game_date:
            poss_key = (player_abbr, game_date)
            game_poss[poss_key]["fga"] += row.get("fga") or 0
            game_poss[poss_key]["fta"] += row.get("fta") or 0
            game_poss[poss_key]["oreb"] += row.get("oreb") or 0
            game_poss[poss_key]["tov"] += row.get("turnover") or 0

    defense: dict[str, dict[str, float]] = defaultdict(dict)
    for (opp_abbr, pos_bucket), obs in observations.items():
        obs.sort(key=lambda x: x[0])
        recent = obs[-cfg.defense_game_window :]
        avg = round(sum(v for _, v in recent) / len(recent), 2)
        defense[opp_abbr][pos_bucket] = avg

    team_game_poss: dict[str, list[float]] = defaultdict(list)
    for (abbr, _), totals in game_poss.items():
        poss = totals["fga"] + cfg.pace_fta_coefficient * totals["fta"] - totals["oreb"] + totals["tov"]
        team_game_poss[abbr].append(poss)

    pace_map: dict[str, float] = {}
    for abbr, games in team_game_poss.items():
        recent = sorted(games)[-cfg.defense_game_window :]
        pace_map[abbr] = round(sum(recent) / len(recent), 1)

    defense_result = dict(defense)

    logger.info(
        "Defense: %d raw rows → %d teams, %d teams with pace",
        len(raw_rows),
        len(defense_result),
        len(pace_map),
    )
    if pace_map:
        paces = list(pace_map.values())
        logger.info(
            "Defense: pace range [%.1f, %.1f], avg=%.1f",
            min(paces),
            max(paces),
            sum(paces) / len(paces),
        )

    return defense_result, pace_map
