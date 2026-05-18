"""Generate prop predictions by applying Normal CDF + Platt calibration.

Takes opportunity results (from build_opportunity_results) and prop lines
(from The Odds API) and produces prop prediction documents per the schema
in docs/prediction.md.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

from bks_pipeline_core.pipeline.platt import (
    apply_platt,
    compute_stat_distributions,
    no_vig_probability,
    prob_over_line,
)
from bks_pipeline_core.sport_config import get_active_config

logger = logging.getLogger(__name__)


def build_prop_predictions(
    results: list[dict[str, Any]],
    prop_lines: dict[str, list[dict[str, Any]]],
    platt_coeffs: dict[str, dict[str, Any]] | None = None,
    sport: str = "nba",
) -> list[dict[str, Any]]:
    """Build prop prediction documents for players with prop lines.

    Args:
        results: Output from ``build_opportunity_results()``.
        prop_lines: Dict keyed by player name → list of prop line dicts.
            Each line: ``{"stat": "pts", "line": 28.5, "over_odds": -105, "under_odds": -121}``.
        platt_coeffs: Optional dict of Platt coefficients keyed by stat type.
            ``{"pts": {"A": -1.2, "B": 0.1}, ...}``.
            If ``None``, raw model probabilities are used uncalibrated.
        sport: Sport identifier (default "nba").

    Returns:
        List of prop prediction documents (one per player with lines).
    """

    def _norm(name: str) -> str:
        """Normalize a player name for fuzzy matching across data sources.

        Strips punctuation, lowercases, and collapses whitespace so that
        "R.J. Barrett", "RJ Barrett", "r.j. barrett" all match.
        """
        return " ".join(name.lower().replace(".", "").replace("-", " ").split())

    # Build a normalized-name→result lookup for matching prop lines to players.
    name_to_result: dict[str, dict[str, Any]] = {}
    for r in results:
        first = r.get("first_name", "")
        last = r.get("last_name", "")
        if first and last:
            full_name = f"{first} {last}"
            name_to_result[_norm(full_name)] = r

    predictions: list[dict[str, Any]] = []
    now_iso = datetime.now(timezone.utc).isoformat()

    for player_name, lines in prop_lines.items():
        result = name_to_result.get(_norm(player_name))
        if result is None:
            logger.debug("prop skip: no matching result for %r", player_name)
            continue

        dists = compute_stat_distributions(result)
        if not dists:
            logger.debug("prop skip: no stat distributions for %r", player_name)
            continue

        calibration_version = "uncalibrated"
        if platt_coeffs:
            calibration_version = "v1"

        prop_line_docs: dict[str, dict[str, Any]] = {}
        for line_info in lines:
            stat = line_info["stat"]
            line_val = float(line_info["line"])
            over_odds = int(line_info["over_odds"])
            under_odds = int(line_info["under_odds"])

            stat_dist = dists.get(stat)
            if stat_dist is None:
                continue

            mu = stat_dist["mu"]
            sigma = stat_dist["sigma"]

            # Raw model probability from Normal CDF.
            model_prob = prob_over_line(mu, sigma, line_val)

            # Platt calibration (if available for this stat).
            calibrated_prob = model_prob
            platt_a: float | None = None
            platt_b: float | None = None
            if platt_coeffs and stat in platt_coeffs:
                coeffs = platt_coeffs[stat]
                platt_a = float(coeffs["A"])
                platt_b = float(coeffs["B"])
                calibrated_prob = apply_platt(model_prob, platt_a, platt_b)

            # No-vig market probability.
            nv_prob = no_vig_probability(over_odds, under_odds)

            # Edge = calibrated model prob - no-vig market prob.
            edge = round(calibrated_prob - nv_prob, 4)
            has_edge = edge >= get_active_config().prop_edge_display_threshold

            line_key = f"{stat}_{line_val}"
            prop_line_docs[line_key] = {
                "stat": stat,
                "line": line_val,
                "market_key": f"player_{_stat_to_market(stat)}",
                "over_odds": over_odds,
                "under_odds": under_odds,
                "no_vig_prob_over": round(nv_prob, 4),
                "model_prob_over": round(model_prob, 4),
                "calibrated_prob_over": round(calibrated_prob, 4),
                "edge": edge,
                "has_edge": has_edge,
                "display_label": f"{round(calibrated_prob * 100)}% Over {line_val} {stat.upper()}",
            }

        if not prop_line_docs:
            continue

        predictions.append(
            {
                "player_id": str(result.get("id", "")),
                "player_name": player_name,
                "game_id": None,  # populated by caller if available
                "date": None,  # populated by caller
                "sport": sport,
                "generated_at": now_iso,
                "calibration_version": calibration_version,
                "stats": {
                    stat: {
                        "dist": info["dist"],
                        "mu": round(info["mu"], 3),
                        "sigma": round(info["sigma"], 3),
                    }
                    for stat, info in dists.items()
                },
                "prop_lines": prop_line_docs,
            }
        )

    logger.info("prop predictions: %d players with %d total lines", len(predictions), sum(len(p["prop_lines"]) for p in predictions))
    return predictions


def _stat_to_market(stat: str) -> str:
    """Map internal stat key to The Odds API market suffix."""
    return {
        "pts": "points",
        "reb": "rebounds",
        "ast": "assists",
        "fg3m": "threes",
        "stl": "steals",
        "blk": "blocks",
        "tov": "turnovers",
        "pts_reb_ast": "points_rebounds_assists",
        "pts_reb": "points_rebounds",
    }.get(stat, stat)
