"""Base types for sport-agnostic configuration.

A SportConfig instance encapsulates all sport-specific constants that the
pipeline layer needs. Swap the active config via set_active_config() to
target a different sport without changing any pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringWeights:
    """Fantasy scoring weights for a single DFS platform."""

    pts: float
    reb: float
    ast: float
    stl: float
    blk: float
    ftm: float = 0.0
    fgm_3pt: float = 0.0
    to: float = 0.0
    pf: float = 0.0
    dd_bonus: float = 0.0
    td_bonus: float = 0.0


@dataclass(frozen=True)
class SportConfig:
    """Sport-specific constants consumed by the pipeline layer.

    All values that differ between sports (scoring weights, position taxonomy,
    stat categories, odds team mapping, pace formula coefficients) live here.
    Operational constants (API keys, URLs, Firestore IDs, cron schedules) stay
    in config.py.
    """

    # Per-minute normalization base (36.0 for NBA — one regulation game = 48 min,
    # but DFS scoring is normalized to 36 min of play time by convention)
    per_minute_base: float

    # FTA coefficient in the possession-estimation formula:
    #   possessions ≈ FGA + coeff * FTA - OREB + TOV
    # NBA standard (Dean Oliver): 0.44
    pace_fta_coefficient: float

    # Fantasy scoring weights keyed by platform ("dk", "fd", …)
    scoring_weights: dict[str, ScoringWeights]

    # Raw position string → canonical bucket (e.g. "G-F" → "SG")
    position_bucket_map: dict[str, str]
    default_position_bucket: str

    # Full team name → 3-letter abbreviation, used when parsing odds API responses
    team_name_to_abbr: dict[str, str]

    # Which box-score stats to trend and whether to normalise per-minute.
    # Each entry: (output_key, stat_key, per_minute_normalise)
    # e.g. ("trend_pts", "pts", True)
    stat_categories: list[tuple[str, str, bool]]

    # Stat keys used in per-game projection (subset of box-score stats)
    projected_stat_keys: list[str]

    # The Odds API sport slug — e.g. "basketball_nba"
    # Used to document which sport slug the api/ provider targets.
    odds_api_sport_slug: str

    # League-average implied team total from over/under lines.
    # Used to normalise Vegas team-total multipliers.
    league_avg_team_total: float
