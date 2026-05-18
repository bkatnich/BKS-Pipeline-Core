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

    # Firestore database ID for this sport's named database.
    firestore_database_id: str = ""

    # External API key (SecretParam) for the sport's stats data provider.
    stats_api_key: object = None  # firebase_functions.params.SecretParam

    # Maximum pages to fetch from paginated APIs.
    max_pages: int = 20

    # Minimum trend_games to qualify for ranked tiers.
    tier_min_games: int = 3

    # Tier percentile thresholds.
    tier_elite_pct: float = 0.90
    tier_good_pct: float = 0.75
    tier_solid_pct: float = 0.50

    # Game-appearances per team used to compute pts allowed per position.
    defense_game_window: int = 15

    # Days after return_date within which is_return_game_window = True.
    injury_return_window_days: int = 14

    # Minutes below which a playoff game counts as a rest game.
    playoff_rest_game_minutes_threshold: int = 15

    # Playoff cold-start rolling weight (rolling-5g avg vs season avg).
    playoff_cold_start_rolling_weight: float = 0.65

    # Playoff elimination multipliers by series result.
    playoff_elimination_mult: dict[str, float] | None = None

    # Playoff rotation tier bands: (min_pct, max_pct, label).
    playoff_rotation_tiers: list[tuple[float, float, str]] | None = None

    # Minimum prop_actuals to fit Platt scaling per stat type.
    platt_min_samples: int = 50

    # Rolling window (days) for Platt training data.
    platt_window_days: int = 30

    # Per-stat Platt window overrides: {stat_key: days}.
    platt_window_days_by_stat: dict[str, int] | None = None

    # Minimum edge (proportion) to flag a prop as actionable.
    prop_edge_display_threshold: float = 0.04

    # Pregame freshness stale thresholds (minutes).
    pregame_injury_stale_minutes: int = 30
    pregame_odds_stale_minutes: int = 60
    pregame_trends_stale_minutes: int = 120
    pregame_dedup_window_minutes: int = 10

    # Minutes before re-polling lineups.
    lineup_stale_minutes: int = 10
