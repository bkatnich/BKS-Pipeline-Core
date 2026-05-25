"""Base types for sport-agnostic configuration.

A SportConfig instance encapsulates all sport-specific constants that the
pipeline layer needs. Swap the active config via set_active_config() to
target a different sport without changing any pipeline code.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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
    # Optional field name override for the primary fantasy points column.
    # Used when a sport provider stores FD points under a different key (e.g. "pts_fd").
    pts_field: str = "pts"


@dataclass(frozen=True)
class BracketConfig:
    """Playoff bracket structure for a sport. Set on SportConfig.playoff_bracket."""

    # Conference/group identifiers — first segment of series_id.
    # e.g. ("east", "west") for NBA; ("al", "nl") for MLB.
    conferences: tuple[str, ...]

    # First-round seed pairings per conference: ((high, low), ...)
    # e.g. ((1,8),(2,7),(3,6),(4,5)) for NBA.
    first_round_matchups: tuple[tuple[int, int], ...]

    # Round number → human-readable name.
    round_names: dict[int, str]

    # Wins required to take a series. Best-of-7 → 4, best-of-5 → 3.
    series_win_threshold: int

    # Home-court pattern keyed by string game number ("1"…"7").
    # True = higher seed is at home. String keys required (Firestore rejects int keys).
    home_court_pattern: dict[str, bool]

    # Total playoff rounds including the championship.
    total_rounds: int

    # Conference key used in the championship series_id (e.g. "nba", "mlb").
    championship_conference_key: str

    # R1 matchup index → (higher_seed, lower_seed).
    # Index order determines bracket pairing: index 0 meets index 1 in R2, etc.
    r1_matchups_index: dict[int, tuple[int, int]]

    # R2 matchup index → (R1 index A, R1 index B) that feed it.
    r2_feeders: dict[int, tuple[int, int]]


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

    # Human-readable sport name used in UI strings, email subjects, and report titles.
    # e.g. "Basketball", "Baseball"
    sport_display_name: str

    # Firestore sub-collection key for sport-scoped documents.
    # e.g. "nba", "mlb"
    sport_collection_key: str

    # Apple App Store bundle ID prefix for IAP product ID construction.
    # e.g. "com.blackkatt.bksbasketball", "com.blackkatt.bksbaseball"
    apple_bundle_id: str

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

    # Playoff bracket structure. None = sport has no playoffs or not yet configured.
    # When None, bracket functions (generate_first_round_matchups, advance_bracket, etc.)
    # will raise RuntimeError. Read-only functions (get_all_series, get_series) are safe.
    playoff_bracket: BracketConfig | None = None

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

    # Per-stat accuracy field mapping for backtesting.
    # Each entry: (predicted_field, actual_field, display_name)
    # predicted_field: key on the prediction snapshot record (e.g. "projected_pts")
    # actual_field: key on the joined record after actuals are prefixed with "actual_"
    #               (e.g. "actual_actual_pts" — actuals doc has "actual_pts", join prefixes again)
    # display_name: label shown in the accuracy report table (e.g. "Hits", "HR")
    # None = use the sport-agnostic default (basketball alias names).
    stat_fields: list[tuple[str, str, str]] | None = None

    # Sport-specific trend field names written per player, beyond the universal set.
    # These are merged into the full trend_fields frozenset by build_trend_fields().
    # Example for MLB: frozenset({"season_hits_pg", "season_avg", "woba_proxy", ...})
    trend_field_extras: frozenset[str] = field(default_factory=frozenset)

    # Subset of trend_field_extras (and/or universal fields) to include in the
    # change-detection hash, beyond the universal hash fields.
    # Example for MLB: ("season_avg", "season_obp", "season_ops", "woba_proxy", ...)
    trend_hash_field_extras: tuple[str, ...] = ()
