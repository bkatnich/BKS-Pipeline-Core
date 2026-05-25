"""Subscription tier entitlements — defines what each tier can access.

Used by endpoint handlers to filter response fields, cap limits, and gate features.
"""

from __future__ import annotations

from typing import Any

from bks_pipeline_core.models.user import (
    TIER_BASIC,
    TIER_EXPIRED,
    TIER_INSIDER,
    TIER_PREMIUM,
    TIER_PRO,
    TIER_TRIAL,
)

# ---------------------------------------------------------------------------
# Field sets — grouped by tier access level
# ---------------------------------------------------------------------------

# Fields available to all tiers (including expired, for minimal roster display)
_ROSTER_FIELDS: set[str] = {
    "id",
    "first_name",
    "last_name",
    "team",
    "position",
    "headshot_url",
    "sport_person_id",
    "nba_person_id",     # basketball-specific external ID used by iOS join
    "mlb_person_id",     # baseball-specific external ID
}

# All fields — available to every paid tier (Trial and above)
_BASIC_FIELDS: set[str] = _ROSTER_FIELDS | {
    # Core predictions
    "opportunity_score",
    "predicted_fp",
    "opportunity_percentile",
    "is_top_pick",
    "top_pick_rank",
    "top_pick_reasons",
    "vegas_prop_lines",
    # Platform projections
    "predicted_fp_dk",
    "predicted_fp_fd",
    "fp_floor_dk",
    "fp_ceiling_dk",
    "fp_floor_fd",
    "fp_ceiling_fd",
    "fp_mu_dk",
    "fp_sigma_dk",
    "fp_mu_fd",
    "fp_sigma_fd",
    "fp_ceiling",
    "fp_floor",
    "fp_mu",
    "fp_sigma",
    "prob_over_dk",
    "prob_under_dk",
    "prob_over_fd",
    "prob_under_fd",
    "avg_fantasy_score_dk",
    "avg_fantasy_score_fd",
    "confidence_score_dk",
    "confidence_score_fd",
    "platform",
    "mode",
    # Trend signals
    "trend_direction",
    "trend_score",
    "trend_acceleration",
    "confidence_score",
    "consistency_score",
    "hot_streak",
    "cold_streak",
    "is_role_change",
    "is_surging",
    "surge_delta",
    "surge_delta_pct",
    # Per-stat trends
    "trend_pts",
    "trend_reb",
    "trend_ast",
    "trend_stl",
    "trend_blk",
    "trend_updated_at",
    # Season averages
    "avg_fantasy_score",
    "season_avg_fantasy_score",
    "avg_minutes",
    "projected_minutes",
    "season_ppg",
    "season_rpg",
    "season_apg",
    "season_spg",
    "season_bpg",
    "season_ftmpg",
    "season_topg",
    "season_games",
    "season_fg3_pct",
    "playoff_fg3_pct",
    "playoff_simplified_per",
    "playoff_true_shooting_pct",
    "playoff_usage_rate_proxy",
    # Game context
    "is_injured",
    "injury_status",
    "is_back_to_back",
    "is_home",
    "opponent_abbr",
    "game_datetime",
    "team_rest_days",
    "opponent_rest_days",
    "opponent_is_b2b",
    "vegas_implied_team_total",
    "vegas_over_under",
    "vegas_spread",
    "def_ratio_at_position",
    "opp_fantasy_pts_allowed",
    "opp_pts_allowed_by_pos",
    "opp_pts_allowed_by_pos_fd",
    "opp_pace",
    # Lineup / starter status
    "is_confirmed_starter",
    "lineup_confirmed_at",
    # Injury trajectory
    "days_since_return",
    "injury_status_changed_at",
    "previous_injury_status",
    "is_return_game_window",
    # Recent game history
    "recent_game_scores",
    "recent_game_scores_fd",
    "recent_game_minutes",
    "recent_fg3_pct",
    # Advanced efficiency
    "simplified_per",
    "true_shooting_pct",
    "usage_rate_proxy",
    # Multipliers
    "matchup_multiplier",
    "vegas_multiplier",
    "b2b_penalty",
    "health_factor",
    "venue_multiplier",
    "opp_rest_multiplier",
    "mean_reversion_signal",
    "mean_reversion_multiplier",
    "is_minutes_driven_disappointment",
    "cat_trend_multiplier",
    "usage_delta_multiplier",
    "usage_delta_pct",
    "shooting_luck_multiplier",
    "role_change_multiplier",
    "pace_multiplier",
    # Variance modeling
    "blowout_prob",
    "minutes_ceiling",
    "minutes_floor",
    "minutes_env_multiplier",
    "game_env_capped",
    "game_env_cap",
    # Stacking
    "stacking_multiplier",
    "stacking_environment",
    "stacking_team_density",
    "stacking_game_density",
    "stacking_signal",
    # Playoff signals
    "playoff_rotation_multiplier",
    "rotation_tier",
    "playoff_trend_trust",
    "playoff_games_played",
    # Per-stat projections
    "projected_stats",
}

# Pro and Premium collapse to Basic — all fields available to every paid tier
_PRO_FIELDS: set[str] = _BASIC_FIELDS
_PREMIUM_FIELDS: set[str] = _BASIC_FIELDS

# ---------------------------------------------------------------------------
# Tier entitlements configuration
# ---------------------------------------------------------------------------

TIER_ENTITLEMENTS: dict[str, dict[str, Any]] = {
    TIER_EXPIRED: {
        "opportunities_limit": 0,
        "projections_lookahead": 0,
        "fields_allowed": _ROSTER_FIELDS,
        "platforms": [],
        "props_access": False,
        "notifications": False,
    },
    TIER_TRIAL: {
        "opportunities_limit": 10,
        "projections_lookahead": 1,
        "fields_allowed": _BASIC_FIELDS,
        "platforms": ["dk"],
        "props_access": False,
        "notifications": False,
    },
    TIER_BASIC: {
        "opportunities_limit": 10,
        "projections_lookahead": 1,
        "fields_allowed": _BASIC_FIELDS,
        "platforms": ["dk"],
        "props_access": False,
        "notifications": False,
    },
    TIER_PRO: {
        "opportunities_limit": 25,
        "projections_lookahead": 3,
        "fields_allowed": _PRO_FIELDS,
        "platforms": ["dk", "fd"],
        "props_access": False,
        "notifications": False,
    },
    TIER_PREMIUM: {
        "opportunities_limit": 50,
        "projections_lookahead": 7,
        "fields_allowed": _PREMIUM_FIELDS,
        "platforms": ["dk", "fd"],
        "props_access": True,
        "notifications": True,
    },
    TIER_INSIDER: {
        "opportunities_limit": 100,
        "projections_lookahead": 7,
        "fields_allowed": _PREMIUM_FIELDS,
        "platforms": ["dk", "fd"],
        "props_access": True,
        "notifications": True,
    },
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def get_entitlements(tier: str) -> dict[str, Any]:
    """Get entitlements for a tier. Defaults to expired if tier is unknown."""
    return TIER_ENTITLEMENTS.get(tier, TIER_ENTITLEMENTS[TIER_EXPIRED])


def get_opportunities_limit(tier: str) -> int:
    """Max number of opportunity results for a tier."""
    return get_entitlements(tier)["opportunities_limit"]


def get_allowed_platforms(tier: str) -> list[str]:
    """Platforms accessible for a tier."""
    return get_entitlements(tier)["platforms"]


def has_props_access(tier: str) -> bool:
    """Whether the tier can access prop predictions."""
    return get_entitlements(tier)["props_access"]


def has_notifications(tier: str) -> bool:
    """Whether the tier receives push notifications."""
    return get_entitlements(tier)["notifications"]


def get_projections_lookahead(tier: str) -> int:
    """Max days of multi-day projections for a tier."""
    return get_entitlements(tier)["projections_lookahead"]


def filter_response_fields(data: dict[str, Any], tier: str) -> dict[str, Any]:
    """Strip fields the tier is not entitled to see.

    Returns a new dict containing only allowed fields for the given tier.
    """
    allowed = get_entitlements(tier)["fields_allowed"]
    return {k: v for k, v in data.items() if k in allowed}


def filter_response_list(data: list[dict[str, Any]], tier: str) -> list[dict[str, Any]]:
    """Filter fields from a list of response dicts."""
    allowed = get_entitlements(tier)["fields_allowed"]
    return [{k: v for k, v in item.items() if k in allowed} for item in data]


def check_platform_access(tier: str, platform: str) -> bool:
    """Check if a tier has access to a specific platform."""
    return platform in get_allowed_platforms(tier)
