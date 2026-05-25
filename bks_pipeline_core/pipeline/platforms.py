import logging

logger = logging.getLogger(__name__)

# Registry of supported fantasy platforms.
# Each entry maps a platform key to the Firestore field names used for that platform's metrics.
# Adding a new platform requires:
#   1. A scoring function in pipeline/scoring.py
#   2. Computed fields in pipeline/trends.py (_compute_trend_fields)
#   3. A new entry here
# Opportunity scoring and orchestrator defaults all pick up new platforms automatically.
PLATFORMS: dict[str, dict[str, str]] = {
    "dk": {
        "label": "DraftKings",
        "avg_fs_field": "avg_fantasy_score",
        "trend_field": "trend_score",
        "trend_dir_field": "trend_direction",
        "streak_field": "hot_streak",
        "surging_field": "is_surging",
        "surge_pct_field": "surge_delta_pct",
        "confidence_field": "confidence_score",
        "consistency_field": "consistency_score",
        "accel_field": "trend_acceleration",
        "home_fs_field": "avg_fantasy_score_home",
        "away_fs_field": "avg_fantasy_score_away",
        "def_pts_field": "pts_allowed_by_position",
    },
    "fd": {
        "label": "FanDuel",
        "avg_fs_field": "avg_fantasy_score",
        "trend_field": "trend_score",
        "trend_dir_field": "trend_direction",
        "streak_field": "hot_streak",
        "surging_field": None,
        "surge_pct_field": None,
        "confidence_field": "confidence_score",
        "consistency_field": "consistency_score",
        "accel_field": "trend_acceleration",
        "home_fs_field": "avg_fantasy_score_home",
        "away_fs_field": "avg_fantasy_score_away",
        "def_pts_field": "pts_allowed_by_position_fd",
    },
}

DEFAULT_PLATFORM = "dk"


def get_platform(key: str) -> dict[str, str]:
    """Return the platform config for key, falling back to DEFAULT_PLATFORM if unknown."""
    if key in PLATFORMS:
        return PLATFORMS[key]
    logger.warning("Unknown platform key %r — falling back to %r", key, DEFAULT_PLATFORM)
    return PLATFORMS[DEFAULT_PLATFORM]
