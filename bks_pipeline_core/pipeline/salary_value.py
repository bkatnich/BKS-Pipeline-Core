"""DraftKings salary-value signal.

Returns a multiplier in [0.90, 1.10] reflecting how under/overpriced a player
is relative to the position-median salary on today's slate.

Dead zone: ±5% of median — tiny price differences are noise. Beyond that,
30% leverage on the discount: a player 20% below median earns a 1.06× boost;
20% above median earns a 0.94× cut.

The signal is dormant (returns 1.0 for all players) on off-nights when
dk_salary is not set, or when the platform is not DK.
"""

from typing import Any

from bks_pipeline_core.pipeline.opportunity_primitives import _clamp

_CLAMP_MIN = 0.90
_CLAMP_MAX = 1.10
_DEAD_ZONE = 0.05  # suppress within ±5% of median
_LEVERAGE = 0.30  # 30% of price discount translates to multiplier deviation


def compute_position_salary_medians(players: list[dict[str, Any]]) -> dict[str, float]:
    """Compute median DK salary by position across the eligible player pool.

    Uses dk_position (from DK data) with a fallback to position (from BDL).
    Returns empty dict when no players have dk_salary set (off-night).
    """
    by_pos: dict[str, list[float]] = {}
    for p in players:
        salary = p.get("dk_salary")
        if not salary:
            continue
        pos = p.get("dk_position") or p.get("position")
        if not pos:
            continue
        # DK multi-position slots (e.g. "PF/C") — attribute to first position only
        primary_pos = pos.split("/")[0]
        by_pos.setdefault(primary_pos, []).append(float(salary))

    medians: dict[str, float] = {}
    for pos, salaries in by_pos.items():
        salaries.sort()
        mid = len(salaries) // 2
        if len(salaries) % 2 == 0 and len(salaries) > 1:
            medians[pos] = (salaries[mid - 1] + salaries[mid]) / 2.0
        else:
            medians[pos] = salaries[mid]
    return medians


def salary_value_multiplier(
    player: dict[str, Any],
    position_salary_medians: dict[str, float],
    signal_clamps: dict[str, dict[str, float]] | None = None,
) -> float:
    """Return a salary-value multiplier for a single player.

    Returns 1.0 when:
    - dk_salary is not set (off-night / no slate)
    - position has no median (position not in today's pool)
    - discount is within the ±5% dead zone
    """
    salary = player.get("dk_salary")
    if not salary:
        return 1.0

    pos = player.get("dk_position") or player.get("position")
    if not pos:
        return 1.0

    primary_pos = pos.split("/")[0]
    median = position_salary_medians.get(primary_pos)
    if not median or median <= 0:
        return 1.0

    discount = (median - float(salary)) / median  # positive = cheaper than median
    if abs(discount) < _DEAD_ZONE:
        return 1.0

    raw = 1.0 + discount * _LEVERAGE
    return float(round(_clamp(raw, "salary_value_multiplier", signal_clamps, _CLAMP_MIN, _CLAMP_MAX), 4))
