"""Central registry for all Claude prompts and the sole call site for the Anthropic API.

Rules:
- All prompt text lives in bks_pipeline_core/prompts/*.toml — edit there, not here.
- All Anthropic API calls MUST go through call_claude(). Direct anthropic imports
  in other modules are forbidden.
- Model/token constants are derived from the TOML [meta] sections at import time
  (lru_cached, so one file read per cold start).
"""

from bks_pipeline_core.prompts.loader import load_prompt, prompt_version, resolve_sport_tokens
from bks_pipeline_core.sport_config import get_active_config

# ---------------------------------------------------------------------------
# Model + token constants — sourced from TOML [meta]
# ---------------------------------------------------------------------------


def _meta(name: str, key: str) -> object:
    cfg = load_prompt(name)
    return (cfg.get("meta") or {})[key]


ANALYSIS_MODEL: str = str(_meta("analysis", "model"))
ANALYSIS_MAX_TOKENS_CACHED: int = int(_meta("analysis", "max_tokens_cached"))
ANALYSIS_MAX_TOKENS_BACKGROUND: int = int(_meta("analysis", "max_tokens_background"))

GAME_INSIGHT_MODEL: str = str(_meta("game_insight", "model"))
GAME_INSIGHT_MAX_TOKENS: int = int(_meta("game_insight", "max_tokens"))

PROP_LLM_TAKE_MODEL: str = str(_meta("prop_llm_take", "model"))
PROP_LLM_TAKE_MAX_TOKENS: int = int(_meta("prop_llm_take", "max_tokens"))

PROP_SLATE_SYNTHESIS_MODEL: str = str(_meta("prop_slate_synthesis", "model"))
PROP_SLATE_SYNTHESIS_MAX_TOKENS: int = int(_meta("prop_slate_synthesis", "max_tokens"))

PROP_COMBINED_MODEL: str = str(_meta("prop_combined", "model"))
PROP_COMBINED_MAX_TOKENS: int = int(_meta("prop_combined", "max_tokens"))

TRANSLATION_MODEL: str = str(_meta("translation", "model"))
TRANSLATION_MAX_TOKENS: int = int(_meta("translation", "max_tokens"))


# ---------------------------------------------------------------------------
# Slate analysis
# ---------------------------------------------------------------------------


def build_analysis_prompts(
    today_et: str,
    game_count: int,
    games_text: str,
    player_rows: str,
    round_description: str = "slate",
    lang_instruction: str = "",
    arena_context_text: str = "",
    series_context_text: str = "",
) -> tuple[str, str]:
    """Return (system, user) prompts for slate synthesis analysis (Stage 2).

    Args:
        today_et: Date string in ET, e.g. "2026-05-09".
        game_count: Number of games on the slate.
        games_text: Pre-formatted game lines block (Away @ Home | Spread | Total).
        player_rows: Pre-formatted opportunity-score rows (top-N players by opp_ranking_score).
        round_description: "slate", "regular season", or "playoff (Round X)".
        lang_instruction: Optional i18n suffix appended to the system prompt.
        arena_context_text: Optional pre-formatted arena factors block.
        series_context_text: Optional pre-formatted playoff series context block.
    """
    cfg = load_prompt("analysis")
    sport = get_active_config()
    ctx = sport.prompt_context

    schema = resolve_sport_tokens(str((cfg["schema"])["text"]), ctx)  # type: ignore[index]
    system = resolve_sport_tokens(
        str((cfg["system"])["text"]),  # type: ignore[index]
        ctx,
    ).format(
        sport_display_name=sport.sport_display_name,
        lang_instruction=lang_instruction,
        schema=schema,
    )

    playoff_context = (
        "Playoff context: factor series score (desperation vs. rest risk), pitcher usage, and bullpen availability when assessing prop direction."
        if "playoff" in round_description.lower()
        else "Factor pace-of-play edges, bullpen fatigue, and weather where relevant."
    )

    arena_context_block = f"\n## Arena Factors\n{arena_context_text}\n" if arena_context_text else ""
    series_context_block = f"\n## Playoff Series Context\n{series_context_text}\n" if series_context_text else ""

    series_rules = (
        "- Series Context provided: factor score (desperation vs. rest risk), game number, and bullpen/rotation usage trends."
        if series_context_text
        else ""
    )
    arena_rules = (
        "- Arena factors provided: apply park and environmental effects to prop direction and ceiling."
        if arena_context_text
        else ""
    )

    user = resolve_sport_tokens(
        str((cfg["user"])["template"]),  # type: ignore[index]
        ctx,
    ).format(
        today_et=today_et,
        game_count=game_count,
        sport_key=sport.sport_collection_key.upper(),
        round_description=round_description,
        playoff_context=playoff_context,
        games_text=games_text,
        arena_context_block=arena_context_block,
        series_context_block=series_context_block,
        player_rows=player_rows,
        arena_rules=arena_rules,
        series_rules=series_rules,
    )
    return system, user


def analysis_prompt_version() -> str:
    """Return the current analysis prompt version string (from TOML [meta])."""
    return prompt_version("analysis")


# ---------------------------------------------------------------------------
# Per-game insights (Stage 1 of map-reduce analysis)
# ---------------------------------------------------------------------------


def build_game_insight_prompt(game_ctx: "dict[str, object]") -> "tuple[str, str]":
    """Return (system, user) prompts for a single-game per-game insight call.

    Args:
        game_ctx: Dict assembled by _generate_game_insights() in handlers.py. Keys:
            game_id, home_team, away_team, game_datetime,
            over_under, home_implied_total, away_implied_total, home_spread,
            ou_open, home_implied_open,  (opening lines — may be None)
            home_probable_pitcher, away_probable_pitcher,  (may be None)
            home_park_label,  (from park_context_label())
            umpire,  (may be None)
            home_bullpen_ip, away_bullpen_ip,  (may be None)
            home_defense, away_defense,  (pts_allowed_by_position dicts — may be None)
            players,  (list of snapshot entry dicts, both teams)
            bk_ou_pick,  ("over"/"under"/None — deterministic model call; None = push zone)
            proj_total,  (blended run total from 10-signal model — may be None)
            bk_ou_edge,  (abs delta between proj_total and Vegas line — may be None)
    """
    cfg = load_prompt("game_insight")
    ctx = get_active_config().prompt_context
    system: str = resolve_sport_tokens(str((cfg["system"])["text"]), ctx)  # type: ignore[index]
    schema: str = resolve_sport_tokens(str((cfg["schema"])["text"]), ctx)  # type: ignore[index]

    home: str = str(game_ctx.get("home_team") or "")
    away: str = str(game_ctx.get("away_team") or "")
    game_dt: str = str(game_ctx.get("game_datetime") or "")
    ou = game_ctx.get("over_under")
    bk_ou_pick = game_ctx.get("bk_ou_pick")
    proj_total = game_ctx.get("proj_total")
    bk_ou_edge = game_ctx.get("bk_ou_edge")
    home_itt = game_ctx.get("home_implied_total")
    away_itt = game_ctx.get("away_implied_total")
    spread = game_ctx.get("home_spread")
    ou_open = game_ctx.get("ou_open")
    home_itt_open = game_ctx.get("home_implied_open")
    home_sp = game_ctx.get("home_probable_pitcher") or "unknown"
    away_sp = game_ctx.get("away_probable_pitcher") or "unknown"
    park_label: str = str(game_ctx.get("home_park_label") or "")
    umpire = game_ctx.get("umpire") or "unknown"
    home_bp = game_ctx.get("home_bullpen_ip")
    away_bp = game_ctx.get("away_bullpen_ip")
    home_def: "dict[str, object] | None" = game_ctx.get("home_defense")  # type: ignore[assignment]
    away_def: "dict[str, object] | None" = game_ctx.get("away_defense")  # type: ignore[assignment]
    players: "list[dict[str, object]]" = list(game_ctx.get("players") or [])  # type: ignore[arg-type]

    ou_str = f"{ou}" if ou is not None else "N/A"
    home_itt_str = f"{home_itt}" if home_itt is not None else "N/A"
    away_itt_str = f"{away_itt}" if away_itt is not None else "N/A"
    spread_str = f"{home} {spread:+.1f}" if spread is not None else "N/A"

    lines_block = f"Lines: O/U {ou_str} | {home} ITT {home_itt_str} | {away} ITT {away_itt_str} | Spread {spread_str}"

    # BK projection block — deterministic model output; shown to ground the LLM's ou_pick.
    if proj_total is not None:
        bk_ou_pick_str = str(bk_ou_pick).lower() if bk_ou_pick is not None else "none"
        bk_ou_edge_str = f" (edge {bk_ou_edge:.1f})" if bk_ou_edge is not None else ""
        lines_block += f"\nBK Projection: proj_total={proj_total} | bk_ou_pick={bk_ou_pick_str}{bk_ou_edge_str}"

    if ou_open is not None and home_itt_open is not None:
        ou_delta = round(float(ou) - float(ou_open), 1) if ou is not None else None
        itt_delta = round(float(home_itt) - float(home_itt_open), 1) if home_itt is not None else None
        ou_delta_str = f" ({ou_delta:+.1f})" if ou_delta is not None else ""
        itt_delta_str = f" ({itt_delta:+.1f})" if itt_delta is not None else ""
        lines_block += f"\nOpening: O/U {ou_open}{ou_delta_str} | {home} ITT {home_itt_open}{itt_delta_str}"

    bp_parts = []
    if home_bp is not None:
        bp_parts.append(f"{home}: {home_bp:.1f} IP")
    if away_bp is not None:
        bp_parts.append(f"{away}: {away_bp:.1f} IP")
    bullpen_block = "Bullpen load (last 3d): " + (" | ".join(bp_parts) if bp_parts else "N/A")

    def _fmt_defense(d: "dict[str, object] | None", label: str) -> str:
        if not d:
            return f"{label} defense: N/A"
        parts = [f"{pos}:{v:.1f}" for pos, v in sorted(d.items()) if v is not None]
        return f"{label} defense (avg PA allowed per game by pos): " + ", ".join(parts)

    defense_block = _fmt_defense(home_def, home) + "\n" + _fmt_defense(away_def, away)

    def _fmt_player(p: "dict[str, object]") -> str:
        name: str = str(p.get("name") or "")
        team: str = str(p.get("team") or "")
        pos: str = str(p.get("position") or "")
        score = p.get("opp_ranking_score")
        floor_fp = p.get("fp_floor")
        ceil_fp = p.get("fp_ceiling")
        bat_ord = p.get("batting_order")
        confirmed_starter = p.get("is_confirmed_starter")
        hot = p.get("hot_streak")
        cold = p.get("cold_streak")
        recent = p.get("avg_pa_per_game")
        season = p.get("season_avg_score")
        injury = str(p.get("injury_status") or "").strip()
        reasons: "list[object]" = list(p.get("top_pick_reasons") or [])  # type: ignore[arg-type]

        parts = [f"{name} ({team}, {pos})"]
        if score is not None:
            parts.append(f"opp={score:.1f}")
        if floor_fp is not None and ceil_fp is not None:
            parts.append(f"range={floor_fp:.1f}-{ceil_fp:.1f}")
        if recent is not None:
            parts.append(f"recent={recent:.1f}")
        if season is not None:
            parts.append(f"season={season:.1f}")
        if bat_ord:
            flag = "✓" if confirmed_starter else "?"
            parts.append(f"bat#{bat_ord}{flag}")
        if hot:
            parts.append(f"hot({hot})")
        elif cold:
            parts.append(f"cold({cold})")
        if injury and injury.lower() not in ("active", ""):
            parts.append(f"[{injury}]")
        if reasons:
            reason_strs = []
            for r in reasons:
                if isinstance(r, dict):
                    rtype = str(r.get("type") or "")
                    if rtype:
                        reason_strs.append(rtype)
                elif isinstance(r, str):
                    reason_strs.append(r)
            if reason_strs:
                parts.append("signals=" + ",".join(reason_strs))
        return " | ".join(parts)

    sorted_players = sorted(
        players,
        key=lambda p: -(float(p.get("opp_ranking_score") or 0)),
    )
    player_block = "\n".join(_fmt_player(p) for p in sorted_players) or "No players available"

    key_player_cards_input: "list[dict[str, object]]" = list(game_ctx.get("key_player_cards_input") or [])  # type: ignore[arg-type]
    if key_player_cards_input:
        kp_lines = []
        for kp in key_player_cards_input:
            pid = str(kp.get("player_id") or "")
            name = str(kp.get("name") or "")
            team = str(kp.get("team") or "")
            pos = str(kp.get("position") or "")
            score = kp.get("opp_ranking_score")
            floor_fp = kp.get("fp_floor")
            ceil_fp = kp.get("fp_ceiling")
            hot = kp.get("hot_streak")
            cold = kp.get("cold_streak")
            parts = [f"player_id={pid}", f"{name} ({team}, {pos})"]
            if score is not None:
                parts.append(f"opp={score:.1f}")
            if floor_fp is not None and ceil_fp is not None:
                parts.append(f"range={floor_fp:.1f}-{ceil_fp:.1f}")
            if hot:
                parts.append(f"hot({hot})")
            elif cold:
                parts.append(f"cold({cold})")
            kp_lines.append(" | ".join(parts))
        key_player_cards_block = "\nkey_players_input (write edge_sentence for each, preserve player_id):\n" + "\n".join(kp_lines)
    else:
        key_player_cards_block = ""

    prop_lines_input: "dict[str, list[dict[str, object]]]" = dict(game_ctx.get("prop_lines_input") or {})  # type: ignore[arg-type]
    if prop_lines_input:
        prop_lines_rows = []
        for pid, markets in prop_lines_input.items():
            for m in markets:
                market = str(m.get("market") or "")
                direction = str(m.get("direction") or "over")
                prob = m.get("prob")
                edge = m.get("edge")
                prob_str = f"{round(float(prob) * 100)}%" if prob is not None else ""
                edge_str = f"+{round(float(edge) * 100)}pp" if edge is not None else ""
                prop_lines_rows.append(f"  player_id={pid} | {market} {direction} | model={prob_str} | edge={edge_str}")
        prop_lines_block = "\nProp lines with edge (use for prop_picks):\n" + "\n".join(prop_lines_rows)
    else:
        prop_lines_block = ""

    user = f"""Game: {away} @ {home} — {game_dt}
{lines_block}
Park: {park_label or "standard"}
Pitchers: {home} SP: {home_sp} | {away} SP: {away_sp}
Umpire: {umpire}
{bullpen_block}
{defense_block}

Players (sorted by opportunity score):
{player_block}{key_player_cards_block}{prop_lines_block}

Output schema (return only this JSON object, no prose):
{schema}"""

    return system, user


# ---------------------------------------------------------------------------
# Per-prop LLM take (batched enrichment of top_prop_opportunities)
# ---------------------------------------------------------------------------


def _build_prop_row(p: "dict[str, object]") -> str:
    """Render a single prop dict as a compact text row for LLM prompts."""
    player_id = str(p.get("player_id") or "")
    player_name = str(p.get("player_name") or "")
    team = str(p.get("team") or "")
    market = str(p.get("market") or "")
    instinct: "dict[str, object]" = dict(p.get("blackkatt_instinct") or {})  # type: ignore[arg-type]
    direction = str(instinct.get("direction") or p.get("direction") or "over")
    predicted = instinct.get("predicted_value") or p.get("predicted_value")
    our_prob = instinct.get("probability")
    mkt_prob = instinct.get("market_probability")
    edge_pp = instinct.get("edge_pp") or p.get("edge_pp")

    predicted_str = f"{predicted}" if predicted is not None else "N/A"
    our_prob_str = f"{round(float(our_prob) * 100)}%" if our_prob is not None else "N/A"
    mkt_prob_str = f"{round(float(mkt_prob) * 100)}%" if mkt_prob is not None else "N/A"
    edge_str = f"+{edge_pp}pp" if edge_pp is not None else "N/A"

    sharp_flag = "Sharp" if p.get("is_sharp") else "Soft"
    parts = [
        f"player_id={player_id}",
        f"{player_name} ({team})" if team else player_name,
        f"market={market}",
        direction,
        f"predicted={predicted_str}",
        f"our_prob={our_prob_str}",
        f"mkt_prob={mkt_prob_str}",
        f"edge={edge_str}",
        sharp_flag,
    ]

    bookmakers: "dict[str, object]" = dict(p.get("bookmakers") or {})  # type: ignore[arg-type]
    if bookmakers:
        bk_parts = []
        for bk_name, bk_data in bookmakers.items():
            if isinstance(bk_data, dict):
                over_odds = bk_data.get("over_odds")
                under_odds = bk_data.get("under_odds")
                bk_parts.append(f"{bk_name}: over {over_odds} / under {under_odds}")
        if bk_parts:
            parts.append(" | ".join(bk_parts))

    ctx_parts: "list[str]" = []
    bat_order = p.get("batting_order")
    confirmed = p.get("is_confirmed_starter")
    if bat_order is not None:
        slot_flag = "✓" if confirmed else "?"
        ctx_parts.append(f"bat#{bat_order}{slot_flag}")
    hot = p.get("hot_streak")
    cold = p.get("cold_streak")
    if hot:
        ctx_parts.append(f"hot({hot})")
    elif cold:
        ctx_parts.append(f"cold({cold})")
    opp_hand = p.get("opp_pitcher_hand")
    season_ops = p.get("season_ops")
    vs_left = p.get("vs_left_ops")
    vs_right = p.get("vs_right_ops")
    if opp_hand in ("L", "R") and season_ops:
        split_ops = vs_left if opp_hand == "L" else vs_right
        if split_ops:
            ratio = round(float(split_ops) / float(season_ops), 2)  # type: ignore[arg-type]
            ctx_parts.append(f"vs{opp_hand}={ratio:.2f}xOPS")
    park = p.get("park_factor_tier")
    elev = p.get("elevation_tier")
    is_home = p.get("is_home")
    if park and park != "neutral":
        ctx_parts.append(f"park={park}")
    if elev and elev == "high":
        ctx_parts.append("elev=high")
    if is_home is not None:
        ctx_parts.append("home" if is_home else "away")
    ump = p.get("umpire_name")
    if ump:
        ctx_parts.append(f"ump={ump}")
    k9 = p.get("opp_sp_k9")
    bb9 = p.get("opp_sp_bb9")
    if k9 is not None:
        ctx_parts.append(f"SP_K9={k9:.1f}")
    if bb9 is not None:
        ctx_parts.append(f"SP_BB9={bb9:.1f}")
    opp = p.get("opponent_abbr")
    if opp:
        ctx_parts.append(f"vs={opp}")
    if ctx_parts:
        parts.append("[" + " ".join(ctx_parts) + "]")

    return " | ".join(parts)


def build_prop_llm_take_prompt(
    props: "list[dict[str, object]]",
    near_misses: "list[dict[str, object]] | None" = None,
) -> "tuple[str, str]":
    """Return (system, user) prompts for a batched prop LLM-take call.

    Args:
        props: The final top_prop_opportunities result list. Each entry must have
               player_id, player_name, team, market, and a blackkatt_instinct dict
               with direction, predicted_value, probability, market_probability, edge_pp.
        near_misses: Optional list of near-miss prop candidates for the nomination task.
                     Each entry has player_id, player_name, team, market, direction,
                     predicted_value, edge_pp, and scoring-engine context fields.
                     When None or empty, the nomination section is omitted and the
                     response schema remains a bare JSON array (backwards-compatible).
    """
    cfg = load_prompt("prop_llm_take")
    ctx = get_active_config().prompt_context
    system: str = resolve_sport_tokens(str((cfg["system"])["text"]), ctx)  # type: ignore[index]
    schema: str = resolve_sport_tokens(str((cfg["schema"])["text"]), ctx)  # type: ignore[index]

    props_block = "\n".join(_build_prop_row(p) for p in props) if props else "(no props)"

    if near_misses:
        nm_block = "\n".join(_build_prop_row(nm) for nm in near_misses)
        user = f"""Props to evaluate ({len(props)} total):
{props_block}

Near-miss props for nomination consideration ({len(near_misses)} total):
{nm_block}

Output schema (return only this JSON object):
{schema}"""
    else:
        # No near-misses: return bare array for backwards compatibility with callers
        # that don't yet handle the wrapped object format.
        bare_schema = schema
        try:
            # Extract just the takes array schema from the object schema
            import re as _re
            m = _re.search(r'"takes":\s*(\[.*?\])', schema, _re.DOTALL)
            if m:
                bare_schema = m.group(1)
        except Exception:
            pass
        user = f"""Props to evaluate ({len(props)} total):
{props_block}

Output schema (return only this JSON array, one object per prop, same order):
{bare_schema}"""

    return system, user


# ---------------------------------------------------------------------------
# Cross-prop slate synthesis (Sonnet, post-Haiku)
# ---------------------------------------------------------------------------


def build_prop_slate_synthesis_prompt(props: "list[dict[str, object]]") -> "tuple[str, str]":
    """Return (system, user) prompts for the Sonnet cross-prop slate synthesis call.

    Fires once per snapshot_prop_predictions run, after Haiku enrichment and
    _stamp_llm_fade. Gives Sonnet the full final prop list plus per-game context
    (O/U, ITT, spread, park, venue) so it can rank by conviction, write a
    slate narrative, and flag structural contradictions across props.

    Args:
        props: The finalized top_prop_opportunities list (post _stamp_llm_fade).
               Each entry is expected to have player_id, player_name, team,
               market, stat, blackkatt_instinct (with direction, predicted_value,
               probability, market_probability, edge_pp, agrees, confidence),
               llm_fade, is_sharp, and optional context fields
               (vegas_implied_team_total, vegas_over_under, vegas_spread,
               park_factor_tier, elevation_tier, is_home, opponent_abbr,
               umpire_name, batting_order, hot_streak, cold_streak).
    """
    cfg = load_prompt("prop_slate_synthesis")
    ctx = get_active_config().prompt_context
    system: str = resolve_sport_tokens(str((cfg["system"])["text"]), ctx)  # type: ignore[index]
    schema: str = resolve_sport_tokens(str((cfg["schema"])["text"]), ctx)  # type: ignore[index]

    # Build deduplicated game lines block from props context fields.
    # Key: (team, opponent, is_home) — one row per unique game side.
    seen_games: "set[tuple[str, str]]" = set()
    game_lines: "list[str]" = []
    for p in props:
        team = str(p.get("team") or "")
        opp = str(p.get("opponent_abbr") or "")
        is_home = p.get("is_home")
        if not team or not opp:
            continue
        home = team if is_home else opp
        away = opp if is_home else team
        game_key = (min(home, away), max(home, away))
        if game_key in seen_games:
            continue
        seen_games.add(game_key)
        ou = p.get("vegas_over_under")
        home_itt = p.get("vegas_implied_team_total") if is_home else None
        spread = p.get("vegas_spread")
        park = p.get("park_factor_tier") or "neutral"
        elev = p.get("elevation_tier") or "normal"
        parts = [f"{away} @ {home}"]
        if ou is not None:
            parts.append(f"O/U {ou}")
        if home_itt is not None:
            parts.append(f"{home} ITT {home_itt}")
        if spread is not None:
            parts.append(f"spread {spread:+.1f}")
        if park != "neutral":
            parts.append(f"park={park}")
        if elev == "high":
            parts.append("elev=high")
        game_lines.append(" | ".join(parts))

    games_block = "\n".join(game_lines) if game_lines else "N/A"

    rows: "list[str]" = []
    for p in props:
        player_id = str(p.get("player_id") or "")
        player_name = str(p.get("player_name") or "")
        team = str(p.get("team") or "")
        market = str(p.get("market") or "")
        instinct: "dict[str, object]" = dict(p.get("blackkatt_instinct") or {})  # type: ignore[arg-type]
        direction = str(instinct.get("direction") or "over")
        edge_pp = instinct.get("edge_pp")
        haiku_confidence = instinct.get("confidence") or "?"
        agrees = instinct.get("agrees")
        fade_flag = " [FADE]" if p.get("llm_fade") else ""
        sharp_flag = "Sharp" if p.get("is_sharp") else "Soft"
        edge_str = f"+{edge_pp}pp" if edge_pp is not None else "N/A"

        ctx_parts: "list[str]" = []
        bat = p.get("batting_order")
        if bat is not None:
            ctx_parts.append(f"bat#{bat}")
        hot = p.get("hot_streak")
        cold = p.get("cold_streak")
        if hot:
            ctx_parts.append(f"hot({hot})")
        elif cold:
            ctx_parts.append(f"cold({cold})")
        ump = p.get("umpire_name")
        if ump:
            ctx_parts.append(f"ump={ump}")
        ctx_str = " ".join(ctx_parts)

        row = f"player_id={player_id} | {player_name} ({team}) | {market} {direction} | edge={edge_str} | {sharp_flag} | haiku={haiku_confidence}{fade_flag}"
        if ctx_str:
            row += f" | [{ctx_str}]"
        rows.append(row)

    props_block = "\n".join(rows) if rows else "(no props)"

    user = f"""Game lines:
{games_block}

Props ({len(props)} total, sorted by conviction tier then edge):
{props_block}

Output schema (return only this JSON object, no prose):
{schema}"""

    return system, user


# ---------------------------------------------------------------------------
# Combined prop evaluation + nomination + synthesis (single Sonnet call)
# ---------------------------------------------------------------------------


def build_prop_combined_prompt(
    props: "list[dict[str, object]]",
    near_misses: "list[dict[str, object]] | None" = None,
) -> "tuple[str, str]":
    """Return (system, user) prompts for the combined Sonnet prop call.

    Replaces the sequential Haiku (prop_llm_take) + Sonnet (prop_slate_synthesis)
    calls with a single Sonnet call that evaluates props, nominates near-misses,
    and synthesizes the slate in one pass.

    Args:
        props: The final top_prop_opportunities list with full context fields.
               Each entry must have player_id, player_name, team, market, and a
               blackkatt_instinct dict with direction, predicted_value, probability,
               market_probability, edge_pp.
        near_misses: Optional near-miss candidates for the nomination task. When
                     None or empty, the nominations key is omitted from the schema
                     and the model skips the nomination task.
    """
    cfg = load_prompt("prop_combined")
    ctx = get_active_config().prompt_context
    system: str = resolve_sport_tokens(str((cfg["system"])["text"]), ctx)  # type: ignore[index]
    schema: str = resolve_sport_tokens(str((cfg["schema"])["text"]), ctx)  # type: ignore[index]

    props_block = "\n".join(_build_prop_row(p) for p in props) if props else "(no props)"

    # Build deduplicated game lines block (same logic as build_prop_slate_synthesis_prompt).
    seen_games: "set[tuple[str, str]]" = set()
    game_lines: "list[str]" = []
    for p in props:
        team = str(p.get("team") or "")
        opp = str(p.get("opponent_abbr") or "")
        is_home = p.get("is_home")
        if not team or not opp:
            continue
        home = team if is_home else opp
        away = opp if is_home else team
        game_key = (min(home, away), max(home, away))
        if game_key in seen_games:
            continue
        seen_games.add(game_key)
        ou = p.get("vegas_over_under")
        home_itt = p.get("vegas_implied_team_total") if is_home else None
        spread = p.get("vegas_spread")
        park = p.get("park_factor_tier") or "neutral"
        elev = p.get("elevation_tier") or "normal"
        parts = [f"{away} @ {home}"]
        if ou is not None:
            parts.append(f"O/U {ou}")
        if home_itt is not None:
            parts.append(f"{home} ITT {home_itt}")
        if spread is not None:
            parts.append(f"spread {spread:+.1f}")
        if park != "neutral":
            parts.append(f"park={park}")
        if elev == "high":
            parts.append("elev=high")
        game_lines.append(" | ".join(parts))

    games_block = "\n".join(game_lines) if game_lines else "N/A"

    if near_misses:
        nm_block = "\n".join(_build_prop_row(nm) for nm in near_misses)
        user = f"""Game lines:
{games_block}

Props to evaluate ({len(props)} total):
{props_block}

Near-miss props for nomination consideration ({len(near_misses)} total):
{nm_block}

Output schema (return only this JSON object, no prose):
{schema}"""
    else:
        user = f"""Game lines:
{games_block}

Props to evaluate ({len(props)} total):
{props_block}

Output schema (return only this JSON object, no prose — omit the nominations key):
{schema}"""

    return system, user


# ---------------------------------------------------------------------------
# API client
# ---------------------------------------------------------------------------


def call_claude(
    *,
    api_key: str,
    model: str,
    max_tokens: int,
    system: str,
    user: str,
    timeout: float = 90.0,
    cache_system: bool = False,
) -> tuple[str, dict[str, int]]:
    """Make a single-turn Claude API call and return (text, usage).

    usage keys: input_tokens, output_tokens, cache_read_input_tokens,
    cache_creation_input_tokens (all int, absent keys default to 0).

    When cache_system=True the system prompt is tagged with cache_control
    ephemeral so Anthropic caches it for up to 5 minutes. Use this for
    background generation paths where the system prompt is static and large.
    Leave False for translation calls (short, varied system prompts).

    Raises anthropic.APIError (and subclasses) on API-level failures.
    Raises RuntimeError if Claude returns no text block.
    Callers own exception handling.
    """
    import anthropic  # lazy — cold-start cost paid only when actually called

    client = anthropic.Anthropic(api_key=api_key, timeout=timeout, max_retries=0)

    system_param: str | list[dict[str, object]]
    if cache_system:
        system_param = [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}]
    else:
        system_param = system

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_param,
        messages=[{"role": "user", "content": user}],
    )
    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        raise RuntimeError(f"Claude returned no text block (model={model})")
    u = response.usage
    usage: dict[str, int] = {
        "input_tokens": getattr(u, "input_tokens", 0) or 0,
        "output_tokens": getattr(u, "output_tokens", 0) or 0,
        "cache_read_input_tokens": getattr(u, "cache_read_input_tokens", 0) or 0,
        "cache_creation_input_tokens": getattr(u, "cache_creation_input_tokens", 0) or 0,
    }
    return text, usage


# ---------------------------------------------------------------------------
# Translation
# ---------------------------------------------------------------------------


def build_translation_prompt(full_lang_name: str, context: str = "") -> str:
    """Return the system prompt for the translation call."""
    cfg = load_prompt("translation")
    sport = get_active_config()
    context_hint = f" Context: {context}." if context else ""
    return resolve_sport_tokens(
        str((cfg["system"])["text"]),  # type: ignore[index]
        sport.prompt_context,
    ).format(
        sport_key=sport.sport_collection_key.upper(),
        full_lang_name=full_lang_name,
        context_hint=context_hint,
    )
