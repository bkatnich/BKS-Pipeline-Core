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
    prop_pick_count: int = 5,
) -> tuple[str, str]:
    """Return (system, user) prompts for slate props analysis.

    Args:
        today_et: Date string in ET, e.g. "2026-05-09".
        game_count: Number of games on the slate.
        games_text: Pre-formatted game lines block (Away @ Home | Spread | Total).
        player_rows: Pre-formatted prop rows block (pre-screened, sharp-first).
        round_description: "slate", "regular season", or "playoff (Round X)".
        lang_instruction: Optional i18n suffix appended to the system prompt.
        arena_context_text: Optional pre-formatted arena factors block.
        series_context_text: Optional pre-formatted playoff series context block.
        prop_pick_count: Number of prop picks to select (default 5).
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
        prop_pick_count=prop_pick_count,
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
    """
    cfg = load_prompt("game_insight")
    ctx = get_active_config().prompt_context
    system: str = resolve_sport_tokens(str((cfg["system"])["text"]), ctx)  # type: ignore[index]
    schema: str = resolve_sport_tokens(str((cfg["schema"])["text"]), ctx)  # type: ignore[index]

    home: str = str(game_ctx.get("home_team") or "")
    away: str = str(game_ctx.get("away_team") or "")
    game_dt: str = str(game_ctx.get("game_datetime") or "")
    ou = game_ctx.get("over_under")
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
    if ou_open is not None or home_itt_open is not None:
        ou_open_str = f"{ou_open}" if ou_open is not None else "N/A"
        home_open_str = f"{home_itt_open}" if home_itt_open is not None else "N/A"
        lines_block += f"\nOpening: O/U {ou_open_str} | {home} ITT {home_open_str}"

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
        return f"{label} defense (avg pts allowed by pos): " + ", ".join(parts)

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


def build_prop_llm_take_prompt(props: "list[dict[str, object]]") -> "tuple[str, str]":
    """Return (system, user) prompts for a batched prop LLM-take call.

    Args:
        props: The final top_prop_opportunities result list as built by
               _build_top_prop_opportunities() in handlers.py. Each entry must
               have player_id, player_name, team, market, and a blackkatt_instinct
               dict with direction, predicted_value, probability, market_probability,
               edge_pp.
    """
    cfg = load_prompt("prop_llm_take")
    ctx = get_active_config().prompt_context
    system: str = resolve_sport_tokens(str((cfg["system"])["text"]), ctx)  # type: ignore[index]
    schema: str = resolve_sport_tokens(str((cfg["schema"])["text"]), ctx)  # type: ignore[index]

    rows: list[str] = []
    for p in props:
        player_id = str(p.get("player_id") or "")
        player_name = str(p.get("player_name") or "")
        team = str(p.get("team") or "")
        market = str(p.get("market") or "")
        instinct: "dict[str, object]" = dict(p.get("blackkatt_instinct") or {})  # type: ignore[arg-type]
        direction = str(instinct.get("direction") or "over")
        predicted = instinct.get("predicted_value")
        our_prob = instinct.get("probability")
        mkt_prob = instinct.get("market_probability")
        edge_pp = instinct.get("edge_pp")

        predicted_str = f"{predicted}" if predicted is not None else "N/A"
        our_prob_str = f"{round(float(our_prob) * 100)}%" if our_prob is not None else "N/A"
        mkt_prob_str = f"{round(float(mkt_prob) * 100)}%" if mkt_prob is not None else "N/A"
        edge_str = f"+{edge_pp}pp" if edge_pp is not None else "N/A"

        parts = [
            f"player_id={player_id}",
            f"{player_name} ({team})" if team else player_name,
            f"market={market}",
            direction,
            f"predicted={predicted_str}",
            f"our_prob={our_prob_str}",
            f"mkt_prob={mkt_prob_str}",
            f"edge={edge_str}",
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

        rows.append(" | ".join(parts))

    props_block = "\n".join(rows) if rows else "(no props)"

    user = f"""Props to evaluate ({len(props)} total):
{props_block}

Output schema (return only this JSON array, one object per prop, same order):
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
