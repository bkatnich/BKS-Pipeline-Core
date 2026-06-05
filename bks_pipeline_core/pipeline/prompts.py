"""Central registry for all Claude prompts and the sole call site for the Anthropic API.

Rules:
- All prompt strings and model/token constants MUST be defined here.
- All Anthropic API calls MUST go through call_claude(). Direct anthropic imports
  in other modules are forbidden.
"""

from bks_pipeline_core.sport_config import get_active_config

# ---------------------------------------------------------------------------
# Model + token constants
# ---------------------------------------------------------------------------

ANALYSIS_MODEL = "claude-sonnet-4-6"
ANALYSIS_MAX_TOKENS_CACHED = 3000  # on-demand path (GET handler, cache miss)
ANALYSIS_MAX_TOKENS_BACKGROUND = 4096  # background generation path

GAME_INSIGHT_MODEL = "claude-haiku-4-5-20251001"
GAME_INSIGHT_MAX_TOKENS = 1500

PROP_LLM_TAKE_MODEL = "claude-haiku-4-5-20251001"
PROP_LLM_TAKE_MAX_TOKENS = 1800

TRANSLATION_MODEL = "claude-haiku-4-5-20251001"
TRANSLATION_MAX_TOKENS = 512

# ---------------------------------------------------------------------------
# Slate analysis
# ---------------------------------------------------------------------------

_ANALYSIS_SCHEMA = """{
  "emerging_plays": [
    // *** THIS IS THE MOST IMPORTANT SECTION. ***
    // Array of 3-5 strings. Non-obvious, mid-tier players representing the sharpest edges.
    //
    // Selection criteria (ALL must apply):
    //   1. Recent Avg FP is notably higher than Season Avg FP (trending up).
    //   2. NOT a consensus top-tier superstar (exclude top ~10 projected players unless
    //      their trend is extreme).
    //   3. Matchup tags or game environment support continued upside.
    //
    // Each string: 2-3 sentences covering the trend (Recent vs Season avg and what's
    // driving it), the matchup edge or game environment today, and the upside case.
  ],
  "top_projections": [
    // Array of 2-3 strings. The slate's highest-projected players.
    // BRIEF — 1-2 sentences each: projected FP, matchup note, done.
  ],
  "key_trends": [
    // Array of 3-5 strings. Each covers one of:
    //   - a matchup edge (soft/tough tags vs. player strength)
    //   - a pace or total-driven opportunity
    //   - an injury-driven or rotation-driven usage bump
    //   - a team-level trend visible in the Recent vs Season splits
    // No bullet prefixes.
  ],
  "data_confidence": "high",
  // "high"  — <10% of players show 0.0 FP or missing tags/trends
  // "medium" — 10-25% have data gaps
  // "low"   — >25% have data gaps or other quality concerns
  "slate_narrative": "Single string — all game paragraphs joined by \\n\\n, then the construction paragraph. **Bold** player names. Kept for backward compat.",
  "slate_narrative_sections": [
    // Ordered array of { "title": string, "body": string } objects.
    //
    // 1. OVERVIEW (required first): title="" (empty string), body=2-3 sentences.
    //    Lead with the best EMERGING play (not highest projection). Identify the best
    //    game environment and the key construction angle.
    //
    // 2. ONE SECTION PER GAME (same order as Game Lines):
    //    title="AWAY @ HOME" (e.g. "BOS @ MIL"), body=3-5 sentences.
    //    Cover total/spread implications, then mid-tier trending players BEFORE stars.
    //
    // 3. LINEUP CONSTRUCTION (required last, title must be exactly "Lineup Construction"):
    //    body=3-5 sentences. Frame the build around emerging plays as differentiators.
  ]
}"""


def build_analysis_prompts(
    today_et: str,
    game_count: int,
    games_text: str,
    player_rows: str,
    round_description: str = "slate",
    lang_instruction: str = "",
    salary_medians_text: str = "",
    arena_context_text: str = "",
    series_context_text: str = "",
) -> tuple[str, str]:
    """Return (system, user) prompts for slate analysis (v3).

    Args:
        today_et: Date string in ET, e.g. "2026-05-09".
        game_count: Number of games on the slate.
        games_text: Pre-formatted game lines block (Away @ Home | Spread | Total).
        player_rows: Pre-formatted player pool block (Name | Pos | Team | Opp | Opp Score | Matchup Tags | Recent Trends | Salary).
        round_description: "slate", "regular season", or "playoff (Round X)".
        lang_instruction: Optional i18n suffix appended to the system prompt.
        salary_medians_text: Optional pre-formatted position salary medians line, e.g.
            "Position salary medians: PG: $7,800, SG: $6,400, ...". Empty string
            when no salary data is available.
        arena_context_text: Optional pre-formatted arena factors block (elevation, travel distance).
            Only populated when at least one game has a meaningful arena factor. Empty string otherwise.
        series_context_text: Optional pre-formatted playoff series context block (series score,
            game number, per-player series averages). Only populated during playoffs. Empty string otherwise.
    """
    system = (
        f"You are a {get_active_config().sport_display_name} analyst writing for a sharp, experienced sports audience. "
        "You MUST base every claim on the data provided below — projections, game lines, "
        "matchup tags, recent performance trends, and salary context where available. "
        "You MUST NOT infer or fabricate injury status or playing-time projections "
        "beyond what is explicitly stated. "
        "Your response MUST be a single raw JSON object. No markdown fences, no preamble, "
        f"no trailing text. The first character of your response must be {{{lang_instruction}"
    )
    playoff_context = (
        "If this is a playoff slate, prioritize identifying role players and mid-tier options "
        "whose usage, minutes, or role has expanded during the postseason. Playoff rotations "
        "tighten to 8-9 players, which concentrates opportunity. Look for non-stars absorbing "
        "meaningful touches — these are often the highest-edge plays on the slate."
        if "playoff" in round_description.lower()
        else "Give weight to pace-of-play edges and rest/back-to-back situations."
    )

    salary_context_block = f"\n## Salary Context\n{salary_medians_text}\n" if salary_medians_text else ""
    arena_context_block = f"\n## Arena Factors\n{arena_context_text}\n" if arena_context_text else ""
    series_context_block = f"\n## Playoff Series Context\n{series_context_text}\n" if series_context_text else ""
    series_rules = (
        "- Playoff Series Context is provided. Factor in series score (desperation vs. close-out), "
        "game number, and per-player series trends when assessing upside. A team trailing 3-1 faces "
        "must-win pressure; a team up 3-1 may rest stars in a blowout. Players whose series avg FP "
        "exceeds their season avg are trending into this matchup — prioritize them."
        if series_context_text
        else ""
    )
    arena_rules = (
        "- Arena factors are provided. Altitude (Denver 5,280 ft) and long road trips suppress "
        "visiting team stamina and shooting efficiency — especially in Q4. Factor this into "
        "ceiling projections for visiting players."
        if arena_context_text
        else ""
    )
    salary_column_note = "<!-- Salary = slate salary. -->\n" if salary_medians_text else "<!-- Salary column is N/A — no slate salary data available. -->\n"
    salary_rules = (
        "- Salary context is provided. Use it to identify value plays (high projection relative to salary) "
        "and salary traps (expensive players whose projection does not justify the price). "
        "Reference specific salaries and the position medians when making value claims."
        if salary_medians_text
        else "- Salary data is not available for this slate. Do not reference salary, price, value, cost, or bargain."
    )

    user = f"""Today is {today_et}. This is a {game_count}-game {get_active_config().sport_collection_key.upper()} {round_description} slate.

{playoff_context}

## Game Lines
<!-- Each row: Away @ Home | Spread | Total -->
{games_text}
{salary_context_block}{arena_context_block}{series_context_block}
## Player Pool
<!-- "Recent Avg FP" = average fantasy points over the last 5 games played. -->
<!-- "Season Avg FP" = full season average fantasy points. -->
<!-- A player whose Recent Avg FP significantly exceeds Season Avg FP is trending UP. -->
{salary_column_note}{player_rows}

## Output Schema
{_ANALYSIS_SCHEMA}

## Rules
- 0.0 FP does not mean inactive. Do not assume injury or rest.
- Every claim must trace back to a number, tag, or line in the provided data.
- Emerging over obvious. The primary value is surfacing NON-OBVIOUS plays. If the output is just a list of big names, it has failed.
- Trend math must be real. When citing Recent vs Season avg, the numbers must match the input data exactly.
- If Recent Avg FP or Season Avg FP is absent for a player, omit those claims for that player. Do not refuse to produce the JSON object.
- Game sections in slate_narrative_sections must follow the same order as Game Lines.
- Bold all player names with double asterisks.
{salary_rules}
{arena_rules}
{series_rules}
- Return only the JSON object, no prose outside it."""
    return system, user


# ---------------------------------------------------------------------------
# Per-game insights (Stage 1 of map-reduce analysis)
# ---------------------------------------------------------------------------

_GAME_INSIGHT_SCHEMA = """{
  "key_players": [
    // Array of 2-4 strings. Highest-edge players for DFS in this game (both teams).
    // Each string: player name + 1 sentence stating the edge (stat signal or matchup).
  ],
  "key_player_cards": [
    // Structured array — one object per player in key_players_input, same order.
    // Do NOT add or remove players from the list provided. Do NOT change player_id values.
    {
      "player_id": "string — copy exactly from key_players_input",
      "edge_sentence": "1 sentence: the sharpest DFS edge for this player today (signal + matchup context)."
    }
  ],
  "matchup_narrative": "2-3 sentences. Cover the total/spread angle and the best DFS construction angle.",
  "game_stack_targets": [
    // Array of 1-3 strings. Team(s) to stack and why (implied total, park, bullpen, SP matchup).
  ],
  "game_environment": "high-scoring" | "pitcher-duel" | "neutral",
  "injury_flags": [
    // Array of strings. ONLY players with non-Active injury status.
    // Format each: "Player Name (Status: comment)". Empty list if all healthy.
  ],
  "prop_picks": [
    // Array of objects — ONLY for players where prop_lines data was provided.
    // Omit entirely (empty array) if no prop lines were supplied for this game.
    // Include only lines where the edge justifies a directional call.
    {
      "player_id": "string — copy exactly from prop_lines input",
      "market": "string — e.g. hits_0.5, total_bases_1.5",
      "direction": "over" | "under",
      "rationale": "1 sentence: why this line has edge given the matchup and model signal."
    }
  ],
  "line_movement_signal": "sharp" | "fade" | "neutral"
  // sharp: home ITT moved up >=0.5 runs (offense bet into)
  // fade:  home ITT moved down >=0.5 runs
  // neutral: <0.5 run movement, or opening lines unavailable
}"""

_GAME_INSIGHT_SYSTEM = (
    "You are a baseball DFS analyst. Produce a compact JSON game analysis object. "
    "Base every claim on the numbers provided. No prose outside the JSON object. "
    "The first character of your response must be {."
)


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
    # Narrow type imports only needed at call time — avoid top-level Any import
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

    # Lines block
    ou_str = f"{ou}" if ou is not None else "N/A"
    home_itt_str = f"{home_itt}" if home_itt is not None else "N/A"
    away_itt_str = f"{away_itt}" if away_itt is not None else "N/A"
    spread_str = f"{home} {spread:+.1f}" if spread is not None else "N/A"

    lines_block = f"Lines: O/U {ou_str} | {home} ITT {home_itt_str} | {away} ITT {away_itt_str} | Spread {spread_str}"
    # Opening lines block — omit entirely if unavailable (avoids confusing Claude)
    if ou_open is not None or home_itt_open is not None:
        ou_open_str = f"{ou_open}" if ou_open is not None else "N/A"
        home_open_str = f"{home_itt_open}" if home_itt_open is not None else "N/A"
        lines_block += f"\nOpening: O/U {ou_open_str} | {home} ITT {home_open_str}"

    # Bullpen block
    bp_parts = []
    if home_bp is not None:
        bp_parts.append(f"{home}: {home_bp:.1f} IP")
    if away_bp is not None:
        bp_parts.append(f"{away}: {away_bp:.1f} IP")
    bullpen_block = "Bullpen load (last 3d): " + (" | ".join(bp_parts) if bp_parts else "N/A")

    # Defense block
    def _fmt_defense(d: "dict[str, object] | None", label: str) -> str:
        if not d:
            return f"{label} defense: N/A"
        parts = [f"{pos}:{v:.1f}" for pos, v in sorted(d.items()) if v is not None]
        return f"{label} defense (avg pts allowed by pos): " + ", ".join(parts)

    defense_block = _fmt_defense(home_def, home) + "\n" + _fmt_defense(away_def, away)

    # Player block — one line per player, sorted by opp_ranking_score desc
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

    # key_player_cards_input: top N players pre-selected server-side — Claude writes edge_sentence only
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
        key_player_cards_block = (
            "\nkey_players_input (write edge_sentence for each, preserve player_id):\n"
            + "\n".join(kp_lines)
        )
    else:
        key_player_cards_block = ""

    # prop_lines_input: {player_id: [{market, line, direction, prob, edge}, ...]} — edge markets only
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
{_GAME_INSIGHT_SCHEMA}"""

    return _GAME_INSIGHT_SYSTEM, user


# ---------------------------------------------------------------------------
# Per-prop LLM take (batched enrichment of top_prop_opportunities)
# ---------------------------------------------------------------------------

_PROP_LLM_TAKE_SCHEMA = """[
  {
    "player_id": "string — copy exactly from input",
    "market": "string — copy exactly from input",
    "agrees": true,
    // true if you agree the model's direction has edge; false if you would fade it.
    "confidence": "high" | "medium" | "low",
    // high: strong evidence, clear edge; medium: directionally sound but some uncertainty;
    // low: marginal edge, close call.
    "suggested_value": 1.5,
    // Your best estimate of the true expected value for this stat today (float, same units as line).
    // May match or differ from predicted_value.
    "rationale": "1 sentence: the sharpest reason to take or fade this prop given the numbers."
  }
]"""

_PROP_LLM_TAKE_SYSTEM = (
    "You are a baseball prop betting analyst. "
    "Evaluate each prop entry and return a JSON array — one object per prop, same order as input. "
    "Base every claim strictly on the numbers provided. "
    "No prose outside the JSON array. The first character of your response must be [."
)


def build_prop_llm_take_prompt(props: "list[dict[str, object]]") -> "tuple[str, str]":
    """Return (system, user) prompts for a batched prop LLM-take call.

    Args:
        props: The final top_prop_opportunities result list as built by
               _build_top_prop_opportunities() in handlers.py. Each entry must
               have player_id, player_name, team, market, stat (via 'stat' key),
               and a blackkatt_instinct dict with direction, predicted_value,
               probability, market_probability, edge_pp.
    """
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
{_PROP_LLM_TAKE_SCHEMA}"""

    return _PROP_LLM_TAKE_SYSTEM, user


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
    """Return the system prompt for the translation call.

    Args:
        full_lang_name: Full language name, e.g. "Spanish".
        context: Short hint for the translator, e.g. "NBA push notification body".
    """
    context_hint = f" Context: {context}." if context else ""
    return (
        f"You are a professional sports translator specializing in {get_active_config().sport_collection_key.upper()} daily fantasy sports.{context_hint} "
        f"Translate the following text to {full_lang_name}. "
        "Preserve all player names, team abbreviations, numbers, and Markdown formatting exactly. "
        "Return only the translated text — no explanation, no preamble."
    )
