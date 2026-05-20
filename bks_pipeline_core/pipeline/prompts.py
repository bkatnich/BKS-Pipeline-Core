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
ANALYSIS_MAX_TOKENS_CACHED = 3000   # on-demand path (GET handler, cache miss)
ANALYSIS_MAX_TOKENS_BACKGROUND = 4096  # background generation path

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
        player_rows: Pre-formatted player pool block (Name | Pos | Team | Opp | Proj FP | Matchup Tags | Recent Avg FP | Season Avg FP | Salary | DK Avg Pts).
        round_description: "slate", "regular season", or "playoff (Round X)".
        lang_instruction: Optional i18n suffix appended to the system prompt.
        salary_medians_text: Optional pre-formatted position salary medians line, e.g.
            "Position salary medians (DK): PG: $7,800, SG: $6,400, ...". Empty string
            when no salary data is available (off-night, FD platform).
        arena_context_text: Optional pre-formatted arena factors block (elevation, travel distance).
            Only populated when at least one game has a meaningful arena factor. Empty string otherwise.
        series_context_text: Optional pre-formatted playoff series context block (series score,
            game number, per-player series averages). Only populated during playoffs. Empty string otherwise.
    """
    system = (
        "You are a DFS analyst writing for a sharp, experienced daily-fantasy audience. "
        "You MUST base every claim on the data provided below — projections, game lines, "
        "matchup tags, recent performance trends, and salary context where available. "
        "You MUST NOT infer or fabricate ownership, injury status, or minutes projections "
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
        else
        "Give weight to pace-of-play edges and rest/back-to-back situations."
    )

    salary_context_block = (
        f"\n## Salary Context\n{salary_medians_text}\n"
        if salary_medians_text
        else ""
    )
    arena_context_block = (
        f"\n## Arena Factors\n{arena_context_text}\n"
        if arena_context_text
        else ""
    )
    series_context_block = (
        f"\n## Playoff Series Context\n{series_context_text}\n"
        if series_context_text
        else ""
    )
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
    salary_column_note = (
        "<!-- Salary = DK salary. DK Avg Pts = DraftKings-reported season average. -->\n"
        if salary_medians_text
        else "<!-- Salary and DK Avg Pts columns are N/A — no slate salary data available. -->\n"
    )
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
- Game sections in slate_narrative_sections must follow the same order as Game Lines.
- Bold all player names with double asterisks.
{salary_rules}
{arena_rules}
{series_rules}
- Return only the JSON object, no prose outside it."""
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
) -> str:
    """Make a single-turn Claude API call and return the text response.

    Raises anthropic.APIError (and subclasses) on API-level failures.
    Raises RuntimeError if Claude returns no text block.
    Callers own exception handling.
    """
    import anthropic  # lazy — cold-start cost paid only when actually called

    client = anthropic.Anthropic(api_key=api_key, timeout=timeout)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    text = next((block.text for block in response.content if block.type == "text"), None)
    if text is None:
        raise RuntimeError(f"Claude returned no text block (model={model})")
    return text


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
