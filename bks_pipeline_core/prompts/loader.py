"""Load prompt TOML files from this directory and resolve sport-specific tokens.

Files are read once per process lifetime via lru_cache — no per-request I/O.
Sport-specific fragments ({sport.<key>}) are resolved at build time by the
prompt builders in pipeline/prompts.py, which pass the active SportConfig's
prompt_context dict.

A missing or unparseable prompt file is a hard failure: it means the deployment
is broken, not that the call should fall back silently.
"""

import re
import tomllib
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent

# Matches {sport.some_key} — the sport-specific injection token.
_SPORT_TOKEN_RE = re.compile(r"\{sport\.([a-z_]+)\}")


@lru_cache(maxsize=None)
def load_prompt(name: str) -> dict[str, object]:
    """Return the raw parsed TOML dict for the named prompt (e.g. 'analysis').

    Raises FileNotFoundError if the file does not exist.
    Raises tomllib.TOMLDecodeError if the file is malformed.
    Both are intentionally unhandled — a bad prompt file is a deployment error.
    """
    path = _PROMPTS_DIR / f"{name}.toml"
    with path.open("rb") as f:
        return tomllib.load(f)


def resolve_sport_tokens(text: str, prompt_context: dict[str, str]) -> str:
    """Replace all {sport.<key>} tokens in text using prompt_context.

    Raises KeyError if a token references a key not present in prompt_context,
    so missing sport configuration is caught at cold-start rather than silently
    producing malformed prompts.
    """
    def _replace(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in prompt_context:
            raise KeyError(
                f"Prompt references {{sport.{key}}} but SportConfig.prompt_context "
                f"has no '{key}' key. Add it to prompt_context in your SportConfig."
            )
        return prompt_context[key]

    return _SPORT_TOKEN_RE.sub(_replace, text)


def prompt_version(name: str) -> str:
    """Return the version string from a prompt file's [meta] section."""
    cfg = load_prompt(name)
    return str((cfg.get("meta") or {}).get("version") or "unknown")
