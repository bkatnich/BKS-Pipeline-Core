"""Load prompt TOML files from this directory.

Files are read once per process lifetime via lru_cache — no per-request I/O.
A missing or unparseable prompt file is a hard failure: it means the deployment
is broken, not that the call should fall back silently.
"""

import tomllib
from functools import lru_cache
from pathlib import Path

_PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load_prompt(name: str) -> dict[str, object]:
    """Return the parsed TOML dict for the named prompt (e.g. 'analysis').

    Raises FileNotFoundError if the file does not exist.
    Raises tomllib.TOMLDecodeError if the file is malformed.
    Both are intentionally unhandled — a bad prompt file is a deployment error.
    """
    path = _PROMPTS_DIR / f"{name}.toml"
    with path.open("rb") as f:
        return tomllib.load(f)


def prompt_version(name: str) -> str:
    """Return the version string from a prompt file's [meta] section."""
    cfg = load_prompt(name)
    return str((cfg.get("meta") or {}).get("version") or "unknown")
