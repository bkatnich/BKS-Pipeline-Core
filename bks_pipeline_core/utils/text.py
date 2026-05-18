import unicodedata


def normalize_name(name: str) -> str:
    """Lowercase and strip accents for fuzzy name matching."""
    return unicodedata.normalize("NFD", name).encode("ascii", "ignore").decode().lower()
