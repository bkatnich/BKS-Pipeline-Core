"""Sports-domain translation via Claude with Firestore caching.

translate() is the single entry point. Returns text unchanged for English.
For other languages, calls claude-haiku-4-5 and caches results in
Firestore at translations/{lang}/{sha256(text)} with a 30-day TTL.
"""

import hashlib
import logging
import threading
from datetime import datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

from bks_pipeline_core.pipeline.prompts import TRANSLATION_MAX_TOKENS, TRANSLATION_MODEL, build_translation_prompt, call_claude

logger = logging.getLogger(__name__)

_CACHE_TTL_DAYS = 30

# Map IETF language tags to full language names for the Claude prompt.
_LANG_NAMES: dict[str, str] = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "de": "German",
    "es": "Spanish",
    "fr": "French",
    "hi": "Hindi",
    "it": "Italian",
    "ja": "Japanese",
    "ko": "Korean",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "pt-br": "Brazilian Portuguese",
    "ru": "Russian",
    "sv": "Swedish",
    "tr": "Turkish",
    "zh": "Chinese (Simplified)",
    "zh-tw": "Chinese (Traditional)",
}


def lang_name(lang: str) -> str:
    """Return the full language name for a given IETF tag, e.g. 'es' -> 'Spanish'."""
    key = lang.lower()
    name = _LANG_NAMES.get(key)
    if name is None:
        logger.warning("i18n: unknown language tag %r — passing raw tag to Claude; add to _LANG_NAMES if recurrent", lang)
        return lang
    return name


def translate(
    text: str,
    target_lang: str,
    context: str = "",
    db: Any = None,
    api_key: str = "",
) -> str:
    """Translate text to target_lang using Claude haiku with Firestore caching.

    Returns text unchanged for English or empty input.
    db and api_key are required for non-English languages.
    context is a short hint for the translator, e.g. "NBA push notification body".
    """
    result, _ = _translate_with_usage(text, target_lang, context=context, db=db, api_key=api_key)
    return result


def _translate_with_usage(
    text: str,
    target_lang: str,
    context: str = "",
    db: Any = None,
    api_key: str = "",
) -> tuple[str, dict[str, int]]:
    """Like translate(), but also returns the Anthropic token_usage dict.

    Returns ({original_text}, {}) on cache hit, English passthrough, or empty input.
    Used by preWarmTranslations to aggregate per-language token totals.
    """
    if not text or not text.strip():
        return text, {}

    # Any English variant (en, en-US, en-GB) passes through unchanged.
    if target_lang.lower().startswith("en"):
        return text, {}

    cache_key = hashlib.sha256(text.encode()).hexdigest()
    lang_lower = target_lang.lower()

    # Firestore cache check — cache hit costs zero tokens
    if db is not None:
        try:
            cache_ref = db.collection("translations").document(lang_lower).collection("cache").document(cache_key)
            cached = cache_ref.get()
            if cached.exists:
                data = cached.to_dict() or {}
                translation = data.get("translation")
                if translation:
                    return translation, {}
        except Exception:
            logger.warning("i18n: cache read failed for lang=%s key=%s", lang_lower, cache_key, exc_info=True)

    # Call Claude haiku for translation
    translation, usage = _translate_via_claude(text, target_lang, context, api_key)

    # Write to cache
    if db is not None and translation and translation != text:
        try:
            ttl = datetime.now(ZoneInfo("America/New_York")) + timedelta(days=_CACHE_TTL_DAYS)
            cache_ref.set(
                {
                    "translation": translation,
                    "source": text,
                    "lang": lang_lower,
                    "context": context,
                    "created_at": datetime.now(ZoneInfo("America/New_York")).isoformat(),
                    "ttl": ttl,
                }
            )
        except Exception:
            logger.warning("i18n: cache write failed for lang=%s key=%s", lang_lower, cache_key, exc_info=True)

    return translation, usage


def translate_dict(
    strings: dict[str, str],
    target_lang: str,
    context: str = "",
    db: Any = None,
    api_key: str = "",
) -> dict[str, str]:
    """Translate a dict of {key: text} values, returning a new dict with translated values.

    Keys are preserved unchanged. Useful for translating notification title + body together.
    """
    return {k: translate(v, target_lang, context=context, db=db, api_key=api_key) for k, v in strings.items()}


def _translate_via_claude(text: str, target_lang: str, context: str, api_key: str) -> tuple[str, dict[str, int]]:
    """Call Claude haiku to translate text. Returns (translated_text, token_usage).

    Returns (original_text, {}) on any failure or missing api_key.
    """
    if not api_key:
        logger.warning("i18n: no API key provided, returning original text")
        return text, {}

    try:
        system = build_translation_prompt(lang_name(target_lang), context)
        translated, usage = call_claude(
            api_key=api_key,
            model=TRANSLATION_MODEL,
            max_tokens=TRANSLATION_MAX_TOKENS,
            system=system,
            user=text,
            timeout=15.0,
        )
        return translated.strip(), usage

    except Exception:
        logger.warning("i18n: translation failed for lang=%s, returning original", target_lang, exc_info=True)
        return text, {}
