"""Normalizes raw LLM output into canonical tokens and validity classes."""

import re
import unicodedata
from typing import Any
from fingerprint.types import NormalizationResult

# Refusal detection regex
REFUSAL_RE = re.compile(
    r"(i can.?t|i cannot|i'm sorry|as an ai|не могу|извин|抱歉|无法|لا أستطيع|عذراً|آسف)",
    re.IGNORECASE,
)

# Arabic-Indic and Eastern Arabic-Indic digit mapping
AR_DIGITS = {
    '٠': '0', '١': '1', '٢': '2', '٣': '3', '٤': '4',
    '٥': '5', '٦': '6', '٧': '7', '٨': '8', '٩': '9',
    '۰': '0', '۱': '1', '۲': '2', '۳': '3', '۴': '4',
    '۵': '5', '۶': '6', '۷': '7', '۸': '8', '۹': '9',
}

ZH_DIGITS = {
    '零': 0, '一': 1, '二': 2, '两': 2, '三': 3,
    '四': 4, '五': 5, '六': 6, '七': 7, '八': 8, '九': 9
}

COIN_MAP = {
    "en": {"heads": "h", "tails": "t"},
    "ru": {"орёл": "h", "орел": "h", "решка": "t"},
    "zh": {"正面": "h", "正": "h", "反面": "t", "反": "t"},
    "ar": {"صورة": "h", "كتابة": "t"},
}


def basic_clean(raw: str) -> str:
    """NFC normalize, strip wrapping quotes/brackets/punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFC", raw)
    # Replace punctuation characters with space
    s = re.sub(r'[«»"“”„\'’‘`().,!?。！？、：:;؛؟\[\]{}*_#-]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()


def zh_number(s: str) -> int | None:
    """Parses Chinese numerals 1-99 (e.g. 七=7, 十=10, 十七=17, 四十二=42)."""
    m = re.match(r"^([零一二两三四五六七八九])?十?([零一二两三四五六七八九])?$", s)
    if not m or (not m.group(1) and not m.group(2) and "十" not in s):
        return None
    if "十" not in s:
        return ZH_DIGITS.get(m.group(1), None) if m.group(1) else None
    tens = ZH_DIGITS.get(m.group(1), 1) if m.group(1) else 1
    ones = ZH_DIGITS.get(m.group(2), 0) if m.group(2) else 0
    return tens * 10 + ones


def _normalize_integer(s: str, lang: str, answer_space: str) -> NormalizationResult:
    n = None
    m = re.search(r"-?\d+", s)
    if m:
        n = int(m.group(0))
    elif lang == "zh":
        n = zh_number(s)

    if n is None:
        return NormalizationResult(normalized=None, answer_class="invalid")

    range_match = re.search(r"(\d+)-(\d+)", answer_space)
    if range_match:
        low, high = int(range_match.group(1)), int(range_match.group(2))
        in_range = low <= n <= high
    else:
        in_range = True

    return NormalizationResult(
        normalized=str(n),
        answer_class="valid" if in_range else "invalid"
    )


def _normalize_binary(s: str, lang: str) -> NormalizationResult:
    word = s.lower().split()[0]
    mapped = COIN_MAP.get(lang, {}).get(word)
    if mapped:
        return NormalizationResult(normalized=mapped, answer_class="valid")
    return NormalizationResult(normalized=None, answer_class="invalid")


def _normalize_text(
    s: str,
    lang: str,
    normalize_as: str,
    category: str,
    color_lexicon: dict[str, Any] | None,
) -> NormalizationResult:
    words = s.lower().split()
    if normalize_as == "word" and len(words) > 3:
        # Off-format sentence residue
        return NormalizationResult(normalized=None, answer_class="invalid")

    first_word = words[0] if words else ""
    if not first_word:
        return NormalizationResult(normalized=None, answer_class="empty")

    if normalize_as == "grapheme":
        if len(first_word) > 1 and lang != "zh":
            single = next((w for w in words if len(w) == 1), None)
            if not single:
                return NormalizationResult(normalized=None, answer_class="invalid")
            first_word = single

    res = NormalizationResult(normalized=first_word, answer_class="valid")

    if category == "color" and color_lexicon and res.normalized:
        color_map = color_lexicon.get("map", {}).get(lang, {})
        res.color_canon = color_map.get(res.normalized, None)

    return res


def normalize_answer(
    raw: str | None,
    lang: str = "en",
    task: dict[str, Any] | None = None,
    color_lexicon: dict[str, Any] | None = None,
) -> NormalizationResult:
    """Normalizes raw model response into canonical output and validity class.
    
    answer_class: 'valid' | 'invalid' | 'refusal' | 'empty'
    """
    if task is None:
        task = {"normalize_as": "word", "answer_space": "", "category": "word"}

    normalize_as = task.get("normalize_as", "word")
    answer_space = task.get("answer_space", "")
    category = task.get("category", "")

    if raw is None or not raw.strip():
        return NormalizationResult(normalized=None, answer_class="empty")

    if REFUSAL_RE.search(raw):
        return NormalizationResult(normalized=None, answer_class="refusal")

    s = basic_clean(raw)
    if not s:
        return NormalizationResult(normalized=None, answer_class="empty")

    # Map Arabic-Indic digits to Latin digits
    s = "".join(AR_DIGITS.get(c, c) for c in s)

    if normalize_as == "integer":
        return _normalize_integer(s, lang, answer_space)

    if normalize_as == "binary":
        return _normalize_binary(s, lang)

    return _normalize_text(s, lang, normalize_as, category, color_lexicon)
