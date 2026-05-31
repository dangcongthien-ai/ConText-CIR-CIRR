"""Safe noun-phrase to tokenizer index alignment."""

from __future__ import annotations

from typing import Any


def _find_span(text: str, phrase: str, start: int = 0) -> tuple[int, int] | None:
    """Find a character span for `phrase` inside `text`.

    This first tries an exact substring match (case-insensitive). If that
    fails it falls back to a looser regex-based search that allows
    punctuation/spacing differences (helps with tokenization artifacts like
    "ca n't" vs "can't"). Returns (start, end) or None.
    """
    import re

    text_lower = text.lower()
    phrase_lower = (phrase or "").lower().strip()
    if not phrase_lower:
        return None

    # Fast exact substring match
    idx = text_lower.find(phrase_lower, start)
    if idx >= 0:
        return idx, idx + len(phrase_lower)

    # Looser tokenized match: allow any non-word chars between phrase tokens.
    tokens = re.findall(r"[\w'-]+", phrase_lower)
    if not tokens:
        return None
    pattern = r"\b" + r"\W*".join(re.escape(tok) for tok in tokens) + r"\b"
    m = re.search(pattern, text_lower)
    if not m:
        return None
    return m.start(), m.end()


def align_noun_phrases(
    text: str,
    noun_phrases: list[str],
    tokenizer: Any,
) -> list[list[int]]:
    """Map noun phrases to token indices using fast tokenizer offsets."""

    if not noun_phrases:
        return []
    encoded = tokenizer(
        text,
        add_special_tokens=True,
        return_offsets_mapping=True,
        truncation=True,
    )
    offsets = encoded.get("offset_mapping", [])
    aligned: list[list[int]] = []
    search_start = 0
    for phrase in noun_phrases:
        span = _find_span(text, phrase, search_start)
        if span is None:
            aligned.append([])
            continue
        start, end = span
        token_indices = [
            idx
            for idx, (tok_start, tok_end) in enumerate(offsets)
            if tok_end > tok_start and tok_start < end and tok_end > start
        ]
        aligned.append(token_indices)
        search_start = end
    return aligned


def align_single_phrase(text: str, phrase: str, tokenizer: Any) -> list[int]:
    return align_noun_phrases(text, [phrase], tokenizer)[0]
