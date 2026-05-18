"""Safe noun-phrase to tokenizer index alignment."""

from __future__ import annotations

from typing import Any


def _find_span(text: str, phrase: str, start: int = 0) -> tuple[int, int] | None:
    text_lower = text.lower()
    phrase_lower = phrase.lower().strip()
    if not phrase_lower:
        return None
    idx = text_lower.find(phrase_lower, start)
    if idx < 0:
        idx = text_lower.find(phrase_lower)
    if idx < 0:
        return None
    return idx, idx + len(phrase_lower)


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
