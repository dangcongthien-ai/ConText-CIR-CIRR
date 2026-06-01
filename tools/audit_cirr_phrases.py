from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

PRONOUNS = {"it", "its", "they", "them", "there", "this", "that", "these", "those", "i"}
PREPOSITIONS = {
    "in",
    "on",
    "at",
    "by",
    "from",
    "to",
    "of",
    "with",
    "near",
    "beside",
    "behind",
    "under",
    "over",
    "inside",
    "outside",
    "around",
    "across",
    "through",
    "into",
    "onto",
    "next",
    "before",
    "after",
}
EDIT_VERBS = {
    "add",
    "include",
    "insert",
    "introduce",
    "plant",
    "remove",
    "delete",
    "exclude",
    "omit",
    "hide",
    "take",
    "make",
    "change",
    "alter",
    "switch",
    "transform",
    "convert",
    "replace",
    "put",
    "place",
    "move",
    "show",
    "shows",
    "display",
    "capture",
    "photograph",
    "shoot",
    "have",
    "has",
    "turn",
    "rotate",
    "flip",
    "tilt",
    "crop",
    "isolate",
    "focus",
    "zoom",
    "paint",
    "color",
    "colour",
    "darken",
    "lighten",
    "blur",
    "fill",
    "open",
    "close",
    "hang",
    "line",
    "arrange",
    "set",
    "slide",
    "stand",
    "sit",
    "lay",
    "lie",
    "face",
    "hold",
    "fold",
    "charge",
    "use",
    "cut",
    "knead",
    "give",
    "grow",
    "leave",
    "keep",
    "preserve",
    "reduce",
    "increase",
    "decrease",
    "lessen",
    "fewer",
    "wipe",
    "tie",
    "stack",
    "roll",
}
ATTRIBUTE_HINTS = {
    "white",
    "black",
    "blue",
    "red",
    "green",
    "yellow",
    "orange",
    "pink",
    "purple",
    "brown",
    "gray",
    "grey",
    "blonde",
    "bigger",
    "smaller",
    "darker",
    "lighter",
    "longer",
    "shorter",
    "fatter",
    "thinner",
    "stand",
    "sit",
    "lie",
    "upside down",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit exported CIRR noun phrases and edit actions.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def _load_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["samples"] if isinstance(payload, dict) and "samples" in payload else payload
    if not isinstance(rows, list):
        raise ValueError(f"Expected samples list in {path}")
    return payload if isinstance(payload, dict) else {}, rows


def _first_token(text: str) -> str:
    tokens = str(text or "").strip().split()
    return tokens[0].lower() if tokens else ""


def _phrase_starts_with_prep(text: str) -> bool:
    return _first_token(text) in PREPOSITIONS


def _pronoun_only(text: str) -> bool:
    lowered = str(text or "").strip().lower()
    return lowered in PRONOUNS and len(lowered.split()) == 1


def _contains_edit_verb(text: str) -> bool:
    tokens = {token.lower() for token in str(text or "").split()}
    return any(token in EDIT_VERBS for token in tokens)


def _caption_has_verb(caption: str, verb: str) -> bool:
    lowered = str(caption or "").lower()
    return lowered.startswith(f"{verb} ") or f" {verb} " in lowered


def _caption_has_attribute_hint(caption: str) -> bool:
    lowered = str(caption or "").lower()
    return any(hint in lowered for hint in ATTRIBUTE_HINTS)


def _safe_pct(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 4) if denominator else 0.0


def _row_quality(row: dict[str, Any]) -> float:
    phrases = row.get("noun_phrases") or []
    warnings = row.get("parser_warnings") or []
    if not phrases:
        return 0.0
    score = 1.0
    if warnings:
        score -= 0.15
    if _phrase_starts_with_prep(phrases[0]):
        score -= 0.35
    if any(_pronoun_only(phrase) for phrase in phrases):
        score -= 0.25
    if any(_contains_edit_verb(phrase) for phrase in phrases):
        score -= 0.15
    if any(action.get("action") == "unknown" for action in (row.get("edit_actions") or [])):
        score -= 0.15
    return max(0.0, score)


def _sample_rows(rows: list[dict[str, Any]], predicate, limit: int, seed: int) -> list[dict[str, Any]]:
    selected = [row for row in rows if predicate(row)]
    rng = random.Random(seed)
    rng.shuffle(selected)
    return selected[:limit]


def _attribute_capture_rate(rows: list[dict[str, Any]], verb: str | None = None) -> float:
    subset = [row for row in rows if verb is None or _caption_has_verb(row.get("caption", ""), verb)]
    subset = [row for row in subset if _caption_has_attribute_hint(row.get("caption", ""))]
    hits = 0
    for row in subset:
        for action in row.get("edit_actions") or []:
            if verb is not None and action.get("action") != verb:
                continue
            if action.get("attribute") or action.get("target"):
                hits += 1
                break
    return _safe_pct(hits, len(subset))


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)
    payload, rows = _load_rows(input_path)
    phrases = [phrase for row in rows for phrase in (row.get("noun_phrases") or [])]
    actions = [action for row in rows for action in (row.get("edit_actions") or [])]
    top_hist = Counter(phrase.lower() for phrase in phrases).most_common(30)
    prep_first = sum(1 for row in rows if (row.get("noun_phrases") or []) and _phrase_starts_with_prep(row["noun_phrases"][0]))
    pronoun_only = sum(1 for phrase in phrases if _pronoun_only(phrase))
    edit_verb_phrase = sum(1 for phrase in phrases if _contains_edit_verb(phrase))
    pp_only = sum(1 for phrase in phrases if _phrase_starts_with_prep(phrase))
    truncated = sum(1 for row in rows if "truncated_by_max_phrases" in (row.get("parser_warnings") or []))
    unknown_actions = sum(1 for action in actions if action.get("action") == "unknown")

    report = {
        "input": str(input_path),
        "dataset": payload.get("dataset"),
        "split": payload.get("split"),
        "backend": payload.get("backend"),
        "num_samples": len(rows),
        "zero_phrase_pct": _safe_pct(sum(1 for row in rows if not (row.get("noun_phrases") or [])), len(rows)),
        "avg_phrases": round(sum(len(row.get("noun_phrases") or []) for row in rows) / len(rows), 4) if rows else 0.0,
        "prep_first_phrase_pct": _safe_pct(prep_first, len(rows)),
        "pronoun_only_phrase_pct": _safe_pct(pronoun_only, len(phrases)),
        "edit_verb_phrase_pct": _safe_pct(edit_verb_phrase, len(phrases)),
        "pp_only_phrase_pct": _safe_pct(pp_only, len(phrases)),
        "top_phrase_histogram": top_hist,
        "caption_truncated_by_max_phrases_pct": _safe_pct(truncated, len(rows)),
        "attribute_capture_rate_make": _attribute_capture_rate(rows, "make"),
        "attribute_capture_rate_change": _attribute_capture_rate(rows, "change"),
        "attribute_capture_rate_turn": _attribute_capture_rate(rows, "turn"),
        "attribute_capture_rate_overall_edit_heavy": _attribute_capture_rate(rows, None),
        "num_unknown_edit_actions": unknown_actions,
        "unknown_edit_action_pct": _safe_pct(unknown_actions, len(actions)),
        "bad_examples": _sample_rows(rows, lambda row: _row_quality(row) < 0.45, 50, args.seed),
        "good_examples": _sample_rows(rows, lambda row: _row_quality(row) >= 0.8, 50, args.seed + 1),
        "edit_heavy_examples": _sample_rows(
            rows,
            lambda row: any(_caption_has_verb(row.get("caption", ""), verb) for verb in ("make", "change", "turn", "remove", "add", "replace")),
            50,
            args.seed + 2,
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
