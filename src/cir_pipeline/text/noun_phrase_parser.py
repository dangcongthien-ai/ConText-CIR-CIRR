"""Noun phrase extraction, edit-action parsing, and lightweight fallbacks."""

from __future__ import annotations

import re
import warnings
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

FALLBACK_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "but",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "make",
    "more",
    "of",
    "on",
    "or",
    "the",
    "to",
    "turn",
    "with",
}

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

PRONOUN_ONLY = {"it", "its", "they", "them", "there", "this", "that", "these", "those", "i"}
EDIT_VERBS = {
    "add",
    "include",
    "insert",
    "remove",
    "delete",
    "take",
    "make",
    "change",
    "replace",
    "put",
    "place",
    "move",
    "show",
    "have",
    "turn",
    "crop",
    "isolate",
    "focus",
    "sit",
}
EDIT_VERB_ACTIONS = {
    "add": "add",
    "include": "add",
    "insert": "add",
    "remove": "remove",
    "delete": "remove",
    "take": "remove",
    "make": "make",
    "change": "change",
    "replace": "replace",
    "put": "put",
    "place": "place",
    "move": "move",
    "show": "show",
    "have": "have",
    "turn": "turn",
    "crop": "crop",
    "isolate": "isolate",
    "focus": "focus",
    "sit": "unknown",
}
CLAUSE_STARTERS = tuple(sorted(EDIT_VERBS | {"stand", "lie"}, key=len, reverse=True))
ATTRIBUTE_STARTERS = {
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
    "taller",
    "wider",
    "narrower",
    "marble",
    "concrete",
    "transparent",
    "opaque",
    "upside",
    "upright",
    "standing",
    "sitting",
    "lying",
    "stand",
    "sit",
    "lie",
}
CAPTION_NORMALIZATION_MAP = {
    "mae": "make",
    "cange": "change",
    "anmial": "animal",
    "colour": "color",
    "concret": "concrete",
}
LOCATION_PREPOSITIONS = (
    "next to",
    "in front of",
    "out of",
    "inside",
    "outside",
    "behind",
    "beside",
    "under",
    "over",
    "near",
    "onto",
    "into",
    "with",
    "from",
    "on",
    "in",
    "at",
    "by",
    "to",
)
LEADING_OBJECT_WORDS = {
    "the",
    "a",
    "an",
    "this",
    "that",
    "these",
    "those",
    "my",
    "your",
    "his",
    "her",
    "its",
    "their",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "another",
    "more",
    "less",
    "fewer",
    "only",
}
TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
CONNECTOR_RE = re.compile(r"\b(?:then|and)\b|[;,.]", re.IGNORECASE)
WHITESPACE_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class PhraseRecord:
    text: str
    confidence: float
    extractor: str
    starts_with_preposition: bool = False
    pronoun_only: bool = False
    contains_edit_verb: bool = False
    pp_only: bool = False
    inside_pp: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "confidence": float(self.confidence),
            "extractor": self.extractor,
            "starts_with_preposition": self.starts_with_preposition,
            "pronoun_only": self.pronoun_only,
            "contains_edit_verb": self.contains_edit_verb,
            "pp_only": self.pp_only,
            "inside_pp": self.inside_pp,
        }

    @classmethod
    def from_any(cls, value: Any) -> "PhraseRecord":
        if isinstance(value, PhraseRecord):
            return value
        if isinstance(value, dict):
            return cls(
                text=str(value.get("text") or "").strip(),
                confidence=float(value.get("confidence", 0.0) or 0.0),
                extractor=str(value.get("extractor") or "legacy"),
                starts_with_preposition=bool(value.get("starts_with_preposition", False)),
                pronoun_only=bool(value.get("pronoun_only", False)),
                contains_edit_verb=bool(value.get("contains_edit_verb", False)),
                pp_only=bool(value.get("pp_only", False)),
                inside_pp=bool(value.get("inside_pp", False)),
            )
        return phrase_record(str(value or ""), extractor="legacy")


@dataclass(frozen=True)
class CaptionParseResult:
    noun_phrases: list[str]
    phrase_records: list[PhraseRecord]
    edit_actions: list[dict[str, Any]]
    parser_warnings: list[str]
    phrase_extractor: str
    normalized_caption: str


def _compact_space(text: str) -> str:
    return WHITESPACE_RE.sub(" ", str(text or "")).strip()


def _normalize_phrase_text(phrase: str) -> str:
    s = _compact_space(phrase)
    if not s:
        return ""
    s = re.sub(r"\s+([.,!?;:])", r"\1", s)
    s = re.sub(r"\b([a-z]+)\s+n'?t\b", r"\1n't", s, flags=re.IGNORECASE)
    s = re.sub(r"\b([a-z]+)\s+'s\b", r"\1's", s, flags=re.IGNORECASE)
    s = re.sub(r"\b([a-z]+)\s+'re\b", r"\1're", s, flags=re.IGNORECASE)
    s = re.sub(r"\b([a-z]+)\s+'ve\b", r"\1've", s, flags=re.IGNORECASE)
    s = re.sub(r"\b([a-z]+)\s+'ll\b", r"\1'll", s, flags=re.IGNORECASE)
    s = re.sub(r"\b([a-z]+)\s+'m\b", r"\1'm", s, flags=re.IGNORECASE)
    return s.strip(" \t\n\r\"'")


def normalize_caption_for_parse(caption: str) -> str:
    normalized = str(caption or "")
    for source, target in CAPTION_NORMALIZATION_MAP.items():
        normalized = re.sub(rf"\b{re.escape(source)}\b", target, normalized, flags=re.IGNORECASE)
    return _compact_space(normalized)


def _tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text)


def _first_token(text: str) -> str:
    tokens = _tokens(text)
    return tokens[0].lower() if tokens else ""


def _starts_with_preposition(text: str) -> bool:
    return _first_token(text) in PREPOSITIONS


def _is_pronoun_only(text: str) -> bool:
    lowered = _normalize_phrase_text(text).lower()
    return lowered in PRONOUN_ONLY and len(lowered.split()) == 1


def _contains_edit_verb(text: str) -> bool:
    token_set = {token.lower() for token in _tokens(text)}
    return any(token in EDIT_VERBS for token in token_set)


def _phrase_signature(text: str) -> str:
    signature_tokens = [token.lower() for token in _tokens(text) if token.lower() not in {"the", "a", "an"}]
    return " ".join(signature_tokens)


def _phrase_quality_score(text: str, *, extractor: str, inside_pp: bool = False) -> float:
    phrase = _normalize_phrase_text(text)
    if not phrase:
        return 0.0
    score = 0.95 if extractor == "constituency_minimal_np" else 0.78
    if extractor == "regex":
        score = 0.62
    if inside_pp:
        score -= 0.45
    if _starts_with_preposition(phrase):
        score -= 0.35
    if _is_pronoun_only(phrase):
        score -= 0.50
    if _contains_edit_verb(phrase):
        score -= 0.25
    if len(_tokens(phrase)) == 1 and phrase.lower() in {"target", "there", "its", "i"}:
        score -= 0.30
    if any(marker in phrase.lower() for marker in (" of ", " with ")):
        score += 0.05
    return max(0.0, min(1.0, score))


def phrase_record(text: str, *, extractor: str, inside_pp: bool = False) -> PhraseRecord:
    phrase = _normalize_phrase_text(text)
    return PhraseRecord(
        text=phrase,
        confidence=_phrase_quality_score(phrase, extractor=extractor, inside_pp=inside_pp),
        extractor=extractor,
        starts_with_preposition=_starts_with_preposition(phrase),
        pronoun_only=_is_pronoun_only(phrase),
        contains_edit_verb=_contains_edit_verb(phrase),
        pp_only=_starts_with_preposition(phrase),
        inside_pp=inside_pp,
    )


def select_clean_phrase_texts(
    noun_phrases: list[str] | None,
    phrase_records: list[Any] | None,
    *,
    clean_only: bool,
    min_confidence: float,
) -> list[str]:
    phrases = list(noun_phrases or [])
    if not clean_only:
        return phrases
    records = [PhraseRecord.from_any(record) for record in (phrase_records or [])]
    if not records:
        records = [phrase_record(phrase, extractor="legacy") for phrase in phrases]
    clean_records = [
        record
        for record in records
        if record.text
        and record.confidence >= float(min_confidence)
        and not record.pp_only
        and not record.pronoun_only
        and not record.contains_edit_verb
    ]
    return [record.text for record in clean_records]


def _starts_new_edit_clause(text: str) -> bool:
    normalized = normalize_caption_for_parse(text).lower()
    return any(normalized.startswith(f"{verb} ") for verb in CLAUSE_STARTERS)


def split_edit_clauses(caption: str) -> list[str]:
    text = _compact_space(caption)
    if not text:
        return []
    pieces: list[str] = []
    start = 0
    for match in CONNECTOR_RE.finditer(text):
        connector = match.group(0)
        next_text = text[match.end() :].lstrip()
        if connector in {",", ";", "."}:
            should_split = _starts_new_edit_clause(next_text)
        else:
            should_split = _starts_new_edit_clause(next_text)
        if not should_split:
            continue
        piece = _compact_space(text[start : match.start()])
        if piece:
            pieces.append(piece)
        start = match.end()
    tail = _compact_space(text[start:])
    if tail:
        pieces.append(tail)
    return pieces or [text]


def _clean_field(text: str | None) -> str | None:
    value = _normalize_phrase_text(text or "")
    return value or None


def _split_location(text: str) -> tuple[str, str | None]:
    compact = _compact_space(text)
    lowered = compact.lower()
    for marker in LOCATION_PREPOSITIONS:
        pattern = f" {marker} "
        idx = lowered.find(pattern)
        if idx > 0:
            return compact[:idx].strip(), compact[idx + 1 :].strip()
    return compact, None


def _looks_like_attribute_start(tokens: list[str], index: int) -> bool:
    if index >= len(tokens):
        return False
    token = tokens[index].lower()
    next_two = " ".join(tokens[index : index + 2]).lower()
    if next_two == "upside down":
        return True
    if token in ATTRIBUTE_STARTERS or token in LOCATION_PREPOSITIONS:
        return True
    if token.endswith("er") and token not in {"water", "flower", "another"}:
        return True
    return False


def _has_substantive_object_tokens(tokens: list[str]) -> bool:
    return any(token.lower() not in LEADING_OBJECT_WORDS for token in tokens)


def _split_make_object_attribute(rest: str) -> tuple[str | None, str | None]:
    compact = _compact_space(rest)
    tokens = compact.split()
    if not tokens:
        return None, None
    for index in range(1, len(tokens)):
        if _looks_like_attribute_start(tokens, index):
            object_tokens = tokens[:index]
            if not _has_substantive_object_tokens(object_tokens):
                continue
            return _clean_field(" ".join(object_tokens)), _clean_field(" ".join(tokens[index:]))
    if len(tokens) > 2 and tokens[0].lower() in LEADING_OBJECT_WORDS:
        return _clean_field(" ".join(tokens[:-1])), _clean_field(tokens[-1])
    return _clean_field(compact), None


def _build_action(
    *,
    action: str,
    raw_clause: str,
    obj: str | None = None,
    attribute: str | None = None,
    target: str | None = None,
    location: str | None = None,
    confidence: float = 0.0,
) -> dict[str, Any]:
    return {
        "action": action,
        "object": _clean_field(obj),
        "attribute": _clean_field(attribute),
        "target": _clean_field(target),
        "location": _clean_field(location),
        "raw_clause": _compact_space(raw_clause),
        "confidence": float(max(0.0, min(1.0, confidence))),
    }


def extract_edit_action_from_clause(clause: str, stanza_sentence: Any = None) -> dict[str, Any]:
    del stanza_sentence
    raw_clause = _compact_space(clause)
    normalized = normalize_caption_for_parse(raw_clause)
    lowered = normalized.lower().strip(" .;,")
    if not lowered:
        return _build_action(action="unknown", raw_clause=raw_clause, confidence=0.0)

    take_match = re.match(r"^take\s+(?P<object>.+?)\s+(?P<location>out of|off|from)\s+(?P<locrest>.+)$", lowered)
    if take_match:
        location = f"{take_match.group('location')} {take_match.group('locrest')}"
        return _build_action(
            action="remove",
            raw_clause=raw_clause,
            obj=take_match.group("object"),
            location=location,
            confidence=0.88,
        )

    remove_match = re.match(r"^(?:remove|delete)\s+(?P<object>.+)$", lowered)
    if remove_match:
        obj, location = _split_location(remove_match.group("object"))
        return _build_action(action="remove", raw_clause=raw_clause, obj=obj, location=location, confidence=0.92)

    add_match = re.match(r"^(?:add|include|insert)\s+(?P<object>.+)$", lowered)
    if add_match:
        obj, location = _split_location(add_match.group("object"))
        return _build_action(action="add", raw_clause=raw_clause, obj=obj, location=location, confidence=0.92)

    make_match = re.match(r"^make\s+(?P<rest>.+)$", lowered)
    if make_match:
        obj, attribute = _split_make_object_attribute(make_match.group("rest"))
        location = None
        if attribute:
            attribute, location = _split_location(attribute)
        return _build_action(
            action="make",
            raw_clause=raw_clause,
            obj=obj,
            attribute=attribute,
            location=location,
            confidence=0.86,
        )

    change_to_match = re.match(r"^change\s+to\s+(?P<target>.+)$", lowered)
    if change_to_match:
        return _build_action(action="change", raw_clause=raw_clause, target=change_to_match.group("target"), confidence=0.70)

    change_match = re.match(r"^change\s+(?P<object>.+?)\s+to\s+(?P<target>.+)$", lowered)
    if change_match:
        return _build_action(
            action="change",
            raw_clause=raw_clause,
            obj=change_match.group("object"),
            target=change_match.group("target"),
            confidence=0.90,
        )

    replace_match = re.match(r"^replace\s+(?P<object>.+?)\s+with\s+(?P<target>.+)$", lowered)
    if replace_match:
        return _build_action(
            action="replace",
            raw_clause=raw_clause,
            obj=replace_match.group("object"),
            target=replace_match.group("target"),
            confidence=0.90,
        )

    instead_match = re.match(r"^(?P<target>.+?)\s+instead of\s+(?P<object>.+)$", lowered)
    if instead_match:
        return _build_action(
            action="replace",
            raw_clause=raw_clause,
            obj=instead_match.group("object"),
            target=instead_match.group("target"),
            confidence=0.74,
        )

    move_like_match = re.match(r"^(?P<verb>put|place|move|show|have|turn|crop|isolate|focus)\s+(?P<rest>.+)$", lowered)
    if move_like_match:
        verb = move_like_match.group("verb")
        obj, location = _split_location(move_like_match.group("rest"))
        action = EDIT_VERB_ACTIONS.get(verb, "unknown")
        if action == "turn" and obj and location is None:
            object_text, attribute = _split_make_object_attribute(obj)
            attribute_location = None
            if attribute:
                attribute, attribute_location = _split_location(attribute)
            return _build_action(
                action="turn",
                raw_clause=raw_clause,
                obj=object_text,
                attribute=attribute,
                location=attribute_location,
                confidence=0.78,
            )
        return _build_action(action=action, raw_clause=raw_clause, obj=obj, location=location, confidence=0.78)

    unknown_object, unknown_location = _split_location(lowered)
    return _build_action(
        action="unknown",
        raw_clause=raw_clause,
        obj=unknown_object,
        location=unknown_location,
        confidence=0.15,
    )


def dedupe_actions(actions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen: set[tuple[Any, ...]] = set()
    for action in actions:
        key = (
            action.get("action"),
            action.get("object"),
            action.get("attribute"),
            action.get("target"),
            action.get("location"),
            action.get("raw_clause"),
        )
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped


def extract_edit_actions(caption: str) -> list[dict[str, Any]]:
    clauses = split_edit_clauses(caption)
    actions = [extract_edit_action_from_clause(clause) for clause in clauses if _compact_space(clause)]
    return dedupe_actions(actions)


@dataclass
class NounPhraseParser:
    """Extract noun phrases, preferring Stanza and cleaner minimal NPs when requested."""

    max_phrases: int = 10
    backend: str = "auto"
    stanza_resources_dir: str | None = None
    stanza_download_method: str = "reuse_resources"
    stanza_np_extractor: str = "constituency"

    def __post_init__(self) -> None:
        self._stanza_pipeline = None
        self._stanza_failed = False
        self._nltk_failed = False

    def extract(self, text: str) -> list[str]:
        return self.analyze(text).noun_phrases

    def analyze(
        self,
        text: str,
        *,
        force_backend: str | None = None,
        allow_fallback: bool = True,
    ) -> CaptionParseResult:
        original = _compact_space(text)
        normalized = normalize_caption_for_parse(original)
        if not original or self.backend == "none":
            return CaptionParseResult([], [], extract_edit_actions(original), [], "none", normalized)

        parser_warnings: list[str] = []
        phrase_records: list[PhraseRecord] = []
        phrase_extractor = force_backend or self.stanza_np_extractor
        for backend in self._backend_order(force_backend):
            phrase_records, phrase_extractor, backend_warnings = self._extract_records(normalized, backend)
            parser_warnings.extend(backend_warnings)
            if phrase_records:
                break
            if not allow_fallback:
                break
        if not phrase_records and allow_fallback and (force_backend or self.backend) != "regex":
            fallback_records, _, fallback_warnings = self._extract_records(normalized, "regex")
            if fallback_records:
                phrase_records = fallback_records
                phrase_extractor = "regex"
                parser_warnings.extend([*fallback_warnings, "used_regex_fallback"])

        phrase_records = self._dedupe_and_rank_records(phrase_records)
        if len(phrase_records) > self.max_phrases:
            parser_warnings.append("truncated_by_max_phrases")
            phrase_records = phrase_records[: self.max_phrases]
        return CaptionParseResult(
            noun_phrases=[record.text for record in phrase_records],
            phrase_records=phrase_records,
            edit_actions=extract_edit_actions(original),
            parser_warnings=sorted(set(parser_warnings)),
            phrase_extractor=phrase_extractor,
            normalized_caption=normalized,
        )

    def _backend_order(self, force_backend: str | None) -> list[str]:
        if force_backend is not None:
            return [force_backend]
        if self.backend in {"stanza", "nltk", "regex"}:
            return [self.backend]
        return ["stanza", "nltk", "regex"]

    def _extract_records(self, text: str, backend: str) -> tuple[list[PhraseRecord], str, list[str]]:
        if backend == "stanza":
            return self._extract_stanza_records(text)
        if backend == "nltk":
            return self._extract_nltk_records(text), "nltk", []
        return self._extract_regex_records(text), "regex", []

    def _ensure_stanza(self):
        if self._stanza_failed:
            return None
        if self._stanza_pipeline is not None:
            return self._stanza_pipeline
        try:
            import stanza

            self._stanza_pipeline = stanza.Pipeline(**self._stanza_pipeline_kwargs())
            return self._stanza_pipeline
        except Exception as exc:
            self._stanza_failed = True
            warnings.warn(
                "Stanza noun phrase parser is unavailable; falling back to NLTK/regex. "
                f"Original error: {exc}"
            )
            return None

    def _stanza_pipeline_kwargs(self) -> dict[str, Any]:
        kwargs: dict[str, Any] = {
            "lang": "en",
            "processors": "tokenize,mwt,pos,lemma,depparse,constituency",
            "use_gpu": False,
            "verbose": False,
        }
        if self.stanza_resources_dir:
            kwargs["dir"] = str(Path(self.stanza_resources_dir))
        kwargs["download_method"] = self._stanza_download_method()
        return kwargs

    def _stanza_download_method(self):
        method = self.stanza_download_method.lower()
        if method == "none":
            return None
        try:
            from stanza.pipeline.core import DownloadMethod
        except Exception:
            return method
        enum_name = {
            "reuse_resources": "REUSE_RESOURCES",
            "download_resources": "DOWNLOAD_RESOURCES",
        }[method]
        return getattr(DownloadMethod, enum_name)

    def _extract_stanza_records(self, text: str) -> tuple[list[PhraseRecord], str, list[str]]:
        nlp = self._ensure_stanza()
        if nlp is None:
            return [], self.stanza_np_extractor, ["stanza_unavailable"]
        try:
            doc = nlp(text)
        except Exception:
            return [], self.stanza_np_extractor, ["stanza_parse_failed"]
        extractor = str(self.stanza_np_extractor).lower()
        if extractor.startswith("dep"):
            return self._extract_dependency_records(doc), "dependency", []
        if extractor == "constituency_minimal_np":
            return self._extract_constituency_minimal_records(doc), "constituency_minimal_np", []
        return self._extract_constituency_records(doc), "constituency", []

    def _build_children_map(self, words: list[Any]) -> dict[int, list[int]]:
        children: dict[int, list[int]] = {}
        for index, word in enumerate(words, start=1):
            head = int(getattr(word, "head", 0) or 0)
            children.setdefault(head, []).append(index)
        return children

    def _is_clausal_or_verb(self, child: Any) -> bool:
        deprel = (getattr(child, "deprel", "") or "").lower()
        upos = (getattr(child, "upos", "") or "").lower()
        return upos == "verb" or deprel.startswith("acl") or deprel in {"advcl", "xcomp", "ccomp", "cop"}

    def _collect_subtree(self, children: dict[int, list[int]], words: list[Any], root_idx: int) -> list[int]:
        stack = [root_idx]
        collected: list[int] = []
        while stack:
            current = stack.pop()
            collected.append(current)
            for child_idx in children.get(current, []):
                child = words[child_idx - 1]
                deprel = (getattr(child, "deprel", "") or "").lower()
                if self._is_clausal_or_verb(child):
                    continue
                if deprel == "conj" or deprel == "punct":
                    continue
                stack.append(child_idx)
        return sorted(set(collected))

    def _assemble_phrase(self, words: list[Any], indices: list[int]) -> str:
        tokens = [getattr(words[index - 1], "text", "") for index in sorted(indices)]
        return _normalize_phrase_text(" ".join(token for token in tokens if token))

    def _extract_dependency_records(self, doc: Any) -> list[PhraseRecord]:
        records: list[PhraseRecord] = []
        for sentence in getattr(doc, "sentences", []) or []:
            words = getattr(sentence, "words", []) or []
            if not words:
                continue
            children = self._build_children_map(words)
            for index, word in enumerate(words, start=1):
                upos = (getattr(word, "upos", "") or "").upper()
                if upos not in {"NOUN", "PROPN", "PRON"}:
                    continue
                phrase = self._assemble_phrase(words, self._collect_subtree(children, words, index))
                record = phrase_record(phrase, extractor="dependency")
                if not record.text or (record.pronoun_only and len(records) > 0):
                    continue
                records.append(record)
        return records

    def _extract_constituency_records(self, doc: Any) -> list[PhraseRecord]:
        records: list[PhraseRecord] = []
        for sentence in getattr(doc, "sentences", []) or []:
            tree = getattr(sentence, "constituency", None)
            if tree is None:
                continue
            queue = deque([tree])
            while queue:
                node = queue.popleft()
                if getattr(node, "label", "") == "NP":
                    record = phrase_record(self._tree_text(node), extractor="constituency")
                    if record.text:
                        records.append(record)
                queue.extend(getattr(node, "children", []) or [])
        return records

    def _extract_constituency_minimal_records(self, doc: Any) -> list[PhraseRecord]:
        candidates: list[PhraseRecord] = []
        for sentence in getattr(doc, "sentences", []) or []:
            tree = getattr(sentence, "constituency", None)
            if tree is None:
                continue
            for phrase, inside_pp in self._collect_minimal_np_candidates(tree, ancestors=()):
                record = phrase_record(phrase, extractor="constituency_minimal_np", inside_pp=inside_pp)
                if record.text:
                    candidates.append(record)
        filtered = [record for record in candidates if record.confidence >= 0.45]
        if filtered:
            return filtered
        return self._dedupe_and_rank_records(candidates)[: max(1, self.max_phrases)]

    def _collect_minimal_np_candidates(self, node: Any, ancestors: tuple[str, ...]) -> list[tuple[str, bool]]:
        label = getattr(node, "label", "")
        current_ancestors = (*ancestors, label) if label else ancestors
        candidates: list[tuple[str, bool]] = []
        if label == "NP":
            phrase = self._tree_text(node)
            inside_pp = "PP" in ancestors
            candidates.append((phrase, inside_pp))
        for child in getattr(node, "children", []) or []:
            candidates.extend(self._collect_minimal_np_candidates(child, current_ancestors))
        return candidates

    def _tree_text(self, node: Any) -> str:
        if hasattr(node, "leaf_labels"):
            return _normalize_phrase_text(" ".join(node.leaf_labels()).strip())
        tokens = re.findall(r"\b[\w'\-]+\b", str(node))
        return _normalize_phrase_text(" ".join(tokens))

    def _dedupe_and_rank_records(self, records: list[PhraseRecord]) -> list[PhraseRecord]:
        ranked = sorted(
            (record for record in records if record.text),
            key=lambda record: (-record.confidence, len(record.text.split()), -len(record.text)),
        )
        deduped: list[PhraseRecord] = []
        for record in ranked:
            if any(self._is_near_duplicate(record, kept) for kept in deduped):
                continue
            deduped.append(record)
        return deduped

    def _is_near_duplicate(self, record: PhraseRecord, kept: PhraseRecord) -> bool:
        if record.text == kept.text:
            return True
        signature = _phrase_signature(record.text)
        kept_signature = _phrase_signature(kept.text)
        if not signature or not kept_signature:
            return False
        if signature == kept_signature:
            return True
        if record.inside_pp and signature in kept_signature:
            return True
        if kept.inside_pp and kept_signature in signature and record.confidence <= kept.confidence:
            return True
        if signature in kept_signature and record.confidence <= kept.confidence and len(signature.split()) <= len(kept_signature.split()):
            return True
        return False

    def _extract_nltk_records(self, text: str) -> list[PhraseRecord]:
        if self._nltk_failed:
            return []
        try:
            import nltk

            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            tree = nltk.RegexpParser("NP: {<DT>?<JJ.*>*<NN.*>+}").parse(tagged)
            records: list[PhraseRecord] = []
            for subtree in tree.subtrees(lambda candidate: candidate.label() == "NP"):
                phrase = " ".join(token for token, _ in subtree.leaves()).strip()
                record = phrase_record(phrase, extractor="nltk")
                if record.text:
                    records.append(record)
            return records
        except Exception as exc:
            self._nltk_failed = True
            warnings.warn(
                "NLTK noun phrase fallback is unavailable; using regex fallback. "
                f"Original error: {exc}"
            )
            return []

    def _extract_regex_records(self, text: str) -> list[PhraseRecord]:
        tokens = re.findall(r"[a-z][a-z'\-]*", text, flags=re.IGNORECASE)
        phrases: list[str] = []
        current: list[str] = []
        for token in tokens:
            lowered = token.lower()
            if lowered in FALLBACK_STOPWORDS:
                self._flush_regex_phrase(current, phrases)
                current = []
                continue
            current.append(token)
            current = self._maybe_emit_long_regex_phrase(current, phrases)
        self._flush_regex_phrase(current, phrases)
        return [phrase_record(phrase, extractor="regex") for phrase in phrases if phrase]

    def _flush_regex_phrase(self, current: list[str], phrases: list[str]) -> None:
        if not current:
            return
        phrase = " ".join(current)
        if phrase.lower() not in FALLBACK_STOPWORDS and phrase not in phrases:
            phrases.append(phrase)

    def _maybe_emit_long_regex_phrase(self, current: list[str], phrases: list[str]) -> list[str]:
        if len(current) < 3:
            return current
        phrase = " ".join(current)
        if phrase not in phrases:
            phrases.append(phrase)
        return current[-1:]


def prepare_stanza_resources(resources_dir: str | None = None) -> bool:
    """Download Stanza English parser resources once before DataLoader workers start."""

    try:
        import stanza

        kwargs: dict[str, Any] = {
            "lang": "en",
            "processors": "tokenize,mwt,pos,lemma,depparse,constituency",
            "verbose": False,
        }
        if resources_dir:
            kwargs["dir"] = str(Path(resources_dir))
        stanza.download(**kwargs)
        return True
    except Exception as exc:
        warnings.warn(f"Could not prepare Stanza resources: {exc}")
        return False
