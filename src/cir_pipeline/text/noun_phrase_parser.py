"""Noun phrase extraction with Stanza and lightweight fallbacks."""

from __future__ import annotations

import re
import warnings
from collections import deque
from dataclasses import dataclass


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


@dataclass
class NounPhraseParser:
    """Extract noun phrases, preferring Stanza constituency parses."""

    max_phrases: int = 10
    backend: str = "auto"

    def __post_init__(self) -> None:
        self._stanza_pipeline = None
        self._stanza_failed = False
        self._nltk_failed = False

    def extract(self, text: str) -> list[str]:
        text = text.strip()
        if not text:
            return []
        for backend, extractor in self._backend_extractors():
            phrases = extractor(text)
            if phrases or self.backend == backend:
                return phrases[: self.max_phrases]
        return self._extract_regex(text)[: self.max_phrases]

    def _backend_extractors(self):
        if self.backend in {"auto", "stanza"}:
            yield "stanza", self._extract_stanza
        if self.backend in {"auto", "nltk"}:
            yield "nltk", self._extract_nltk

    def _ensure_stanza(self):
        if self._stanza_failed:
            return None
        if self._stanza_pipeline is not None:
            return self._stanza_pipeline
        try:
            import stanza

            self._stanza_pipeline = stanza.Pipeline(
                lang="en",
                processors="tokenize,pos,constituency",
                use_gpu=False,
                verbose=False,
            )
            return self._stanza_pipeline
        except Exception as exc:  # pragma: no cover - depends on local models
            self._stanza_failed = True
            warnings.warn(
                "Stanza noun phrase parser is unavailable; falling back to NLTK/regex. "
                f"Original error: {exc}"
            )
            return None

    def _extract_stanza(self, text: str) -> list[str]:
        nlp = self._ensure_stanza()
        if nlp is None:
            return []
        try:
            doc = nlp(text)
            phrases: list[str] = []
            for sentence in doc.sentences:
                self._append_sentence_noun_phrases(sentence, phrases)
            return phrases
        except Exception:  # pragma: no cover
            return []

    def _append_sentence_noun_phrases(self, sentence, phrases: list[str]) -> None:
        tree = getattr(sentence, "constituency", None)
        if tree is None:
            return
        queue = deque([tree])
        while queue and len(phrases) < self.max_phrases:
            node = queue.popleft()
            self._append_node_phrase(node, phrases)
            queue.extend(getattr(node, "children", []) or [])

    def _append_node_phrase(self, node, phrases: list[str]) -> None:
        if getattr(node, "label", "") != "NP":
            return
        phrase = self._tree_text(node)
        if phrase and phrase not in phrases:
            phrases.append(phrase)

    def _tree_text(self, node) -> str:
        if hasattr(node, "leaf_labels"):
            return " ".join(node.leaf_labels()).strip()
        text = str(node)
        tokens = re.findall(r"\b[\w'-]+\b", text)
        return " ".join(tokens).strip()

    def _extract_nltk(self, text: str) -> list[str]:
        if self._nltk_failed:
            return []
        try:
            import nltk

            tokens = nltk.word_tokenize(text)
            tagged = nltk.pos_tag(tokens)
            grammar = "NP: {<DT>?<JJ.*>*<NN.*>+}"
            tree = nltk.RegexpParser(grammar).parse(tagged)
            phrases: list[str] = []
            for subtree in tree.subtrees(lambda t: t.label() == "NP"):
                phrase = " ".join(token for token, _ in subtree.leaves()).strip()
                if phrase and phrase not in phrases:
                    phrases.append(phrase)
            return phrases
        except Exception as exc:
            self._nltk_failed = True
            warnings.warn(
                "NLTK noun phrase fallback is unavailable; using regex fallback. "
                f"Original error: {exc}"
            )
            return []

    def _extract_regex(self, text: str) -> list[str]:
        tokens = re.findall(r"[A-Za-z][A-Za-z'-]*", text)
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
        return phrases

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
