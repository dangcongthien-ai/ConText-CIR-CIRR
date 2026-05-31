"""Noun phrase extraction with Stanza and lightweight fallbacks."""

from __future__ import annotations

import re
import warnings
from collections import deque
from dataclasses import dataclass
from pathlib import Path


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
    stanza_resources_dir: str | None = None
    stanza_download_method: str = "reuse_resources"
    stanza_np_extractor: str = "constituency"

    def __post_init__(self) -> None:
        self._stanza_pipeline = None
        self._stanza_failed = False
        self._nltk_failed = False

    def extract(self, text: str) -> list[str]:
        text = text.strip()
        if not text or self.backend == "none":
            return []
        for backend, extractor in self._backend_extractors():
            phrases = extractor(text)
            if phrases:
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

            pipeline_kwargs = self._stanza_pipeline_kwargs()
            self._stanza_pipeline = stanza.Pipeline(**pipeline_kwargs)
            return self._stanza_pipeline
        except Exception as exc:
            self._stanza_failed = True
            warnings.warn(
                "Stanza noun phrase parser is unavailable; falling back to NLTK/regex. "
                f"Original error: {exc}"
            )
            return None

    def _stanza_pipeline_kwargs(self) -> dict:
        processors = "tokenize,pos,constituency"
        if str(self.stanza_np_extractor).lower().startswith("dep"):
            processors = "tokenize,mwt,pos,lemma,depparse"
        kwargs = {
            "lang": "en",
            "processors": processors,
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

    def _extract_stanza(self, text: str) -> list[str]:
        if str(self.stanza_np_extractor).lower().startswith("dep"):
            return self._extract_stanza_dependency(text)

        nlp = self._ensure_stanza()
        if nlp is None:
            return []
        try:
            doc = nlp(text)
            phrases: list[str] = []
            for sentence in doc.sentences:
                self._append_sentence_noun_phrases(sentence, phrases)
            return phrases
        except Exception:
            return []

    def _build_children_map(self, words: list) -> dict[int, list[int]]:
        children: dict[int, list[int]] = {}
        for i, w in enumerate(words, start=1):
            head = int(getattr(w, "head", 0) or 0)
            children.setdefault(head, []).append(i)
        return children

    def _is_clausal_or_verb(self, child) -> bool:
        deprel = (getattr(child, "deprel", "") or "").lower()
        upos = (getattr(child, "upos", "") or "").lower()
        return (
            upos == "verb"
            or deprel.startswith("acl")
            or deprel in {"advcl", "xcomp", "ccomp", "cop"}
        )

    def _collect_subtree(self, children: dict[int, list[int]], words: list, root_idx: int) -> list[int]:
        """Collect dependency subtree indices for a given head, skipping clausal/verb children."""
        stack = [root_idx]
        collected: list[int] = []
        while stack:
            cur = stack.pop()
            collected.append(cur)
            for c in children.get(cur, []):
                child = words[c - 1]
                deprel = (getattr(child, "deprel", "") or "").lower()
                if self._is_clausal_or_verb(child):
                    continue
                if deprel == "conj" or deprel == "punct":
                    continue
                stack.append(c)
        return sorted(set(collected))

    def _assemble_phrase(self, words: list, indices: list[int]) -> str:
        if not indices:
            return ""
        phrase_tokens = [getattr(words[j - 1], "text", "") for j in sorted(indices)]
        phrase = " ".join(t for t in phrase_tokens if t).strip()
        return self._normalize_phrase_text(phrase)

    def _normalize_phrase_text(self, phrase: str) -> str:
        """Small normalizations to make phrases match caption substrings.

        Collapses common tokenization artifacts (e.g. "ca n't" -> "can't") and
        removes stray spaces before punctuation.
        """
        s = (phrase or "").strip()
        if not s:
            return ""
        s = re.sub(r"\s+([.,!?;:])", r"\1", s)
        s = re.sub(r"\b([a-z]+)\s+n'?t\b", r"\1n't", s, flags=re.IGNORECASE)
        s = re.sub(r"\b([a-z]+)\s+'s\b", r"\1's", s, flags=re.IGNORECASE)
        s = re.sub(r"\b([a-z]+)\s+'re\b", r"\1're", s, flags=re.IGNORECASE)
        s = re.sub(r"\b([a-z]+)\s+'ve\b", r"\1've", s, flags=re.IGNORECASE)
        s = re.sub(r"\b([a-z]+)\s+'ll\b", r"\1'll", s, flags=re.IGNORECASE)
        s = re.sub(r"\b([a-z]+)\s+'m\b", r"\1'm", s, flags=re.IGNORECASE)
        return s.strip()

    def _extract_stanza_dependency(self, text: str) -> list[str]:
        """Extract noun chunks using Stanza dependency parses.

        Strategy: for each noun/proper-noun/pronoun token, collect a small
        subtree of dependents that are not clausal (exclude verbal/clausal
        relations and punctuation). This keeps phrases compact (no verbs)
        and focuses CC on object-like spans.
        """
        nlp = self._ensure_stanza()
        if nlp is None:
            return []
        try:
            doc = nlp(text)
            phrases: list[str] = []
            for sentence in doc.sentences:
                self._append_dependency_sentence_phrases(sentence, phrases)
                if len(phrases) >= self.max_phrases:
                    break
            return phrases
        except Exception:
            return []

    def _append_dependency_sentence_phrases(self, sentence, phrases: list[str]) -> None:
        words = getattr(sentence, "words", []) or []
        if not words:
            return

        children = self._build_children_map(words)
        for index, word in enumerate(words, start=1):
            if len(phrases) >= self.max_phrases:
                break
            phrase = self._dependency_phrase_for_word(words, children, index, word)
            if phrase and phrase not in phrases:
                phrases.append(phrase)

    def _dependency_phrase_for_word(
        self,
        words: list,
        children: dict[int, list[int]],
        word_index: int,
        word,
    ) -> str:
        upos = (getattr(word, "upos", "") or "").upper()
        if upos not in {"NOUN", "PROPN", "PRON"}:
            return ""
        indices = self._collect_subtree(children, words, word_index)
        phrase = self._assemble_phrase(words, indices)
        if self._is_trivial_dependency_phrase(phrase):
            return ""
        return phrase

    def _is_trivial_dependency_phrase(self, phrase: str) -> bool:
        if not phrase:
            return True
        lowered = phrase.lower()
        if lowered in FALLBACK_STOPWORDS:
            return True
        pronouns = {"it", "them", "they", "this", "that", "these", "those"}
        return lowered in pronouns and len(phrase.split()) == 1

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
        tokens = re.findall(r"\b[\w'\-]+\b", text)
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


def prepare_stanza_resources(resources_dir: str | None = None) -> bool:
    """Download Stanza English parser resources once before DataLoader workers start."""
    try:
        import stanza

        kwargs = {
            "lang": "en",
            "processors": "tokenize,pos,constituency,depparse",
            "verbose": False,
        }
        if resources_dir:
            kwargs["model_dir"] = str(Path(resources_dir))
        stanza.download(**kwargs)
        return True
    except Exception as exc:
        warnings.warn(f"Could not prepare Stanza resources; parser will fall back if needed. Original error: {exc}")
        return False
