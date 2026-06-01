from __future__ import annotations

from cir_pipeline.text.noun_phrase_parser import (
    NounPhraseParser,
    extract_edit_actions,
    normalize_caption_for_parse,
    phrase_record,
    select_clean_phrase_texts,
    split_edit_clauses,
)


def test_normalize_caption_for_parse_fixes_known_typos():
    assert normalize_caption_for_parse("mae the concret colour darker") == "make the concrete color darker"


def test_split_edit_clauses_only_splits_on_new_edit_clause():
    assert split_edit_clauses("make the dog white and add a hat") == [
        "make the dog white",
        "add a hat",
    ]
    assert split_edit_clauses("make the dog white and black") == ["make the dog white and black"]


def test_extract_edit_actions_make_splits_object_attribute_and_location():
    actions = extract_edit_actions("make the needle upside down in the hand and add a hat")
    assert actions[0]["action"] == "make"
    assert actions[0]["object"] == "the needle"
    assert actions[0]["attribute"] == "upside down"
    assert actions[0]["location"] == "in the hand"
    assert actions[1]["action"] == "add"
    assert actions[1]["object"] == "a hat"


def test_extract_edit_actions_keeps_substantive_make_object():
    action = extract_edit_actions("make the red square darker")[0]
    assert action["object"] == "the red square"
    assert action["attribute"] == "darker"


def test_select_clean_phrase_texts_filters_noisy_supervision():
    phrases = ["the needle", "in the hand", "its", "show three bottles"]
    records = [
        phrase_record("the needle", extractor="constituency_minimal_np"),
        phrase_record("in the hand", extractor="constituency_minimal_np", inside_pp=True),
        phrase_record("its", extractor="constituency_minimal_np"),
        phrase_record("show three bottles", extractor="regex"),
    ]
    assert select_clean_phrase_texts(phrases, records, clean_only=True, min_confidence=0.7) == ["the needle"]


def test_backend_none_returns_actions_without_noun_phrases():
    parser = NounPhraseParser(backend="none")
    result = parser.analyze("add a hat")
    assert result.noun_phrases == []
    assert result.edit_actions[0]["action"] == "add"
