"""Normalization tests.

Every string comparison in the project routes through this module, so a change
here moves every number in the results table. The cases below are the ones I
actually hit while looking at model output.
"""

from __future__ import annotations

import pytest

from unc_bench.normalize import (
    clean_model_answer,
    count_distinct,
    exact_match,
    first_line,
    is_abstention,
    mean_pairwise_token_f1,
    normalize_answer,
    strip_answer_prefix,
    token_f1,
    tokenize,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Paris", "paris"),
        ("  Paris  ", "paris"),
        ("PARIS", "paris"),
        ("The Beatles", "beatles"),
        ("A Tale of Two Cities", "tale of two cities"),
        ("an apple", "apple"),
        ("Paris, France", "paris france"),
        ("Mr. Smith", "mr smith"),
        ("1945.", "1945"),
        ("", ""),
        ("   ", ""),
        ("!!!", ""),
    ],
)
def test_normalize_answer_cases(raw: str, expected: str) -> None:
    assert normalize_answer(raw) == expected


def test_unicode_is_folded_and_all_punctuation_is_stripped() -> None:
    # The curly apostrophe is the one that mattered. NFKC does NOT fold U+2019
    # to an ASCII apostrophe, and string.punctuation does not contain it, so an
    # ASCII-only strip left the two spellings unequal. Models emit the curly form.
    assert normalize_answer("O\u2019Brien") == normalize_answer("O'Brien") == "obrien"
    assert normalize_answer("New\u00a0York") == normalize_answer("New York")
    assert normalize_answer("\uff11\uff19\uff14\uff15") == "1945"
    # Em dash, en dash, ellipsis, guillemets, CJK full stop.
    assert normalize_answer("Paris\u2014France") == "parisfrance"
    assert normalize_answer("Paris\u3002") == "paris"
    assert normalize_answer("\u00abParis\u00bb") == "paris"


def test_article_stripping_is_word_bounded() -> None:
    # "the" inside a word must survive. Losing this turns "Theodore" into
    # "odore" and breaks every name starting with those letters.
    assert normalize_answer("Theodore Roosevelt") == "theodore roosevelt"
    assert normalize_answer("Athens") == "athens"
    assert normalize_answer("Anne") == "anne"


def test_normalization_does_not_conflate_distinct_places() -> None:
    # The reason normalization stays conservative.
    assert normalize_answer("Paris, France") != normalize_answer("Paris, Texas")
    assert not exact_match("Paris, Texas", ["Paris, France"])


def test_exact_match_against_aliases() -> None:
    gold = ["William Shakespeare", "Shakespeare"]
    assert exact_match("shakespeare", gold)
    assert exact_match("  William   Shakespeare ", gold)
    assert not exact_match("Christopher Marlowe", gold)


def test_exact_match_on_empty_prediction_is_false() -> None:
    # An empty generation must never match, even against an empty alias.
    assert not exact_match("", ["Paris"])
    assert not exact_match("   ", [""])


def test_abstention_detection_is_exact_not_substring() -> None:
    assert is_abstention("UNKNOWN")
    assert is_abstention("unknown")
    assert is_abstention(" Unknown. ")
    # "unknown soldier" is a real answer. Treating it as an abstention would
    # silently drop a scorable row.
    assert not is_abstention("The Unknown Soldier")
    assert not is_abstention("unknown region of Mars")
    assert not is_abstention("")


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [
        ("Paris", "Paris", 1.0),
        ("Paris", "Lyon", 0.0),
        ("", "", 1.0),
        ("Paris", "", 0.0),
        ("New York City", "New York", 0.8),  # 2 overlap, p=2/3, r=2/2
        # 2 of 3 tokens overlap both ways -> p=r=2/3 -> F1=2/3. Deliberately
        # not single letters: a bare "a" is an article and normalizes away.
        ("xx yy zz", "yy zz ww", 2 / 3),
    ],
)
def test_token_f1_values(a: str, b: str, expected: float) -> None:
    assert token_f1(a, b) == pytest.approx(expected)


def test_token_f1_is_symmetric() -> None:
    pairs = [("New York City", "New York"), ("a b", "b c d"), ("Paris", "")]
    for a, b in pairs:
        assert token_f1(a, b) == pytest.approx(token_f1(b, a))


def test_token_f1_uses_multiset_counts() -> None:
    # "xx xx" vs "xx": overlap 1, precision 1/2, recall 1/1, F1 = 2/3.
    assert token_f1("xx xx", "xx") == pytest.approx(2 / 3)


def test_bare_articles_normalize_away() -> None:
    # Documented consequence of article stripping. Real gold spans are never a
    # bare article, but it makes single-letter fixtures misleading.
    assert normalize_answer("a") == ""
    assert normalize_answer("the") == ""
    assert tokenize("a b c") == ["b", "c"]


def test_mean_pairwise_f1_of_identical_answers_is_one() -> None:
    assert mean_pairwise_token_f1(["Paris"] * 6) == pytest.approx(1.0)


def test_mean_pairwise_f1_of_disjoint_answers_is_zero() -> None:
    assert mean_pairwise_token_f1(["Paris", "Lyon", "Nice"]) == pytest.approx(0.0)


def test_mean_pairwise_f1_of_a_single_answer_is_one() -> None:
    # A set of one is trivially self-consistent. Returning 0.0 would read as
    # maximal disagreement and invert the signal for N=1 in the ablation.
    assert mean_pairwise_token_f1(["Paris"]) == pytest.approx(1.0)
    assert mean_pairwise_token_f1([]) == pytest.approx(1.0)


def test_mean_pairwise_f1_hand_computed() -> None:
    # Pairs: (Paris,Paris)=1, (Paris,Lyon)=0, (Paris,Lyon)=0 -> 1/3
    assert mean_pairwise_token_f1(["Paris", "Paris", "Lyon"]) == pytest.approx(1 / 3)


def test_count_distinct_uses_normalized_form() -> None:
    assert count_distinct(["Paris", "paris", " PARIS. "]) == 1
    assert count_distinct(["Paris", "Lyon", "the Paris"]) == 2


def test_strip_answer_prefix() -> None:
    assert strip_answer_prefix("A: Paris") == "Paris"
    assert strip_answer_prefix("Answer: Paris") == "Paris"
    assert strip_answer_prefix("The answer is Paris") == "Paris"
    assert strip_answer_prefix("Paris") == "Paris"
    # Must not eat a real answer that happens to start with the letter a.
    assert strip_answer_prefix("Amsterdam") == "Amsterdam"


def test_first_line_skips_leading_blanks() -> None:
    assert first_line("\n\n  Paris \nsome explanation") == "Paris"
    assert first_line("") == ""


def test_clean_model_answer_end_to_end() -> None:
    assert clean_model_answer("A: Paris\nBecause it is the capital.") == "Paris"
    assert clean_model_answer("UNKNOWN") == "UNKNOWN"
    assert clean_model_answer("  unknown  ") == "UNKNOWN"
    assert clean_model_answer("The Unknown Soldier") == "The Unknown Soldier"


def test_tokenize_returns_empty_list_for_blank() -> None:
    assert tokenize("") == []
    assert tokenize("  ...  ") == []
    assert tokenize("New York") == ["new", "york"]
