"""Answer matching (D6), including the cases the Gate 1 hand check found."""

import pytest

from saerag.answers import contains_answer, is_correct

# Correct replies the old matcher missed (Gate 1 hand check, 2026-11-10).
HAND_CHECK_MISSES = [
    ("Daniel Patrick Moynihan", ['Daniel Patrick "Pat" Moynihan']),  # quoted nickname
    ("-273.15 °C", ["−273.15"]),                                 # Unicode minus sign
    ("Oh, What a Beautiful Mornin’", ["Oh, What a Beautiful Mornin'"]),  # curly apostrophe
    ("Melpoméne", ["Melpomene"]),                                     # accent
    ("Claudio López", ["Claudio Javier López"]),                      # first + last
    ("Howard Shore", ["Howard Leslie Shore"]),
    ("Nancy Cartwright", ["Nancy Jean Cartwright"]),
    ("Clarence Tinker", ["Major General Clarence L. Tinker"]),        # title + middle initial
    ("Major Clarence L. Tinker", ["Major General Clarence L. Tinker"]),
    ("Explosion", ["in an explosion"]),                               # leading preposition
]

# Wrong replies that must stay wrong.
STILL_WRONG = [
    ("AR-15", ["AR"]),
    ("1990", ["1"]),
    ("Paris Hilton", ["Paris France"]),
    ("Lake Michigan", ["Lake Superior"]),
    ("Shore", ["Howard Leslie Shore"]),           # surname alone is not enough
    ("Leslie Shore", ["Howard Leslie Shore"]),
    ("Texas", ["in Texas, USA"]),
    ("General Tinker", ["Major General Clarence L. Tinker"]),
]


@pytest.mark.parametrize("output,aliases", HAND_CHECK_MISSES)
def test_hand_check_misses_now_match(output, aliases):
    assert is_correct(output, aliases)


@pytest.mark.parametrize("output,aliases", STILL_WRONG)
def test_wrong_replies_stay_wrong(output, aliases):
    assert not is_correct(output, aliases)


def test_basics_unchanged():
    assert is_correct("It was Paris, France", ["Paris"])
    assert is_correct("The Yellow River", ["the Yellow River"])
    assert not is_correct("anything", [])
    assert not is_correct("", ["Paris"])


def test_contains_answer_unchanged_for_passages():
    # Passage matching (hole detection) keeps the old, stricter behaviour.
    assert not contains_answer("Howard Shore wrote the score", ["Howard Leslie Shore"])
    assert contains_answer("Howard Leslie Shore wrote the score", ["Howard Leslie Shore"])


def test_never_stricter_than_contains_answer():
    # A symbol-only alias is emptied by Unicode folding; the old match must still count.
    assert is_correct("⊂", ["⊂"])
