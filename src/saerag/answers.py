"""Answer normalisation and matching (D6).

Label correctness for every dataset is token-level alias containment: some normalised
alias appears as a contiguous token span of the normalised output. Token spans (not
substrings) stop "1" matching "1990". EM and F1 are reported for HotpotQA only.

is_correct (labels and accuracy) is slightly more forgiving than contains_answer, after the
Gate 1 hand check found the matcher missed correct replies far more often than it accepted
wrong ones (20 of 21 label errors): it also folds Unicode punctuation, symbols and accents,
and tries a few alias variants (quoted nickname dropped, leading title dropped, leading
preposition dropped, "first last" for a three-word name). contains_answer is unchanged,
so passage matching for hole detection and the gold stand-in stays as it was.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)


def normalize(s: str) -> str:
    """SQuAD-style: lowercase, drop punctuation and articles, collapse whitespace."""
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def _contains_span(haystack: list[str], needle: list[str]) -> bool:
    n = len(needle)
    return n > 0 and any(haystack[i : i + n] == needle for i in range(len(haystack) - n + 1))


def contains_answer(output: str, aliases: list[str]) -> bool:
    out = normalize(output).split()
    return any(_contains_span(out, normalize(a).split()) for a in aliases)


def exact_match(output: str, aliases: list[str]) -> bool:
    out = normalize(output)
    return any(out == normalize(a) for a in aliases)


def f1(output: str, aliases: list[str]) -> float:
    def one(pred: str, gold: str) -> float:
        p, g = normalize(pred).split(), normalize(gold).split()
        common = sum((Counter(p) & Counter(g)).values())
        if common == 0:
            return 0.0
        prec, rec = common / len(p), common / len(g)
        return 2 * prec * rec / (prec + rec)

    return max((one(output, a) for a in aliases), default=0.0)


_QUOTED = re.compile(r"[\"\u201c\u201d][^\"\u201c\u201d]+[\"\u201c\u201d]")
_TITLES = {"sir", "dame", "dr", "mr", "mrs", "ms", "rev", "reverend", "major", "general",
           "colonel", "captain", "lieutenant", "lt", "admiral", "sergeant", "professor", "prof"}
_PREPS = {"in", "on", "at", "by", "from", "within", "during"}


def match_normalize(s: str) -> str:
    """normalize() plus Unicode folding: accents stripped (NFKD), and every Unicode
    punctuation or symbol character dropped like ASCII punctuation ("\u2212273.15" and
    "-273.15" agree, so do "Mornin\u2019" and "Mornin'")."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(ch for ch in s if not unicodedata.category(ch).startswith(("M", "P", "S")))
    return normalize(s)


def alias_variants(alias: str) -> list[list[str]]:
    """Token lists that count as this alias. The full alias always comes first."""
    out = [match_normalize(alias).split()]
    no_nick = match_normalize(_QUOTED.sub(" ", alias)).split()
    if no_nick and no_nick != out[0]:
        out.append(no_nick)
    for toks in list(out):
        while len(toks) > 2 and toks[0] in _TITLES:   # "Major General Clarence L Tinker"
            toks = toks[1:]
            out.append(toks)
        if len(toks) > 1 and toks[0] in _PREPS:        # "in an explosion" -> "explosion"
            out.append(toks[1:])
        if len(toks) == 3 and all(t.isalpha() for t in toks):  # "Howard Leslie Shore"
            out.append([toks[0], toks[2]])
    return out


def is_correct(output: str, aliases: list[str]) -> bool:
    """The single correctness rule used for labels and accuracy."""
    if not aliases:
        return False
    if contains_answer(output, aliases):   # never stricter than the old rule (e.g. alias "⊂")
        return True
    out = match_normalize(output).split()
    return any(_contains_span(out, v) for a in aliases for v in alias_variants(a))
