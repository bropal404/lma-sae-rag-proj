"""Answer normalisation and matching (D6).

Label correctness for every dataset is token-level alias containment: some normalised
alias appears as a contiguous token span of the normalised output. Token spans (not
substrings) stop "1" matching "1990". EM and F1 are reported for HotpotQA only.
"""

from __future__ import annotations

import re
import string
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


def is_correct(output: str, aliases: list[str]) -> bool:
    """The single correctness rule used for labels and accuracy."""
    return bool(aliases) and contains_answer(output, aliases)
