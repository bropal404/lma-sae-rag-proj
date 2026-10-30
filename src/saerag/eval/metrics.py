"""Metrics from the proposal's metrics table. Owner: Person 4.

Routing outcome per question: the router's action plus the matching generation run.
ANSWER uses the closed-book generation, RETRIEVE the BM25 (or dense) generation, ABSTAIN
counts as not answered. So end-to-end accuracy needs no new generation.

Every function takes plain dicts keyed by qid, so it works on any subset (a dataset,
hole questions only). Router output is `probs`: qid -> {"ANSWER": p, "RETRIEVE": p,
"ABSTAIN": p}, the format every router and probe writes.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score

from ..schema import ACTIONS, Action

Probs = dict[str, dict[str, float]]


def outcomes(actions: dict[str, Action], closed_correct: dict[str, bool],
             rag_correct: dict[str, bool]) -> dict[str, bool | None]:
    """qid -> True/False if attempted, None if abstained."""
    out: dict[str, bool | None] = {}
    for q, a in actions.items():
        out[q] = None if a == "ABSTAIN" else (closed_correct[q] if a == "ANSWER" else rag_correct[q])
    return out


# ---------------------------------------------------------------- routing decisions

def argmax_actions(probs: Probs) -> dict[str, Action]:
    """Plain argmax; ties break in ACTIONS order (ANSWER, RETRIEVE, ABSTAIN)."""
    return {q: max(ACTIONS, key=lambda a: p[a]) for q, p in probs.items()}


def route(probs: Probs, abstain_threshold: float | None = None) -> dict[str, Action]:
    """ABSTAIN when p(ABSTAIN) >= threshold, else the likelier of ANSWER and RETRIEVE.

    With no threshold this is the argmax. The threshold comes from `tune_abstain_threshold`
    on val, never on the file being scored.
    """
    if abstain_threshold is None:
        return argmax_actions(probs)
    return {q: "ABSTAIN" if p["ABSTAIN"] >= abstain_threshold
            else ("ANSWER" if p["ANSWER"] >= p["RETRIEVE"] else "RETRIEVE")
            for q, p in probs.items()}


def tune_abstain_threshold(p_abstain: dict[str, float], should_abstain: dict[str, bool],
                           max_false_refusal: float = 0.10) -> float:
    """Lowest threshold whose false-refusal rate on answerable questions is <= the budget.

    A question is refused when p_abstain >= threshold. Lower thresholds refuse more, so
    the lowest admissible one maximises abstention recall within the budget.
    """
    neg = np.sort([p_abstain[q] for q, s in should_abstain.items() if not s])[::-1]
    if len(neg) == 0:
        return 0.0
    allowed = int(np.floor(max_false_refusal * len(neg)))  # answerable questions we may refuse
    # Refusing exactly the top `allowed` negatives needs a threshold strictly above neg[allowed].
    if allowed >= len(neg):
        return 0.0
    return float(np.nextafter(neg[allowed], np.inf))


# ---------------------------------------------------------------- answer quality

def accuracy(out: dict[str, bool | None]) -> float:
    """Correct over all questions; abstentions count as not correct."""
    return _mean([bool(o) for o in out.values()])


def selective_accuracy(out: dict[str, bool | None]) -> float:
    """Correct over attempted questions (NaN if nothing was attempted)."""
    return _mean([o for o in out.values() if o is not None])


def coverage(out: dict[str, bool | None]) -> float:
    return _mean([o is not None for o in out.values()])


def hallucination_rate(out: dict[str, bool | None]) -> float:
    """Wrong answers over attempted answers. On index-hole questions: the unsafe-answer rate."""
    attempted = [o for o in out.values() if o is not None]
    return _mean([not o for o in attempted])


# ---------------------------------------------------------------- tool calling

def retrieval_call_rate(actions: dict[str, Action]) -> float:
    return _mean([a == "RETRIEVE" for a in actions.values()])


def call_necessity(actions: dict[str, Action], labels: dict[str, Action]) -> dict[str, float]:
    """Precision/recall of RETRIEVE against the labels.

    unnecessary_call_rate = calls the label did not need (1 - precision);
    missed_call_rate = needed calls not made (1 - recall).
    """
    qs = list(labels)
    pred = np.array([actions[q] == "RETRIEVE" for q in qs])
    gold = np.array([labels[q] == "RETRIEVE" for q in qs])
    tp = int((pred & gold).sum())
    precision = tp / pred.sum() if pred.sum() else float("nan")
    recall = tp / gold.sum() if gold.sum() else float("nan")
    return {"precision": precision, "recall": recall,
            "unnecessary_call_rate": 1 - precision, "missed_call_rate": 1 - recall}


# ---------------------------------------------------------------- abstention

def abstention_recall(actions: dict[str, Action],
                      should_abstain: dict[str, bool]) -> dict[str, float]:
    """Share of should-abstain questions abstained on, and share of answerable ones refused."""
    pos = [actions[q] == "ABSTAIN" for q, s in should_abstain.items() if s]
    neg = [actions[q] == "ABSTAIN" for q, s in should_abstain.items() if not s]
    return {"abstention_recall": _mean(pos), "false_refusal_rate": _mean(neg)}


def risk_coverage(out_full: dict[str, bool], confidence: dict[str, float]) -> dict:
    """Selective risk as coverage grows, answering the most confident questions first.

    `out_full` is the correctness of the router's non-abstain action for every question
    (route with no ABSTAIN), `confidence` usually 1 - p(ABSTAIN). Tied confidences are
    admitted together, so a hard router gives a single point at full coverage.
    AURC is the mean selective risk over the n coverage levels (lower is better).
    """
    qs = sorted(out_full, key=lambda q: -confidence[q])
    conf = np.array([confidence[q] for q in qs])
    wrong = np.array([not out_full[q] for q in qs], dtype=float)
    n = len(qs)
    if n == 0:
        return {"coverage": [], "risk": [], "aurc": float("nan")}
    cum_risk = np.cumsum(wrong) / np.arange(1, n + 1)
    # At a tie, every position in the group takes the risk at the end of the group.
    group_end = np.r_[np.flatnonzero(conf[1:] != conf[:-1]), n - 1]
    end_of = np.searchsorted(group_end, np.arange(n))
    risk_at = cum_risk[group_end[end_of]]
    cov_points = (group_end + 1) / n
    return {"coverage": cov_points.tolist(), "risk": cum_risk[group_end].tolist(),
            "aurc": float(risk_at.mean())}


def full_outcomes(probs: Probs, closed_correct: dict[str, bool],
                  rag_correct: dict[str, bool]) -> dict[str, bool]:
    """Correctness if the router never abstained (the likelier of ANSWER and RETRIEVE)."""
    acts = route(probs, abstain_threshold=float("inf"))
    return {q: bool(o) for q, o in outcomes(acts, closed_correct, rag_correct).items()}


# ---------------------------------------------------------------- router quality

def macro_f1(actions: dict[str, Action], labels: dict[str, Action]) -> float:
    """Averaged over classes present in the labels or predictions, so a subset with no
    RETRIEVE labels (hole questions) does not score an absent class as F1 = 0."""
    qs = list(labels)
    gold, pred = [labels[q] for q in qs], [actions[q] for q in qs]
    present = [a for a in ACTIONS if a in gold or a in pred]
    return float(f1_score(gold, pred, labels=present, average="macro", zero_division=0))


def expected_calibration_error(probs: Probs, labels: dict[str, Action], n_bins: int = 15) -> float:
    """Top-label ECE with equal-width bins on the max class probability."""
    qs = list(labels)
    conf = np.array([max(probs[q].values()) for q in qs])
    hit = np.array([max(ACTIONS, key=lambda a: probs[q][a]) == labels[q] for q in qs], dtype=float)
    if len(qs) == 0:
        return float("nan")
    bins = np.clip(np.ceil(conf * n_bins).astype(int) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = bins == b
        if m.any():
            ece += m.mean() * abs(hit[m].mean() - conf[m].mean())
    return float(ece)


def confusion(actions: dict[str, Action], labels: dict[str, Action]) -> dict[str, dict[str, int]]:
    """label -> predicted -> count, for the failure analysis."""
    m = {g: {p: 0 for p in ACTIONS} for g in ACTIONS}
    for q, g in labels.items():
        m[g][actions[q]] += 1
    return m


# ---------------------------------------------------------------- latency

STAGES = ("controller", "retrieval", "generation")


def route_latency(actions: dict[str, Action], controller_ms: dict[str, float],
                  retrieval_ms: dict[str, float], closed_gen_ms: dict[str, float],
                  rag_gen_ms: dict[str, float]) -> dict[str, dict[str, float]]:
    """Per-question stage times for the chosen action. ABSTAIN costs only the controller.

    Hard routers with no forward pass (B1, B2) pass an empty `controller_ms`.
    """
    lat = {}
    for q, a in actions.items():
        lat[q] = {"controller": controller_ms.get(q, 0.0),
                  "retrieval": retrieval_ms[q] if a == "RETRIEVE" else 0.0,
                  "generation": (closed_gen_ms[q] if a == "ANSWER"
                                 else rag_gen_ms[q] if a == "RETRIEVE" else 0.0)}
    return lat


def latency_percentiles(lat: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    """p50 and p95 per stage and end to end (stage percentiles do not add up to the total)."""
    rows = list(lat.values())
    res = {}
    for stage in (*STAGES, "total"):
        xs = np.array([sum(r.values()) if stage == "total" else r.get(stage, 0.0) for r in rows])
        res[stage] = ({"p50": float(np.percentile(xs, 50)), "p95": float(np.percentile(xs, 95))}
                      if len(xs) else {"p50": float("nan"), "p95": float("nan")})
    return res


# ---------------------------------------------------------------- everything at once

def score(probs: Probs, labels: dict[str, Action], closed_correct: dict[str, bool],
          rag_correct: dict[str, bool], should_abstain: dict[str, bool],
          abstain_threshold: float | None = None) -> dict[str, float]:
    """The scalar metrics for one router on one set of questions (no latency)."""
    acts = route(probs, abstain_threshold)
    out = outcomes(acts, closed_correct, rag_correct)
    full = full_outcomes(probs, closed_correct, rag_correct)
    conf = {q: 1 - p["ABSTAIN"] for q, p in probs.items()}
    nec = call_necessity(acts, labels)
    ab = abstention_recall(acts, should_abstain)
    return {
        "n": len(labels),
        "accuracy": accuracy(out),
        "selective_accuracy": selective_accuracy(out),
        "coverage": coverage(out),
        "hallucination_rate": hallucination_rate(out),
        "retrieval_call_rate": retrieval_call_rate(acts),
        "retrieve_precision": nec["precision"],
        "retrieve_recall": nec["recall"],
        "abstention_recall": ab["abstention_recall"],
        "false_refusal_rate": ab["false_refusal_rate"],
        "aurc": risk_coverage(full, conf)["aurc"],
        "macro_f1": macro_f1(acts, labels),
        "ece": expected_calibration_error(probs, labels),
    }


def _mean(xs) -> float:
    return float(np.mean(xs)) if len(xs) else float("nan")
