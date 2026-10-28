"""SAE and dense probes (B7) on the same activation. Owner: Person 3.

Multinomial logistic regression, class-weighted, trained on NQ+TriviaQA train,
tuned on val. Grid: layer x {sae width, dense} x label budget {50,100,500,1000,5000}
x seed. Saves per-question class probabilities so Person 4's metrics can score any probe:
    runs/probes/{probe_id}/probs_{file}.jsonl   {"qid", "p": {"ANSWER":..,"RETRIEVE":..,"ABSTAIN":..}}
"""

from __future__ import annotations


def main() -> None:
    # TODO(P3): sklearn LogisticRegression(class_weight="balanced"); L1 and L2 variants.
    raise NotImplementedError


if __name__ == "__main__":
    main()
