"""Record types passed between pipeline stages (D9).

Every stage reads and writes JSONL keyed by ``qid``. This file is the contract between
the four work streams: change a field here only after telling the others.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Iterator
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Literal

Dataset = Literal["nq", "triviaqa", "popqa", "hotpotqa", "selfaware"]
Split = Literal["train", "val", "test", "pilot"]
Condition = Literal["closed", "bm25", "gold", "dense"]
Action = Literal["ANSWER", "RETRIEVE", "ABSTAIN"]
ACTIONS: tuple[Action, ...] = ("ANSWER", "RETRIEVE", "ABSTAIN")


@dataclass
class Question:
    qid: str                    # "{dataset}:{source_id}", stable across all stages
    dataset: Dataset
    split: Split
    question: str
    answers: list[str]          # all accepted aliases; empty for unanswerable
    unanswerable: bool = False  # Self-Aware: ABSTAIN by construction
    hole: bool = False          # D3: retrieval runs with gold evidence filtered out
    gold_titles: list[str] = field(default_factory=list)  # HotpotQA supporting titles
    meta: dict = field(default_factory=dict)              # e.g. PopQA s_pop, HotpotQA level


@dataclass
class Passage:
    pid: str
    title: str
    text: str
    score: float


@dataclass
class Retrieval:
    qid: str
    retriever: str              # "bm25" | "dense" | "gold"
    hole: bool
    passages: list[Passage]
    removed_pids: list[str] = field(default_factory=list)  # audit trail for index holes
    latency_ms: float | None = None


@dataclass
class Generation:
    qid: str
    condition: Condition
    output: str
    correct: bool
    latency_ms: float | None = None


@dataclass
class Label:
    qid: str
    variant: str                # "default" | "miss_as_retrieve"
    label: Action
    closed_correct: bool
    rag_correct: bool
    gold_correct: bool | None   # None when no gold passage exists
    abstain_reason: str | None  # "retriever_miss" | "beyond_model" | "unanswerable" | None
    hole: bool


def _from_dict[T](cls: type[T], d: dict) -> T:
    if cls is Retrieval:
        d = {**d, "passages": [Passage(**p) for p in d["passages"]]}
    names = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in d.items() if k in names})


def write_jsonl(path: str | Path, records: Iterable) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.writelines(
            json.dumps(asdict(r) if hasattr(r, "__dataclass_fields__") else r) + "\n" for r in records
        )


def read_jsonl[T](path: str | Path, cls: type[T] | None = None) -> Iterator[T]:
    with open(path) as f:
        for line in f:
            if line.strip():
                d = json.loads(line)
                yield _from_dict(cls, d) if cls else d
