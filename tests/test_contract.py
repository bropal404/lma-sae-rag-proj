from saerag.answers import contains_answer, exact_match, f1, is_correct, normalize
from saerag.data.build import assign_hole
from saerag.labels import label_rule
from saerag.retrieval.holes import apply_hole
from saerag.schema import Passage, Question, Retrieval, read_jsonl, write_jsonl


def test_normalize():
    assert normalize("The  Eiffel Tower!") == "eiffel tower"


def test_containment_is_token_level():
    assert contains_answer("It was built in 1889.", ["1889"])
    assert not contains_answer("It was built in 1990.", ["1"])
    assert contains_answer("Paris, France", ["paris"])
    assert not is_correct("anything", [])


def test_em_f1():
    assert exact_match("The Beatles", ["beatles"])
    assert 0 < f1("the beatles band", ["beatles"]) < 1


def test_label_rule():
    assert label_rule(True, False, None) == ("ANSWER", None)
    assert label_rule(True, True, True, unanswerable=True) == ("ABSTAIN", "unanswerable")
    assert label_rule(False, True, None) == ("RETRIEVE", None)
    assert label_rule(False, False, True) == ("ABSTAIN", "retriever_miss")
    assert label_rule(False, False, True, variant="miss_as_retrieve") == ("RETRIEVE", None)
    assert label_rule(False, False, False) == ("ABSTAIN", "beyond_model")
    assert label_rule(False, False, None) == ("ABSTAIN", "beyond_model")


def test_hole_assignment_is_deterministic():
    flags = [assign_hole(f"nq:train-{i}", 13, 0.3) for i in range(5000)]
    assert flags == [assign_hole(f"nq:train-{i}", 13, 0.3) for i in range(5000)]
    assert 0.27 < sum(flags) / len(flags) < 0.33


def test_apply_hole_drops_evidence():
    q = Question("nq:x", "nq", "test", "capital of france?", ["Paris"], hole=True)
    ps = [Passage("1", "Paris", "Paris is the capital.", 9.0),
          Passage("2", "Lyon", "Lyon is a city.", 8.0),
          Passage("3", "Nice", "Nice is on the coast.", 7.0)]
    kept, removed = apply_hole(ps, q, k=3)
    assert [p.pid for p in kept] == ["2", "3"] and removed == ["1"]


def test_jsonl_roundtrip(tmp_path):
    r = Retrieval("nq:x", "bm25", False, [Passage("1", "T", "text", 1.0)])
    write_jsonl(tmp_path / "r.jsonl", [r])
    assert list(read_jsonl(tmp_path / "r.jsonl", Retrieval)) == [r]
