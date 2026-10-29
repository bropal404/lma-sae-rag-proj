import json
from argparse import Namespace
from types import SimpleNamespace

import pytest

from saerag.data.build import assign_hole, to_question, _hotpot_metadata
from saerag.retrieval.bm25 import BM25Tool, parse_passage
from saerag.retrieval.holes import apply_hole, gold_passages, is_evidence
from saerag.retrieval.run import process_file, process_question
from saerag.schema import Passage, Question, Retrieval, read_jsonl, write_jsonl


def q(hole=False):
    return Question('nq:x', 'nq', 'test', 'capital of france?', ['Paris'], hole=hole)


def ps():
    return [Passage('1', 'Paris', 'Paris is capital', 3),
            Passage('2', 'Lyon', 'Lyon is in France', 2),
            Passage('3', 'Marseille', 'A city in France', 1)]


@pytest.mark.parametrize('raw,title,text', [
    ('{"contents":"\\"Paris\\"\\nParis is the capital"}', 'Paris', 'Paris is the capital'),
    ('{"title":"Paris","text":"Paris is the capital"}', 'Paris', 'Paris is the capital'),
    ('{"contents":"Paris\\nParis is the capital"}', 'Paris', 'Paris is the capital'),
    ('plain standalone text', '', 'plain standalone text'),
])
def test_parse_passage(raw, title, text):
    assert parse_passage('p1', 1.5, raw) == Passage('p1', title, text, 1.5)


def test_batch_search_fake_backend():
    tool = object.__new__(BM25Tool)
    tool.threads = 2
    tool.searcher = SimpleNamespace(
        batch_search=lambda questions, qids, k, threads: {qids[0]: [SimpleNamespace(docid='p1', score=1)]},
        doc=lambda _: SimpleNamespace(raw=lambda: '{"title":"T","text":"paragraph"}')
    )
    assert list(tool.batch_search({'a': 'some question', 'b': 'other'}, 3)) == ['a', 'b']
    assert tool.batch_search({'a': 'q', 'b': 'q'}, 3)['a'][0].text == 'paragraph'
    assert tool.batch_search({'a': 'q', 'b': 'q'}, 3)['b'] == []


def test_gold_independent_of_holes():
    b, g = process_question(q(hole=True), ps(), 2, 12.1)
    assert [p.pid for p in b.passages] == ['2', '3']
    assert b.removed_pids == ['1']
    assert b.latency_ms == 12.1
    assert g.passages[0].pid == '1'
    assert g.retriever == 'gold'


def test_title_normalization_and_hotpot_gold():
    hot = Question('hotpotqa:x', 'hotpotqa', 'test', 'q', ['Yes'],
                   gold_titles=['A_Special Article'],
                   meta={'gold_paragraphs': [{'title':'A_Special Article', 'text': 'some evidence'}]})
    assert is_evidence(Passage('p', 'A Special Article', 'irrelevant', 1), hot)
    assert gold_passages([], hot)[0].text == 'some evidence'
    assert gold_passages([], Question('hotpotqa:y','hotpotqa','test','q',['yes'])) is None


def test_qid_and_deterministic_hole():
    row = {'question':'What is the capital of France?', 'answer':['Paris']}
    assert to_question('nq', 'train', row, 13, .3).qid == to_question('nq', 'test', row, 13, .3).qid
    assert 0.28 < sum(assign_hole(f'nq:{i}', 13, .3) for i in range(5000)) / 5000 < .32


def test_hotpot_metadata():
    titles, meta = _hotpot_metadata({
        'supporting_facts':{'title':['A','A','B']},
        'context':{'title':['A','X','B'], 'sentences':[['a1','a2'], ['x1'],['b1']]},
    })
    assert titles == ['A','B']
    assert [p['title'] for p in meta['gold_paragraphs']] == ['A','B']


class FakeBM25:
    def search(self, query, k):
        return ps()
    def batch_search(self, queries, k):
        return {qid: ps() for qid in queries}


@pytest.mark.parametrize('mode', ['sequential', 'batch'])
def test_end_to_end_without_index(tmp_path, mode):
    questions = [q(hole=False), Question('nq:y','nq','test','capital?', ['Paris'],hole=True)]
    root = tmp_path / 'data'
    write_jsonl(root / 'questions' / 'smoke.jsonl', questions)
    cfg = {'paths': {'questions':str(root/'questions'), 'retrieval':str(root/'retrieval')},
           'retrieval': {'top_k':2, 'hole_fetch_k':100}}
    args = Namespace(index='fake', mode=mode, threads=1, batch_size=2, overwrite=False,
                     limit=0, verbose=False)
    process_file(cfg, 'smoke', FakeBM25(), args)
    bm = list(read_jsonl(root/'retrieval'/'bm25'/'smoke.jsonl', Retrieval))
    gold = list(read_jsonl(root/'retrieval'/'gold'/'smoke.jsonl', Retrieval))
    stats = json.loads((root/'retrieval'/'stats'/'smoke.json').read_text())
    assert [p.pid for p in bm[1].passages] == ['2','3']
    assert bm[1].removed_pids == ['1']
    assert gold[1].passages[0].pid == '1'
    assert stats['hole_questions'] == 1
    assert stats['hole_removed_passages'] == 1
    assert (bm[0].latency_ms is None) == (mode == 'batch')
    with pytest.raises(FileExistsError):
        process_file(cfg, 'smoke', FakeBM25(), args)


def test_builder_creates_consistent_pilot_with_fake_hf(tmp_path, monkeypatch):
    import sys
    from saerag.data.build import build
    def mock_load_dataset(repo, subset=None, split=None):
        tag = 'nq' if 'nq_open' in repo else 'triviaqa'
        n = 12 if split == 'train' else 5
        rows = []
        for i in range(n):
            row = {'question': f'{tag} {split} question {i}',
                   'answer': ['France'] if tag == 'nq' else {'aliases':['France']}}
            if tag == 'triviaqa':
                row['question_id'] = f'{split}-{i}'
            rows.append(row)
        return rows
    monkeypatch.setitem(sys.modules, 'datasets', SimpleNamespace(load_dataset=mock_load_dataset))
    cfg = {
        'seed': 13,
        'paths': {'questions': str(tmp_path / 'questions')},
        'retrieval': {'hole_fraction': .3},
        'datasets': {'nq': {'train':4,'val':2,'test':2},
                     'triviaqa': {'train':4,'val':2,'test':2}},
        'pilot': {'n':4},
    }
    build(cfg, ['nq', 'triviaqa'])
    qpath = tmp_path / 'questions'
    assert len(list(read_jsonl(qpath / 'pilot.jsonl', Question))) == 4
    train = list(read_jsonl(qpath / 'nq_train.jsonl', Question))
    val = list(read_jsonl(qpath / 'nq_val.jsonl', Question))
    test = list(read_jsonl(qpath / 'nq_test.jsonl', Question))
    assert len(set(q.qid for q in train + val + test)) == 8
    assert all(row.split == 'pilot' for row in read_jsonl(qpath / 'pilot.jsonl', Question))
    assert set(q.qid for q in read_jsonl(qpath / 'pilot.jsonl', Question)).issubset(
        {q.qid for q in list(read_jsonl(qpath / 'nq_train.jsonl', Question)) +
         list(read_jsonl(qpath / 'triviaqa_train.jsonl', Question))})
