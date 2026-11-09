# Person 2 latency at batch size 1

## Generation per condition (2x Tesla T4, torch.float32, files nq_test, triviaqa_test, 200 questions with a gold passage)

```
              p50     p95    mean    n
closed_ms   430.5   763.6   475.0  200
bm25_ms    1429.1  1952.3  1481.6  200
gold_ms     760.1  1130.3   792.7  200
```

gold / BM25-RAG time ratio on this GPU: p50 0.53, p95 0.58

This ratio does not carry over to the L4: in fp32 on a T4 the prompt (prefill) dominates, while on the L4 in bf16 BM25-RAG costs only ~1.14x closed-book.

Mean prompt tokens: closed 42, gold 198, BM25 495 (gold sits 34% of the way from closed to BM25).

**Gold generation on the L4 (estimate): p50 211.0 ms, p95 438.9 ms**
