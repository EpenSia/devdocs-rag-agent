
### before — 2026-11-08 22:34

| before (30문항) | Hit@1 | Hit@5 | MRR | Noise |
|---|---|---|---|---|
| vector | 63% | 93% | 0.761 | 7% |
| bm25 | 60% | 83% | 0.678 | 5% |
| hybrid | 70% | 97% | 0.787 | 7% |

### after — 2026-11-08 22:38

| after (30문항) | Hit@1 | Hit@5 | MRR | Noise |
|---|---|---|---|---|
| vector | 67% | 90% | 0.778 | 3% |
| bm25 | 60% | 80% | 0.671 | 4% |
| hybrid | 63% | 93% | 0.740 | 4% |

### inline — 2026-11-08 22:59

| inline (30문항) | Hit@1 | Hit@5 | MRR | Noise |
|---|---|---|---|---|
| vector | 67% | 90% | 0.778 | 3% |
| bm25 | 60% | 80% | 0.671 | 4% |
| hybrid | 63% | 93% | 0.740 | 4% |

### split — 2026-11-08 23:03

| split (30문항) | Hit@1 | Hit@5 | MRR | Noise |
|---|---|---|---|---|
| vector | 63% | 97% | 0.769 | 5% |
| bm25 | 57% | 83% | 0.665 | 5% |
| hybrid | 67% | 93% | 0.748 | 6% |

### v100 — 2026-11-08 23:47

| v100 (100문항) | Hit@1 | Hit@5 | MRR | Noise |
|---|---|---|---|---|
| vector | 70% | 95% | 0.809 | 8% |
| bm25 | 58% | 88% | 0.694 | 7% |
| hybrid | 77% | 96% | 0.839 | 6% |

hybrid vs vector (질문별 순위 비교): hybrid 우세 15 · vector 우세 11 · 같음 74 → p=0.557, 차이가 있다고 말할 근거 부족

### v100 — 2026-11-14 18:00

| v100 (100문항) | Hit@1 | Hit@5 | MRR | Noise |
|---|---|---|---|---|
| vector | 70% | 95% | 0.809 | 8% |
| bm25 | 58% | 88% | 0.694 | 7% |
| hybrid | 77% | 96% | 0.839 | 6% |

hybrid vs vector (질문별 순위 비교): hybrid 우세 15 · vector 우세 11 · 같음 74 → p=0.557, 차이가 있다고 말할 근거 부족

### 답변 평가: trial — 2026-11-14 19:54 (judge: claude-sonnet-5-5, search: hybrid)

| trial (5문항) | Faithfulness | Relevance | Retrieved | Cited | Turns | Tools | Tokens | Seconds |
|---|---|---|---|---|---|---|---|---|
| 평균 | 3.80 | 5.00 | 100% | 100% | 2.0 | 1.4 | 7,538 | 9.8 |

### 답변 평가: hybrid — 2026-11-14 20:00 (judge: claude-sonnet-5-5, search: hybrid)

| hybrid (20문항) | Faithfulness | Relevance | Retrieved | Cited | Turns | Tools | Tokens | Seconds |
|---|---|---|---|---|---|---|---|---|
| 평균 | 3.60 | 5.00 | 100% | 100% | 2.0 | 1.5 | 6,362 | 9.4 |

### 답변 평가: grounded — 2026-11-14 20:28 (judge: claude-sonnet-5-5, search: hybrid)

| grounded (20문항) | Faithfulness | Relevance | Retrieved | Cited | Labeled | ReadDoc | Turns | Tools | Tokens | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|
| 평균 | 4.45 | 4.70 | 100% | 100% | 70% | 0.5 | 2.5 | 1.8 | 11,353 | 10.1 |

### 답변 평가: grounded2 — 2026-11-14 20:38 (judge: claude-sonnet-5-5, search: hybrid)

| grounded2 (20문항) | Faithfulness | Relevance | Retrieved | Cited | Korean | Truncated | Labeled | ReadDoc | Turns | Tools | Tokens | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 평균 | 4.55 | 5.00 | 100% | 100% | 100% | 0개 | 10% | 0.5 | 2.5 | 1.8 | 11,347 | 10.8 |

### 답변 평가: final — 2026-11-14 20:48 (judge: claude-sonnet-5-5, search: hybrid)

| final (20문항) | Faithfulness | Relevance | Retrieved | Cited | Korean | Truncated | Labeled | ReadDoc | Turns | Tools | Tokens | Seconds |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 평균 | 4.50 | 5.00 | 100% | 100% | 100% | 0개 | 5% | 0.5 | 2.6 | 1.7 | 11,816 | 10.6 |

언어 가드레일이 다시 쓰게 한 답변: 2개
