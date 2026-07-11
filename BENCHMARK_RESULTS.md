# MultiHop-RAG benchmark · 2026-07-11

This report records the first reproducible benchmark of Linki against the
official [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) dataset.

## Adaptive upgrade checkpoint · commit `51811fe`

This checkpoint evaluates the retrieval and governance pieces added by the
second upgrade plan. It does **not** replace the historical N=12 end-to-end
result below; the 120-question end-to-end experiment is recorded separately
when complete.

### Frozen protocol

- Dataset: all 2,556 official questions; 2,255 are answerable.
- Corpus/index: 609 articles, 10,398 parent chunks, 47,593 child vectors.
- Corpus snapshot: `55dd1b00d0df848a7e8fbf62`; dataset fingerprint:
  `40f309a270bbb4e1`.
- Execution: serial, one local machine, embedded Qdrant, warm-up excluded.
- Reranker score cache: disabled for every measured request.
- Graph source: entity↔document edges extracted only from corpus text. The
  graph retriever cannot accept benchmark gold sources.
- `legacy_k5`: frozen top-5 hybrid/full-parent baseline.
- `rerank_pack`: 30 hybrid candidates → local cross-encoder → 8 verbatim
  supporting spans.
- `rerank_ppr`: the same 30-candidate budget after hybrid + bounded two-hop
  corpus PPR fusion → cross-encoder → 8 spans.

All three reports have `complete=true` and exactly 2,556 item records. The
supporting-span variants achieved 100% source-offset validity.

### Full retrieval matrix (N=2,556)

| Variant | Recall | Precision | Strict all-support | MRR | Mean evidence tokens | p95 latency |
|---|---:|---:|---:|---:|---:|---:|
| `legacy_k5` | 0.5858 | **0.4532** | 0.2625 | 0.7539 | **744** | **0.672 s** |
| `rerank_pack` | 0.5994 | 0.3262 | 0.2678 | 0.7520 | 1,278 | 1.076 s |
| `rerank_ppr` | **0.6348** | 0.3373 | **0.3038** | **0.7807** | 1,620 | 2.397 s |

Relative to `legacy_k5`, plain reranking gained 1.36 percentage points of
recall and 0.53 points of strict all-support recall, while increasing p95 by
60% and evidence volume by 72%. PPR gained 4.90 and 4.13 points respectively,
while increasing p95 by 257% and evidence volume by 118%. PPR is therefore an
evidence-recall option for selected expensive paths, not a default fast path.

### Strict all-support recall by answerable question type

| Variant | Comparison (856) | Inference (816) | Temporal (583) |
|---|---:|---:|---:|
| `legacy_k5` | 0.3797 | 0.1078 | 0.3070 |
| `rerank_pack` | 0.3505 | 0.1360 | 0.3310 |
| `rerank_ppr` | **0.4077** | **0.1434** | **0.3756** |

The graph pilot improves every type over the frozen baseline, but inference
remains the weakest slice: only 14.34% of questions retrieve the entire support
chain. This is not sufficient to claim that graph retrieval has solved
multi-hop inference.

### Stability and governed-memory checks

- Retrieval stability: 120 stratified questions × 3 runs, score cache off;
  top-k Jaccard `1.0`, exact-ranking rate `1.0`, error rate `0.0`, p50 `0.935 s`,
  p95 `1.095 s`, mean latency variance `0.00094`.
- Answer-claim Jaccard was not measured in that run and is stored as `null`,
  rather than being inferred from retrieval stability.
- Project-owned memory governance suite: 5/5 status decisions correct; write
  precision/recall, update correctness and deletion completeness `1.0`; stale
  recall rate `0.0`. This is a deterministic regression suite, **not** a claimed
  LongMemEval or LoCoMo score.

### Preliminary end-to-end pipeline smoke (N=3)

The three rows validate real provider telemetry, checkpointing and all four
system slots; all happened to be comparison questions, so they are not a
quality conclusion.

| System | Calls | Mean tokens | Mean latency | Faithfulness | All-support |
|---|---:|---:|---:|---:|---:|
| Fair single pass | 1.00 | 1,650 | 2.67 s | 3.67 | 0.667 |
| Legacy full graph | 6.00 | 6,128 | 17.25 s | 4.67 | 1.000 |
| Adaptive auto | 1.33 | 2,507 | 4.95 s | 4.00 | 0.667 |
| Adaptive deep | 5.00 | 9,412 | 28.44 s | 5.00 | 1.000 |

Provider-returned token usage was available for 100% of measured system calls;
judge scoring calls are deliberately excluded from system telemetry. Confidence
intervals from N=3 are too wide for release decisions.

### Known measurement limitations

1. MultiHop-RAG covers comparison, inference, temporal and null queries; it does
   not cover the planned single, multi-turn or cross-KB strata.
2. The full retrieval run predates persistence of the per-item
   `rerank_backend`. A separate preflight observed the declared
   `Xenova/ms-marco-MiniLM-L-6-v2` backend, but the full report cannot prove that
   every item avoided lexical fallback. The evaluator now persists this field.
3. Embedded Qdrant warns that local mode is not recommended above 20,000 points.
   These latency values must not be presented as server-profile capacity.
4. Retrieval stability does not establish answer stability; the answer run is a
   separate, optional protocol because it incurs three model calls per question.

### Reproduction of the adaptive checkpoint

```bash
linki bench-retrieval --benchmark multihop-rag --variant all
linki bench-stability --benchmark multihop-rag --limit 120 --seed 42 --repeats 3
linki bench-memory
linki bench-fair --benchmark multihop-rag --fold-size 40
```

---

## Experimental setup

- Official corpus: 609 full news articles.
- Questions: 2,556 total; 2,255 answerable and 301 `null_query` items.
- Index: 10,398 parent chunks and 47,593 child vectors.
- Retriever: `BAAI/bge-small-en-v1.5` + `Qdrant/bm25` hybrid search, top-k 5.
- Runtime: embedded local Qdrant.
- Agent limits: two grade/refine rounds and two verifier attempts.
- Main model: DeepSeek provider default; judge: `deepseek-reasoner`.
- End-to-end sample: 12 deterministic stratified questions, seed 42, three
  questions from each official type (`comparison`, `inference`, `temporal`,
  `null`). This sample is diagnostic, not a statistically definitive leaderboard.

The fair baseline uses the same corpus, retriever, model, evidence-only rule,
refusal rule, and citation requirement as Linki. `agentic_no_reflow` limits both
retrieval and verification to one pass; `agentic` enables the complete workflow.

## Full retrieval-only result (all 2,556 questions)

| Question type | N | Retrieval recall | Retrieval precision | All-support recall | Mean latency |
|---|---:|---:|---:|---:|---:|
| Overall answerable | 2,255 | 0.5858 | 0.4532 | 0.2625 | 0.6836 s |
| Comparison | 856 | 0.6544 | 0.4615 | 0.3797 | 0.6632 s |
| Inference | 816 | 0.4961 | 0.4348 | 0.1078 | 0.7127 s |
| Temporal | 583 | 0.6106 | 0.4669 | 0.3070 | 0.6663 s |

`All-support recall` is strict: every one of the question's 2-4 gold documents
must be retrieved. The largest current weakness is inference retrieval, where the
complete evidence chain is found for only 10.78% of questions.

## End-to-end strict result (stratified N=12)

| Metric | Fair naive | Agentic, no reflow | Full agentic |
|---|---:|---:|---:|
| Retrieval recall | 0.611 | 0.648 | **0.685** |
| Retrieval precision | 0.519 | **0.694** | 0.567 |
| All-support recall | 0.444 | **0.556** | **0.556** |
| Gold answer contained | 0.167 | **0.250** | **0.250** |
| Refusal correctness | 0.583 | **0.667** | **0.667** |
| Citation presence | 0.333 | 0.833 | **0.917** |
| Citation index validity | 1.000 | 1.000 | 1.000 |
| Judge faithfulness (1-5) | 4.583 | **4.917** | 4.583 |
| Judge quality (1-5) | 3.333 | 3.667 | **3.750** |
| Mean latency | **2.326 s** | 17.490 s | 21.092 s |
| Mean LLM calls | **1.000** | 5.750 | 6.667 |
| Mean total tokens | **1,013** | 6,158 | 7,643 |

## Interpretation

1. Agentic planning improves evidence recall and answer coverage, but it does not
   solve the central retrieval problem: even in the stratified sample, 44.4% of
   questions still miss at least one required document.
2. Full verifier reflow did not improve strict all-support recall or gold-answer
   containment over the no-reflow ablation in this sample. It added roughly 21%
   more tokens and 21% more latency. Reflow should therefore be conditional and
   measured by “failed before, correct after,” rather than enabled blindly.
3. Agentic execution costs about 9x the baseline latency and 7.5x the tokens.
   Quality gains are real but currently expensive.
4. The local configuration's 200-character child chunks expand 609 documents to
   47,593 vectors. Qdrant warns that embedded local mode is not recommended above
   20,000 points. Larger corpora should use Qdrant server/cloud and a chunk-size
   sweep rather than this demo-oriented setting.
5. Citation presence improved substantially, but index validity only proves that
   `[n]` points to a retrieved item. Claim-to-citation entailment still needs a
   dedicated claim-level metric such as RAGChecker.

## Reproduction

```bash
linki bench-prep --benchmark multihop-rag
linki bench-ingest --benchmark multihop-rag
linki bench-retrieval --benchmark multihop-rag
linki eval --benchmark multihop-rag --strict --limit 12 --seed 42
```

Local raw reports are written below
`.linki/benchmarks/multihop-rag/runtime/eval/` and intentionally ignored by Git
because they include full generated answers and retrieved URLs.
