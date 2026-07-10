# MultiHop-RAG benchmark · 2026-07-11

This report records the first reproducible benchmark of Linki against the
official [MultiHop-RAG](https://github.com/yixuantt/MultiHop-RAG) dataset.

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
