<p align="center">
  <img alt="Linki logo" src="assets/logo.png" width="320px">
</p>

<h1 align="center">Linki · Agentic RAG</h1>

<p align="center">
  <strong>A reliable knowledge assistant that routes, retrieves, grades, cites — and honestly refuses when the knowledge base has no answer.</strong>
</p>

<p align="center">
  <a href="#what-is-linki">What is Linki</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#reliability-features">Reliability</a> •
  <a href="#install">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="#configuration">Configuration</a> •
  <a href="#testing">Testing</a> •
  <a href="#roadmap">Roadmap</a>
</p>

---

## What is Linki

Linki turns a plain "retrieve-once, then answer" RAG pipeline into an **Agentic RAG**:
retrieval becomes a decision process, not a single function call. The model decides
whether to retrieve at all, rewrites the query, grades the evidence, retrieves again
when it is not enough, generates an answer **constrained to the evidence** with
per-claim citations, verifies that answer, and — when the knowledge base genuinely
lacks the answer — says so instead of hallucinating.

This project is a **fusion of three assets**:

| Source | Contributes |
| --- | --- |
| [`linki-agent-i`](https://github.com/Linkbreathe) (a multi-agent coding assistant) | The **agentic backbone**: intent routing, tool registry + executor pipeline, provider factory, session, hooks/trace patterns. |
| [`agentic-rag-for-dummies`](https://github.com/GiovanniPasq/agentic-rag-for-dummies) (upstream, under `project/`) | The **RAG internals**: hierarchical parent/child chunking, embedded Qdrant hybrid search, parent store. |
| The upgrade design (`docs/superpowers/specs/`) | The **reliability delta**: explicit grader, evidence-cited answering, and a verifier reflow loop. |

The result lives in `src/linki/` — a self-contained package that reuses the proven
pieces and adds the reliability layer that makes answers trustworthy.

---

## Architecture

The Phase-5 graph (planned, parallel retrieval with verification reflow):

```
router ─chat──▶ chat_responder ─▶ END
  │  ─clarify─▶ clarify ─▶ END
  └  ─retrieve▶ rewrite ─▶ planner ─Send×N▶ retrieve(loop) ─▶ answer ─▶ verifier
                              ▲                                      ├ pass ──▶ final ─▶ END
                              └──────── retry with issues ───────────┤
                                                                     └ giveup▶ final_with_warning ─▶ END

retrieve(loop): retrieve ─▶ grade ─(insufficient)▶ refine ↺
                         └(sufficient / give up)▶ collect
```

- **router** — chat / retrieve / clarify. On any doubt, prefer retrieve.
- **rewrite** — resolve pronouns and ellipsis from history into a standalone, retrieval-friendly query.
- **planner** — produce a minimal `QueryPlan` (1-4 sub-queries), choose each target knowledge-base tool from its description, and fan out independent searches with LangGraph `Send`.
- **retrieve loop** — hybrid (dense + sparse) search with parent expansion, then a
  **grader** decides sufficiency; if not enough it **refines** the query and retrieves
  again (bounded by `max_rounds`), with global de-duplication so each branch/round explores new ground.
- **answer** — generate **only** from the numbered evidence, marking every claim with `[n]`, and disclosing gaps.
- **verifier** — check the answer against the evidence claim-by-claim; on failure, send structured issues back to the planner for supplemental retrieval (bounded by `max_attempts`), else degrade transparently with a warning.

Embeddings are **local** (`fastembed`, ONNX — no torch, no GPU). Qdrant runs in
**embedded on-disk mode** — no Docker, no server.

---

## Reliability features

- **Evidence-constrained generation** — answers are built only from retrieved evidence; the model's own pretraining knowledge is not used to fill gaps.
- **Per-claim citations** — every claim ends with `[n]`; a Sources panel maps `[n]` back to file · section.
- **Honest refusal** — when the knowledge base does not cover the question, Linki says "not found in the knowledge base" instead of inventing an answer.
- **Retrieval grading + self-correction** — weak retrieval is caught before generation, and the query is refined and retried.
- **Answer verification** — a separate judge model checks the answer against the evidence and can trigger another retrieval pass.
- **Multi-turn** — conversation history is fed to the router/rewrite so follow-ups like "and how does it compare?" resolve correctly.

---

## Install

Requires **Python 3.12+**. [`uv`](https://docs.astral.sh/uv/) recommended.

```bash
git clone https://github.com/Linkbreathe/LinkiRAG-AgenticRAG.git
cd LinkiRAG-AgenticRAG
uv venv --python 3.12
uv pip install -e '.[ui]'          # omit [ui] if you don't need the web interface
```

Configure your LLM provider in a `.env` file (git-ignored):

```dotenv
# DeepSeek
LINKI_PROVIDER=deepseek
DEEPSEEK_API_KEY=sk-...
DEEPSEEK_MODEL=deepseek-chat
LINKI_JUDGE_MODEL=deepseek-reasoner   # judge/verifier; a different model is recommended

# ...or OpenAI
# LINKI_PROVIDER=openai
# OPENAI_API_KEY=sk-...
# LINKI_MODEL=gpt-4o-mini
```

Embeddings download automatically on first ingest (small ONNX models).

---

## Usage

```bash
# Ingest documents (PDF or Markdown) into a knowledge base
linki ingest ./docs/fastapi.md --kb default

# Ask one question through the full agentic graph
linki ask "How do I configure TrustedHostMiddleware?" --debug

# Multi-turn REPL
linki

# Web UI (TypeScript console: topics, document upload, chat, trace, evidence)
linki web            # http://127.0.0.1:7860
```

Example (`--debug` shows the routing/retrieval decisions):

```
[router] retrieve — needs knowledge-base content
[retrieve] 2 chunk(s); verified=True
╭─────────────────────────── Linki ───────────────────────────╮
│ To configure TrustedHostMiddleware, pass allowed_hosts ...[1]│
│ Uvicorn workers do NOT share memory ...[2]                   │
│ Sources: [1] fastapi.md · Middleware > TrustedHostMiddleware │
│          [2] fastapi.md · Server Workers                     │
╰──────────────────────────────────────────────────────────────╯
```

---

## Configuration

Copy `linki.yaml.example` to `linki.yaml` to override defaults (chunk sizes,
retrieval `k`, loop caps, embedding models, and the knowledge-base registry). Model
provider and API keys live in `.env`, never in yaml.

Each knowledge base has a `usage_hint` — this is the only routing signal the planner
sees (Phase 3), so write it as "what this KB holds / which questions belong here".

---

## Testing

```bash
uv run pytest            # unit tests: planner fan-out, grade→refine dedup, citations, verifier reflow
```

The graph and CLI import without the heavy RAG stack (Qdrant/embeddings are lazy),
so the reliability logic is unit-tested with fakes — no API key or model download needed.

### Reproducible MultiHop-RAG benchmark

Linki includes an adapter for the official MultiHop-RAG corpus (2,556 questions,
609 news articles). The benchmark index is isolated under `.linki/benchmarks/`
and uses the same production chunker and hybrid retriever as normal queries.

```bash
# Download the official questions and full corpus, then build the isolated index.
linki bench-prep --benchmark multihop-rag
linki bench-ingest --benchmark multihop-rag

# Retrieval-only run over all 2,556 questions (no API calls).
linki bench-retrieval --benchmark multihop-rag

# Stratified end-to-end run with all four official query types.
# Strict mode compares a fair single-shot baseline, a no-reflow ablation,
# and the complete agentic workflow while recording latency/calls/tokens.
linki eval --benchmark multihop-rag --strict --limit 40 --seed 42
```

The baseline has the same model, retriever, evidence-only policy, refusal rule,
and citation requirement as Linki; only the orchestration differs. Reports include
retrieval recall/precision, strict all-support recall, answer token F1, gold-answer
containment, refusal correctness, citation validity, faithfulness, latency, model
calls, and token usage. Empty retrieval is counted as a failure rather than skipped.

---

## Project structure

```
src/linki/
├── config.py            Settings + knowledge-base registry (linki.yaml / env)
├── core/                providers (OpenAI/DeepSeek + judge), session, json utils
├── ingestion/           loader (PDF/MD) · chunker (parent/child) · indexer (embedded Qdrant + parent store)
├── tools/               hybrid retrieval → Evidence · tool registry
├── graph/               state · prompts · evidence(citations) · nodes · subgraph(grade→refine) · workflow
├── ui/                  FastAPI backend + React/TypeScript web console
└── cli/app.py           linki ingest / ask / web / REPL

docs/superpowers/specs/  design document
project/                 upstream agentic-rag-for-dummies reference implementation
tests/                   unit tests (fakes; no network)
```

---

## Roadmap

- ✅ **Phase 0–5** — package scaffold, hierarchical hybrid ingestion/retrieval, router/rewrite/clarify, planner fan-out with LangGraph `Send`, graded refine loops, evidence-only answers, citations, and verifier reflow.
- ✅ **Phase 6** — pre/post-retrieve hooks, persistent trace/timeline, and fair naive-vs-agentic evaluation.
- ✅ **Benchmark hardening** — official MultiHop-RAG adapter, full-corpus indexing, strict retrieval metrics, no-reflow ablation, and cost/latency telemetry.

---

## Credits

- RAG internals adapted from **[agentic-rag-for-dummies](https://github.com/GiovanniPasq/agentic-rag-for-dummies)** by Giovanni Pasqualino.
- Agentic backbone and design patterns from the **linki-agent-i** project.

## License

MIT — see [LICENSE](LICENSE).
