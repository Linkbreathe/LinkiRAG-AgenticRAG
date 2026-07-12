<p align="center">
  <img alt="Linki logo" src="assets/logo.png" width="320px">
</p>

<h1 align="center">Linki · Adaptive Agentic RAG</h1>

<p align="center">
  <strong>Evidence-grounded answers, governed memory and knowledge, and a release-gated path to self-evolution.</strong>
</p>

<p align="center">
  <a href="#what-linki-is">Overview</a> •
  <a href="#architecture">Architecture</a> •
  <a href="#benchmark-status">Benchmarks</a> •
  <a href="#install">Install</a> •
  <a href="#usage">Usage</a> •
  <a href="#evaluation">Evaluation</a>
</p>

---

## What Linki is

Linki is a Python 3.12 knowledge assistant built around LangGraph, local
FastEmbed embeddings and Qdrant hybrid search. It separates four concerns that
are often mixed together in RAG projects:

1. **Answer execution** — choose a bounded P0–P3 path, retrieve evidence, answer
   with citations, and verify only when risk justifies the cost.
2. **Long-term memory** — version user-scoped preferences and episodes through a
   governed state machine; do not inject the whole conversation history.
3. **Organizational knowledge** — retain immutable source artifacts and
   bitemporal claims, then build cited Wiki and temporal graph projections.
4. **Controlled evolution** — turn feedback and failures into a backlog and
   offline candidates; require test, shadow, canary and rollback gates before
   production promotion.

The implementation plan and its open-source architecture references are in
[`docs/Linki-AgenticRAG二次升级规划.md`](docs/Linki-AgenticRAG二次升级规划.md).

> **Release status:** the adaptive runtime is implemented, but `auto` remains on
> the legacy path by default because the N=120 quality gate did not pass. Linki
> still records the local P0–P3 decision as `shadow_policy`. A caller can opt in
> per request with `--mode fast|balanced|deep`, or an operator can enable
> `adaptive_enabled` after calibrating against its own corpus.

## Architecture

```text
Request + tenant/user/ACL + immutable KB snapshot
                         │
                  local policy router
             shadow only │ or enabled/explicit mode
        ┌────────────────┼──────────────────────────┐
        │                │              │           │
       P0               P1             P2          P3
 local/chat       retrieve + answer   plan once   bounded plan/
 0–1 call             1 call         ≤3 calls    grade/verify ≤7
        └────────────────┴──────────────┴───────────┘
                         │
      hybrid candidates (30) ── optional corpus PPR pilot
                         │
      local cross-encoder → supporting spans → token-budgeted Evidence Pack
                         │
       cited answer + deterministic risk gate + optional verifier/reflow
                         │
       answer, evidence offsets, policy reason, cache/snapshot versions,
              per-node model/token/latency telemetry and trace

Governed side planes
  Episodes → memory candidates → proposed/review/active/superseded/deleted
  Sources  → claim versions → approval → Wiki + temporal graph projections
  Feedback → knowledge gaps → offline candidate → test → shadow → canary
                                                       └→ promote / rollback
```

### Runtime and cost controls

- Local, explainable P0–P3 routing runs before the first model call.
- Every path has `max_model_calls`, input-token, evidence, round and deadline
  budgets. Budget exhaustion degrades explicitly instead of looping forever.
- The primary server entry point is async; identical in-flight requests use
  single-flight coalescing.
- Exact answer/retrieval caches are versioned by prompt, model, index, memory,
  tenant, user and ACL dimensions. Semantic answer caching is off by default.
- SQLite is the local cache backend; a Redis adapter is available through the
  `cache` extra. Qdrant can run embedded or against Server/Cloud.
- `NodeCost` attributes model, prompt version, policy path, provider/estimated
  token usage and latency to every model call. Traces and cost reports are
  separate, optional persistence channels.

### Retrieval and evidence integrity

- Dense + sparse hybrid search gathers 30 cheap child candidates.
- `Xenova/ms-marco-MiniLM-L-6-v2` reranks locally through FastEmbed, with an
  explicit lexical fallback that is surfaced in evidence metadata.
- The model receives verbatim supporting spans, not silently rewritten text;
  each span keeps source version, content hash and validated character offsets.
- Evidence Packs enforce source diversity, sub-query coverage and path-specific
  token budgets (`1200 / 2400 / 4000`).
- `GraphRetriever` is an adapter boundary. The included two-hop PPR pilot is
  corpus-only and remains an expensive experimental path, not a default claim
  that “graph automatically fixes recall.”

### Memory, knowledge and self-evolution boundaries

- Explicit user memory can become active; inferred memory starts as a candidate
  and sensitive content requires review. Edit creates a newer version; forget
  tombstones content and redacts recoverable episode payloads.
- Memory is isolated by tenant/user/ACL and injected only after retrieval under
  a fixed token budget.
- A model output cannot become organizational fact. Claims require a source
  artifact, valid source span, schema checks and explicit approval.
- Claim queries support both valid time and system-known time. Wiki and graph
  are reproducible projections with staging, atomic promotion and rollback.
- Feedback mining creates knowledge-gap backlog items, never invented answers.
  Prompt/policy/index candidates move through immutable evaluation results,
  deterministic canary buckets and constraint gates.

## Benchmark status

The current release was measured on the official MultiHop-RAG corpus: 609
articles, 2,556 questions and 47,593 vectors. Full reports, exact protocol and
limitations are in [`BENCHMARK_RESULTS.md`](BENCHMARK_RESULTS.md).

### Full retrieval, N=2,556

| Variant | Recall | Strict all-support | MRR | Evidence tokens | p95 latency |
|---|---:|---:|---:|---:|---:|
| Frozen `legacy_k5` | 0.5858 | 0.2625 | 0.7539 | **744** | **0.672 s** |
| `rerank_pack` | 0.5994 | 0.2678 | 0.7520 | 1,278 | 1.076 s |
| `rerank_ppr` | **0.6348** | **0.3038** | **0.7807** | 1,620 | 2.397 s |

PPR improved strict support-chain recall by 4.13 percentage points but raised
p95 latency by 257% and evidence volume by 118%. It is therefore suitable for a
selective deep path, not the default fast path.

### Counterbalanced end-to-end evaluation, N=120

Three disjoint 40-question folds contain 30 comparison, inference, temporal and
null questions each. All four systems use the same main/judge models and corpus
snapshot; caches and memory are disabled, system order rotates per row, judge
calls are excluded from system cost, and provider token usage was available for
100% of measured calls.

| System | Calls | Mean tokens | p50 / p95 latency | Faithfulness | Quality | All-support | Refusal correct |
|---|---:|---:|---:|---:|---:|---:|---:|
| Fair single pass | **1.00** | **1,654** | **2.40 / 3.08 s** | **4.567** | 3.233 | 0.244 | 0.692 |
| Legacy full graph | 6.92 | 9,222 | 22.43 / 62.78 s | 4.417 | 3.592 | 0.433 | 0.650 |
| Adaptive auto | 1.26 | 2,440 | 4.14 / 6.95 s | 4.342 | **4.008** | 0.322 | **0.892** |
| Adaptive deep | 5.23 | 10,262 | 21.11 / 45.91 s | 4.267 | 3.475 | **0.467** | 0.675 |

Adaptive auto beat the legacy efficiency targets: mean tokens fell 73.5%, p50
latency fell 81.5%, and calls fell 81.8%. It also improved quality, gold-answer
containment and refusal correctness. It did **not** pass the strict quality
floor: paired all-support delta was `-0.111` with 95% CI `[-0.222, 0.000]`,
well below the allowed `-0.02` floor. On the P1 subset, one-call execution passed
but faithfulness (`4.271`) was below the fair single-pass value (`4.469`). This
is why adaptive auto remains shadowed by default.

Known scope limits: MultiHop-RAG does not measure the planned single,
multi-turn or cross-KB strata; the project memory suite is not a claimed
LongMemEval/LoCoMo result; embedded Qdrant warns above 20,000 points; retrieval
stability does not imply answer stability.

## Install

Python **3.12+** and [`uv`](https://docs.astral.sh/uv/) are recommended.

```bash
git clone https://github.com/Linkbreathe/LinkiRAG-AgenticRAG.git
cd LinkiRAG-AgenticRAG
uv venv --python 3.12
uv pip install -e '.[ui]'

# Optional Redis adapter and evaluation dependencies:
uv pip install -e '.[ui,cache,eval]'
```

Create a git-ignored `.env`:

```dotenv
# DeepSeek
LINKI_PROVIDER=deepseek
DEEPSEEK_API_KEY=...
DEEPSEEK_MODEL=deepseek-chat
LINKI_JUDGE_PROVIDER=deepseek
LINKI_JUDGE_MODEL=deepseek-reasoner

# Or OpenAI / an OpenAI-compatible gateway:
# LINKI_PROVIDER=openai
# OPENAI_API_KEY=...
# LINKI_MODEL=gpt-4o-mini
# LINKI_PROVIDER=gateway
# LINKI_GATEWAY_BASE_URL=http://localhost:4000/v1
# LINKI_GATEWAY_API_KEY=...
```

Models for dense, sparse and cross-encoder retrieval download on first use.

## Usage

```bash
# Ingest PDF or Markdown; a new immutable KB snapshot is promoted on success.
linki ingest ./docs/handbook.md --kb default --tenant acme --acl public

# Default auto uses the release-gated legacy path and records shadow_policy.
linki ask "What is the release policy?" --debug --tenant acme --user alice

# Explicit modes opt in to adaptive execution for this request.
linki ask "What is the release policy?" --mode fast
linki ask "Compare policy A and B" --mode balanced
linki ask "Strictly verify the complete evidence chain" --mode deep

# Multi-turn terminal session and web console.
linki
linki web  # http://127.0.0.1:7860
```

### Governance CLI

```bash
# Every command has detailed --help and JSON output where appropriate.
linki memory --help
linki memory remember "Use TypeScript examples" --tenant acme --user alice
linki memory list --tenant acme --user alice

linki knowledge --help   # source-add, claim-propose/approve/reject, query
linki wiki --help        # list/show/approve cited pages
linki evolve --help      # feedback, gaps, release gates, canary, rollback
```

The React/TypeScript console exposes execution mode, tenant/user scope, node
costs, evidence, memory controls, Wiki pages and feedback. The FastAPI backend
also provides `/api/chat`, `/api/memory`, `/api/wiki`, `/api/feedback` and
`/api/gaps` endpoints.

## Configuration

Copy [`linki.yaml.example`](linki.yaml.example) to `linki.yaml`. Secrets stay in
`.env`; YAML contains runtime policy only.

| Setting | Default | Purpose |
|---|---:|---|
| `adaptive_enabled` | `false` | Keep auto on legacy while recording shadow policy; explicit modes still opt in. |
| `candidate_k / rerank_k` | `30 / 8` | Separate high-recall candidates from prompt evidence. |
| `evidence_budget_*` | `1200/2400/4000` | Fast, balanced and deep Evidence Pack limits. |
| `graph_retrieval` | `none` | `ppr_pilot` requires an injected graph adapter. |
| `enable_answer_cache` | `true` | Versioned exact cache for approved stable paths. |
| `enable_semantic_cache` | `false` | Enable only after corpus-specific false-hit calibration. |
| `enable_memory` | `true` | Governed recall/formation; tenant/user scoped. |
| `qdrant_url` | unset | Use Server/Cloud instead of embedded storage. |

For the benchmark-sized 47,593-vector index, use Qdrant Server/Cloud for
capacity testing. Embedded mode remains convenient for local development, not a
production performance claim.

## Evaluation

```bash
# Offline tests use fakes; no provider key is needed.
uv run pytest -q

# Production frontend type-check and bundle.
cd src/linki/ui/frontend && npm run build

# Reproduce the public benchmark in an isolated namespace.
linki bench-prep --benchmark multihop-rag
linki bench-ingest --benchmark multihop-rag
linki bench-retrieval --benchmark multihop-rag --variant all
linki bench-fair --benchmark multihop-rag --fold-size 40
linki bench-stability --benchmark multihop-rag --limit 120 --seed 42 --repeats 3
linki bench-memory
```

Benchmark commands checkpoint progress and reject checkpoints whose dataset or
protocol fingerprint differs. Raw answers and retrieved URLs stay under
`.linki/benchmarks/.../runtime/eval/` and are intentionally ignored by Git.

## Project layout

```text
src/linki/
├── routing/       local P0–P3 policy and deterministic risk gates
├── retrieval/     candidates, reranker, spans, Evidence Pack and graph protocol
├── graph/         adaptive/legacy LangGraph workflows and cited answer nodes
├── cache/         SQLite, Redis protocol and single-flight
├── memory/        immutable episodes, candidates, policy, consolidation, recall
├── knowledge/     sources, bitemporal claims, Wiki/graph projections, snapshots
├── evolution/     feedback, gap mining, candidate gates, canary and rollback
├── core/          provider factory, RequestContext, telemetry and trace
├── ingestion/     PDF/Markdown chunking, parent store and Qdrant indexing
├── eval/          fair E2E, retrieval, memory and stability protocols
├── ui/            FastAPI backend and bundled React/TypeScript console
└── cli/           Typer command surface
```

## Provenance and license

Linki combines the agentic patterns of the Linkbreathe `linki-agent-i` work with
hierarchical RAG ideas adapted from
[`agentic-rag-for-dummies`](https://github.com/GiovanniPasq/agentic-rag-for-dummies).
The current adaptive, governance and evaluation layers live in this repository.

MIT — see [`LICENSE`](LICENSE).
