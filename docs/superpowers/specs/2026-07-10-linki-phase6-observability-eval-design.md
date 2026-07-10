# Linki Phase 6 — Hook · Trace · Eval 设计

> 阶段六：把"全流程黑盒"变成"可观测 + 可证明"。这是升级规划六阶段的最后一环，前五阶段（知识底座 / Router / Planner / 检索循环 / 证据生成+验证）已在 `src/linki/` 落地。

## 目标与非目标

**目标**
1. **Hook 体系**：在检索这个唯一的高频副作用操作前后各留一个挂点，把缓存 / 去重 / 记录这类横切逻辑从检索节点里剥离出来。
2. **持久化 Trace**：把当前只喂给 UI 的临时事件流，固化成 `.linki/traces/<run_id>.jsonl` + 可读的 `timeline.md`。
3. **评测**：搭出 naive vs agentic 的对照评测骨架（数据后填），用数字口径证明升级价值。

**非目标（YAGNI）**
- 敏感词过滤 / 超长证据压缩 Hook —— 只留扩展点 + no-op stub，不实现（当前无真实痛点）。
- 真实标注题集 —— dataset 只放 2–3 条占位样本，真实语料由使用者后填。
- 跨进程 / 跨会话的持久缓存 —— Cache 只在单次 run 内 memoize。
- 全会话上下文压缩 —— 不在本阶段范围。

## 现状（设计的出发点）

- 图已全线贯通：`router → rewrite → planner ─Send×N→ retrieve(grade/refine 循环) → answer → verifier ─(失败回流)→ planner`（`src/linki/graph/workflow.py`）。
- 事件流是**临时的**：节点通过 LangGraph 的 `stream_mode=["updates","custom"]` 吐事件，只有 `src/linki/ui/app.py` 把它们攒进一个内存 `trace` 列表，**不落盘**。
- 检索路径直连：`subgraph.retrieval_node` 直接调 `hybrid_search`，没有 Hook 层；父块去重逻辑散落在检索循环里。
- `core/session.py` 只持久化 Q/A 对，没有逐 run 的事件轨迹。
- `Settings`（`config.py`，frozen dataclass）是所有配置旋钮的落点。

## 核心架构决策：run-scoped 句柄如何抵达图内部

图在 `Send` 下会**并行 fan-out**，Hook（缓存、去重 seen-set）和 Tracer 必须能被并行分支共享地读到。

**选定方案 A — `contextvars` 环境上下文**（对照 B/C 见附录）：
- 在入口（`answer_question` / UI）处**每 run 构造一次** `HookManager`（cache + seen-set）和 `Tracer`，塞进 `contextvars.ContextVar`，检索路径与节点从中读取。
- 单一 `emit(event)` **扇出两个 sink**：现有 custom-stream writer（喂实时 UI）+ 持久化 JSONL（Tracer）。
- 为什么不是把句柄塞进 `LinkiGraphState`：`Send` 会**逐分支拷贝** state，cache 和 seen-set 无法真正共享，句柄也不易序列化——那样 Cache/Dedup 形同虚设。
- 为什么不是只在 tool 边界包一层：那样 trace 会漏掉 router/rewrite/answer/verifier 的事件，时间线不完整。

这条 contextvar 是 Hook 与 Trace 之间的**闭环接点**：一次 `emit`，实时 UI 和落盘 JSONL 同时拿到。

---

## 组件一：`hooks/` — 检索拦截层

### `hooks/base.py`

- `HookContext`（run 级）：`run_id`、`cache: dict`、`seen_parents: set[str]`、`tracer` 句柄。存于模块级 `ContextVar[HookContext | None]`，入口 set、`finally` reset。
- 两个协议：
  - `PreRetrieveHook.before(query: str, kb: str, ctx: HookContext) -> PreResult`，`PreResult` 可以是「改写后的 query」或「命中的缓存 hits（短路检索）」。
  - `PostRetrieveHook.after(query: str, kb: str, hits: list[Evidence], ctx: HookContext) -> list[Evidence]`。
- `HookManager(pre: list, post: list)`：`run_pre(...)` 顺序执行 pre-hook（任一返回 hits 即短路）；`run_post(...)` 顺序执行 post-hook，链式变换 hits。
- `run_retrieval(query, kb) -> list[Evidence]`：**唯一的检索缝合点**。逻辑：`run_pre → 若无缓存短路则 hybrid_search → run_post`。从 `ctx = current_hooks.get()`；ctx 为 None（未在 run 内，如单测直接调）时退化为直连 `hybrid_search`，保证可独立测试。
- `subgraph.retrieval_node` 改为调用 `run_retrieval(...)` 而非 tool 的 raw func。

### `hooks/builtin.py`

- `CacheHook`(Pre)：key `(query, kb)` → `ctx.cache` 内 memoize。命中则短路 `hybrid_search`，并 `emit` 一条 `cache_hit` 事件。在 grade→refine 重试与 verifier 回流重检时真实省 embedding+查库开销。
- `DedupHook`(Post)：丢弃 `parent_id` 已在 `ctx.seen_parents` 中的 hit（因 contextvar 进程内共享，去重同时覆盖**跨轮**与**跨并行子查询**）。把原先散在循环里的去重抽成命名、可测的 hook。
- `TraceLogHook`(Post)：向 Tracer `emit` 一条结构化检索事件（`query, kb, round, n_hits, top_score, latency_ms`）。**Hook↔Trace 闭环的落点。**
- `SensitiveWordHook` / `CompressHook`：仅 docstring + no-op stub，标注为扩展点，不接入默认链。

### 默认 Hook 链装配

- `hooks/builtin.py` 提供 `default_hooks(settings) -> HookManager`：pre=`[CacheHook()]`，post=`[DedupHook(), TraceLogHook()]`。
- 可通过 `Settings` 开关（`enable_cache: bool = True` 等）裁剪，默认全开。

---

## 组件二：`core/trace.py` — 持久化

- `Tracer(run_id, trace_dir)`：
  - `emit(event: dict)`：补全 `ts / run_id`，**扇出两 sink**——(1) 若存在 LangGraph custom-stream writer 则 push（实时 UI，行为与现状一致）；(2) 追加一行到 `.linki/traces/<run_id>.jsonl`。
  - `finalize()`：渲染 `.linki/traces/<run_id>.timeline.md`——按节点分组成可读链路（router 裁决 → rewrite → planner 子查询 → 逐轮 retrieve/grade → answer 引用数 → verifier 结论/回流）。落盘版的"UI 实时视图"孪生。
- 事件 schema（宽松、可选字段）：
  `{ts, run_id, node, type, round?, query?, kb?, n_hits?, top_score?, latency_ms?, detail?}`。
- **节点改动最小**：节点继续 emit 它们本来就 emit 的那些事件，只是把出口从"裸 stream writer"换成 `emit()`。UI 消费逻辑不变，落盘白捡。
- 装配：`graph/workflow.py` 的 `answer_question` 与 `ui/app.py` 各自每 run 构造 `Tracer` + `HookContext`、set contextvar、`finally` 里 `finalize()` + reset。

---

## 组件三：`eval/` — 用数字证明

- `eval/dataset.jsonl`：2–3 条**占位**样本（数据后填），每类问题一条，schema：
  `{id, question, type: chat|single|multihop|out_of_kb, expect: {should_refuse?, expect_sources?, expect_keywords?, gold?}}`。
- `eval/naive.py`：baseline 路径 `embed → top-k → 塞 prompt → LLM`，复用**同一套** retriever/embeddings，但绕过 router/planner/grader/verifier/引用。这是 agentic 的对照组。
- `eval/run_eval.py`：每题**同时**跑 naive 与 agentic，用现有 `judge_provider` LLM-judge 打分：
  - **faithfulness**（答案是否有证据支撑）
  - **citation coverage**（引用覆盖率）
  - **refusal correctness**（该拒答时是否正确拒答）
  - **answer quality**（1–5）
  输出 per-question + 汇总的 naive vs agentic **对照表**到 stdout，并写 `.linki/eval/<ts>.json`。KB 未入库时优雅打印 `N/A (需真实语料)`，保证骨架今天就能在占位数据上跑通。
- `linki eval` CLI 命令（`cli/app.py`）。

---

## 测试与隔离

- `hooks/`：用 fake hook + fake `HookContext` 测（无 LLM / 无 qdrant）；`run_retrieval` 在 ctx 为 None 时退化直连，便于单测。
- `core/trace.py`：emit 若干事件 → 读回 JSONL 断言 → 断言 `timeline.md` 渲染。
- `eval/`：用 fake workflow + fake judge 跑占位 dataset，断言对照表结构。
- 每个单元单一职责、接口清晰；重依赖（torch/qdrant/fastembed）沿用仓库既有的 lazy-import 约定，保证 graph/CLI/eval 骨架无重依赖可测。

## 交付物清单

- `src/linki/hooks/__init__.py`、`base.py`、`builtin.py`
- `src/linki/core/trace.py`
- `src/linki/eval/__init__.py`、`dataset.jsonl`、`naive.py`、`run_eval.py`
- `subgraph.py` 接 `run_retrieval`；`workflow.py` / `ui/app.py` 接 Tracer+HookContext；`config.py` 加开关；`cli/app.py` 加 `linki eval`
- 新增测试：`tests/test_hooks.py`、`tests/test_trace.py`、`tests/test_eval.py`

## 附录：被否决的接线方案

- **B. 塞进 `LinkiGraphState`**：`Send` 逐分支拷贝 state → cache/seen-set 不共享、句柄难序列化，Cache/Dedup 失效。否决。
- **C. 只包 tool 边界**：trace 漏掉 router/rewrite/answer/verifier 事件，时间线不完整。否决。
