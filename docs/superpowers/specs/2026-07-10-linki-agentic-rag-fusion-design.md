# 设计文档 · 让 linki-agent-i 充当 Agentic 底座，融合出可靠的 Agentic RAG

> 日期：2026-07-10
> 目标读者：本项目开发者
> 状态：待 review

---

## 1. 背景与目标

`Linki-AgenticRAG升级规划.md` 描述了把一个线性 RAG 升级为 Agentic RAG 的六阶段蓝图。
本文回答一个更具体的问题：**不重复造"Agentic 脚手架"，而是把已经成熟的
`~/code-space/linki-agent-i` 当作 Agentic 底座，与本仓库已有的 RAG 参考实现
（`project/`）融合，产出一个"真实可用"的知识助手产品。**

核心洞察：这不是"从零搭 RAG"，而是**三份资产的融合**——大部分难点已经存在，
净新增工作集中在"可靠性增量"和"融合管线"两处。

| 资产 | 提供 | 成熟度 |
|------|------|--------|
| `linki-agent-i`（约 10K LOC） | Agentic **底座**：session、hooks 策略层、trace/timeline、checkpoint/recovery、tool 注册表+executor 管线、provider 工厂（OpenAI/DeepSeek）、intent_router 与 verifier 范式、compaction、CLI+TUI | 成熟 |
| `project/`（约 1.1K LOC 参考实现） | RAG **内核**：`DocumentChunker`（父/子切分）、`VectorDbManager`（**本地磁盘 Qdrant，无需 Docker**）、`ParentStoreManager`、`ToolFactory` 检索、rewrite+拆解（`QueryAnalysis`）、`interrupt_before` 澄清、上下文压缩 | 可运行 |
| `升级规划.md` | **目标架构** = 参考实现所缺的"可靠性增量" | 规范 |

**关键复审**：参考实现的 `orchestrator` 是一个 ReAct 工具调用循环——评估是*隐式*的，
**没有显式 grader、没有 verifier、没有逐句引用、没有多知识库选择**。
这几样恰是本产品的核心价值，也是本次要补的"Agentic 增量"。

---

## 2. 决策记录（本次已确认）

| # | 决策 | 选定 | 影响 |
|---|------|------|------|
| D1 | 项目目标 | **真实可用的产品**，最大化复用 | 架构偏"工程可复用"而非"教学可演示" |
| D2 | 代码归属 | **把 linki-agent-i 可复用模块拷贝进本仓库**，做自包含 fork | 交付物住在 `LinkiRAG-AgenticRAG/src/linki/`；不依赖外部 repo |
| D3 | 编排风格 | **C 混合**：可靠性关键路径用显式节点，基建与并行原语复用 | 见 §4 |
| D4 | 模型 provider | **复用 linki-agent-i 的 provider 工厂**，默认读现有 key（OpenAI/DeepSeek）；judge 用不同型号；embedding 一律本地 `fastembed` | 云端 LLM + 本地向量，离线只差 LLM |
| D5 | 首期里程碑 | **Phase 0→1→2**：能引用、能拒答的最小可靠单查询 RAG | planner/并行/TUI 延后到 Phase 3-4 |

---

## 3. 为什么是"C 混合"

- **可靠性契约必须结构化强制**：证据约束生成、逐句 `[n]` 引用、诚实拒答、grader 拦截，
  是产品的核心价值。放进自由 ReAct 循环里，模型"可能引用也可能不引用"——对
  "错答代价 > 延迟代价"的团队知识库场景不可接受。故 `grader / answer / verifier`
  必须是**显式图节点**。
- **并行检索与选库**复用 LangGraph `Send` 原语（子查询彼此独立、生命周期极短，
  比动用重型 swarm/AgentTool 更贴切；规划文档亦如此论证）。
- **纯复用**：intent_router、session、hooks、trace、checkpoint、compaction、
  tools 注册表+executor 管线、provider 工厂、CLI/TUI 外壳。

---

## 4. 目标架构

### 4.1 最终形态（Phase 4 完成）

```
router ──(retrieve)── planner ─Send▶ [检索子图 q1]
   │  \                              ├▶ [检索子图 q2]
 chat  clarify(interrupt)           └▶ [检索子图 q3]
   │                    检索子图: rewrite→retrieve→grade─(不足)→refine↺(≤MAX_ROUNDS)
   │                                                    └(足/认输)→collect
   ▼                    evidence 汇聚(Annotated add + retrieval_keys 去重)
 chat 回复                        │
                                  ▼
                     answer(仅证据·逐句[n]·缺则明说)
                                  ▼
                     verifier(逐条核对) ─pass→ 终答+来源面板
                                  └fail→ 回流 planner（带 issues 补检索，≤MAX_ATTEMPTS）
                                          └耗尽→ final_with_warning 透明降级

复用底座: intent_router 范式 · session · hooks(Pre/PostRetrieve) · trace · checkpoint
```

### 4.2 首期形态（Phase 2 完成，无 planner/并行）

```
router ──(retrieve)── rewrite ──▶ 检索子图(retrieve→grade→refine↺) ──▶ answer([n]) ──▶ verifier
   │                                                                                    │
 chat / clarify                                                          pass→终答  fail→回流 rewrite
```

首期以**单查询**跑通"检索→评估→引用生成→验证"闭环。verifier 失败时回流到
`rewrite`（尚无 planner），以 issue 作为改进查询重检，受 `MAX_ATTEMPTS` 约束。

---

## 5. 模块布局与来源映射

目标包：`src/linki/`（小写，与规划文档一致）。逐模块标注 **COPY**（原样拷贝）/
**ADAPT**（改造）/ **BUILD**（新写）及来源。

```
src/linki/
├── core/            COPY  ← linki-agent-i/src/Linki/core:
│                          state, session, trace, hooks, checkpoint,
│                          compact, context, paths, providers(openai_provider)
├── ingestion/       ADAPT ← project/:
│   ├── loader.py           pymupdf4llm PDF→Markdown（notebooks/pdf_to_markdown 逻辑）
│   ├── chunker.py          DocumentChunker（父/子 + HEADERS_TO_SPLIT_ON）
│   └── indexer.py          VectorDbManager(本地 Qdrant path) + ParentStoreManager
├── tools/           FUSE  ← linki registry+executor  ＋  Retrieve_kb_* 工具
│   ├── registry.py         复用 linki 注册表；build_retrieval_tools 生成每库一实例
│   ├── retrieve_tool.py    ADAPT ToolFactory：hybrid_search + fetch_parents
│   └── web_search_tool.py  COPY linki WebSearch（兜底）
├── graph/           BUILD (C 混合):
│   ├── state.py            LinkiGraphState（融合两边字段，见 §5.1）
│   ├── nodes.py            router/rewrite/answer/verifier
│   ├── subgraph.py         retrieve→grade→refine 循环
│   └── workflow.py         组图（首期 4.2 形态；Phase3 升 4.1）
├── hooks/           ADAPT ← linki hooks → Pre/PostRetrieve(cache/dedup/compress/trace)
├── eval/            REUSE ← notebooks/curated_ragas_qa.json + ragas + 自建对比脚本
└── cli/             ADAPT ← linki CLI/TUI 外壳：ask / ingest / eval
```

**丢弃**：编码专属件——`bash_tool`、`file_tools`(write/edit)、`code_agent`、
`grep_tool`、swarm 调度（首期用 Send 即可）。

### 5.1 状态字段融合

以 linki 的 `LinkiGraphState`(TypedDict) 为骨，并入 RAG 字段（借鉴 `project` 的
reducer）：

```python
class Evidence(TypedDict):        # 检索证据的统一结构
    chunk_id: str; parent_id: str; kb: str; source: str
    heading_path: str; text: str; score: float

class LinkiGraphState(TypedDict, total=False):
    # —— 复用 linki 现有：session_context, intent_route, ask_budget,
    #     attempts, max_attempts, provider, model, runtime ...
    question: str
    route: str                                    # chat|retrieve|clarify
    clarify_question: str
    rewritten_query: str
    sub_queries: list[SubQuery]                    # Phase3 起
    evidence: Annotated[list[Evidence], operator.add]
    retrieval_keys: Annotated[set[str], set_union] # 全局去重（沿用 project set_union）
    gaps: Annotated[list[str], operator.add]
    citations: list[Citation]
    answer: str
    verified: bool
    verify_issues: list[dict]                      # {claim, problem, fix_instruction}
```

---

## 6. 可靠性增量规格（本次净新增的核心）

### 6.1 grader（检索子图内，生成*前*）

- 输入：单个（子）查询 + 本轮*新*证据（去重后）。
- 输出 JSON：`{sufficient, relevant_chunk_ids, missing, refined_query}`。
- 标准（写死进 prompt）：证据须**直接**支撑，弱相关不算；宁可判不足，不可放行垃圾。
- 路由：`sufficient` 或 `rounds≥MAX_ROUNDS(=2)` 或 `too_similar(refined,原查询)` → collect；
  否则 → refine（用 `refined_query` 重检）。
- `collect` 只带走 `relevant_chunk_ids` 命中块；未 sufficient 时把 `missing` 写入 `gaps`。
- 解析失败 fallback：`sufficient=true`（评估挂了不卡死流程）。

### 6.2 answer（证据约束生成 + 逐句引用）

- 三条铁律逐字写死：①只用 `<evidence>`，禁用模型自身知识补全；②每个论断句末标 `[n]`；
  ③证据不足处明确输出"知识库中未找到 XX"。
- 注入 `<gaps>`：上游诚实记录的缺口 = 下游"未找到"声明的依据（两阶段闭环）。
- `number_evidence` 按来源聚合去重赋 `[1..n]`；`build_citations` 正则提取 `[n]` 回映
  `{index, source, heading_path, chunk_id}`，CLI 用 rich Panel 渲染"来源"面板。

### 6.3 verifier（生成*后*，只读不检）

- 输出 JSON：`{passed, issues:[{claim,problem,fix_instruction}], coverage, summary}`。
- 只依据给定证据判断，不引入外部知识；"未找到"类诚实声明视为合规。
- 路由：`verified` → final；`attempts≥MAX_ATTEMPTS(=2)` → final_with_warning（前置
  "⚠️ 以下回答未完全通过验证"并列 issues）；否则回流（首期→rewrite，Phase3→planner）。
- judge/verifier 使用与主模型**不同**型号（D4）。

### 6.4 grader vs verifier 分工

| | grader | verifier |
|---|---|---|
| 时机 | 生成前 | 生成后 |
| 对象 | 检索证据质量 | 答案与证据一致性 |
| 视角 | 单个子查询 | 用户原问题全局 |
| 动作 | 不足→改写重检 | 不过→回流补检索 |

---

## 7. 数据流

**入库（`linki ingest <path> --kb <name>`）**
`loader`(PDF→MD 保标题层级) → `chunker`(父块≤~4000/子块~500 overlap100，子块带
`parent_id/kb/source/heading_path`) → `indexer`(本地 Qdrant：dense=fastembed +
sparse=Qdrant/bm25；父块原文存 `parent_store/`)。

**问答（`linki ask "<q>"` 或多轮 REPL）**
`session_context` 注入 → router → rewrite → 检索子图(hybrid RRF 融合 top-k 子块
→ 按 `parent_id` 扩展父块 → grade → 不足则 refine 重检，`retrieval_keys` 全局去重)
→ answer(引用) → verifier → 终答+来源面板 / 诚实拒答 / 透明降级。

---

## 8. 配置与 provider

- `linki.yaml`：`knowledge_bases[{name,title,usage_hint}]`（description 即选库依据）、
  qdrant 本地路径、chunk 参数、hooks 启停顺序、`MAX_ROUNDS`/`MAX_ATTEMPTS`。
- provider：复用 `core/providers`，`.env` 读 `OPENAI_API_KEY`/`DEEPSEEK_*`；
  `JUDGE_MODEL` 单列。embedding 走本地 `fastembed`，与 LLM provider 解耦。
- **无 Docker**：`VectorDbManager` 用 `QdrantClient(path=QDRANT_DB_PATH)` 本地磁盘模式
  （已在参考实现 `config.py` 中如此配置），无需起 Qdrant 服务。

---

## 9. 错误处理与降级

| 场景 | 处理 |
|------|------|
| router JSON 解析失败 | fallback `retrieve`（宁可多查） |
| grader JSON 解析失败 | fallback `sufficient=true`（不卡死） |
| 检索多轮仍不足 | 认输，如实写 `gaps`，交 answer 诚实拒答 |
| verifier 判不过且 attempts 耗尽 | `final_with_warning` 透明降级，标注未验证部分 |
| 非法 `target_kb`（Phase3） | 回退默认库并记 trace |
| refine 与原查询过近 | 视为认输，防原地打转 |

---

## 10. 测试策略

- 单测（pytest，沿用 linki `tests/` 约定）：chunker 父/子结构、hybrid_search 返回
  Evidence 结构完整、grader 两轮判定（第二轮 hits 与第一轮无交集、gaps 认输非空）、
  answer 引用回映正确、verifier 对构造幻觉论断判 false 且定位到该论断、注册表长度=库数+1。
- 端到端（Phase2 验收）：①正常问题带来源面板；②库中无答案→诚实拒答；
  ③故意高温诱发一次幻觉→verifier 拦截并回流修正。
- 评测（Phase4）：`curated_ragas_qa.json` + ragas，`linki eval --mode naive|agentic`
  输出对比表（Recall@k / Faithfulness / 端到端正确率 / 拒答正确率 / 轮数 / token）。

---

## 11. 分阶段交付计划

| Phase | 交付物 | 主要动作 |
|------|--------|---------|
| **0 · 脚手架** | linki 底座拷入 `src/linki/`，砍编码工具，接 provider+config，`linki` CLI 可启动 | COPY |
| **1 · 垂直切片** | `linki ingest`(切分→本地 Qdrant 混合) → 单查询 `retrieve→answer` 端到端跑通（暂无 agentic） | ADAPT `project/` |
| **2 · 可靠性增量** | 显式 grader(grade→refine 去重) + answer 逐句 `[n]` 引用 + verifier 回流 + 诚实拒答 —— **首期终点** | **BUILD** |
| 3 · Planner+并行 | router(chat/retrieve/clarify) + planner(QueryPlan 多库) + `Send` 并行子图 | BUILD+复用 |
| 4 · 产品化 | Pre/PostRetrieve hooks、trace/timeline、session/checkpoint、TUI、`linki eval` 对比表 | 复用 linki |

**首期 = Phase 0→1→2**，每个 Phase 独立可演示、留下可用产品。

---

## 12. 风险与开放问题

1. **包名/大小写冲突**：linki-agent-i 是 `src/Linki/`（大写），本项目定为 `src/linki/`。
   拷贝时需统一改包名与 import，避免两套并存。**动手前第一步先做全局重命名脚本**。
2. **Python 3.13 依赖**：本机 3.13.5，`fastembed`/`sentence-transformers` 在 3.13 的
   wheel 需实测；必要时锁定次版本或降到 3.12 venv。
3. **linki provider 可拆性**：`core/providers/openai_provider.py` 与 linki 其它模块的
   耦合度需在 Phase0 拷贝时验证，可能要顺带拷 `core/state.RuntimeState`。
4. **检索走 executor 管线**：`Retrieve_kb_*` 须经 linki `execute_tool()` 才能让
   hooks/trace/approval 生效——这是"融合"而非"并列"的关键，Phase1 就要接对。
5. **embedding 型号**：参考实现用 `Qwen/Qwen3-Embedding-0.6B`(fastembed)；可换更小的
   `bge-small` 起步，入库后抽查 10 块人工确认"子块单一语义、父块完整可读"。

---

## 13. 附录：完整六阶段愿景

本设计的 Phase 3-4 覆盖规划文档的阶段二~六（router/rewrite/clarify、planner/Send、
grader/refine、answer/verifier、hooks/trace/eval），阶段一（知识底座）由 Phase 1
的 ingestion 承接。规划文档作为该愿景的详细讲解与验收脚本参考保留。
