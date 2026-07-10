# Linki 项目篇 · Agentic RAG 升级规划

> 从"检索一次就回答"，到"会思考的知识助手" —— 六个阶段，六次进化

---

## 开篇：最重要的问题 —— Linki 引入 Agentic RAG，究竟要解决什么？

### Linki 现状

Linki 当前是一个基于 LangGraph 的**知识库问答助手**，服务于个人/团队的技术知识场景：框架官方文档、项目 Wiki、API 手册、会议纪要。用户在终端里提问，Linki 从知识库里找答案。

它现在的 RAG 是最普通的线性管线：

```
用户问题 → Embedding → 向量库 top-k → 塞进 Prompt → LLM 生成答案
```

一条直线，四步走完。**任何一步出了问题，后面的步骤都没有补救机会** —— 这就是一切痛点的总根源。

### 普通 RAG 的七宗罪（在 Linki 真实场景中的表现）

| # | 问题 | Linki 中的真实翻车现场 | 根因 |
|---|------|----------------------|------|
| 1 | 复杂问题单次检索找不全 | 问"对比 FastAPI 和 Flask 的部署方式"，top-k 全被 FastAPI 的 chunk 占满，答案缺了一半 | 一个 query 只检索一次，对比类/多跳类问题天然需要多次检索 |
| 2 | 表达不准确时不能拆解或改写 | 多轮对话里追问"那它的性能呢？"——代词"它"直接进了 Embedding，召回一堆无关内容 | 检索词 = 用户原话，没有指代消解、没有查询改写 |
| 3 | 检索质量差仍然直接生成 | top-k 全是弱相关 chunk，LLM 硬着头皮"编"了一个像模像样的答案 | 流程里没有"评估"环节，垃圾进、垃圾出 |
| 4 | 信息不足时不会补充检索 | 召回的 3 条 chunk 只覆盖问题的一半，就着半份材料写出半截答案 | 检索只有一轮，没有"不够就再查"的循环 |
| 5 | 多知识源无法动态选择 | 明明该查 API 手册的问题，跑去会议纪要库里捞了半天 | 检索目标写死在代码里，不会按问题选库 |
| 6 | 生成后缺少证据与可靠性检查 | 答案里混进了模型自己的"预训练知识"，用户分不清哪句来自文档、哪句是编的 | 生成即终点，没有验证、没有来源标注 |
| 7 | 固定流程不能根据中间结果调整 | 上面所有问题的公共放大器 | 管线是写死的 DAG 直线，不是带决策的循环 |

### Agentic RAG 的答案：把"检索"从一次函数调用，变成一个决策过程

一句话说清核心转变：

> **普通 RAG 里，检索是流程中的"一步"；Agentic RAG 里，检索是 Agent 手中的"一个工具"——什么时候用、怎么用、用哪个、用几次、结果算不算好，全部由模型根据中间结果动态决定。**

七个问题与七种能力一一对应：

| 普通 RAG 的问题 | Agentic RAG 的解法 | 对应 Linki 新增节点 |
|----------------|------------------|-------------------|
| 1. 单次检索找不全 | **任务拆解**：复杂问题拆成多个子查询 | `planner` |
| 2. 表达不准确 | **查询改写 + 主动澄清**：指代消解、歧义反问 | `rewrite` / `clarify` |
| 3. 检索差仍生成 | **结果评估**：检索后先打分，不合格不放行 | `grader` |
| 4. 信息不足不补检 | **多轮检索**：评估不足 → 改写查询 → 再检索 | `grader → refine → retrieve` 循环 |
| 5. 无法动态选知识源 | **工具选择**：每个知识库封装为工具，由模型挑选 | 统一 Tool 注册表 + `planner` 路由 |
| 6. 缺少答案检查 | **答案验证**：核对证据一致性与来源完整性 | `verifier` |
| 7. 固定流程 | **重新规划**：条件路由 + 失败回流 | LangGraph 条件边 + 回流 `planner` |

### 升级完成后的 Linki（最终形态一图流）

```
                        用户问题
                           │
                           ▼
                    ┌─────────────┐
                    │   Router    │ ← 要不要检索？要不要先问清楚？
                    └──┬───┬───┬──┘
              chat ────┘   │   └──── clarify（反问用户，补充后回来）
                           │ retrieve
                           ▼
                    ┌─────────────┐
                    │   Planner   │ ← 拆子查询 + 给每个子查询选知识库
                    └──────┬──────┘
                           │ Send 并行分发
              ┌────────────┼────────────┐
              ▼            ▼            ▼
        ┌──────────┐ ┌──────────┐ ┌──────────┐
        │ 检索子图  │ │ 检索子图  │ │ 检索子图  │   每个子图内部：
        │  (q1)    │ │  (q2)    │ │  (q3)    │   改写→检索→评估
        └────┬─────┘ └────┬─────┘ └────┬─────┘   →不足则改写重检
              │            │            │
              └────────────┼────────────┘
                           ▼ evidence 汇聚
                    ┌─────────────┐
                    │   Answer    │ ← 只基于证据生成，逐句标注来源
                    └──────┬──────┘
                           ▼
                    ┌─────────────┐
                    │  Verifier   │ ← 答案与证据逐条核对
                    └──┬───────┬──┘
                passed │       │ failed（带着缺口回流 Planner 补检索）
                       ▼       │
                  最终答案+来源 ◀┘（达到重试上限则透明降级）
```

### 技术栈

| 组件 | 选型 | 为什么 |
|------|------|--------|
| LLM 调用 | `langchain` + `langchain-openai` | 与 Linki 现有代码一致，tool binding 开箱即用 |
| 工作流引擎 | `langgraph` | 条件路由 + Send 并行 + interrupt 人机交互，天然适配"评估→重试"循环 |
| 向量库 | `qdrant-client`（Docker 本地起） | 原生支持稠密 + 稀疏（BM25）混合检索与 RRF 融合 |
| Embedding | `fastembed`（dense: bge-small；sparse: bm25） | 本地推理，零 API 成本，入库快 |
| 文档解析 | `pymupdf4llm` | PDF → Markdown 保留标题层级，父子切分的前提 |
| CLI | `typer` + `rich` | 与 Linki 现有交互层一致 |
| 评测 | `pytest` + 自建 eval 脚本 | 用数字证明升级价值，而不是"感觉变好了" |

### 项目架构（最终形态）

```
src/linki/
├── ingestion/               # 知识入库
│   ├── loader.py            # PDF/MD 加载（pymupdf4llm）
│   ├── chunker.py           # 父子层级切分
│   └── indexer.py           # Qdrant 混合索引构建
├── tools/                   # 统一工具层
│   ├── registry.py          # 工具注册表（参考 Claude Code Tool 系统）
│   ├── retrieve_tool.py     # 知识库检索工具（每库一个实例）
│   └── web_search_tool.py   # Web 兜底检索
├── graph/                   # LangGraph 状态图
│   ├── workflow.py          # 主图：router→planner→检索→answer→verifier
│   ├── subgraph.py          # 检索子图：rewrite→retrieve→grade→refine 循环
│   ├── nodes.py             # 所有节点实现
│   └── state.py             # LinkiGraphState / SubQueryState
├── hooks/                   # 检索前后钩子
│   ├── base.py              # Hook 协议与执行器
│   └── builtin.py           # Cache / Dedup / Compress / TraceLog
├── core/                    # 基础设施
│   ├── state.py             # RuntimeState（知识库配置、路径）
│   ├── trace.py             # JSONL 事件流 + timeline.md
│   └── session.py           # 多轮会话历史
├── prompts/                 # 各阶段 Prompt
├── eval/                    # 评测
│   ├── dataset.jsonl        # 评测集（四类问题）
│   └── run_eval.py          # naive vs agentic 对比评测
└── cli/
    └── app.py               # typer 入口：ask / ingest / eval
```

---

## 第一部分：参考 Claude Code，适度升级 Linki

选型总原则：**不是复刻 Claude Code，而是对它的每一个机制问同一个问题——"Agentic RAG 需要它吗？需要多大剂量？"**

Claude Code 是通用编程 Agent（50+ 工具、95+ 命令、多任务后台调度），Linki 是垂直知识助手。照搬是过度设计，关键是识别出哪些机制是 Agentic RAG 的**必要前提**，然后做减法。

### 选型决策表

| Claude Code 机制 | 在 Claude Code 中的形态 | 它解决什么问题 | Linki 要不要 | 简化方案 |
|-----------------|----------------------|--------------|:---:|---------|
| **Tool 工具系统** | 50+ 工具统一注册，name/description/schema 标准化 | 模型自主选工具的前提是"所有工具长得一样" | ✅ 必须 | 一个 `registry.py`，每个知识库封装成一个 `Retrieve_xxx` 工具 + WebSearch 兜底 |
| **任务规划（TodoWrite / Plan Mode）** | 完整 todo 生命周期 + 计划模式切换 | 复杂任务先拆解、可追踪、可修订 | ✅ 简化 | 不要 todo 生命周期，只要一个 `QueryPlan`（子查询列表 + 目标库）存进图状态 |
| **多 Agent（AgentTool 子代理调度）** | 通用子 Agent 动态创建与调度 | 不同技能隔离上下文、互不干扰 | ✅ 简化 | 固定角色节点：Router / Planner / Grader / Answerer / Verifier，不需要动态子代理 |
| **Hook 机制（权限与安全）** | PreToolUse/PostToolUse 钩子 + 权限分级审批 | 横切逻辑（拦截、记录、改写）与工具执行解耦 | ✅ 简化 | 只保留两个挂点：`PreRetrieveHook`（缓存/改写/敏感词）、`PostRetrieveHook`（去重/压缩/记录） |
| **状态管理（AppStateStore）** | 全局状态仓库，UI 与引擎共享 | "根据中间结果调整下一步"的前提是中间结果被结构化记录 | ✅ 简化 | 一个 `LinkiGraphState`（TypedDict）+ `retrieval_keys` 去重集合，LangGraph 原生托管 |
| **上下文压缩（auto-compact）** | 全会话自动压缩 + 8 段式结构化摘要 | 长会话不爆上下文 | ✅ 大幅简化 | 只压检索证据：PostRetrieveHook 里做去重 + 超长压缩；不做全会话 compact（知识问答的会话远短于编程会话） |
| **AskUserQuestionTool** | 工具化的用户提问 | 歧义时主动澄清而不是瞎猜 | ✅ | 用 LangGraph `interrupt` 实现 clarify 节点，单次提问、够用 |
| Slash Commands / MCP / LSP | 命令系统、协议扩展 | 生态扩展 | ❌ 不要 | 与检索质量无关，是产品化阶段的事 |
| Bash 沙箱 / 文件写工具 | 代码执行与修改 | 编程场景专属 | ❌ 不要 | Linki 只读知识、不改文件，砍掉整类风险面 |
| 多任务后台调度（Task 系列工具） | 并发任务管理 | 长时编程任务 | ❌ 不要 | 检索的并行用 LangGraph `Send` 原语即可，无需任务系统 |

### 五个"要"的设计详解

**1. 统一 Tool 调用机制 —— Agentic 的地基**

- **解决什么**：第二部分工作流里的"选择合适的知识库或 Tool"这一步，前提是所有检索源以统一 schema 暴露给模型。没有这层，"动态选择工具"无从谈起。
- **为什么适合 Linki**：Linki 有多个异构知识源（API 文档 / Wiki / 会议纪要 / Web），差异只在 description 和底层 collection，非常适合"一个工具类 + 多实例"的轻量注册表。
- **Claude Code 给的最大启示**：工具的 **description 就是路由依据**。Claude Code 每个工具都有精心撰写的长描述，模型靠读描述决定用谁——Linki 的每个知识库描述要写"这个库装什么、什么问题该来查我"。

**2. 轻量任务规划 —— QueryPlan 而非 Todo 系统**

- **解决什么**：问题 1（复杂问题拆解）。Plan Mode 的精神是"先想清楚再动手"，对应到 RAG 就是"先拆解再检索"。
- **为什么适合 Linki**：检索子任务生命周期极短（秒级），不需要 in_progress/blocked 这类状态机，一个 `{id, query, target_kb}` 列表就是全部计划。**砍掉状态机，保留"结构化计划"这个内核。**

**3. 分工 Agent —— 固定角色而非动态调度**

- **解决什么**：检索、评估、回答、验证是四种不同技能，塞进一个 Prompt 会互相干扰（评估要求苛刻挑剔，回答要求流畅综合，天然冲突）。
- **为什么适合 Linki**：Agentic RAG 的角色集合是**固定且已知**的，不需要 Claude Code 那种"临时拉一个子 Agent 干活"的通用调度器。每个角色一个节点、一个专属 Prompt、一套专属输出 schema，复杂度砍掉一个数量级。

**4. Hook 机制 —— 检索前后的两个挂点**

- **解决什么**：缓存、敏感词过滤、结果去重、超长压缩、trace 记录……这些逻辑跟"检索本身"无关，写死在检索节点里会让节点迅速腐烂。
- **为什么适合 Linki**：检索是 Linki 唯一的"高频副作用操作"（对应 Claude Code 里的工具执行），在它前后各留一个挂点，就覆盖了 90% 的横切需求。Claude Code 的 Pre/PostToolUse 思想，剂量减半后正好。

**5. 任务状态与中间结果记录 —— 可调整的前提**

- **解决什么**：问题 7（固定流程）。"根据中间结果调整下一步"这句话的隐含前提是：**中间结果必须被结构化地记下来**——哪些子查询完成了、各召回了什么、评估结论是什么、还缺什么。
- **为什么适合 Linki**：LangGraph 的 State 本身就是共享内存，只需精心设计字段（而不是引入独立状态仓库）；再加一条 JSONL trace 流，调试和评测就都有了抓手。

---

## 第二部分：Linki 与 Agentic RAG 的结合方式

### 十步工作流 → 图节点映射

以 `agentic-rag-for-dummies` 的流程为参考骨架，Linki 的完整 Agentic RAG 工作流如下：

| # | 工作流步骤 | Linki 节点/机制 | 引入阶段 |
|---|-----------|----------------|:---:|
| 1 | 接收用户问题 | CLI 入口 + session 历史注入 | 阶段二 |
| 2 | 判断是否需要检索 | `router` 节点（chat / retrieve / clarify 三路） | 阶段二 |
| 3 | 分析并拆解复杂问题 | `planner` 节点输出 QueryPlan | 阶段三 |
| 4 | 改写检索查询 | 子图内 `rewrite` 节点（指代消解 + 检索友好化） | 阶段二/四 |
| 5 | 选择合适的知识库或 Tool | planner 为每个子查询标注 `target_kb`（基于工具 description） | 阶段一/三 |
| 6 | 执行检索 | `retrieve` 节点：Qdrant 混合检索 + 父块扩展 | 阶段一 |
| 7 | 判断检索结果是否足够 | `grader` 节点：结构化评估（充分性 / 相关性 / 缺口） | 阶段四 |
| 8 | 不足时重新检索或调整查询 | `grader → refine → retrieve` 循环（≤ max_rounds） | 阶段四 |
| 9 | 基于证据生成答案 | `answer` 节点：只用证据、逐句标注来源、缺则明说 | 阶段五 |
| 10 | 检查答案完整性、可靠性与来源 | `verifier` 节点：逐条核对，不过则带缺口回流 planner | 阶段五 |

### 升级前后能力对比

| 能力维度 | 升级前（普通 RAG） | 升级后（Agentic Linki） |
|---------|------------------|----------------------|
| 闲聊输入 | 也去翻一遍知识库，慢且滑稽 | router 直接短路回复 |
| 多轮追问"它/这个" | 代词进 Embedding，召回随缘 | rewrite 结合历史消解指代 |
| 歧义问题 | 瞎猜一个方向答 | clarify 反问一次再检索 |
| 对比/多跳问题 | 单次检索顾此失彼 | planner 拆解 + Send 并行检索 |
| 跨知识库问题 | 只会查默认库 | 按工具 description 动态选库 |
| 检索质量差 | 照样生成，答案像真的 | grader 拦截，改写重检 |
| 知识库没有答案 | 模型用预训练知识硬编 | 诚实回答"知识库中未找到" |
| 答案可信度 | 无从判断 | 逐句来源引用 + verifier 核对 |
| 过程可观测 | 黑盒 | trace 全链路事件 + timeline |
| 效果可证明 | "感觉好了" | 评测集 + 消融实验数字 |

### 这些能力对应的真实使用场景

1. **新人入职问跨文档问题**："上周会议定的限流方案，在我们 API 文档里对应哪个配置？" —— 需要跨 `kb_meetings` 和 `kb_api` 两库拆解检索（能力 3+5）。
2. **口语化连环追问**："FastAPI 怎么部署？" → "那它性能调优呢？" → "生产环境要注意啥？" —— 每一问都依赖指代消解与改写（能力 4）。
3. **偏门细节问题**：第一轮检索只召回边缘内容，grader 发现缺口后用更精确的术语重检（能力 7+8）。
4. **知识库覆盖不到的问题**：与其编造，不如诚实拒答并给出"你可以补充哪类文档"（能力 9+10）——这在团队场景里是**信任的来源**。

---

## 升级路线图：六个阶段总览

| 阶段 | 主题 | 围绕的核心问题 | 交付物 |
|------|------|--------------|--------|
| 一 | 知识底座：层级索引 + 混合检索 + 工具化 | 检索本身质量差，且检索是死代码不是工具 | ingestion 管线 + 统一工具注册表 |
| 二 | Router：先想清楚"要不要查、查什么" | 什么都检索 + 表达不准确 | router / rewrite / clarify 三节点 |
| 三 | Planner：复杂问题分而治之 | 单次检索找不全 + 多库无法动态选择 | QueryPlan + Send 并行检索 |
| 四 | 检索循环：评估与自我修正 | 检索差仍生成 + 不足不补检 | grader + refine 循环 + 去重 |
| 五 | 证据生成 + 答案验证 | 生成缺少可靠性检查与来源 | answer 引用体系 + verifier 回流 |
| 六 | Hook、Trace 与评测 | 全流程黑盒，无法量化提升 | Hook 体系 + trace + 评测集与消融 |

每个阶段都遵循同一条问题链：**当前问题 → Agentic RAG 如何解决 → Linki 需要增加什么能力 → 最终如何演示**。

---

## 阶段一：知识底座 —— 层级索引 + 混合检索 + 检索工具化

> **当前问题**：Linki 的检索又粗又死——固定大小切分让 chunk 要么碎得没上下文、要么长得没重点；单路向量检索对专有名词（API 名、版本号、报错信息）经常失灵；检索还是写死的函数调用，后续任何"让模型选工具"的能力都无从挂载。
> **Agentic RAG 如何解决**：层级索引（子块负责"被搜到"，父块负责"给模型看"）+ 稠密/稀疏混合检索互补短板 + 把每个知识库封装为统一 schema 的工具。
> **Linki 需要增加什么能力**：ingestion 管线（解析→父子切分→双路索引）、混合检索函数、工具注册表。
> **最终如何演示**：同一个问题分别用"普通切分 vs 父子切分"、"纯向量 vs 混合检索"跑一遍，肉眼可见的召回差异。

### 🎯 设计目标

在动任何"Agentic"的心思之前，先把"检"本身做扎实。**检索层是地基：地基是歪的，后面的评估、循环、验证全是在垃圾上打转。** 本阶段完成三件事：

1. 文档入库质量：PDF → Markdown（保留标题层级）→ 父子层级切分
2. 检索召回质量：Qdrant 稠密 + BM25 稀疏双路召回，RRF 融合
3. 检索工具化：每个知识库一个工具实例，统一注册（为阶段三的"动态选库"埋好钩子）

### 🏗️ 架构设计

```
            linki ingest ./docs/fastapi.pdf --kb kb_api
                           │
                           ▼
              ┌─────────────────────────┐
              │  loader.py               │  pymupdf4llm
              │  PDF ──▶ Markdown        │  保留 # ## ### 标题层级
              └───────────┬─────────────┘
                          ▼
              ┌─────────────────────────┐
              │  chunker.py              │
              │  按标题切【父块】≤1500tok │  ←"给模型看"：上下文完整
              │  父块内递归切【子块】~300 │  ←"被搜到"：语义集中
              │  子块携带 parent_id       │
              └───────────┬─────────────┘
                          ▼
              ┌─────────────────────────┐
              │  indexer.py → Qdrant     │
              │  dense 向量（bge-small）  │  ←语义相似
              │  sparse 向量（BM25）      │  ←精确词匹配
              └─────────────────────────┘

            检索时（retrieve_tool.py）：
            query ──▶ dense 召回 top20 ─┐
                  └─▶ bm25  召回 top20 ─┤─▶ RRF 融合 ─▶ top5 子块
                                              │
                                              ▼
                                    按 parent_id 换取父块 ─▶ 返回
```

**为什么是"父子"而不是"一刀切"**：检索和生成对 chunk 的需求天然矛盾——检索希望块小而纯（一个块一个语义点，向量才不糊），生成希望块大而全（模型要看到完整上下文才不断章取义）。父子结构让两边各取所需：**用子块搜，用父块答**。

**工具注册表（参考 Claude Code Tool 工具系统）**：

| 工具 | description（示意） | 底层 |
|------|--------------------|------|
| `Retrieve_kb_api` | "FastAPI/Flask 等框架官方文档。查 API 用法、配置参数、部署方式来这里" | Qdrant collection `kb_api` |
| `Retrieve_kb_wiki` | "团队项目 Wiki。查架构决策、内部规范、踩坑记录来这里" | Qdrant collection `kb_wiki` |
| `Retrieve_kb_meetings` | "会议纪要。查某次会议的结论、待办、讨论过程来这里" | Qdrant collection `kb_meetings` |
| `WebSearch` | "知识库都没有时的兜底：查最新版本、社区讨论" | Tavily |

### 🎬 演示效果

```bash
# 入库
$ linki ingest ./docs/fastapi.pdf --kb kb_api
📄 解析 fastapi.pdf → 312 段 Markdown
🧩 父块 87 个（avg 1180 tok）/ 子块 542 个（avg 290 tok）
📦 已写入 Qdrant：dense(768d) + sparse(bm25)

# 对照实验一：普通切分 vs 父子切分
$ linki ask "uvicorn 多 worker 部署时怎么共享内存？" --debug --no-parent
命中子块#217：\"...--workers 参数指定进程数...\"（上下文断裂，答案只提到参数名）
$ linki ask "uvicorn 多 worker 部署时怎么共享内存？" --debug
命中子块#217 → 扩展为父块#31《Deployment: Server Workers》全节
✅ 答案完整覆盖：进程模型、为什么不共享内存、推荐用外部存储

# 对照实验二：纯向量 vs 混合检索
$ linki ask "TrustedHostMiddleware 怎么配？" --debug --dense-only
❌ 召回的全是"中间件概述"类内容（专有名词被语义向量稀释）
$ linki ask "TrustedHostMiddleware 怎么配？" --debug
✅ BM25 精确命中该词条，RRF 融合后排第一
```

### 💡 讲解要点

- **检索与生成的矛盾**：为什么 chunk 大小是个两难，父子结构如何"两个都要"
- **稠密向量的盲区**：语义相似 ≠ 字面命中。`TrustedHostMiddleware`、`--workers`、报错堆栈这类 token 在向量空间里会被"稀释"，BM25 恰好补上
- **RRF 融合一句话**：不比较两路的分数（量纲不同没法比），只比较**排名**——两边都排前面的赢
- **Claude Code 的启示**：50+ 工具能被模型准确选用，靠的是统一 schema + 精心撰写的 description。Linki 的知识库 description 要按"这个库装什么、什么问题该来查我"的句式写——**description 写得好不好，直接决定阶段三选库准不准**

### ⭐ 核心要点

1. 子块 ~300 token / 父块 ≤1500 token 是起步经验值，入库后抽查 10 个块人工确认"子块单一语义、父块完整可读"
2. 每个块的 payload 必须带全元数据：`chunk_id / parent_id / kb / source / heading_path`——来源引用（阶段五）和去重（阶段四）全靠它
3. 工具返回**结构化 JSON**（`[{chunk_id, text, score, source}]`）而非拼接纯文本，下游节点才能程序化处理
4. 一个知识库 = 一个 collection = 一个工具实例，新增知识库零代码（配置驱动）

### 🔍 核心代码

**父子切分 — `ingestion/chunker.py`**

```python
def hierarchical_chunk(md_text: str, source: str) -> tuple[list[Chunk], list[Chunk]]:
    parents = split_by_headers(md_text, max_tokens=1500)      # 按 #/##/### 切父块
    children = []
    for p in parents:
        for piece in split_recursive(p.text, chunk_size=300, overlap=50):
            children.append(Chunk(
                id=new_id(), parent_id=p.id, text=piece,
                source=source, heading_path=p.heading_path,   # 如 "Deployment > Workers"
            ))
    return parents, children
```

**混合检索 + 父块扩展 — `tools/retrieve_tool.py`**

```python
def hybrid_search(query: str, kb: str, top_k: int = 5) -> list[Evidence]:
    hits = client.query_points(
        collection_name=kb,
        prefetch=[
            Prefetch(query=dense_embed(query), using="dense", limit=20),
            Prefetch(query=bm25_embed(query),  using="bm25",  limit=20),
        ],
        query=FusionQuery(fusion=Fusion.RRF),                 # 双路排名融合
        limit=top_k,
    ).points
    return fetch_parents(hits)   # 子块命中 → 返回对应父块内容（去重）
```

**统一工具注册 — `tools/registry.py`**

```python
def build_retrieval_tools(state: RuntimeState) -> list[StructuredTool]:
    tools = []
    for kb in state.knowledge_bases:          # 来自 linki.yaml 配置
        tools.append(StructuredTool.from_function(
            name=f"Retrieve_{kb.name}",
            description=f"检索「{kb.title}」。{kb.usage_hint}",   # ← 阶段三选库的唯一依据
            func=lambda query, kb=kb: hybrid_search(query, kb.name),
        ))
    tools.append(build_web_search_tool(state))
    return tools
```

### 🤖 Vibe Coding Prompt

**Step 1：初始化项目与 ingestion 管线**

```
帮我创建 Python 项目 Linki（typer CLI），实现知识入库管线。结构：

src/linki/
├── __init__.py / __main__.py
├── core/state.py          # RuntimeState：knowledge_bases 配置、qdrant 地址、路径
├── ingestion/
│   ├── loader.py          # load_document(path)->str：pymupdf4llm 把 PDF 转 Markdown；.md 直接读
│   ├── chunker.py         # hierarchical_chunk：按标题切父块(≤1500tok)，父块内递归切子块(300tok/overlap50)
│   └── indexer.py         # 建 collection（dense 768d cosine + sparse bm25/IDF），fastembed 编码后 upsert
└── cli/app.py             # linki ingest <path> --kb <name>

要求：
1. 子块 payload 带 chunk_id/parent_id/kb/source/heading_path；父块原文另存本地 parents.jsonl 供扩展查询
2. 知识库清单从 linki.yaml 读取：name/title/usage_hint
3. 依赖：qdrant-client, fastembed, pymupdf4llm, typer, rich, langchain, langchain-openai
```

**Step 2：混合检索与父块扩展**

```
在 src/linki/tools/retrieve_tool.py 实现：
1. hybrid_search(query, kb, top_k=5)：Qdrant query_points，prefetch 两路
   （dense limit=20 + bm25 limit=20），FusionQuery(RRF) 融合取 top_k
2. fetch_parents(hits)：按 parent_id 从 parents.jsonl 取父块内容，
   同一父块多个子块命中时只保留一份（按最高分），返回 Evidence 列表
   Evidence = {chunk_id, parent_id, kb, source, heading_path, text, score}
3. CLI 加 linki ask "<q>" --debug：先只做"检索并打印命中块"（暂不接 LLM 生成），
   --debug 打印每路召回与融合排名，加 --dense-only / --no-parent 两个消融开关
```

**Step 3：统一工具注册表**

```
在 src/linki/tools/registry.py 实现 build_retrieval_tools(state)：
- 遍历 state.knowledge_bases，每库生成一个 StructuredTool，
  name=f"Retrieve_{kb.name}"，description=f"检索「{kb.title}」。{kb.usage_hint}"
- 追加 WebSearch 工具（tavily-python，description 写明"知识库无结果时的兜底"）
- 工具函数返回 JSON 序列化的 Evidence 列表
写 pytest：注册表长度=库数+1；description 含 usage_hint；hybrid_search 返回结构完整
```

**Step 4：测试运行**

```bash
linki ingest ./docs/fastapi.pdf --kb kb_api
linki ask "TrustedHostMiddleware 怎么配置" --debug            # 混合检索
linki ask "TrustedHostMiddleware 怎么配置" --debug --dense-only  # 对照组
```

---

## 阶段二：Router —— 先想清楚"要不要查、查什么、要不要先问清楚"

> **当前问题**：Linki 对所有输入一视同仁地检索——用户说"你好"它也去翻文档（慢、贵、滑稽）；多轮对话里的"它/这个/上面说的"直接进 Embedding，召回随缘；遇到歧义问题（"帮我看看那个配置"）只会瞎猜一个方向。
> **Agentic RAG 如何解决**：检索前置三个决策——意图路由（chat/retrieve/clarify）、查询改写（指代消解 + 检索友好化）、主动澄清（human-in-the-loop）。
> **Linki 需要增加什么能力**：router / rewrite / clarify 三个节点，多轮 session 历史，LangGraph interrupt 机制。
> **最终如何演示**：三连测试——闲聊秒回、指代消解后精准召回、歧义问题反问一次再答。

### 🎯 设计目标

把"接到问题就检索"改成"接到问题先思考"。这是 Linki 第一次拥有**决策**：同一个入口，三条出路。同时引入多轮会话，让 rewrite 有历史可用。

### 🏗️ 架构设计

```
                 ┌─────────────┐
                 │    START    │
                 └──────┬──────┘
                        ▼
                 ┌─────────────┐     session 历史（最近 N 轮）
                 │   router    │◀────一并注入判断
                 └──┬────┬───┬─┘
            chat    │    │   │  clarify
          ┌─────────┘    │   └──────────┐
          ▼              │retrieve      ▼
   ┌──────────────┐      │       ┌──────────────┐
   │chat_responder │      │       │ clarify 节点  │
   │ 直接对话回复   │      │       │ interrupt()  │←── 图暂停
   └──────┬───────┘      │       │ 等用户补充    │    用户回答后恢复
          │              ▼       └──────┬───────┘
          │       ┌──────────────┐      │ 补充信息并入问题
          │       │   rewrite    │◀─────┘
          │       │ 指代消解+改写 │
          │       └──────┬───────┘
          │              ▼
          │        （检索主流程：本阶段先直连
          │          hybrid_search → 简单生成，
          │          阶段三起替换为 planner）
          ▼              │
   ┌─────────────────────┴──┐
   │          END            │
   └─────────────────────────┘
```

**router 的三分类**：

| 路由 | 判断依据 | 去向 |
|------|---------|------|
| `chat` | 寒暄、闲聊、与知识库明显无关 | 直接对话回复，不碰检索 |
| `retrieve` | 需要知识库支撑的问题 | rewrite → 检索主流程 |
| `clarify` | 指代无法从历史消解 / 存在多种合理理解 | interrupt 反问，补充后转 retrieve |

**rewrite 的本质**：让检索词**独立于对话上下文**——把"那它的性能呢？"改写成"FastAPI 的性能表现与调优方式"，改写后的 query 单独拿出来也能被正确检索。

### 🎬 演示效果

```bash
$ linki
🔗 Linki · 知识助手（多轮模式）

> 你好呀
[router] chat（置信 0.97）—— 跳过检索
你好！我是 Linki，可以帮你查 FastAPI 文档、团队 Wiki 和会议纪要，试试问我技术问题？

> FastAPI 怎么做后台任务？
[router] retrieve → [rewrite] "FastAPI 后台任务 BackgroundTasks 使用方式"
✅ （检索并回答，引用《Background Tasks》一节）

> 那它跟 Celery 比呢？
[router] retrieve → [rewrite] "FastAPI BackgroundTasks 与 Celery 的对比与选型"
                              ↑ "它" 被消解，且补全了对比意图
✅ （召回文档中 "BackgroundTasks vs Celery" 建议段落）

> 帮我看看那个配置怎么改
[router] clarify —— 图已暂停
🤔 你指的是哪个配置？我可以查：① uvicorn 部署参数 ② CORS 中间件 ③ 团队 Wiki 里的网关配置
> 第 2 个
[恢复] → [rewrite] "FastAPI CORS 中间件 CORSMiddleware 配置方法"
✅ （精准命中）
```

### 💡 讲解要点

- **路由首先是成本与体验问题**：每次都检索 = 每次都慢、都花 token；闲聊也翻文档是普通 RAG 最直观的"蠢"
- **rewrite 的判据**：改写后的 query 脱离对话单独读，是否仍然明确、完整、检索友好——这就是验收标准
- **interrupt 是 LangGraph 版 human-in-the-loop**：图在节点内暂停、状态落盘、用户输入后从断点恢复——与 Claude Code 的 `AskUserQuestionTool` 精神一致：**歧义时提问不是能力不足，而是避免自信地答错方向**
- **clarify 要克制**：宁可少问。只有"多种合理理解且答案会截然不同"时才反问，否则问个不停比答错更烦人

### ⭐ 核心要点

1. router 输出必须是 JSON，且要有解析失败的 fallback：**默认 `retrieve`**——宁可多查一次，不可把该查的当闲聊漏掉
2. rewrite 只在 `route=retrieve` 且存在会话历史时启用，首轮问题若已明确则原样通过（避免过度改写扭曲原意）
3. clarify 每次任务最多触发一次；反问时给出可选方向（枚举候选），比开放式反问对用户友好得多
4. session 历史只注入**最近 N 轮的问答摘要**而非全文，router/rewrite 的判断不需要全量细节

### 🔍 核心代码

**Router 节点 — `graph/nodes.py`**

```python
ROUTER_PROMPT = """你是 Linki 的路由器。结合对话历史判断用户最新输入，只返回 JSON：
{"route": "chat" | "retrieve" | "clarify",
 "reason": "...",
 "clarify_question": "route=clarify 时给出反问，附 2-3 个候选方向"}
判定规则：
- chat：寒暄/闲聊/与知识库明显无关
- retrieve：需要知识库内容支撑
- clarify：指代无法从历史消解，或存在多种合理理解且答案截然不同
"""

def router_node(state: LinkiGraphState) -> dict:
    decision = extract_json(model.invoke([
        SystemMessage(ROUTER_PROMPT),
        HumanMessage(f"对话历史：\n{state.get('session_context','（无）')}\n\n最新输入：{state['question']}"),
    ]).content, fallback={"route": "retrieve"})          # ← 解析失败宁可多查
    return {"route": decision["route"],
            "clarify_question": decision.get("clarify_question", "")}
```

**Clarify 节点（interrupt）与 Rewrite — `graph/nodes.py`**

```python
def clarify_node(state: LinkiGraphState) -> dict:
    supplement = interrupt({"question": state["clarify_question"]})   # 图在此暂停
    return {"question": f"{state['question']}（用户补充：{supplement}）",
            "route": "retrieve"}

REWRITE_PROMPT = """把用户最新问题改写成一条独立、明确、适合检索的查询：
- 结合历史消解代词（它/这个/上面说的），补全省略的主语与对比对象
- 保留专有名词原文，不要翻译、不要扩写成多句
只输出改写后的查询。"""
```

**图构建 — `graph/workflow.py`**

```python
graph.add_conditional_edges("router", lambda s: s["route"], {
    "chat": "chat_responder",
    "retrieve": "rewrite",
    "clarify": "clarify",
})
graph.add_edge("clarify", "rewrite")
# checkpointer 必须开启，interrupt 依赖它保存断点
app = graph.compile(checkpointer=SqliteSaver.from_conn_string(".linki/checkpoints.db"))
```

### 🤖 Vibe Coding Prompt

**Step 1：图状态与 session**

```
在 src/linki/graph/state.py 定义：
class LinkiGraphState(TypedDict, total=False):
    question: str; route: str; clarify_question: str
    rewritten_query: str; session_context: str
    evidence: list[Evidence]; answer: str
在 src/linki/core/session.py 实现 Session：追加问答对到 .linki/session.json，
build_session_context(n=5) 返回最近 n 轮"问-答摘要"文本。
```

**Step 2：三节点与图**

```
在 src/linki/graph/nodes.py 实现 router_node / clarify_node / rewrite_node /
chat_responder_node（按上文 Prompt），extract_json 带 fallback 参数。
在 workflow.py 组图：START→router→(chat_responder|rewrite|clarify→rewrite)，
rewrite 之后暂时直连 simple_retrieve_answer 节点（hybrid_search + 朴素生成），
compile 时挂 SqliteSaver checkpointer。
```

**Step 3：CLI 多轮交互与 interrupt 恢复**

```
改造 cli/app.py：linki 无参数进入多轮 REPL。
- 每轮：注入 session_context → astream 图事件 → rich 打印 [router]/[rewrite] 决策
- 捕获 interrupt：打印 clarify_question，读取用户输入，用 Command(resume=输入) 恢复图
- 回答后把本轮写回 session
```

**Step 4：三连测试**

```bash
linki
> 你好                      # 期望 chat 秒回
> FastAPI 怎么做后台任务？    # 期望 retrieve
> 那它跟 Celery 比呢？        # 期望 rewrite 消解"它"
> 帮我看看那个配置怎么改       # 期望 clarify 反问
```

---

## 阶段三：Planner —— 复杂问题分而治之，多知识库动态路由

> **当前问题**：一个 query 一次检索，对"对比 A 和 B"、"X 方案在 Y 文档里怎么落地"这类问题天然无解——top-k 会被单一主题占满；多个知识库形同虚设，检索目标写死在代码里。
> **Agentic RAG 如何解决**：任务拆解（复杂问题 → 最少必要的子查询集合）+ 工具选择（每个子查询由模型基于工具 description 挑选目标知识库）+ 并行执行（子查询相互独立，没有理由串行等待）。
> **Linki 需要增加什么能力**：planner 节点输出 QueryPlan、Send 并行分发、检索子图骨架、汇聚型状态字段。
> **最终如何演示**：一个对比类问题被拆成两路并行检索；一个跨库问题被正确分派到两个不同知识库。

### 🎯 设计目标

给 Linki 装上"先拆解、再检索"的大脑。planner 是本阶段唯一的新决策者：它读一遍问题和所有工具的 description，产出一份结构化的 QueryPlan——**这正是 Claude Code TodoWrite"先计划后执行"精神的最小化落地**（砍掉状态机，保留结构化计划这个内核）。

### 🏗️ 架构设计

```
                  rewrite（阶段二产物）
                        │
                        ▼
                 ┌─────────────┐   读取全部工具 description
                 │   planner   │──▶ 输出 QueryPlan（1~4 个子查询）
                 └──────┬──────┘
                        │ dispatch：Send × N（并行）
         ┌──────────────┼──────────────┐
         ▼              ▼              ▼
  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐
  │ 检索子图 q1  │ │ 检索子图 q2  │ │ 检索子图 q3  │
  │ target_kb=  │ │ target_kb=  │ │ target_kb=  │
  │  kb_api     │ │  kb_api     │ │  kb_meetings│
  │（本阶段先只： │ │             │ │             │
  │ 检索→收集）  │ │             │ │             │
  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘
         │               │               │
         └───────────────┼───────────────┘
                         ▼
              evidence: Annotated[list, operator.add]
                （并行结果自动汇聚，无需手工合并）
                         │
                         ▼
                  answer（本阶段仍是朴素生成，
                   阶段五升级为证据生成）
```

**QueryPlan 结构（TodoWrite 的减法版）**：

```python
class SubQuery(TypedDict):
    id: str            # "q1"
    query: str         # 子查询文本
    target_kb: str     # 目标工具名，如 "Retrieve_kb_api"
    reason: str        # 为什么拆出这条、为什么选这个库
```

**拆解的克制原则**：简单问题**不拆**（1 条直通），对比类按对象拆，跨库类按知识源拆，上限 4 条。过度拆解 = 噪声证据 + 成本翻倍。

### 🎬 演示效果

```bash
> 对比一下 FastAPI 和 Flask 的部署方式
[planner] QueryPlan（2 条）：
  q1 "FastAPI 生产环境部署方式 uvicorn/gunicorn" → Retrieve_kb_api（该库含 FastAPI 官方文档）
  q2 "Flask 生产环境部署方式 WSGI"              → Retrieve_kb_api
[Send] 并行执行 2 个检索子图 ... q1 命中 4 块 / q2 命中 3 块（耗时 ≈ 单查询）
✅ 答案分两节对比，两边证据齐备

> 上周会议定的限流方案，在我们 API 文档里对应哪个实现？
[planner] QueryPlan（2 条）：
  q1 "限流方案 结论"            → Retrieve_kb_meetings（会议结论应查纪要库）
  q2 "API 限流 rate limit 实现" → Retrieve_kb_api
✅ 跨库拆解正确：先从纪要里找到"令牌桶+网关层"结论，再从 API 文档定位对应中间件

> FastAPI 怎么读环境变量？
[planner] QueryPlan（1 条）：简单问题，不拆解，直通 Retrieve_kb_api
```

### 💡 讲解要点

- **planner 的输入里最关键的不是问题，而是工具 description 清单**——模型对"哪个库该查什么"的全部认知都来自它。这是阶段一埋的钩子第一次兑现，也复述了 Claude Code 的核心经验：工具描述即路由
- **为什么用 Send 并行**：子查询相互独立、无先后依赖，串行是纯浪费。对比 MokioClaw 式 CodeAgent 的 supervisor **串行**调度——那边的工程子任务有依赖（先搜资料后写代码），这边的检索子任务没有。**场景决定架构，能讲清这个对比就懂了两种编排**
- **`Annotated[list, operator.add]` 汇聚**：多个并行分支同时写同一个字段必然冲突，reducer 声明"这个字段用追加合并"，LangGraph 自动处理并发写
- **本阶段子图只是骨架**（检索→收集），阶段四会在同一位置装入"评估→修正"循环——先并行、后循环，复杂度逐层加

### ⭐ 核心要点

1. planner Prompt 中注入 `{kb_descriptions}` 时使用与注册表**完全相同**的文案，避免"计划里的库名"和"真实工具名"对不上
2. 子查询硬上限 4 条写进 Prompt 也写进代码（超出截断），双保险
3. `target_kb` 必须校验存在于注册表，非法值 fallback 到默认库并记 trace——planner 也会写错别字
4. 主图状态与子图状态分离：子图私有字段（rounds 等）不污染主图；只有 `evidence` 通过 reducer 汇回

### 🔍 核心代码

**Planner 节点 — `graph/nodes.py`**

```python
PLANNER_PROMPT = """你是 Linki 的查询规划器。把问题拆解为【最少必要】的子查询（1~4 条），
每条指定目标知识库工具。

可用知识库工具：
{kb_descriptions}

只返回 JSON：
{"sub_queries": [{"id":"q1","query":"...","target_kb":"Retrieve_kb_api","reason":"..."}]}

规则：
- 简单问题输出 1 条，禁止过度拆解
- 对比类按对象拆；跨知识源按库拆
- query 要检索友好：保留专有名词，一条只问一件事
"""

def planner_node(state: LinkiGraphState) -> dict:
    plan = extract_json(model.invoke([
        SystemMessage(PLANNER_PROMPT.format(kb_descriptions=render_tool_descriptions())),
        HumanMessage(state["rewritten_query"]),
    ]).content, fallback={"sub_queries": [{"id": "q1",
        "query": state["rewritten_query"], "target_kb": DEFAULT_KB, "reason": "fallback"}]})
    subs = [validate_kb(sq) for sq in plan["sub_queries"][:4]]   # 上限+库名校验
    return {"sub_queries": subs}
```

**Send 并行分发与汇聚 — `graph/workflow.py` / `graph/state.py`**

```python
def dispatch_retrieval(state: LinkiGraphState) -> list[Send]:
    return [Send("retrieval_subgraph",
                 {"sub_query": sq, "retrieval_keys": state["retrieval_keys"]})
            for sq in state["sub_queries"]]

graph.add_conditional_edges("planner", dispatch_retrieval, ["retrieval_subgraph"])
graph.add_edge("retrieval_subgraph", "answer")

class LinkiGraphState(TypedDict, total=False):
    ...
    sub_queries: list[SubQuery]
    evidence: Annotated[list[Evidence], operator.add]   # ← 并行分支自动追加合并
    retrieval_keys: Annotated[set[str], lambda a, b: a | b]  # 去重集合并集合并
```

### 🤖 Vibe Coding Prompt

**Step 1：planner 与计划结构**

```
在 src/linki/graph/nodes.py 实现 planner_node（Prompt 如上）：
- render_tool_descriptions() 从工具注册表提取 name+description 渲染成清单
- extract_json fallback 为单条直通计划；validate_kb 校验 target_kb 在注册表内，
  非法则替换为默认库并 log warning
- 状态新增 sub_queries；写 pytest：简单问题 1 条 / 对比问题 2 条 / 非法库名被纠正
```

**Step 2：检索子图骨架与并行分发**

```
在 src/linki/graph/subgraph.py 定义子图（本阶段两节点）：
- SubQueryState：sub_query, hits, retrieval_keys, evidence(Annotated add)
- retrieve_node：按 sub_query.target_kb 调 hybrid_search，写入 hits
- collect_node：hits→Evidence 列表写入 evidence（该字段与主图同名同 reducer，自动汇回）
在 workflow.py：planner 后接 dispatch_retrieval 的 Send 条件边，子图出口连 answer。
answer 节点暂用朴素生成（把 evidence 文本拼进 prompt）。
```

**Step 3：测试运行**

```bash
linki
> 对比一下 FastAPI 和 Flask 的部署方式          # 期望拆 2 条并行
> 上周会议定的限流方案，文档里对应哪个实现？      # 期望跨库分派
> FastAPI 怎么读环境变量？                      # 期望不拆解
# --debug 下确认：并行耗时≈单查询耗时，evidence 汇聚条数=各子图之和
```

---

## 阶段四：检索循环 —— 评估、修正与"不够就再查"

> **当前问题**：检索完直接生成——top-k 是弱相关也照样答（问题 3）；信息只覆盖一半也不会补查（问题 4）；整个流程无法根据中间结果调整（问题 7 在检索环节的体现）。
> **Agentic RAG 如何解决**：结果评估（grader 对召回做结构化打分）+ 多轮检索（不足 → 说清缺什么 → 改写查询 → 再检索）+ 重新规划（这一切通过条件路由动态发生，而非固定流程）。
> **Linki 需要增加什么能力**：子图内 grade / refine 节点与循环边、全局 retrieval_keys 去重、轮次上限与"诚实认输"机制。
> **最终如何演示**：一个偏门问题第一轮召回不足，trace 显示 grader 指出缺口 → refined_query 第二轮补齐；重复子查询自动跳过已见 chunk。

### 🎯 设计目标

这是整个升级中**最"Agentic"的一步**：检索子图从直线（检索→收集）变成带判断的环（检索→评估→不够→修正→再检索）。Linki 第一次具备"对自己的工作结果不满意"的能力。

### 🏗️ 架构设计

```
              检索子图（每个子查询一份实例）
   ┌───────────────────────────────────────────────┐
   │            sub_query 进入                       │
   │                 │                              │
   │                 ▼                              │
   │          ┌─────────────┐                       │
   │   ┌─────▶│  retrieve   │ 混合检索 + 去重 + 父块扩展│
   │   │      └──────┬──────┘                       │
   │   │             ▼                              │
   │   │      ┌─────────────┐                       │
   │   │      │   grader    │ 结构化评估：            │
   │   │      └──────┬──────┘ sufficient? 缺什么?     │
   │   │             │        哪些块真相关?           │
   │   │      ┌──────┴──────┐                       │
   │   │      │ sufficient? │                       │
   │   │      └──┬───────┬──┘                       │
   │   │     no  │       │ yes（或 rounds≥MAX）      │
   │   │         ▼       ▼                          │
   │   │   ┌─────────┐ ┌─────────┐                  │
   │   └───│ refine  │ │ collect │ 只带走 grader      │
   │       │用missing │ │ 认可的块 │ 认可的块；未满足    │
   │       │改写query │ └────┬────┘ 时如实记录 gaps    │
   │       └─────────┘      │                       │
   └────────────────────────┼───────────────────────┘
                            ▼
                    evidence 汇回主图

   全局去重：retrieval_keys（主图共享，Annotated 并集合并）
   ── 任何子图检到已见 chunk_id 直接丢弃，refine 后的第二轮
      不会重复捞第一轮的内容，逼检索走向新区域
```

**grader 的三个输出，各有去处**：

| 字段 | 用途 |
|------|------|
| `sufficient: bool` | 决定路由：collect 还是 refine |
| `relevant_chunk_ids` | collect 时只带走这些（噪声块就地淘汰，兼职做了 rerank） |
| `missing` + `refined_query` | refine 的原料：缺什么 → 下一轮怎么查 |

### 🎬 演示效果

```bash
> FastAPI 里怎么让某个接口跳过全局依赖注入？        # 偏门问题
[q1·round1] retrieve：4 块（全是"依赖注入概述"）
[q1·round1] grader：sufficient=false
            missing="缺少'覆盖/绕过已声明依赖'的具体机制"
            refined_query="FastAPI dependency_overrides 覆盖依赖"
[q1·round2] retrieve：3 块（去重后全新）——命中《Testing: dependency_overrides》
[q1·round2] grader：sufficient=true，relevant=[#88,#91]
✅ 答案给出 dependency_overrides 方案；trace 完整记录两轮决策

> （某个知识库确实没有的问题）
[q1·round1] grader：sufficient=false，missing="..."
[q1·round2] grader：sufficient=false —— 达到 MAX_ROUNDS=2
[collect] 如实记录 gaps=["未找到 XX 的直接说明"]，交给下游诚实处理
```

### 💡 讲解要点

- **grader 是 RAG 的免疫系统**：普通 RAG 的病根是"无条件信任召回"，grader 补上"先检疫再放行"这一环
- **评估标准要苛刻**："证据必须直接支撑答案，弱相关不算数；宁可误判不足，不可放行垃圾"——判不足的代价是多查一轮，放行垃圾的代价是幻觉答案，不对称
- **refined_query 为什么由 grader 出**：它刚读完全部召回，最清楚"缺的是什么"，由它一并给出下一轮查询，比另起一个改写节点少一次信息转手
- **死循环防护与诚实认输**：MAX_ROUNDS 兜底；认输时**如实记录 gaps** 而不是假装够了——"诚实的失败"是阶段五拒答能力的上游原料
- **与 Claude Code 上下文压缩的呼应**：多轮检索必然带来证据膨胀，去重（retrieval_keys）+ 只带走 relevant 块，就是检索场景下的"上下文管理"

### ⭐ 核心要点

1. grader 评估的对象是**子查询**而非用户原问题——子图只对自己的任务负责，全局完整性由阶段五 verifier 把关
2. `rounds` 是子图私有状态；`retrieval_keys` 是主图共享状态（并集 reducer）——一私一公，别弄反
3. 去重要在 grader **之前**做：让它只评新证据，省 token 也防止旧块反复刷分
4. refine 后的查询若与原查询几乎相同（编辑距离过小），直接视为认输——防"原地打转"式循环

### 🔍 核心代码

**Grader 节点与路由 — `graph/subgraph.py`**

```python
GRADER_PROMPT = """你是检索质量评估员。判断以下证据能否【直接、充分】地回答子查询。
只返回 JSON：
{"sufficient": bool,
 "relevant_chunk_ids": ["确实支撑答案的块 id"],
 "missing": "还缺什么信息（sufficient=false 时必填）",
 "refined_query": "针对缺口的改进查询（sufficient=false 时必填）"}
标准：证据必须直接支撑，弱相关不算；宁可判不足，不可放行垃圾。"""

def grader_node(state: SubQueryState) -> dict:
    verdict = extract_json(model.invoke([
        SystemMessage(GRADER_PROMPT),
        HumanMessage(f"子查询：{state['sub_query']['query']}\n\n证据：\n{render(state['hits'])}"),
    ]).content, fallback={"sufficient": True,          # 评估挂了别卡死流程
                          "relevant_chunk_ids": [h.chunk_id for h in state["hits"]]})
    return {"verdict": verdict, "rounds": state.get("rounds", 0) + 1}

def grader_route(state: SubQueryState) -> str:
    if state["verdict"]["sufficient"]:            return "collect"
    if state["rounds"] >= MAX_ROUNDS:             return "collect"   # 认输也走收集
    if too_similar(state["verdict"]["refined_query"], state["sub_query"]["query"]):
        return "collect"                                             # 防原地打转
    return "refine"
```

**去重检索与诚实收集 — `graph/subgraph.py`**

```python
def retrieve_node(state: SubQueryState) -> dict:
    hits = hybrid_search(state["sub_query"]["query"], state["sub_query"]["target_kb"])
    fresh = [h for h in hits if h.chunk_id not in state["retrieval_keys"]]   # 全局去重
    return {"hits": fresh,
            "retrieval_keys": {h.chunk_id for h in fresh}}   # 并集 reducer 汇入主图

def collect_node(state: SubQueryState) -> dict:
    keep = [h for h in state["hits"]
            if h.chunk_id in state["verdict"]["relevant_chunk_ids"]]
    gaps = [] if state["verdict"]["sufficient"] else [state["verdict"]["missing"]]
    return {"evidence": keep, "gaps": gaps}      # gaps 同样用 add reducer 汇回主图
```

### 🤖 Vibe Coding Prompt

**Step 1：子图升级为评估循环**

```
改造 src/linki/graph/subgraph.py：
- SubQueryState 增加 verdict/rounds/gaps 字段（gaps 用 Annotated[list, operator.add]）
- 新增 grader_node（Prompt 如上）与 refine_node（把 sub_query.query 替换为
  verdict.refined_query，rounds 保留）
- 图：retrieve→grader→条件边 grader_route（collect / refine→retrieve）
- MAX_ROUNDS=2；too_similar 用简单的字符重合率>0.85 判定
```

**Step 2：全局去重与状态收口**

```
- retrieve_node 检索后按主图 retrieval_keys 过滤，新 id 通过并集 reducer 写回
- collect_node 只保留 relevant_chunk_ids 命中的块；未 sufficient 时写入 gaps
- 主图 LinkiGraphState 增加 gaps: Annotated[list[str], operator.add]
- pytest：模拟 grader 两轮判定，断言最终 evidence 只含 relevant 块、
  第二轮 hits 与第一轮无交集、gaps 在认输场景非空
```

**Step 3：测试运行**

```bash
linki --debug
> FastAPI 里怎么让某个接口跳过全局依赖注入？   # 观察 round1 missing → round2 补齐
> （问一个知识库确实没有的问题）              # 观察两轮后 gaps 被如实记录
```

---

## 阶段五：证据生成 + Verifier —— 有据可查，不对就重来

> **当前问题**：生成是流程终点，没有任何检查（问题 6）——答案里混入模型预训练知识、来源无法追溯、知识库没有的内容被"编"出来，用户无从分辨真假。
> **Agentic RAG 如何解决**：证据约束生成（只用 evidence、逐句标注来源、缺就明说）+ 答案验证（verifier 逐条核对答案与证据的一致性和覆盖度）+ 重新规划（验证不过 → 带着具体缺口回流 planner 补检索）。
> **Linki 需要增加什么能力**：answer 节点的引用体系、verifier 节点与回流路由、"知识库中未找到"的诚实拒答路径。
> **最终如何演示**：正常回答带来源面板；知识库没有的问题得到诚实拒答；故意构造一次幻觉被 verifier 抓住并纠正。

### 🎯 设计目标

补上普通 RAG 缺失的"最后一公里"：**生成不是终点，验证通过才是**。本阶段把 Linki 的回答从"看起来对"升级成"可以查证"——每个论断有来源编号，每个来源可以点开原文，答不了的部分明说。

### 🏗️ 架构设计

```
        evidence + gaps（阶段三/四汇聚产物）
                     │
                     ▼
              ┌─────────────┐
              │   answer    │ 三条铁律：只用证据 /
              └──────┬──────┘ 逐句标 [n] / 缺就明说
                     ▼
              ┌─────────────┐
              │  verifier   │ 逐条核对（不重新检索）：
              └──────┬──────┘ ① 每个论断有证据支撑吗？
                     │        ② 与证据矛盾吗？
              ┌──────┴──────┐ ③ 覆盖了问题的所有子问题吗？
              │  passed?    │ ④ 来源编号标对了吗？
              └──┬───────┬──┘
            yes  │       │ no
                 ▼       ▼
          ┌──────────┐ ┌──────────────────────┐
          │  final   │ │ attempts < MAX?       │
          │ 答案+来源 │ └──┬──────────────┬────┘
          └──────────┘   yes│           no│
                            ▼             ▼
                     ┌──────────┐  ┌──────────────┐
                     │ planner  │  │final_with_    │
                     │带issues回流│  │warning 透明降级│
                     │→补检索    │  │（标注未验证部分）│
                     └──────────┘  └──────────────┘
```

**grader 与 verifier 的分工（面试必问）**：

| | grader（阶段四） | verifier（本阶段） |
|---|----------------|------------------|
| 时机 | 生成**前** | 生成**后** |
| 对象 | 检索证据的质量 | 答案与证据的一致性 |
| 视角 | 单个子查询 | 用户原问题全局 |
| 动作 | 不足 → 改写重检 | 不过 → 回流 planner 补检索/重写 |

**回流为什么到 planner 而不是 answer**：verifier 发现的问题多数是"证据缺口"（某论断没证据、某子问题没覆盖），重写答案无济于事——**缺证据要补检索，不是换措辞**。issues 进入 planner 后变成新的补充子查询。

### 🎬 演示效果

```bash
> FastAPI 的后台任务和生命周期事件分别怎么用？
✅ 答案（节选）：
   后台任务通过 BackgroundTasks 注入并 add_task 注册 [1]；
   生命周期推荐使用 lifespan 上下文管理器，startup/shutdown 事件已不推荐 [2]。
   ⚠️ 知识库中未找到两者执行顺序的直接说明。
📚 来源：[1] fastapi.pdf · Background Tasks   [2] fastapi.pdf · Lifespan Events
[verifier] passed=true（3 论断均有支撑；子问题覆盖 2/2；来源编号正确）

> 我们项目的灰度发布流程是什么？        # 知识库确实没有
[collect] gaps=["未找到灰度发布相关内容"]
✅ 知识库中没有找到灰度发布流程的相关文档。以下最接近的内容供参考：
   Wiki 中《部署规范》提到了发布审批流程 [1]，但未涉及灰度策略。
   建议：如果团队有相关文档，可以 linki ingest 补充进 kb_wiki。

# 幻觉拦截演示（把 answer 温度调高故意诱发一次）
[verifier] passed=false
   issues=[{claim:"BackgroundTasks 底层基于 Celery",
            problem:"无证据支撑，且与证据[1]（基于 starlette）矛盾"}]
[回流 planner] 补充子查询 "BackgroundTasks 实现机制 starlette"
✅ 第二轮答案修正，verifier 通过
```

### 💡 讲解要点

- **引用不是装饰**：它把"要不要信这句话"的判断权从模型手里交还给用户——用户可以点开 [1] 自己核对。这是知识助手在团队里建立信任的方式
- **诚实拒答是能力不是缺陷**：普通 RAG"有问必答"恰恰是最大的产品缺陷；"知识库中未找到 + 最接近的内容 + 补充建议"三段式，比编一个答案有价值得多
- **verifier 只读不检**：它只看"答案 vs 证据"，不重新检索——职责单一才能判断锐利；补证据是 planner 的活
- **上限后透明降级**：MAX_ATTEMPTS 用尽仍未通过时，不是死循环也不是假装通过，而是输出答案并**明确标注未验证部分**——把不确定性交给用户，而不是藏起来

### ⭐ 核心要点

1. answer Prompt 的三条铁律逐字写死：只依据 `<evidence>`；每个论断标 `[n]`；证据不足的部分输出"知识库中未找到"，禁止用模型自身知识补全
2. gaps（阶段四产物）要注入 answer Prompt——上游诚实记录的缺口，就是下游"未找到"声明的依据，两阶段在此闭环
3. verifier 的 `issues` 必须结构化（claim/problem/fix_instruction），回流时 planner 才能把它翻译成补充子查询
4. `attempts` 计数在主图，MAX_ATTEMPTS=2；回流路径与阶段四的检索循环是**两个不同层级的环**（内环修检索、外环修答案），图上别绕混

### 🔍 核心代码

**Answer 节点 — `graph/nodes.py`**

```python
ANSWER_PROMPT = """依据 <evidence> 回答用户问题。铁律：
1. 只使用 <evidence> 中的内容，禁止使用你自己的知识补全
2. 每个论断句末标注来源编号 [n]，n 对应证据条目
3. 证据不足以回答的部分，明确写"知识库中未找到 XX 的相关内容"
4. <gaps> 中列出的缺口必须如实告知用户
输出：答案正文 + 末尾"来源"清单（编号/文件/章节）。"""

def answer_node(state: LinkiGraphState) -> dict:
    numbered = number_evidence(state["evidence"])      # 赋 [1][2]... 并建 id 映射
    reply = model.invoke([SystemMessage(ANSWER_PROMPT), HumanMessage(
        f"问题：{state['question']}\n\n<evidence>\n{render(numbered)}\n</evidence>\n"
        f"<gaps>{state.get('gaps') or '（无）'}</gaps>"
    )]).content
    return {"answer": reply, "citations": build_citations(reply, numbered)}
```

**Verifier 节点与路由 — `graph/nodes.py`**

```python
VERIFIER_PROMPT = """你是答案审核员。逐句核对答案与证据，只返回 JSON：
{"passed": bool,
 "issues": [{"claim":"答案中的论断","problem":"无证据支撑|与证据矛盾|来源标错",
             "fix_instruction":"需要补充检索什么或如何修正"}],
 "coverage": "是否覆盖了用户问题的全部子问题，缺哪个",
 "summary": "一句话结论"}
注意：你只依据给定证据判断，不引入外部知识；"未找到"类诚实声明视为合规。"""

def verifier_route(state: LinkiGraphState) -> str:
    if state["verified"]:                          return "final"
    if state["attempts"] >= MAX_ATTEMPTS:          return "final_with_warning"
    return "planner"        # issues 随状态回流，planner 据此追加补充子查询
```

### 🤖 Vibe Coding Prompt

**Step 1：证据编号与 answer 节点**

```
在 src/linki/graph/nodes.py 实现：
- number_evidence(evidence)：按来源聚合去重后赋 [1..n]，返回带编号列表与 id 映射
- answer_node（Prompt 如上）；build_citations 从答案正则提取 [n] 并映射回
  {index, source, heading_path, chunk_id}
- CLI 输出用 rich Panel 渲染"来源"面板，--show-source n 可打印第 n 条证据原文
```

**Step 2：verifier 与双环路由**

```
- 实现 verifier_node（Prompt 如上）：写入 verified/verify_issues，attempts+1
- planner_node 增加分支：state 里有 verify_issues 时，把每条 fix_instruction
  转译为补充子查询（保留既有 evidence，不清空）
- workflow.py：answer→verifier→条件边（final / planner / final_with_warning），
  final_with_warning 在答案前加"⚠️ 以下回答未完全通过验证"并列出 issues
- pytest：构造含幻觉论断的假答案，断言 verifier 判 false 且 issues 定位到该论断
```

**Step 3：测试运行**

```bash
linki
> FastAPI 的后台任务和生命周期事件分别怎么用？   # 期望通过 + 来源面板
> 我们项目的灰度发布流程是什么？                # 期望诚实拒答 + 建议
# 临时把 answer 温度调到 1.2 诱发幻觉，观察 verifier 拦截与回流修正
```

---

## 阶段六：Hook、Trace 与评测 —— 工程化收尾，用数字证明价值

> **当前问题**：能力都有了，但横切逻辑（缓存/去重/压缩/记录）散落在各节点里越写越乱；全流程仍是黑盒，出了错只能靠 print；最致命的是——**没有数字能证明 Agentic 版比普通版好在哪**，说服不了面试官，也说服不了自己。
> **Agentic RAG 如何解决**：这一步不是 Agentic RAG 的新能力，而是让前五阶段的能力**可维护、可观测、可证明**——任何号称做过 Agent 工程的人都绕不开的三件事。
> **Linki 需要增加什么能力**：Pre/PostRetrieveHook 挂点、JSONL trace + timeline、四类问题评测集与消融开关。
> **最终如何演示**：一条命令跑出 naive vs agentic 的指标对比表；一份 timeline 完整展示某次提问的全部决策链。

### 🎯 设计目标

把散落的横切逻辑收编进 Hook 体系（借 Claude Code Pre/PostToolUse 之魂），把不可见的决策链落成 trace，把"感觉变好了"变成一张有数字的表。**这一阶段产出的评测表，就是简历上那几个百分号的出生地。**

### 🏗️ 架构设计

```
① Hook 体系（检索是 Linki 唯一高频副作用点，前后各一个挂点即覆盖 90% 横切需求）

     retrieve_node
     ┌────────────────────────────────────────────┐
     │  for h in HOOKS: ctx = h.pre(ctx)           │ ← PreRetrieveHook
     │      CacheHook   命中缓存直接短路             │
     │      GuardHook   敏感词/越权查询拦截           │
     │  hits = hybrid_search(ctx.query, ctx.kb)     │
     │  for h in HOOKS: hits = h.post(ctx, hits)    │ ← PostRetrieveHook
     │      DedupHook     retrieval_keys 去重        │
     │      CompressHook  单块>800tok 摘要压缩        │
     │      TraceHook     记录 query/命中/分数/耗时    │
     └────────────────────────────────────────────┘
     节点保持纯净，横切逻辑可插拔、可测试、可按需开关

② Trace（.linki/traces/<run_id>/）
     events.jsonl   逐事件：router_decision / plan / retrieve / grade /
                    refine / answer / verify / fallback，含耗时与 token
     timeline.md    人类可读的一次提问决策全景
     summary.json   本次运行统计（轮数/token/命中率）

③ 评测（eval/）
     dataset.jsonl  30~50 条，四类各占一档：
       single        单跳事实题     → 考基础检索
       multi_hop     对比/组合题    → 考拆解与并行
       cross_kb      跨知识库题     → 考工具选择
       unanswerable  库中无答案题   → 考诚实拒答
     run_eval.py    --mode naive|agentic，逐条跑图并打分
```

**指标定义（每个都要能当面说清算法）**：

| 指标 | 算法 | 考察 |
|------|------|------|
| Recall@5 | 命中 gold_sources 的比例 | 检索质量（阶段一/四） |
| Faithfulness | LLM-as-judge 逐论断核对证据支撑率 | 幻觉控制（阶段五） |
| 端到端正确率 | judge 对照 gold_answer 判等价 | 综合 |
| 拒答正确率 | unanswerable 类中诚实拒答的比例 | 诚实性（阶段四/五） |
| 平均检索轮数 / token 成本 | trace 统计 | **代价**——诚实展示 tradeoff |

### 🎬 演示效果

```bash
$ linki eval --mode naive && linki eval --mode agentic
┌────────────────┬─────────┬─────────┬────────┐
│ 指标            │ naive   │ agentic │ Δ      │
├────────────────┼─────────┼─────────┼────────┤
│ Recall@5       │ 61.2%   │ 84.7%   │ +23.5  │
│ Faithfulness   │ 72.4%   │ 93.1%   │ +20.6  │
│ 端到端正确率     │ 54.0%   │ 82.0%   │ +28.0  │
│ 拒答正确率       │ 12.5%   │ 87.5%   │ +75.0  │
│ 平均检索轮数     │ 1.0     │ 1.8     │ +0.8   │
│ 平均 token 成本  │ 1.0x    │ 2.6x    │ +1.6x  │ ← 代价也如实报告
└────────────────┴─────────┴─────────┴────────┘
（示意格式，数字以你的实测为准——评测就是为了得到真实数字）

$ linki eval --mode agentic --ablate no-hybrid     # 消融：关混合检索
$ linki eval --mode agentic --ablate no-grader     # 消融：关评估循环
$ linki eval --mode agentic --ablate no-verifier   # 消融：关答案验证

$ cat .linki/traces/run_042/timeline.md            # 某次提问的完整决策链
## 09:14:02 router → retrieve（0.4s）
## 09:14:03 plan → 2 sub_queries（q1→kb_meetings, q2→kb_api）
## 09:14:05 q1 grade → insufficient，missing="缺少结论性表述" → refine
## 09:14:07 q1 round2 → sufficient ✔
...
```

### 💡 讲解要点

- **Hook 思想的迁移**：Claude Code 用 Pre/PostToolUse 把"权限审批、日志"从工具执行里解耦，Linki 把同一思想按到检索上——**识别系统里唯一的高频副作用点，在它前后留口子**，这个方法论可复用到任何 Agent 系统
- **消融实验是设计决策的照妖镜**：每个消融开关对应一个阶段的核心设计，关掉它、指标掉多少，"为什么要做 X"从此有数字答案（面试反杀利器）
- **诚实报告代价**：Agentic 的检索轮数和 token 成本确实更高——能主动讲出 tradeoff 并说明"什么场景值得付这个代价"，比只报喜不报忧可信十倍
- **评测集为什么必须有 unanswerable 类**：不设陷阱题，永远测不出"系统会不会编"——而这恰恰是普通 RAG 和 Agentic RAG 差距最大的一档

### ⭐ 核心要点

1. 评测集 30~50 条就够，但每条必须有 `gold_sources`（否则 Recall 无从算起）；四类比例建议 4:3:2:1
2. LLM-as-judge 用与被测系统**不同的模型**，并抽 10 条人工复核 judge 的判断——评测器自己也要被评测
3. trace 事件在节点内通过统一 `tracer.emit()` 发出，禁止 print——事件即数据，timeline 只是它的一种渲染
4. Hook 执行顺序显式声明（Cache 必须最先、Trace 必须最后），并允许 `linki.yaml` 配置启停

### 🔍 核心代码

**Hook 协议与执行器 — `hooks/base.py`**

```python
class RetrieveHook(Protocol):
    def pre(self, ctx: RetrieveContext) -> RetrieveContext | ShortCircuit: ...
    def post(self, ctx: RetrieveContext, hits: list[Evidence]) -> list[Evidence]: ...

def run_with_hooks(ctx: RetrieveContext, hooks: list[RetrieveHook]) -> list[Evidence]:
    for h in hooks:
        ctx = h.pre(ctx)
        if isinstance(ctx, ShortCircuit):        # 如缓存命中
            return ctx.hits
    hits = hybrid_search(ctx.query, ctx.kb)
    for h in hooks:
        hits = h.post(ctx, hits)
    return hits
```

**评测脚本骨架 — `eval/run_eval.py`**

```python
def run_case(case: dict, mode: str, ablate: set[str]) -> CaseResult:
    graph = build_graph(mode=mode, ablate=ablate)     # naive=线性RAG基线
    out = graph.invoke({"question": case["question"], ...})
    return CaseResult(
        recall=hit_rate(out["citations"], case["gold_sources"]),
        faithful=judge_faithfulness(out["answer"], out["evidence"]),   # 异模型 judge
        correct=judge_equivalence(out["answer"], case["gold_answer"]),
        honest_refusal=(case["type"] == "unanswerable"
                        and contains_refusal(out["answer"])),
        rounds=out["trace_stats"]["retrieve_rounds"],
        tokens=out["trace_stats"]["total_tokens"],
    )
```

### 🤖 Vibe Coding Prompt

**Step 1：Hook 体系**

```
在 src/linki/hooks/ 实现 base.py（协议+run_with_hooks+ShortCircuit）与 builtin.py：
CacheHook（query+kb 归一化后哈希，TTL 10min）/ DedupHook（读主图 retrieval_keys）/
CompressHook（单块>800tok 用 LLM 压到≤300tok，保留数字与专有名词）/
TraceHook（emit retrieve 事件）。retrieve_node 改为调 run_with_hooks。
hooks 清单与启停从 linki.yaml 读取，顺序：Cache→Guard→(检索)→Dedup→Compress→Trace。
```

**Step 2：Trace**

```
在 src/linki/core/trace.py 实现 TraceRecorder：
- emit(type, **payload) 追加写 events.jsonl（含 ts/node/耗时/token 估算）
- 事件类型：router_decision/plan/retrieve/grade/refine/answer/verify/fallback
- finalize() 生成 summary.json 与 timeline.md（按时间线渲染，决策类事件加粗）
所有节点接入 tracer；CLI 加 linki trace <run_id> 直接渲染 timeline。
```

**Step 3：评测集与对比评测**

```
- 手工构建 eval/dataset.jsonl：40 条 = single 16 / multi_hop 12 / cross_kb 8 /
  unanswerable 4，字段 question/gold_answer/gold_sources/type
- run_eval.py：--mode naive|agentic（naive=单次混合检索+直接生成的基线图），
  --ablate no-hybrid|no-grader|no-verifier 逐项开关；
  汇总输出 rich 表格并落盘 eval/results/<ts>.json
- judge 模型从环境变量 JUDGE_MODEL 读取，须与主模型不同
```

**Step 4：完整验收**

```bash
linki eval --mode naive
linki eval --mode agentic
linki eval --mode agentic --ablate no-grader     # 每个消融跑一遍，记下 Δ
linki trace <最近一次 run_id>                     # 检查决策链完整可读
```

---

## 总结：六个阶段的进化路径

```
Stage 1        Stage 2        Stage 3        Stage 4        Stage 5        Stage 6
知识底座        Router         Planner        检索循环        证据+验证       工程化
──────────────────────────────────────────────────────────────────────────────────
层级索引        意图三分类      任务拆解        Grader评估     引用体系        Hook体系
混合检索        查询改写        多库工具选择    Refine修正     诚实拒答        Trace链路
工具注册表      主动澄清        Send并行       全局去重        Verifier回流    评测+消融
   │              │              │              │              │              │
   ▼              ▼              ▼              ▼              ▼              ▼
"检得准"      "想清楚再查"    "分而治之"     "不够再查"      "有据可查"      "可证明"
```

**每个阶段的核心增量与其解决的原始问题**：

| 阶段 | 核心增量 | 解决开篇七宗罪中的 |
|------|---------|------------------|
| 一 | 父子索引 + 混合检索 + 工具化 | 检索质量地基 + 问题 5 的前提 |
| 二 | router / rewrite / clarify | 问题 2 |
| 三 | QueryPlan + Send 并行 + 动态选库 | 问题 1、5 |
| 四 | grader + refine 循环 + 去重 | 问题 3、4、7 |
| 五 | 证据生成 + verifier 回流 + 诚实拒答 | 问题 6、7 |
| 六 | Hook + trace + 评测消融 | 让以上一切可维护、可观测、可证明 |

### 演示任务速查（录视频/面试演示用）

| 阶段 | 推荐演示 | 为什么 |
|------|---------|--------|
| 一 | `TrustedHostMiddleware 怎么配` 开关 `--dense-only` 对照 | 一眼看懂混合检索价值 |
| 二 | "你好" / "那它呢" / "那个配置" 三连 | 三条路由各走一次 |
| 三 | "对比 FastAPI 和 Flask 部署" + 跨库限流问题 | 拆解与选库同屏可见 |
| 四 | 偏门问题看两轮 refine | Agentic 的灵魂时刻 |
| 五 | 诱发一次幻觉看 verifier 拦截 | 最有戏剧性的 demo |
| 六 | naive vs agentic 评测表 | 收尾的"证据之王" |

### 面试讲解主线（一分钟版本）

> "Linki 原来是线性 RAG，我梳理了它在真实使用里的七类失败（复杂问题、指代、垃圾召回、单轮检索、多库、无验证、固定流程），然后参考 Claude Code 的工具系统、任务规划、Hook 和状态管理做了减法式借鉴，按六个阶段把它升级为 Agentic RAG：检索工具化 → 路由与改写 → 拆解与并行 → 评估与自修正 → 证据生成与验证 → Hook/Trace/评测。最终在自建的 40 条四类评测集上，端到端正确率提升 X 个点、拒答正确率从 X% 到 X%，代价是平均 1.8 轮检索和 2.6 倍 token——这个 tradeoff 在团队知识库这种'错答代价高于延迟代价'的场景里是划算的。"

（数字以你的实测为准。能把最后那句 tradeoff 讲出来的人，会被面试官记住。）

---

> 🔗 Linki —— 从"检索一次就回答"，到"会思考的知识助手"。
> 参考：Claude Code 源码学习专题（xuanyuancode.com/learn-claude-code）· agentic-rag-for-dummies（GiovanniPasq）
