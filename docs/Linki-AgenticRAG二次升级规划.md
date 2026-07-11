# Linki 项目篇 · 低成本、低延迟与 Memory/Knowledge 自进化升级规划

> 从“所有问题都跑完整 Agentic 链”，到“按问题难度付费；在证据约束下持续沉淀” —— 七个阶段，两条主线，一套可回滚的进化闭环

---

## 开篇：最重要的问题 —— Linki 的下一次升级，究竟要解决什么？

### 这不是重做 Phase 0–6，而是修正它们暴露出的第二层矛盾

现有 `src/linki/` 已经落地了 Router、Rewrite、Planner、并行检索、Grader、Refine、证据引用、Verifier、Hook、Trace 和评测。第一份规划解决了“线性 RAG 不会补救”的问题；现在真实基准揭示了新的矛盾：

1. **质量链越完整，平均成本越高**：简单问题也经过 Router、Planner、Grader、Answer、Verifier，系统把“最坏情况保险”变成了“每次请求固定税”。
2. **会记忆不等于会进化**：如果把对话、模型总结或一次成功回答直接写回知识库，短期看像“越用越聪明”，长期会积累重复、矛盾、过期事实和模型幻觉。
3. **图不是召回率的自动答案**：GraphRAG、LightRAG、HippoRAG、KAG 各自解决不同问题；直接引入一套图框架并不能保证 Linki 当前最弱的 inference / all-support recall 自动提高。
4. **Wiki 不是另一份真相**：自动生成的 Wiki 如果脱离原始证据、版本和引用，会成为第二个难以校验的知识库。

因此，本规划的核心不是“继续增加 Agent 节点”，而是完成两次架构转向：

> **在线链路：从 Full-Agentic-by-default 转成 Adaptive-Agentic-by-risk。**  
> **知识链路：从直接写库转成 Evidence-ledger → Candidate → Validate → Promote → Project。**

### Linki 当前基线：先承认真实数字，再谈优化

以下数字来自仓库现有 [`BENCHMARK_RESULTS.md`](../BENCHMARK_RESULTS.md)，不是推测：

| 维度 | 当前结果 | 它说明什么 |
|---|---:|---|
| 全量检索集 | 2,556 个问题 / 609 篇文章 | 检索侧已有可复现实验基础 |
| Answerable Retrieval Recall | 0.5858 | 仍有大量支持文档未被召回 |
| Answerable All-support Recall | 0.2625 | 多跳问题常常只找到证据链的一部分 |
| Inference All-support Recall | **0.1078** | 当前最明确的检索短板 |
| 检索平均耗时 | 0.6836 s | 检索不是 21 秒端到端延迟的唯一或主要来源 |
| Fair naive：延迟 / LLM 调用 / token | 2.326 s / 1.0 / 1,013 | 单次链路的成本下界参照 |
| Full agentic：延迟 / LLM 调用 / token | 21.092 s / 6.667 / 7,643 | 约为 naive 的 9.1× 延迟、7.5× token |
| No-reflow → Full：All-support Recall | 0.556 → 0.556 | 当前 12 题样本里，Verifier 回流没有增加完整证据召回 |
| No-reflow → Full：延迟 / token | +20.6% / +24.1% | 回流目前有可见代价，收益尚未被证明 |

需要同时强调两个限制：

- 2,556 题的 retrieval-only 结果足以定位检索短板；
- 端到端 N=12 只是**诊断样本**，不能当正式结论。新优化必须先扩大分层样本、记录置信区间，再决定默认策略。

### 新规划完成后的 Linki：读写两条链路分离

```text
                                   ┌──────── 在线读取面 ────────┐
用户问题 ─▶ ACL/租户边界 ─▶ 版本化缓存 ─▶ 本地策略路由器
                                      │
             ┌────────────────────────┼────────────────────────┐
             ▼                        ▼                        ▼
       P0 · no-RAG              P1 · fast-RAG           P2/P3 · deep-RAG
       闲聊/明确指令              单跳事实题               多跳/低置信/高风险
       0 次检索                  1 次检索                  Planner + 并行补检
             │                        │                        │
             └────────────────────────┴──────────┬─────────────┘
                                                 ▼
                                  多路候选 → 本地 rerank → Evidence Pack
                                                 │
                                                 ▼
                                           流式证据回答
                                                 │
                            规则检查通过 ─────────┴──────── 风险命中
                                  │                              │
                                  ▼                              ▼
                                返回                     条件式 Verifier/回流

                                   ┌──────── 后台写入面 ────────┐
原始文档 / 对话 / 用户纠错 / 失败 trace
             │
             ▼
不可变 Source/Episode Ledger ─▶ 候选抽取 ─▶ 去重/实体对齐/冲突检查/权限检查
                                                   │
                                      reject ◀─────┼─────▶ review
                                                   │ promote
                                                   ▼
                     Active Claims / Memory ─▶ Wiki 投影 + Temporal Graph + 搜索索引
                                                   │
                                                   ▼
                                   离线评测 → Shadow → Canary → Promote/Rollback
```

### 七条不可妥协的设计原则

1. **问题复杂度决定计算预算**：简单问题不为复杂问题的保险机制买单。
2. **先减少调用，再优化单次调用**：少一次 Planner/Grader/Verifier，通常比微调 prompt 省得更多。
3. **先提高证据密度，再压缩 token**：先 rerank、选 supporting spans，再考虑 LLMLingua；不能把压缩后的模型摘要伪装成原始证据。
4. **Memory 与 Knowledge 分域**：用户偏好、对话经历、团队事实、系统规则不能混在一个向量 collection。
5. **模型只能提出知识变更，不能直接成为事实来源**：自动写入先进入 candidate/quarantine。
6. **Wiki 与 Graph 都是可重建投影**：最终事实依据是带来源、时间和版本的 Claim/Episode Ledger。
7. **所有“自进化”都必须可评测、可灰度、可回滚**：没有回归门禁的自动优化只是自动漂移。

---

## 第一部分：成熟开源架构给 Linki 的答案

### 1.1 选型方法：Star 只作成熟度信号，不作技术结论

本次只采信项目官方仓库、官方文档或论文。GitHub Star 是 2026-07-11 的快照，用来说明社区规模；最终选型还同时看持续发布、文档、测试、与 Linki 痛点的直接相关性。

#### 低 token / 低延迟方向

| 项目 | 可核验成熟度 | 它真正解决什么 | Linki 的借鉴方式 |
|---|---|---|---|
| [vLLM](https://github.com/vllm-project/vllm) | 约 85.9k Star；持续发布 | PagedAttention、continuous batching、prefix caching、chunked prefill、speculative decoding | **只在自托管模型时引入**；先做调用削减，再 A/B 推理后端 |
| [LiteLLM](https://github.com/BerriAI/litellm) | 约 53.2k Star | 统一 Provider 网关、路由、fallback、预算、缓存与成本观测 | 作为可选 Gateway，不把业务路由逻辑塞进 Provider SDK |
| [Haystack](https://github.com/deepset-ai/haystack) | 约 25.9k Star；235+ releases | 条件路由与依赖就绪后的异步并发 | 借鉴 [`AsyncPipeline`](https://docs.haystack.deepset.ai/docs/asyncpipeline) 的并发原则，改造当前同步 `.invoke` 热路径 |
| [Adaptive-RAG](https://github.com/AsH1605/Adaptive-RAG) | NAACL 2024 官方实现 | 按问题复杂度选择 no-retrieval / single-step / iterative | 作为 Linki 三档执行路径的学术骨架 |
| [Semantic Router](https://github.com/aurelio-labs/semantic-router) | 约 3.6k Star | 用本地 embedding/规则在毫秒级路由，避免先调 LLM 才知道是否要调 LLM | 替代当前每题一次 LLM Router；不确定时升级路径 |
| [RouteLLM](https://github.com/lm-sys/RouteLLM) | LMSYS 官方实现 | 在强/弱模型间学习成本—质量路由阈值 | 对 Planner/Grader/Memory extractor 做模型级联；阈值必须用 Linki 数据校准 |
| [RedisVL Semantic Cache](https://redis.io/docs/latest/develop/ai/redisvl/api/cache/) | Redis 官方开源组件 | exact/semantic response cache、TTL、距离阈值和 metadata filter | 只缓存稳定、可版本化、可做 ACL 隔离的回答 |
| [Sentence Transformers Retrieve & Re-rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html) | 广泛使用的两阶段检索实现 | bi-encoder 高召回候选 + cross-encoder 精排 | 优先解决当前 inference/all-support recall 和上下文噪声 |
| [Qdrant Hybrid/Multi-stage Query](https://qdrant.tech/documentation/search/hybrid-queries/) | 当前已使用的向量底座 | dense+sparse 融合、候选预取、ColBERT 等多阶段重排 | 保留 Qdrant；先扩大候选再本地 rerank，不急于换库 |
| [Microsoft LLMLingua](https://github.com/microsoft/LLMLingua) | 经典 prompt compression 项目 | 长上下文压缩与 token budget 控制 | 放在 rerank 之后做可选消融；必须保留原文 offset 与引用回映 |

#### Memory / Knowledge / Wiki / Graph 方向

| 项目 | 可核验成熟度 | 核心机制 | Linki 应该学什么、不该照搬什么 |
|---|---|---|---|
| [Mem0](https://github.com/mem0ai/mem0) | 约 60.6k Star | 从消息抽取事实并搜索长期记忆；当前官方 add 流程是 additive pipeline | 学“抽取与检索分层”；**不能沿用旧文章里未经核验的 UPDATE/DELETE 假设** |
| [Letta](https://github.com/letta-ai/letta) | 约 23.7k Star | 常驻 context 的 memory blocks、agent-managed 与 read-only 边界 | 只把极少量高频 profile 放常驻块；组织策略必须 read-only |
| [LangMem](https://github.com/langchain-ai/langmem) | 与现有 LangGraph 原生衔接 | semantic / episodic / procedural 分类；hot-path 与 background formation | 作为最小改造起点；优先后台沉淀，避免增加用户等待 |
| [Graphiti](https://github.com/getzep/graphiti) | 约 28.6k Star；196 releases | Episode 溯源、增量更新、事实有效期、temporal graph、hybrid retrieval | 借鉴双时间与 supersede 模型；是否引入完整依赖需先做 pilot |
| [Microsoft GraphRAG](https://github.com/microsoft/graphrag) | 约 34.3k Star | entity/relation/claim、community detection、多层社区报告、local/global/DRIFT search | 适合离线全局概览；官方明确提示 indexing 昂贵，不能成为每库默认方案 |
| [LightRAG](https://github.com/HKUDS/LightRAG) | 约 37.5k Star | 图+向量、增量插入、混合查询、reranker、引用与删除后重建 | 作为 graph retrieval 对照组；不直接替换现有生产检索器 |
| [Cognee](https://github.com/topoteretes/cognee) | 约 27.6k Star | Extract/Cognify/Load 式知识图与长期记忆管线 | 借鉴流水线模块化；避免一次性引入完整平台 |
| [KAG](https://github.com/OpenSPG/KAG) | 约 8.8k Star | schema 约束、chunk↔knowledge mutual indexing、逻辑形式混合推理 | 对专业团队 Wiki 很有价值；先实现互索引，不先做完整符号推理引擎 |
| [HippoRAG 2](https://github.com/OSU-NLP-Group/HippoRAG) | 约 3.8k Star；NeurIPS/ICML 系列工作 | Knowledge Graph + Personalized PageRank 的多跳关联召回 | 针对 Linki inference 子集做实验，是解决证据链缺失的候选，不是既定答案 |
| [RAPTOR](https://github.com/parthsarthi03/raptor) | 约 1.7k Star；ICLR 2024 | 递归聚类与多层摘要树 | 借鉴 Wiki 的层级摘要；摘要必须回链原始 Claim，不能独立成为事实 |

### 1.2 最终选型：采用、试验、暂不采用

#### 立即采用

1. Adaptive-RAG 式 **P0/P1/P2/P3 分级执行**。
2. 本地规则 + embedding Router，LLM 只处理边界样本。
3. hybrid candidate → local cross-encoder rerank → token-budgeted Evidence Pack。
4. 条件式 Verifier 与条件式 reflow，取消“每题必审、失败必回流”。
5. Background Memory Formation + Candidate State Machine。
6. Source/Episode/Claim Ledger + Wiki/Graph 可重建投影。
7. 全链路版本号、ACL、快照、回滚与回归门禁。

#### 受控试验

1. Semantic response cache：只有在 KB snapshot、ACL、prompt/model 版本完全匹配时才可命中。
2. LLMLingua：只有在引用回映与答案质量不下降时才启用。
3. HippoRAG / LightRAG / Graphiti：通过统一 `GraphRetriever` 接口做 inference 子集 A/B。
4. vLLM / SGLang：仅当自托管吞吐与成本模型成立时比较；不是当前第一优先级。
5. DSPy prompt optimizer：只在离线 train/dev 上产出候选，不能在线自动替换 prompt。

#### 现在不做

1. 不把所有文档默认跑完整 GraphRAG Standard indexing。
2. 不允许 Agent 直接修改组织知识、系统规则或生产 prompt。
3. 不把全部历史对话塞进 prompt，也不把全部 Memory 设为 always-visible。
4. 不跨租户做 semantic cache，不用“问题相似”替代权限和版本一致。
5. 不为了“有 Graph”提前替换 Qdrant；先证明 graph channel 对目标子集有增益。

---

## 第二部分：Linki 的目标架构

### 2.1 在线执行路径：最少必要推理，而不是最完整推理

| 路径 | 典型问题 | 默认动作 | 预计 LLM 调用结构 |
|---|---|---|---|
| **P0 · no-RAG** | 寒暄、UI 指令、明确不需知识的问题 | 本地路由 → 小模型/模板回复 | 0–1 次 |
| **P1 · fast-RAG** | 单实体、单事实、单文档可答 | 直接检索 → rerank → Evidence Pack → Answer → 规则校验 | 通常 1 次 |
| **P2 · planned-RAG** | 对比、跨库、2–3 个独立子问题 | 一次 Planner → 并行检索/rerank → Answer → 风险门 | 通常 2 次；必要时 +1 Verifier |
| **P3 · deep-RAG** | 多跳、低召回置信、高风险、用户显式 `--deep` | Planner → 有界 refine → Answer → Verifier；证据缺口才回流 | 按预算上限执行 |

这里的“预计”是架构预算，不是现有实测结果。最终阈值和平均调用数必须在阶段七的基准上验收。

### 2.2 路由不是一个标签，而是可解释的 Policy Decision

```python
PolicyDecision = {
    "path": "p0" | "p1" | "p2" | "p3",
    "risk": "low" | "medium" | "high",
    "needs_rewrite": bool,
    "needs_verifier": bool,
    "max_subqueries": int,
    "max_retrieval_rounds": int,
    "evidence_token_budget": int,
    "reason_codes": list[str],
    "policy_version": str,
}
```

路由特征分三类，按便宜到昂贵依次使用：

1. **确定性特征**：是否寒暄、是否有显式 topic、是否含比较/时间/跨库词、是否含未消解代词、用户是否要求 deep。
2. **本地语义特征**：Semantic Router embedding 相似度、问题复杂度分类器、历史追问相似度。
3. **运行时证据信号**：top score、score margin、来源覆盖、reranker 分数、是否覆盖所有子问题、是否出现冲突 Claim。

原则是：**低置信时只允许升级，不能静默降级**。例如 P1 检索后发现 score margin 很小或问题含两个未覆盖实体，应升级 P2，而不是硬生成。

### 2.3 Memory 与 Knowledge 的边界

| 层 | 存什么 | 生命周期 | 能否自动写入 Active | 是否默认进 prompt |
|---|---|---|:---:|:---:|
| Thread State | 当前会话最近轮次、临时任务状态 | 单 thread | ✅ | 摘要后少量注入 |
| User Profile | 明确偏好、称谓、长期约束 | 跨会话，可撤销/过期 | 仅低风险且用户明确表达 | 只注入与当前问题相关部分 |
| Episodic Memory | 发生过什么、成功/失败任务摘要 | 跨会话，衰减 | 进入 candidate 后可自动晋级 | 检索命中才注入 |
| Semantic Memory | 经验证的用户/项目事实 | 版本化、可 supersede | 需证据或明确确认 | 检索命中才注入 |
| Organization Knowledge | 文档、会议结论、架构决策 | 强治理、可审计 | ❌，需来源/审批 | 通过 RAG 取证 |
| Procedural Memory | Prompt、路由规则、工作流策略 | 发布版本 | ❌，必须离线评测+审批 | 由系统版本固定加载 |

### 2.4 知识真相模型：一份 Ledger，多个 Projection

```text
SourceArtifact（原文/文件/消息/反馈，不可变）
  └─ Episode（一次发生或一次摄入；event_time + ingested_at）
      └─ ClaimCandidate（模型/规则抽取，尚未生效）
          └─ ClaimVersion（验证后生效，可被 supersede）
              ├─ Search Projection：BM25 / dense / rerank fields
              ├─ Wiki Projection：Domain / Topic / Page / Section / backlinks
              └─ Graph Projection：Entity / Claim / Relation / temporal edges
```

**Ledger 是事实与审计入口；Wiki 是给人读的目录；Graph 是给机器走关系的索引。** 三者不能互相冒充。

---

## 升级路线图：阶段七到阶段十三

| 阶段 | 主题 | 先解决什么 | 关键交付物 |
|---|---|---|---|
| 七 | 成本与延迟基线 | 不知道 token/延迟究竟花在哪个节点 | NodeCost、分层数据集、预算门禁 |
| 八 | Adaptive Agentic | 简单题也跑完整链 | P0–P3 Policy Router、模型级联、条件式验证 |
| 九 | 高召回、低上下文检索 | inference 证据链缺失，父块又吞 token | 两阶段 rerank、Evidence Pack、graph retrieval pilot |
| 十 | 并发、缓存与 serving | 重复计算、同步等待、扩展性边界 | async/single-flight/版本化缓存/Qdrant server 迁移 |
| 十一 | Memory 候选管线 | 会话没有长期沉淀，直接写又会污染 | Memory Ledger、background formation、state machine |
| 十二 | Knowledge → Wiki + Temporal Graph | 知识缺少可读结构、关系与时间版本 | Claim Store、Wiki 投影、Graph 投影、双向索引 |
| 十三 | 受控自进化 | 反馈不能自动转化为安全改进 | Gap Miner、离线优化、shadow/canary/rollback |

依赖关系必须按以下顺序推进，不能跳步：

```text
阶段七（先能量化）
   ├─▶ 阶段八（先减少无效调用）
   │      └─▶ 阶段十（再缓存、并发、优化 serving）
   └─▶ 阶段九（先把证据质量做稳）
          └─▶ 阶段十二（图通道必须与强检索基线对比）

阶段十一（Memory 候选与治理） ─▶ 阶段十二（统一 Claim/时间/溯源）
阶段八 + 九 + 十一 + 十二 ─────▶ 阶段十三（有数据、有门禁后才能自进化）
```

---

## 阶段七：成本与延迟基线 —— 先知道每一枚 token 为什么存在

> **当前问题**：Trace 能展示决策链，benchmark 能给端到端总量，但还不能稳定回答“Planner、每个 Grader、Answer、Verifier 各用了多少 input/output token、排队多久、TTFT 多久、哪种题最贵”。N=12 也不足以决定默认关闭或开启某个节点。  
> **成熟架构如何解决**：LiteLLM 把 provider usage/cost/latency 统一到网关；vLLM 暴露 serving 指标；成熟评测把质量与成本放在同一条 Pareto 曲线上。  
> **Linki 需要增加什么**：NodeCost schema、复杂度标签、可复现实验矩阵、质量—成本发布门禁。  
> **最终如何演示**：打开一份报告，能指出“简单题 68% 的 token 花在 verifier 输入”“P2 的 p95 被最慢分支拖住”等真实结论。

### 🎯 设计目标

1. 每次模型调用记录真实 provider usage，不再仅靠字符估算。
2. 区分 queue / retrieval / prefill(TTFT) / decode / total latency。
3. 按 `question_type × policy_path × node × model` 聚合。
4. 同一 corpus snapshot、prompt version、model version 下可重复比较。

### 🏗️ 数据结构

```python
NodeCost = {
    "run_id": str,
    "node": str,
    "call_index": int,
    "model": str,
    "prompt_version": str,
    "policy_path": str,
    "input_tokens": int,
    "cached_input_tokens": int | None,
    "output_tokens": int,
    "queue_ms": float | None,
    "ttft_ms": float | None,
    "total_ms": float,
    "cost_usd": float | None,
    "status": str,
}
```

`RunCost` 汇总时必须同时包含质量结果：retrieval recall、all-support recall、claim faithfulness、citation correctness、refusal correctness。只看 token 最少会把系统优化成“最快的错误答案”。

### 📋 实施步骤

1. 在 `core/providers.py` 的统一出口捕获 usage、model、cache token、重试和首 token 时间。
2. 给 Router/Planner/Grader/Answer/Verifier 分配稳定的 `prompt_version`。
3. 扩充端到端样本：保留 MultiHop-RAG 全量 retrieval-only；另做不少于 120 题的分层 E2E 集，包含 single / comparison / inference / temporal / null / multi-turn / cross-kb。
4. 固定 corpus snapshot、随机种子、temperature 与 provider；报告 bootstrap confidence interval，而不是只报单个均值。
5. 建立 `naive / current_full / no_reflow / future_p1 / future_p2 / future_p3` 实验槽位。
6. 所有后续阶段先跑 shadow，不覆盖当前默认路径。

### ✅ 验收门槛

- 100% 的 LLM 调用都能归属到 node、model、prompt version 和 policy path。
- 报告同时给出 p50/p95 TTFT、总延迟、token、调用数与质量；不再只给 mean。
- 当前 `BENCHMARK_RESULTS.md` 可按固定命令复现；新报告明确区分全量 retrieval 与分层 E2E。
- 阶段八开始前，先冻结一版 `baseline-v2`；后续目标百分比都以它为分母。

---

## 阶段八：Adaptive Agentic —— 让简单问题走短路，让复杂问题买保险

> **当前问题**：当前首轮明确问题虽然跳过 rewrite，但仍会经历 LLM Router → Planner → Grader → Answer → Verifier，最少也有多次模型往返。  
> **成熟架构如何解决**：Adaptive-RAG 在 no-retrieval、single-step、iterative 之间选择；Semantic Router 用本地向量做快速决策；RouteLLM 用校准阈值在强弱模型之间路由。  
> **Linki 需要增加什么**：本地 Policy Router、P0–P3 四条路径、Risk Gate、role-based model map、用户显式模式。  
> **最终如何演示**：同一套系统里，单跳题一次 LLM 调用返回；多跳题自动升级；低置信检索触发 deep path，且 debug 面板说明为什么。

### 🎯 设计目标

把“是否检索、是否规划、是否重检、是否验证”从四次独立 LLM 决策，收口成一个**便宜、可解释、可校准的策略层**。

### 🏗️ 路径设计

```text
request
  │
  ├─ exact command / greeting ─────────────────────────▶ P0
  │
  ├─ explicit topic + one intent + no unresolved ref ─▶ P1
  │
  ├─ compare / cross-kb / multiple facets ─────────────▶ P2
  │
  └─ high-risk / low retrieval confidence / conflict ─▶ P3

P1 检索后若 coverage 失败 ─▶ 升级 P2/P3
P2 回答后若 risk gate 命中 ─▶ Verifier
任何路径都受 max_calls / max_tokens / deadline 约束
```

### ⚙️ 模型分工

| 角色 | 默认模型策略 | 原因 |
|---|---|---|
| Route / complexity | 规则 + 本地 embedding；边界样本用小模型 | 不为“决定是否调用模型”先调用大模型 |
| Rewrite | 仅未消解指代时调用小模型 | 首轮清晰问题零调用 |
| Planner | P2/P3 才调用；结构化输出 | 复杂题才支付拆解成本 |
| Grader | cross-encoder + coverage rule；边界样本用 judge | 把每轮生成式 grading 改成例外机制 |
| Answer | 主回答模型 | 保留语言质量与证据整合能力 |
| Verifier | 高风险/低置信/抽样审计才调用强 judge | 把保险从固定税改成风险保费 |
| Memory extractor | 后台小模型 | 不阻塞用户热路径 |

### 🧠 Risk Gate

以下任一条件可触发 Verifier 或 P3：

- 高风险 domain 或用户要求“严格核验”；
- Evidence Pack 中存在互相 `CONTRADICTS` 的 active claims；
- 子问题 coverage 未全部满足；
- reranker top score 低于校准阈值或 top1/top2 margin 太小；
- Answer 出现无 citation 的事实性句子；
- citation index 合法，但本地 entailment/NLI 得分低；
- 系统随机抽样（例如固定小比例）用于持续估计漏检率。

阈值不能凭经验写死。先从 baseline-v2 上做 precision/recall 曲线，选择“漏掉高风险错误”的成本可接受点。

### 📋 实施步骤

1. 新建 `routing/policy.py`，输出 `PolicyDecision`，纯函数优先、可单测。
2. 给 CLI/UI 增加 `auto | fast | balanced | deep`；`auto` 由策略决定，用户可强制升级。
3. 拆出 P1 直连图：`retrieve → rerank → pack → answer → deterministic_validate`。
4. P2 合并“路由+规划”信息：本地路由确认复杂后，只做一次 Planner 调用。
5. 将 `verifier_route` 改成 Risk Gate 条件边；回流必须有明确 `missing_support`，禁止因措辞问题重复检索。
6. 引入每路径 budget：`max_model_calls / max_input_tokens / max_rounds / deadline_ms`。
7. shadow 记录“新策略会选什么路径”，先不影响用户答案；校准后再逐档开放。

### ✅ 目标门槛（待实测，不是既有结果）

- P1 单跳题中位数模型调用 ≤ 1，且 citation/faithfulness 不低于公平 naive 基线。
- 全体 E2E 平均 token 相比 `current_full` 至少下降 40%，p50 延迟至少下降 35%。
- All-support Recall、拒答正确率的 95% CI 下界不低于 `baseline-v2 - 2 个百分点`。
- Reflow 单独报告 `reflow_benefit_rate = 失败后被纠正的题 / 触发 reflow 的题`；收益未证明的题型默认关闭。

---

## 阶段九：高召回、低上下文检索 —— 召回更多候选，只给模型更少证据

> **当前问题**：当前 `retrieval_k=5` 后立即父块扩展；这既可能错过多跳支持文档，又可能把多个大父块原文同时送给 Answer 和 Verifier。Grader 看到每块前 600 字符，Answer/Verifier 却可能看到完整父块，评估对象并不完全一致。  
> **成熟架构如何解决**：Sentence Transformers 的经典 retrieve-and-rerank 先高召回候选、再 cross-encoder 精排；Qdrant 支持 dense+sparse prefetch 与多阶段重排；HippoRAG 用 PPR 扩展多跳关系。  
> **Linki 需要增加什么**：CandidateSet、local reranker、source diversity、supporting span、Evidence Pack、GraphRetriever 实验接口。  
> **最终如何演示**：同一 inference 问题先召回 20–50 个便宜候选，经精排后只给模型 3–6 个紧凑 supporting spans；完整证据链召回上升，输入 token 反而下降。

### 🎯 设计目标

将“检索数量”和“进 prompt 的数量”彻底解耦：

```text
Hybrid Recall(top-N children)
  → parent/source expansion
  → local cross-encoder rerank
  → source diversity / sub-question coverage
  → sentence/span selection
  → token-budgeted Evidence Pack(top-M)
  → answer/verifier 共用同一份 pack
```

### 🧱 Evidence Pack 结构

```python
EvidenceUnit = {
    "evidence_id": str,
    "chunk_id": str,
    "parent_id": str,
    "source_id": str,
    "source_version": str,
    "heading_path": str,
    "quote": str,
    "char_start": int,
    "char_end": int,
    "retrieval_score": float,
    "rerank_score": float,
    "supports": list[str],       # sub_query_id / claim slot
    "token_count": int,
    "content_hash": str,
}
```

Answer、Verifier、citation UI 都必须使用同一个 immutable `evidence_pack_id`。Verifier 不得重新渲染另一套截断证据。

### 🔎 三路候选与融合

1. **Lexical**：BM25 处理 API 名、错误码、版本和人名。
2. **Dense**：处理同义表达和语义相似。
3. **Graph（实验）**：只在 entity/multi-hop/inference 路由命中时，通过 PPR 或受限邻居扩展补充候选。

三路先产生 doc/claim id，再用 RRF/DBSF 或校准分数融合，最后统一 cross-encoder rerank。Graph 结果不能直接越过 reranker 和来源验证进入 prompt。

### 🗜️ 上下文缩减顺序

严格按以下顺序，不能一上来就让 LLM 摘要：

1. chunk/parent 去重；
2. source diversity 与 sub-question coverage；
3. cross-encoder 排序；
4. 选原文 supporting sentence/span；
5. 计算 token budget，超额时剔除低边际价值证据；
6. 只有仍超预算时才 A/B LLMLingua；压缩结果保留原文 offset，不作为独立来源。

建议先测试 `fast=1.2k / balanced=2.4k / deep=4k` 三档 evidence token budget；这些是**实验网格起点**，不是声称存在通用最优值。

### 🧪 Graph Retrieval Pilot

统一接口：

```python
class GraphRetriever(Protocol):
    def retrieve(self, query, seed_entities, snapshot_id, limit) -> list[Candidate]: ...
```

实现三个对照组：

- `none`：当前 dense+sparse 强基线；
- `ppr_pilot`：参考 HippoRAG 的 entity seed + Personalized PageRank；
- `lightrag_adapter` 或 `graphiti_adapter`：只选一个进入第二轮，不同时堆两套系统。

只在 MultiHop-RAG inference/temporal 子集和真实项目多跳题上比较：all-support recall、候选 precision、p95 latency、index cost、update correctness。没有统计增益就不进入默认链路。

### 📋 实施步骤

1. 把 `Retriever.retrieve(k=5)` 拆成 `retrieve_candidates(top_n)` 与 `build_evidence_pack(token_budget)`。
2. 候选阶段返回 child text、parent id、source id，而不是立即加载所有父块全文。
3. 引入本地 `CrossEncoderReranker`，批量评分并缓存 `(query_hash, chunk_hash, reranker_version)`。
4. 实现 supporting span 抽取与 offset 校验；citation 点击仍展示原始父块上下文。
5. 对 chunk size / candidate N / rerank M / evidence budget 做网格消融。
6. 在 `eval/retrieval_eval.py` 增加 nDCG@k、MRR、source diversity、evidence token、strict all-support recall。
7. 最后才接 GraphRetriever pilot。

### ✅ 目标门槛（待实测）

- inference all-support recall 必须相对当前 0.1078 有统计显著提升；不预写虚假目标结果。
- 进入 Answer 的 evidence token 至少比“完整父块 top-5”下降 30%，claim recall 不下降。
- citation quote 对原文 offset 校验通过率 100%。
- Graph pilot 只有在质量增益覆盖其索引/延迟成本时晋级；否则保留为研究分支。

---

## 阶段十：并发、缓存与 Serving —— 把剩余等待从系统层消掉

> **当前问题**：当前图节点主要使用同步 `.invoke`；Cache 只在单次 run 内按 `(query, kb)` 记忆；47,593 child vectors 已超过当前报告中 Qdrant embedded mode 的建议规模；相同知识快照上的重复问题无法跨 run 复用。  
> **成熟架构如何解决**：Haystack AsyncPipeline 并发运行独立分支；RedisVL 提供带 TTL/阈值/filter 的语义缓存；vLLM/SGLang 用 prefix cache、continuous batching 和 speculative decoding 优化自托管推理。  
> **Linki 需要增加什么**：async 热路径、single-flight、四层版本化缓存、Qdrant server profile、可选 LLM Gateway/自托管后端。  
> **最终如何演示**：并发请求命中同一个 in-flight 任务；第二次相同问题在同一知识快照上直接命中；更新文档后缓存自动失效；UI 首 token 提前出现。

### 🏗️ 缓存层次

| 层 | 缓存内容 | Key 必须包含 | 主要风险 |
|---|---|---|---|
| L0 Single-flight | 同时发生的相同计算 Future | tenant + normalized input + operation version | 并发风暴 |
| L1 Embed/Retrieve/Rerank | query embedding、candidate ids、rerank score | tenant/ACL + kb snapshot + model/index version | 过期索引 |
| L2 Exact Answer | Evidence Pack + answer + citations | tenant/user scope + question + snapshot + policy/prompt/model version | 权限泄漏、旧答案 |
| L3 Semantic Answer | 相似问题回答 | L2 全部字段 + semantic threshold + route class | 错误复用、语义近但约束不同 |
| L4 Provider/KV Prefix | system prompt、稳定 schema、共享前缀 KV | provider/model/cache namespace | 前缀不稳定、跨租户侧信道 |

`cache_key` 示例：

```text
sha256(
  tenant_id | user_scope | acl_hash | normalized_query |
  kb_snapshot_id | memory_snapshot_id | policy_version |
  prompt_version | model_id | evidence_pack_version
)
```

语义缓存额外规则：

- 默认只对稳定 KB 问答开放；personal memory、时间敏感问题、高风险问题关闭。
- 命中后仍校验 source versions 存在且用户仍有权限。
- threshold 在专门的 false-hit 集上校准；Redis 默认值不是 Linki 的答案。
- 文档更新不做“遍历删除全部 key”，而是发布新 snapshot id，让旧 key 自然不可达并按 TTL 回收。

### ⚡ Async 与流式响应

1. Provider 全部提供 `ainvoke/astream`；同步 adapter 只能在受控线程池运行。
2. Planner 的独立子查询、dense/sparse、memory/knowledge channel 并发，设置 bounded semaphore。
3. 使用 deadline propagation：上游剩余时间传给每个分支，超时分支返回 partial result + reason。
4. 最慢分支不是无限等待条件；已满足 coverage 时取消低价值分支。
5. Answer 流式输出以降低 TTFT，但 citation mapping 必须在输出前固定；不能边流式边更换 Evidence Pack。

### 🖥️ Serving 分层决策

1. **当前云 API 路径**：先完成调用削减、LiteLLM-compatible gateway interface、streaming 与 provider prompt cache 观测。
2. **自托管候选路径**：同模型、同硬件、同 workload 下比较 vLLM 与 SGLang；测试 prefix hit、TTFT、tokens/s、p95、显存与并发。
3. vLLM 的 prefix caching 只减少共享前缀的 prefill，不会减少生成 token；speculative decoding 也必须按低/中 QPS workload 实测，不能写成必然加速。
4. 只有当月请求量、GPU 利用率和运维成本形成正收益时才切换默认后端。

### 📋 实施步骤

1. 改造 `answer_question` 为 async 主入口，CLI 用 `asyncio.run` 包装，UI 直接消费 stream。
2. `HookContext` 中加入 async single-flight registry，避免重复 embedding/retrieval。
3. 新建 `cache/`，先实现 SQLite/dev + Redis/prod adapter；业务代码只依赖协议。
4. 引入 `SnapshotManifest`，每次 ingest 生成 immutable `kb_snapshot_id`。
5. 把 benchmark 的 47,593 vectors 迁到 Qdrant server profile，比较 embedded/server 的稳定性和 p95。
6. 压测 `1/4/16/64` 并发，分别报告冷/热 cache。
7. 最后建立 vLLM/SGLang benchmark，不提前改变默认 provider。

### ✅ 目标门槛（待实测）

- cache correctness 测试覆盖 snapshot、ACL、tenant、prompt/model version；跨租户误命中必须为 0。
- 热 exact-cache p95 不经过 LLM；semantic cache false-hit rate 低于在评测集上预先批准的阈值。
- 并发 16 时无重复相同 in-flight 调用，错误/取消不会污染缓存。
- streaming TTFT、E2E p95 和吞吐均单独报告；不能用 TTFT 改善掩盖总耗时退化。

---

## 阶段十一：Memory 候选管线 —— 会记，但不把每句话都当真

> **当前问题**：`core/session.py` 主要保存对话历史；它能支持多轮，却没有跨会话的结构化长期记忆。若直接把每轮总结写回 Qdrant，会把助手回答与用户事实混淆。  
> **成熟架构如何解决**：LangMem 区分 semantic/episodic/procedural memory，并明确 hot-path 与 background formation 的取舍；Letta 用 read-only block 划权限；Mem0 当前 add 流程先抽取再 additive storage。  
> **Linki 需要增加什么**：Memory namespace、不可变 episode、候选状态机、后台 extraction/consolidation、显式确认与删除。  
> **最终如何演示**：用户说“以后代码示例优先 Python”后，该偏好在后台成为 user-scoped memory；换租户不可见；用户改口后旧版本被 supersede 而不是静默覆盖。

### 🧠 Memory 状态机

```text
OBSERVED
   │ extract
   ▼
PROPOSED ──duplicate──▶ MERGED
   │
   ├─low confidence / sensitive──▶ REVIEW_REQUIRED
   ├─contradiction───────────────▶ CONFLICT
   ├─policy reject───────────────▶ REJECTED
   └─validated───────────────────▶ ACTIVE
                                      │
                         newer fact ──┴──▶ SUPERSEDED
                         ttl/decay ──────▶ EXPIRED
                         user delete ───▶ DELETED/TOMBSTONED
```

### 🧱 MemoryItem Schema

```python
MemoryItem = {
    "memory_id": str,
    "type": "profile" | "semantic" | "episodic" | "procedural",
    "scope": "user" | "team" | "organization" | "agent",
    "namespace": tuple[str, ...],
    "content": dict,
    "status": str,
    "source_episode_ids": list[str],
    "source_spans": list[dict],
    "event_time": datetime | None,
    "ingested_at": datetime,
    "valid_from": datetime | None,
    "valid_to": datetime | None,
    "importance": float,
    "confidence": float,
    "last_accessed_at": datetime | None,
    "access_count": int,
    "pii_class": str | None,
    "version": int,
    "supersedes": str | None,
}
```

`confidence` 只是抽取器置信度，**不是事实真值**。Promotion 仍取决于来源类型、用户确认、冲突与 policy。

### 🔐 不同 Memory 的晋级规则

| 类型 | 例子 | 自动晋级策略 |
|---|---|---|
| 显式用户偏好 | “以后回答简洁一些” | 可自动 ACTIVE；保留原句，可随时撤销 |
| 推断偏好 | 用户多次要求 Python 示例 | 先 PROPOSED；达到重复/反馈阈值或询问确认 |
| 用户事实 | “我在巴黎办公” | 需要明确陈述；含敏感信息时进入 REVIEW/禁存 |
| Episode | 某次任务失败、采取了什么修复 | 后台生成结构化摘要；设 retention/decay |
| 团队事实 | “生产使用 PostgreSQL 16” | 必须关联文档/会议证据或人工批准 |
| Procedural | “遇到 X 永远跳过 verifier” | 永不直接晋级；进入阶段十三离线优化 |

### 🌙 Hot Path 与 Background

默认后台流程：

```text
回答已返回
  → enqueue episode
  → extract candidates
  → search near-duplicates/current beliefs
  → consolidate / detect conflict
  → policy validate
  → promote or review
  → refresh memory index
```

Hot path 只保留两类动作：

1. 用户明确使用“记住/忘记/修改我的偏好”；
2. 当前回答必须立即使用的新约束，但写入仍先生成可审计 event。

### 🔎 Memory Recall

Recall 先做 namespace/ACL/time/status 过滤，再做 BM25+dense，最后按以下因素 rerank：

```text
score = relevance
      + task_match
      + importance
      + recency_if_relevant
      + access_strength
      - staleness
      - contradiction_penalty
```

不要把所有 memory 拼进 system prompt。只有极少量高频 profile 可常驻；其余按 query 检索并纳入独立的 `memory_token_budget`。

### 📋 实施步骤

1. 用 LangGraph `BaseStore`/LangMem 做最小接口验证，但将业务 schema 保持框架无关。
2. 新建 `memory/ledger.py`、`extractor.py`、`policy.py`、`consolidator.py`、`retriever.py`。
3. 先实现 user-scoped profile + episodic 两类，不一开始做 procedural auto-learning。
4. Worker 采用 idempotency key：`source_episode_hash + extractor_version`。
5. CLI/UI 提供“为什么记住”“来源”“编辑”“忘记”和 namespace 导出。
6. 删除走 tombstone + 索引清理；涉及法定删除时清理 raw payload、vector、cache 和 projection，并保留非敏感审计证明。
7. 增加 LongMemEval 的 information extraction / knowledge update / temporal / abstention 子集，以及真实项目偏好与纠错集。

### ✅ 验收门槛

- 无来源 Episode 的 MemoryItem 不得进入 ACTIVE。
- 团队/组织知识自动晋级率为 0，除非满足明确配置的可信来源规则。
- memory write precision、update correctness、stale-memory rate、deletion completeness 分开报告。
- Memory Recall 提升不能以显著增加每题 prompt token 为代价；报告 `useful_memory_tokens / injected_memory_tokens`。

---

## 阶段十二：Knowledge 自沉淀 —— 从原始证据到 Wiki 与 Temporal Graph

> **当前问题**：当前知识主要是 parent/child chunks 与向量 payload，适合搜索原文，却不具备“这个结论何时生效、被什么替代、对应哪些实体、用户如何浏览”的结构。  
> **成熟架构如何解决**：Graphiti 用 Episode、来源和事实有效期处理动态知识；KAG 做 chunk↔knowledge mutual indexing；GraphRAG 用社区层级支持全局概览；RAPTOR 用树状摘要提供不同抽象层。  
> **Linki 需要增加什么**：Source/Episode/Claim Ledger、实体对齐、双时间事实、Wiki 投影、Graph 投影、snapshot promotion。  
> **最终如何演示**：一次会议结论进入 candidate，经审批后生成 Wiki 页面更新与图关系；新决议到来时旧 Claim 变为 superseded；用户仍能回答“上个月当时的方案是什么”。

### 🧱 核心数据模型

#### SourceArtifact

原始文件、URL、会议记录、用户纠错或系统事件。保存 content hash、权限、版本、解析器版本和原始定位；不可被模型覆写。

#### Episode

一次知识事件，区分：

- `event_time`：现实中何时发生；
- `ingested_at`：系统何时知道；
- `source_id/source_span`：从哪里来；
- `tenant/scope/ACL`：谁能看。

#### ClaimVersion

```python
ClaimVersion = {
    "claim_id": str,
    "subject_entity_id": str,
    "predicate": str,
    "object_entity_id": str | None,
    "literal_value": object | None,
    "qualifiers": dict,
    "status": "candidate" | "active" | "superseded" | "rejected",
    "valid_from": datetime | None,
    "valid_to": datetime | None,
    "recorded_at": datetime,
    "source_spans": list[dict],
    "confidence": float,
    "supersedes": str | None,
    "schema_version": str,
}
```

### 📚 Wiki 结构：给人读的知识投影

```text
Knowledge Space
└── Domain（如 Deployment）
    └── Topic（如 Rate Limiting）
        └── Wiki Page
            ├── TL;DR（由 active claims 生成）
            ├── Current Decision
            ├── How It Works
            ├── Constraints / Exceptions
            ├── Change History
            ├── Open Questions
            ├── Sources（精确到 source span）
            └── Backlinks / Related Entities
```

每页必须带：

- `page_id / slug / version / snapshot_id`；
- `generated | reviewed | approved | stale` 状态；
- 每个事实段的 claim ids 与 source citations；
- `last_verified_at` 与 stale reason；
- 人工编辑也作为新 Episode 进入 Ledger，不能直接绕开版本链。

Wiki TL;DR 与 GraphRAG community summary 只能服务导航/overview。事实回答仍优先返回原始 Claim 和 SourceArtifact，避免“模型摘要引用模型摘要”的证据漂移。

### 🕸️ Graph 结构：给机器走关系的投影

建议的最小节点：

- `Entity`：人、系统、服务、API、概念、决策；
- `Claim`：可验证的事实版本；
- `Episode`：原始发生/摄入事件；
- `Source` / `Chunk`：原文与定位；
- `WikiPage` / `Section`：人类可读投影；
- `Community`：离线聚类后的主题群，可选。

建议的最小边：

```text
Entity ─RELATES_TO/DEPENDS_ON/PART_OF─▶ Entity
Claim  ─SUBJECT/OBJECT────────────────▶ Entity
Claim  ─SUPPORTED_BY──────────────────▶ SourceSpan
Claim  ─OBSERVED_IN───────────────────▶ Episode
Claim  ─SUPERSEDES/CONTRADICTS────────▶ Claim
WikiPage ─SUMMARIZES──────────────────▶ Claim
Chunk  ─MENTIONS──────────────────────▶ Entity
Entity ─ALIAS_OF──────────────────────▶ Entity
```

关系边必须带 `valid_from / valid_to / recorded_at / source`。没有 provenance 的模型推断边只能是 candidate，不能参与默认事实回答。

### 🔄 沉淀流水线

```text
1. Ingest SourceArtifact（hash 去重、ACL、版本）
2. Parse → chunks/source spans
3. Extract entity/claim candidates（结构化输出）
4. Normalize（类型、单位、时间、别名）
5. Entity resolution（exact alias → embedding candidate → rule/human resolve）
6. Detect duplicate / conflict / supersession
7. Validate（source policy、schema、置信、人工审批）
8. Write immutable ClaimVersion
9. Build search/wiki/graph projections to staging snapshot
10. Run canary retrieval + citation integrity + regression eval
11. Atomically switch active snapshot alias
12. Failure → rollback old snapshot；新 ledger 记录仍保留供审计
```

### 🔎 查询路由

| 问题类型 | 优先通道 | 原因 |
|---|---|---|
| 精确 API/错误码 | BM25 + dense + rerank | 图没有必要 |
| 单实体当前事实 | Claim/entity local search | 可直接命中 active claim 与来源 |
| “A 与 B 有什么关系” | graph local neighborhood + source rerank | 关系约束明确 |
| 多跳 inference | graph PPR/path + hybrid text | 补足跨文档链 |
| “整体有哪些主题/趋势” | community/Wiki summary | 适合 GraphRAG 式全局层 |
| 历史状态 | temporal claim filter | 必须按 valid time 查询 |

所有通道最终都输出统一 Candidate/EvidenceUnit；Graph/Wiki 不建立第二套回答协议。

### 🗄️ 存储策略

第一步不强制引入完整图数据库：

1. Ledger/claims/wiki metadata：dev 可 SQLite，production adapter 目标 PostgreSQL。
2. Text/vector：继续 Qdrant server。
3. Graph：先实现 `GraphStore` 协议与小规模 pilot；若 Graphiti 路径胜出再选择 Neo4j/FalkorDB 等支持后端。
4. 每一层都写 snapshot manifest；任何索引都可从 Ledger 重建。

### ✅ 验收门槛

- Active Claim 的 source provenance coverage = 100%。
- 同一 SourceArtifact 重复 ingest 幂等；worker 至少一次投递不会产生重复 active claim。
- 新事实覆盖旧事实时，当前查询与历史时间查询都返回正确版本。
- Wiki 每个事实段可回链 Claim 和 source span；stale 页面不会进入默认回答。
- staging snapshot 未通过 retrieval/citation 回归时，不能切 active alias。

---

## 阶段十三：受控自进化 —— 让系统改进，但不允许它悄悄改坏

> **当前问题**：Trace、用户反馈和 verifier failure 已经产生大量“系统哪里不够好”的信号，但目前不会转成稳定的知识或配置改进；另一方面，直接在线改 prompt/graph 会导致不可解释漂移。  
> **成熟架构如何解决**：LangMem 支持后台 consolidation 与 prompt optimization primitives；DSPy 用明确 metric 在 train/dev 上搜索 instruction/few-shot 组合；成熟发布流程使用 shadow/canary/rollback。  
> **Linki 需要增加什么**：Feedback Ledger、Gap Miner、change proposal、离线 optimizer、版本注册表、灰度和自动回滚。  
> **最终如何演示**：一组反复未回答的问题自动形成“知识缺口清单”，但不生成虚假答案；补充文档后 candidate 通过评测成为新 snapshot；如果线上指标回退，一键切回旧版本。

### 🧬 自进化不是一个循环，而是七道门

```text
Observation
   → Diagnosis
      → Change Proposal
         → Offline Train/Dev Evaluation
            → Shadow Replay
               → Canary
                  → Promote or Rollback
```

任何一步都只能产出下一步的 candidate，不能越级修改 production active state。

### 📥 可用反馈信号

| 信号 | 可以推导什么 | 不能直接推导什么 |
|---|---|---|
| 用户明确纠错 + 给出来源 | Claim correction candidate | 不能直接覆盖组织事实 |
| 多次 `not found` | Knowledge gap cluster | 不能自动生成缺失知识 |
| Verifier issue | retrieval/prompt failure sample | 不能证明 verifier 自己必然正确 |
| citation 点击/展开 | 证据有用性弱信号 | 不能单独证明答案正确 |
| thumbs up/down | 端到端偏好信号 | 不能定位是哪一节点出错 |
| 成功 trace | planner/retrieval 示例候选 | 不存模型私有推理，只存结构化动作与结果 |
| cache hit/miss | 重复 workload 与缓存策略 | 不能牺牲版本/ACL 正确性追 hit rate |

### 🕳️ Gap Miner

后台按以下字段聚类失败：normalized query、entities、target KB、missing support、retrieved source ids、policy path。输出：

```python
KnowledgeGap = {
    "gap_id": str,
    "cluster_label": str,
    "example_questions": list[str],
    "frequency": int,
    "affected_users": int,
    "business_impact": str,
    "missing_entities_or_claims": list[str],
    "nearest_sources": list[str],
    "suggested_source_to_add": str,
    "status": "open" | "documented" | "wont_fix",
}
```

Gap Miner 只创建 Wiki backlog/知识运营任务，禁止自动写答案。

### 🧪 可自动优化的对象与权限

| 对象 | 自动生成候选 | 自动上线 | 必要门禁 |
|---|:---:|:---:|---|
| Router threshold | ✅ | 小流量可 | 路由 confusion matrix + 成本/质量 |
| Retrieval N/M/budget | ✅ | 小流量可 | recall/precision/token/p95 |
| Prompt instruction/few-shot | ✅（DSPy/搜索） | ❌ | 独立 test set + 人工 review |
| Memory consolidation rule | ✅ | ❌ | write precision/update/deletion tests |
| Wiki summary | ✅ | 仅 generated 状态 | citation coverage + stale check |
| Claim/Graph edge | ✅ candidate | ❌ | provenance/schema/conflict/review |
| Model weights | 暂不做 | ❌ | 数据规模、许可、安全均未成熟 |

### 📊 Release Gate

每个 candidate release 都要同时满足五类指标：

1. **质量**：all-support recall、claim recall/precision、faithfulness、citation entailment、refusal。
2. **效率**：p50/p95 TTFT、E2E latency、LLM calls、input/output tokens、cost。
3. **Memory**：write precision、memory recall、knowledge update accuracy、stale/contradiction rate、delete completeness。
4. **Knowledge**：entity resolution accuracy、provenance coverage、index freshness lag、snapshot rollback success。
5. **稳定性**：多次运行 top-k Jaccard、answer claim Jaccard、错误率方差、超时/降级率。

优化目标使用约束式 Pareto，而不是把所有指标揉成一个可被投机的总分：

```text
minimize(tokens, p95_latency, cost)
subject to:
  all_support_recall >= approved_baseline_floor
  faithfulness       >= approved_baseline_floor
  citation_integrity == 100%
  privacy_leakage    == 0
  rollback_test      == pass
```

### 📋 实施步骤

1. 建立 `feedback/ledger.py`，反馈也版本化、带来源与权限。
2. 实现 Gap Miner，只输出 backlog；先由人补文档。
3. 建立 `PromptRegistry / PolicyRegistry / IndexRegistry`，每个版本绑定 train/dev/test 结果。
4. 用 DSPy MIPROv2/BootstrapFewShot 只在离线 train/dev 产生 prompt candidate；锁定 test set 防过拟合。
5. Shadow replay 最近真实 trace，比较 candidate 与 active，但不影响用户。
6. Canary 按 tenant/user hash 固定分桶，防同一用户来回切版本。
7. 设自动回滚条件：质量 floor、p95、错误率、memory conflict 任一越界即撤回。
8. 每次 promote 生成 model card 式变更说明：改了什么、为何、数据范围、已知限制、回滚版本。

### ✅ 最终验收

- 能完整演示“失败信号 → gap → 新来源 → candidate claims → staging snapshot → eval → promote → rollback”。
- 任何 production answer 都能还原：policy/prompt/model/index/memory snapshot/evidence pack 版本。
- 没有通过 test + shadow 的 prompt、route threshold、graph extractor 不得上线。
- “系统不知道”会增加 backlog，不会自动变成“系统编了一个知识点”。

---

## 第三部分：召回率与稳定性，应该怎样严肃地评测

### 3.1 不能只看 Recall@5

| 层 | 主指标 | 为什么 |
|---|---|---|
| Candidate retrieval | Recall@N、MRR、nDCG、source diversity | 判断召回器是否把正确来源带入候选 |
| Evidence Pack | strict all-support recall、context precision、support coverage、evidence tokens | 判断压缩后是否保住完整证据链 |
| Answer | claim recall/precision、faithfulness、citation entailment、answer correctness | citation index 合法不代表该引用支撑论断 |
| Refusal | answerable recall、unanswerable refusal precision/recall | 防止系统靠“多拒答”刷 faithfulness |
| Memory | write precision、retrieval recall、update/temporal accuracy、abstention | 评估记得对不对、旧知识是否失效 |
| Stability | top-k/claim Jaccard、p95 variance、重复运行一致率 | 平均分高但结果漂移也不可用 |

RAG 诊断可参考 [RAGChecker](https://github.com/amazon-science/RAGChecker) 的 claim-level entailment；通用 context precision/recall/faithfulness 可参考 [Ragas 指标](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)。Memory 评测使用 [LongMemEval](https://github.com/xiaowu0162/LongMemEval) 的 extraction、multi-session、update、temporal、abstention 五类能力，并用 [LoCoMo](https://github.com/snap-research/locomo) 作长对话补充；不能只引用某个项目自己公布的单一总分。

### 3.2 测试集分层

```text
A. 现有公开集
   - MultiHop-RAG 全量 retrieval
   - MultiHop-RAG 分层 E2E
   - BEIR 选取与目标语言/领域相近的 retrieval 子集
   - LongMemEval / LoCoMo memory 子集

B. Linki 项目集
   - single fact
   - comparison / cross-kb
   - inference / temporal
   - unanswerable
   - multi-turn anaphora
   - user preference update/delete
   - organization claim conflict/supersede

C. 对抗与稳定集
   - query paraphrase
   - prompt injection in source
   - stale cache / ACL change
   - duplicate ingest / out-of-order event
   - graph entity alias collision
   - provider timeout / partial branch
```

### 3.3 稳定性协议

1. 固定 snapshot/prompt/model 的情况下，每题重复运行至少 3 次；检索本身尽量确定性。
2. 报告 top-k doc/claim Jaccard 与最终 answer claims Jaccard，不用字符串完全相同作唯一标准。
3. 索引增量更新前后跑 unaffected-query set，确认无关问题没有大面积排名漂移。
4. 对每个 snapshot 做可重建测试：从 Ledger 重建后，active claims、Wiki citations、graph provenance 数量一致。
5. 模型或 embedding 版本切换一律新建索引/缓存 namespace，不在原 collection 上静默混写。

---

## 第四部分：代码布局与最小增量

```text
src/linki/
├── routing/
│   ├── policy.py              # P0-P3 PolicyDecision
│   ├── complexity.py          # 规则/本地语义分类
│   └── risk.py                # coverage/conflict/citation risk gate
├── retrieval/
│   ├── candidates.py          # 高召回候选
│   ├── rerank.py              # local cross-encoder
│   ├── spans.py               # supporting span + offset
│   ├── evidence_pack.py       # token budget / diversity / coverage
│   └── graph.py               # GraphRetriever protocol + pilots
├── cache/
│   ├── base.py                # exact/semantic/single-flight protocol
│   ├── sqlite.py              # dev
│   └── redis.py               # production adapter
├── memory/
│   ├── ledger.py              # episode/memory immutable events
│   ├── extractor.py           # background candidate extraction
│   ├── consolidator.py        # dedup/conflict/supersede
│   ├── policy.py              # promotion/PII/scope rules
│   └── retriever.py
├── knowledge/
│   ├── sources.py             # SourceArtifact/source spans
│   ├── claims.py              # ClaimVersion
│   ├── entities.py            # normalization/entity resolution
│   ├── wiki.py                # Wiki materialized projection
│   ├── graph.py               # Graph projection/store protocol
│   └── snapshots.py           # stage/promote/rollback
├── evolution/
│   ├── feedback.py
│   ├── gap_miner.py
│   ├── optimizer.py           # offline candidate only
│   └── release.py             # shadow/canary/rollback
└── eval/
    ├── cost_eval.py
    ├── memory_eval.py
    ├── stability_eval.py
    └── datasets/
```

现有模块不需要推翻：

- `graph/` 继续负责编排，但读取 `PolicyDecision` 选择子图；
- `tools/retrieve.py` 下沉为 candidate backend；
- `graph/evidence.py` 演进为 Evidence Pack 引用层；
- `hooks/` 接入跨 run cache 与 token/latency telemetry；
- `core/session.py` 继续做 thread state，不兼任长期 Memory；
- `eval/` 扩展现有 MultiHop-RAG adapter，而不是另建不可比较的评测系统。

---

## 第五部分：每阶段演示与 Stop/Go 决策

| 阶段 | 推荐演示 | Go 条件 | Stop/回退条件 |
|---|---|---|---|
| 七 | NodeCost 瀑布图 + 分题型 Pareto | 调用、token、TTFT 全可归因 | telemetry 缺失或样本不可复现 |
| 八 | 同一 CLI 连问 single / compare / low-confidence | 短路显著降本且质量 floor 保持 | router 漏掉复杂题/高风险题 |
| 九 | inference 题候选→rerank→pack | all-support 提升、context token 下降 | graph/rerank 只加延迟无增益 |
| 十 | 冷/热 cache + 16 并发 + 文档更新失效 | 无跨租户误命中、p95 改善 | cache 错答或 snapshot 不一致 |
| 十一 | 记住偏好→改口→忘记 | version/supersede/delete 全可审计 | memory 污染组织知识或注入过量 |
| 十二 | 新会议决议更新 Wiki/Graph，查询过去/现在 | provenance 100%、可 rollback | 无来源 active claim、历史被覆盖 |
| 十三 | gap→补文档→shadow→canary→回滚 | candidate 经门禁后稳定改进 | 自动变更绕过 test/review |

---

## 总结：Linki 的第二次进化路径

```text
Stage 7       Stage 8       Stage 9       Stage 10
先量化        按需思考       高召回低上下文   并发缓存Serving
   │             │             │             │
   └──────▶ “更少调用、更少证据、更快首 token” ◀──────┘

Stage 11      Stage 12      Stage 13
Memory候选     Wiki+时序图     受控自进化
   │             │             │
   └──────▶ “有来源地沉淀、按版本地演进、可回滚” ◀────┘
```

最终要证明的不是“Linki 加了多少个流行组件”，而是四件可量化的事：

1. **同等可靠性下，简单题不再支付完整 Agentic 成本。**
2. **同等 token budget 下，多跳问题能召回更完整的证据链。**
3. **系统能从对话与反馈中形成候选 Memory/Knowledge，但不会把模型输出直接当事实。**
4. **每一次知识、路由、prompt 或索引变化都能被解释、评测、灰度和回滚。**

一句话概括新的产品定位：

> **Linki 不只是“会思考的知识助手”，而是一个按风险分配推理预算、以证据账本为核心、能够安全沉淀和受控进化的知识系统。**

---

## 参考资料（均为官方仓库、官方文档或论文）

### 效率与检索

- [vLLM 官方仓库](https://github.com/vllm-project/vllm) · [Automatic Prefix Caching](https://docs.vllm.ai/en/stable/design/prefix_caching/) · [Speculative Decoding](https://docs.vllm.ai/en/stable/features/speculative_decoding/)
- [SGLang 官方仓库](https://github.com/sgl-project/sglang)
- [LiteLLM 官方文档](https://docs.litellm.ai/)
- [Haystack AsyncPipeline](https://docs.haystack.deepset.ai/docs/asyncpipeline) · [ConditionalRouter](https://docs.haystack.deepset.ai/docs/conditionalrouter)
- [Adaptive-RAG 官方实现](https://github.com/AsH1605/Adaptive-RAG)
- [Semantic Router 官方仓库](https://github.com/aurelio-labs/semantic-router)
- [RouteLLM 官方仓库](https://github.com/lm-sys/RouteLLM)
- [RedisVL Semantic Cache](https://redis.io/docs/latest/develop/ai/redisvl/api/cache/)
- [Sentence Transformers: Retrieve & Re-rank](https://www.sbert.net/examples/sentence_transformer/applications/retrieve_rerank/README.html)
- [Qdrant Hybrid and Multi-stage Queries](https://qdrant.tech/documentation/search/hybrid-queries/)
- [Microsoft LLMLingua](https://github.com/microsoft/LLMLingua)

### Memory、Knowledge、Wiki 与 Graph

- [LangGraph Memory](https://docs.langchain.com/oss/python/langgraph/add-memory) · [LangMem Core Concepts](https://langchain-ai.github.io/langmem/concepts/conceptual_guide/)
- [Mem0 官方仓库](https://github.com/mem0ai/mem0) · [Mem0 Add Memory 流程](https://docs.mem0.ai/core-concepts/memory-operations/add)
- [Letta Memory Blocks](https://docs.letta.com/guides/core-concepts/memory/memory-blocks)
- [Graphiti 官方仓库](https://github.com/getzep/graphiti) · [Adding Episodes](https://help.getzep.com/graphiti/core-concepts/adding-episodes)
- [Microsoft GraphRAG Index](https://microsoft.github.io/graphrag/index/overview/) · [Query Engine](https://microsoft.github.io/graphrag/query/overview/)
- [LightRAG 官方仓库](https://github.com/HKUDS/LightRAG)
- [Cognee 官方仓库](https://github.com/topoteretes/cognee)
- [OpenSPG KAG 官方仓库](https://github.com/OpenSPG/KAG)
- [HippoRAG 2 官方仓库](https://github.com/OSU-NLP-Group/HippoRAG)
- [RAPTOR 官方仓库](https://github.com/parthsarthi03/raptor)

### 评测与离线优化

- [RAGChecker](https://github.com/amazon-science/RAGChecker)
- [Ragas Metrics](https://docs.ragas.io/en/stable/concepts/metrics/available_metrics/)
- [BEIR](https://github.com/beir-cellar/beir)
- [LongMemEval](https://github.com/xiaowu0162/LongMemEval)
- [LoCoMo](https://github.com/snap-research/locomo)
- [DSPy MIPROv2](https://github.com/stanfordnlp/dspy/blob/main/docs/docs/api/optimizers/MIPROv2.md)
