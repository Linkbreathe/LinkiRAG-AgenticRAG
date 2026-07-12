import * as Select from "@radix-ui/react-select";
import * as Tabs from "@radix-ui/react-tabs";
import * as Tooltip from "@radix-ui/react-tooltip";
import {
  AlertCircle,
  AlertTriangle,
  ArrowRight,
  Bot,
  BookOpen,
  Brain,
  Check,
  CheckCircle2,
  ChevronDown,
  Clock3,
  Compass,
  Cpu,
  Database,
  FileUp,
  GitBranch,
  Info,
  Library,
  Loader2,
  MessageSquare,
  Plus,
  RefreshCw,
  SendHorizontal,
  Settings2,
  ShieldAlert,
  ThumbsDown,
  ThumbsUp,
  Trash2,
  Upload,
  X
} from "lucide-react";
import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import {
  clearTopic,
  createTopic,
  forgetMemory,
  getMemory,
  getTopics,
  getWiki,
  sendChat,
  sendFeedback,
  uploadDocuments
} from "./api";
import type { ChatMessage, ChatResponse, ExecutionMode, MemoryItem, Topic, TraceStep, WikiPage } from "./types";

type NoticeTone = "success" | "error" | "info";
type NoticeState = { message: string; tone: NoticeTone } | null;
type PendingAction = "refresh" | "create" | "upload" | "clear" | "chat" | "memory" | null;

function errorMessage(error: unknown) {
  return error instanceof Error ? error.message : String(error);
}

function formatCount(value: number, singular: string, plural = `${singular}s`) {
  return `${value} ${value === 1 ? singular : plural}`;
}

function stepTone(step: TraceStep) {
  if (step.kind === "verify" && step.status === "failed") return "danger";
  if (step.kind === "grade" && step.status === "insufficient") return "warning";
  if (step.kind === "retrieve" && step.status === "gaps") return "warning";
  if (step.kind === "verify" && step.status === "passed") return "success";
  return "neutral";
}

function formatTarget(target?: string) {
  return (target || "selected topic").replace(/^Retrieve_/, "");
}

function runSummary(result: ChatResponse | null) {
  return {
    policy: result?.policy_path?.toUpperCase() ?? "—",
    calls: result?.cost?.llm_calls ?? 0,
    tokens: result?.cost?.total_tokens ?? 0,
    latency: result?.cost?.latency_ms ?? 0,
    citations: result?.citations.length ?? 0
  };
}

function SelectBox({
  value,
  onChange,
  topics,
  label,
  hint
}: {
  value: string;
  onChange: (value: string) => void;
  topics: Topic[];
  label: string;
  hint?: string;
}) {
  if (!topics.length) {
    return (
      <label className="field">
        <span>{label}</span>
        <div className="select-trigger disabled">No topics available</div>
        {hint ? <small>{hint}</small> : null}
      </label>
    );
  }

  return (
    <label className="field">
      <span>{label}</span>
      <Select.Root value={value} onValueChange={onChange}>
        <Select.Trigger className="select-trigger" aria-label={label}>
          <Select.Value placeholder="Select a topic" />
          <Select.Icon>
            <ChevronDown size={15} />
          </Select.Icon>
        </Select.Trigger>
        <Select.Portal>
          <Select.Content className="select-content" position="popper" sideOffset={6}>
            <Select.Viewport>
              {topics.map((topic) => (
                <Select.Item className="select-item" value={topic.name} key={topic.name}>
                  <Select.ItemText>
                    {topic.title} - {topic.collection}
                  </Select.ItemText>
                </Select.Item>
              ))}
            </Select.Viewport>
          </Select.Content>
        </Select.Portal>
      </Select.Root>
      {hint ? <small>{hint}</small> : null}
    </label>
  );
}

const executionModes: { value: ExecutionMode; label: string; hint: string }[] = [
  { value: "auto", label: "Auto", hint: "Route locally by complexity and risk" },
  { value: "fast", label: "Fast", hint: "Prefer the one-call factual path" },
  { value: "balanced", label: "Balanced", hint: "Always include a query plan" },
  { value: "deep", label: "Deep", hint: "Grade evidence and verify the answer" }
];

function ModeSelect({ value, onChange }: { value: ExecutionMode; onChange: (value: ExecutionMode) => void }) {
  const selected = executionModes.find((item) => item.value === value);
  return (
    <label className="field">
      <span>Execution mode</span>
      <Select.Root value={value} onValueChange={(next) => onChange(next as ExecutionMode)}>
        <Select.Trigger className="select-trigger" aria-label="Execution mode">
          <Select.Value />
          <Select.Icon><ChevronDown size={15} /></Select.Icon>
        </Select.Trigger>
        <Select.Portal>
          <Select.Content className="select-content mode-content" position="popper" sideOffset={6}>
            <Select.Viewport>
              {executionModes.map((item) => (
                <Select.Item className="select-item mode-item" value={item.value} key={item.value}>
                  <Select.ItemText>{item.label} — {item.hint}</Select.ItemText>
                </Select.Item>
              ))}
            </Select.Viewport>
          </Select.Content>
        </Select.Portal>
      </Select.Root>
      <small>{selected?.hint}</small>
    </label>
  );
}

function IconTip({ label, children }: { label: string; children: ReactNode }) {
  return (
    <Tooltip.Root delayDuration={250}>
      <Tooltip.Trigger asChild>{children}</Tooltip.Trigger>
      <Tooltip.Portal>
        <Tooltip.Content className="tooltip" sideOffset={8}>
          {label}
          <Tooltip.Arrow className="tooltip-arrow" />
        </Tooltip.Content>
      </Tooltip.Portal>
    </Tooltip.Root>
  );
}

function NoticeBanner({ notice, onClose }: { notice: Exclude<NoticeState, null>; onClose: () => void }) {
  const NoticeIcon = notice.tone === "success" ? CheckCircle2 : notice.tone === "error" ? AlertCircle : Info;

  return (
    <div className={`notice ${notice.tone}`} role={notice.tone === "error" ? "alert" : "status"}>
      <NoticeIcon size={18} aria-hidden="true" />
      <span>{notice.message}</span>
      <button type="button" onClick={onClose} aria-label="Dismiss notification">
        <X size={16} />
      </button>
    </div>
  );
}

function InspectorSection({
  title,
  description,
  children,
  defaultOpen = false
}: {
  title: string;
  description: string;
  children: ReactNode;
  defaultOpen?: boolean;
}) {
  return (
    <details className="inspector-section" open={defaultOpen || undefined}>
      <summary>
        <span>
          <strong>{title}</strong>
          <small>{description}</small>
        </span>
        <ChevronDown size={16} aria-hidden="true" />
      </summary>
      <div className="inspector-section-body">{children}</div>
    </details>
  );
}

function TopicItem({
  topic,
  activeUpload,
  activeSearch,
  onChoose
}: {
  topic: Topic;
  activeUpload: boolean;
  activeSearch: boolean;
  onChoose: () => void;
}) {
  const active = activeUpload || activeSearch;

  return (
    <button type="button" className={active ? "topic-item active" : "topic-item"} onClick={onChoose}>
      <div className="topic-row">
        <strong>{topic.title}</strong>
        <div className="topic-tags">
          {activeUpload ? <span>Upload</span> : null}
          {activeSearch ? <span>Search</span> : null}
        </div>
      </div>
      <code>{topic.collection}</code>
      <div className="topic-meta">
        <span>{formatCount(topic.document_count, "source")}</span>
        <span>{topic.vector_count === null ? "Index not built" : formatCount(topic.vector_count, "vector")}</span>
      </div>
    </button>
  );
}

function RunSummary({ result }: { result: ChatResponse | null }) {
  const summary = runSummary(result);

  return (
    <div className="run-summary" aria-label="Latest run summary">
      <div className="run-metric">
        <GitBranch size={15} />
        <span>Policy</span>
        <strong>{summary.policy}</strong>
      </div>
      <div className="run-metric">
        <Cpu size={15} />
        <span>Calls</span>
        <strong>{result ? summary.calls : "—"}</strong>
      </div>
      <div className="run-metric">
        <Database size={15} />
        <span>Tokens</span>
        <strong>{result ? summary.tokens.toLocaleString() : "—"}</strong>
      </div>
      <div className="run-metric">
        <Clock3 size={15} />
        <span>Latency</span>
        <strong>{result ? `${Math.round(summary.latency)} ms` : "—"}</strong>
      </div>
      <div className="run-metric">
        <MessageSquare size={15} />
        <span>Cites</span>
        <strong>{summary.citations || "—"}</strong>
      </div>
    </div>
  );
}

function PlanRows({ result }: { result: ChatResponse | null }) {
  const rows = (result?.trace ?? []).flatMap((step) => step.sub_queries ?? []);
  if (!rows.length) {
    return <div className="empty">No plan yet. Ask a question to see how Linki decomposes retrieval.</div>;
  }

  return (
    <div className="plan-list">
      {rows.map((item, index) => (
        <div className="plan-row" key={`${item.id}-${item.query}-${index}`}>
          <code>{item.id || `q${index + 1}`}</code>
          <div>
            <strong>{item.query}</strong>
            <span>{item.reason || "Planned retrieval query"}</span>
          </div>
          <small>{formatTarget(item.target_kb)}</small>
        </div>
      ))}
    </div>
  );
}

function StepIcon({ step }: { step: TraceStep }) {
  const tone = stepTone(step);
  if (tone === "danger") return <ShieldAlert size={11} />;
  if (tone === "warning") return <AlertTriangle size={11} />;
  return <Check size={11} />;
}

function TracePanel({ result }: { result: ChatResponse | null }) {
  if (!result?.trace.length) {
    return <div className="empty">Run a question to see routing, retrieval, grading, and verification steps.</div>;
  }

  return (
    <div className="trace-list">
      {result.trace.map((step, index) => {
        const tone = stepTone(step);
        return (
        <details
          className={`trace-step ${step.kind} ${tone}`}
          key={`${step.title}-${index}`}
          open={index === result.trace.length - 1 || tone !== "neutral"}
        >
          <summary>
            <span className="dot">
              <StepIcon step={step} />
            </span>
            <span>{step.title}</span>
            <small>{step.status}</small>
          </summary>
          {step.sub_queries?.length ? (
            <div className="plan-list inline-plan">
              {step.sub_queries.map((item, itemIndex) => (
                <div className="plan-row" key={`${item.id}-${itemIndex}`}>
                  <code>{item.id || `q${itemIndex + 1}`}</code>
                  <div>
                    <strong>{item.query}</strong>
                    <span>{item.reason || "Planned retrieval query"}</span>
                  </div>
                  <small>{formatTarget(item.target_kb)}</small>
                </div>
              ))}
            </div>
          ) : null}
          <pre>{step.detail}</pre>
          {step.issues?.length ? (
            <div className="issue-list">
              {step.issues.map((issue, issueIndex) => (
                <div className="issue-row" key={`${issue.claim}-${issueIndex}`}>
                  <strong>{issue.problem || "verification issue"}</strong>
                  <span>{issue.claim || "Unsupported claim"}</span>
                  {issue.fix_instruction ? <small>{issue.fix_instruction}</small> : null}
                </div>
              ))}
            </div>
          ) : null}
          {step.hits?.length ? (
            <div className="hit-list">
              {step.hits.map((hit) => (
                <div className="hit" key={`${hit.chunk_id}-${hit.score}`}>
                  <code>{hit.chunk_id || "chunk"}</code>
                  <span>{hit.heading_path || hit.source || "Untitled passage"}</span>
                  <strong>{typeof hit.score === "number" ? hit.score.toFixed(3) : ""}</strong>
                </div>
              ))}
            </div>
          ) : null}
        </details>
        );
      })}
    </div>
  );
}

function EvidencePanel({ result }: { result: ChatResponse | null }) {
  if (!result?.evidence.length) {
    return <div className="empty">No evidence yet. Ask a question to inspect the passages behind the answer.</div>;
  }

  return (
    <div className="evidence-list">
      {result.evidence.map((item, index) => (
        <div className="evidence-item" key={`${item.chunk_id}-${index}`}>
          <div className="evidence-top">
            <span>[{index + 1}]</span>
            <strong>{item.source || "Unknown source"}</strong>
            <code>{typeof item.score === "number" ? item.score.toFixed(3) : ""}</code>
          </div>
          {item.heading_path ? <small>{item.heading_path}</small> : null}
          <p>{item.text?.slice(0, 320) || "No preview text returned."}</p>
          <div className="evidence-meta">
            <span>{item.token_count ? `${item.token_count} tokens` : "token count unavailable"}</span>
            <span>
              {typeof item.rerank_score === "number" ? `rerank ${item.rerank_score.toFixed(3)}` : item.rerank_backend || "retrieval score"}
            </span>
            {typeof item.char_start === "number" && typeof item.char_end === "number" ? (
              <span>chars {item.char_start}–{item.char_end}</span>
            ) : null}
          </div>
        </div>
      ))}
    </div>
  );
}

function SourcesPanel({ result }: { result: ChatResponse | null }) {
  if (!result?.citations.length) {
    return <div className="empty">No citations yet. Sources appear after an answer is generated.</div>;
  }

  return (
    <div className="source-list">
      {result.citations.map((source) => (
        <div className="source-item" key={`${source.index}-${source.chunk_id}`}>
          <code>[{source.index}]</code>
          <span>{source.label}</span>
        </div>
      ))}
    </div>
  );
}

function MemoryPanel({
  items,
  busy,
  onForget
}: {
  items: MemoryItem[];
  busy: boolean;
  onForget: (memoryId: string) => void;
}) {
  if (!items.length) {
    return (
      <div className="empty">
        No active memory in this user scope. Say “Remember that …” to create an auditable preference.
      </div>
    );
  }
  return (
    <div className="governance-list">
      {items.map((item) => (
        <div className="governance-row" key={item.memory_id}>
          <div className="governance-icon"><Brain size={15} /></div>
          <div>
            <strong>{String(item.content.value ?? item.content.outcome ?? item.content.kind ?? "Memory")}</strong>
            <span>{item.type} · {item.status} · v{item.version}</span>
            <code>{item.memory_id}</code>
          </div>
          <IconTip label="Forget and tombstone this memory">
            <button
              type="button"
              className="icon-button compact danger-text"
              onClick={() => onForget(item.memory_id)}
              disabled={busy}
              aria-label={`Forget memory ${item.memory_id}`}
            >
              <Trash2 size={14} />
            </button>
          </IconTip>
        </div>
      ))}
    </div>
  );
}

function WikiPanel({ pages, snapshotId }: { pages: WikiPage[]; snapshotId: string | null }) {
  if (!pages.length) {
    return (
      <div className="empty">
        No published Wiki projection for this tenant. Approve sourced claims, then publish a knowledge snapshot.
      </div>
    );
  }
  return (
    <div className="governance-list">
      {pages.map((page) => (
        <div className="governance-row wiki-row" key={page.page_id}>
          <div className="governance-icon"><BookOpen size={15} /></div>
          <div>
            <strong>{page.title}</strong>
            <span>{page.domain} / {page.topic} · {page.status} · {page.claim_ids.length} claims</span>
            <code>{page.slug}</code>
          </div>
        </div>
      ))}
      <div className="snapshot-note">Active snapshot <code>{snapshotId}</code></div>
    </div>
  );
}

function CostPanel({ result }: { result: ChatResponse | null }) {
  if (!result) return <div className="empty">Run a question to inspect node-level usage and versions.</div>;
  return (
    <div className="cost-panel">
      <div className="run-identity">
        <span>Run <code>{result.run_id}</code></span>
        <span>Pack <code>{result.evidence_pack_id || "none"}</code></span>
        <span>{result.cache.hit ? "Exact cache hit" : result.cache.coalesced ? "Single-flight reuse" : "Fresh execution"}</span>
      </div>
      {result.cost.node_costs.length ? (
        <div className="cost-list">
          {result.cost.node_costs.map((row, index) => (
            <div className="cost-row" key={`${row.node}-${index}`}>
              <div><strong>{row.node}</strong><span>{row.prompt_version} · {row.model}</span></div>
              <code>{row.input_tokens + row.output_tokens} tok</code>
              <span>{Math.round(row.total_ms)} ms</span>
            </div>
          ))}
        </div>
      ) : <div className="empty compact">No model call was needed for this run.</div>}
    </div>
  );
}

function TopicContext({
  topic,
  onManage
}: {
  topic: Topic | undefined;
  onManage: () => void;
}) {
  if (!topic) {
    return (
      <div className="context-strip empty-context">
        <div>
          <strong>No active topic</strong>
          <span>Create a knowledge topic and add source documents before asking.</span>
        </div>
        <button type="button" className="button secondary" onClick={onManage}>
          Open Knowledge
          <ArrowRight size={15} />
        </button>
      </div>
    );
  }

  return (
    <div className="context-strip">
      <div className="context-heading">
        <Compass size={16} />
        <div>
          <strong>{topic.title}</strong>
          <span>Active knowledge topic</span>
        </div>
      </div>
      <div className="context-grid">
        <span>
          Collection
          <code>{topic.collection}</code>
        </span>
        <span>
          Sources
          <strong>{topic.document_count}</strong>
        </span>
        <span>
          Vectors
          <strong>{topic.vector_count ?? "Not built"}</strong>
        </span>
      </div>
    </div>
  );
}

export function App() {
  const [activeView, setActiveView] = useState("rag");
  const [topics, setTopics] = useState<Topic[]>([]);
  const [uploadTopic, setUploadTopic] = useState("default");
  const [searchTopic, setSearchTopic] = useState("default");
  const [mode, setMode] = useState<ExecutionMode>("auto");
  const [tenant, setTenant] = useState("default");
  const [user, setUser] = useState("anonymous");
  const [newTopic, setNewTopic] = useState("");
  const [topicHint, setTopicHint] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [selectedFiles, setSelectedFiles] = useState("No files selected");
  const [lastResult, setLastResult] = useState<ChatResponse | null>(null);
  const [notice, setNotice] = useState<NoticeState>(null);
  const [clearArmed, setClearArmed] = useState(false);
  const [createTopicOpen, setCreateTopicOpen] = useState(false);
  const [pendingAction, setPendingAction] = useState<PendingAction>(null);
  const [memoryItems, setMemoryItems] = useState<MemoryItem[]>([]);
  const [wikiPages, setWikiPages] = useState<WikiPage[]>([]);
  const [wikiSnapshot, setWikiSnapshot] = useState<string | null>(null);
  const [feedbackSent, setFeedbackSent] = useState<"up" | "down" | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);
  const messagesRef = useRef<HTMLDivElement>(null);
  const busy = pendingAction !== null;
  const chatBusy = pendingAction === "chat";

  const currentUploadTopic = useMemo(
    () => topics.find((topic) => topic.name === uploadTopic) ?? topics[0],
    [topics, uploadTopic]
  );
  const currentSearchTopic = useMemo(
    () => topics.find((topic) => topic.name === searchTopic) ?? topics[0],
    [topics, searchTopic]
  );
  const totalSources = useMemo(() => topics.reduce((sum, topic) => sum + topic.document_count, 0), [topics]);
  const totalVectors = useMemo(() => topics.reduce((sum, topic) => sum + (topic.vector_count ?? 0), 0), [topics]);
  const suggestedPrompts = useMemo(() => {
    if (!currentSearchTopic?.document_count) return [];
    return [
      "Summarize the source set",
      "List key facts with citations",
      "Which sources support the answer?"
    ];
  }, [currentSearchTopic]);

  async function refreshTopics(nextTopic?: string) {
    const data = await getTopics();
    setTopics(data.topics);
    if (!data.topics.length) setCreateTopicOpen(true);
    const fallback = data.topics[0]?.name ?? "default";
    const chosen = nextTopic && data.topics.some((topic) => topic.name === nextTopic) ? nextTopic : fallback;
    setUploadTopic((current) => (data.topics.some((topic) => topic.name === current) ? current : chosen));
    setSearchTopic((current) => (data.topics.some((topic) => topic.name === current) ? current : chosen));
  }

  async function refreshGovernance() {
    const [memory, wiki] = await Promise.all([
      getMemory(tenant || "default", user || "anonymous"),
      getWiki(tenant || "default")
    ]);
    setMemoryItems(memory.items.filter((item) => item.status !== "DELETED"));
    setWikiPages(wiki.pages);
    setWikiSnapshot(wiki.snapshot_id);
  }

  useEffect(() => {
    refreshTopics().catch((error) => setNotice({ message: errorMessage(error), tone: "error" }));
  }, []);

  useEffect(() => {
    const timeout = window.setTimeout(() => {
      refreshGovernance().catch((error) => setNotice({ message: errorMessage(error), tone: "error" }));
    }, 200);
    return () => window.clearTimeout(timeout);
  }, [tenant, user]);

  useEffect(() => {
    const frame = window.requestAnimationFrame(() => {
      messagesRef.current?.scrollTo({ top: messagesRef.current.scrollHeight, behavior: "auto" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [messages, chatBusy]);

  useEffect(() => {
    setClearArmed(false);
  }, [uploadTopic]);

  useEffect(() => {
    if (!clearArmed) return;
    const timeout = window.setTimeout(() => setClearArmed(false), 8000);
    return () => window.clearTimeout(timeout);
  }, [clearArmed]);

  async function handleRefreshTopics() {
    setPendingAction("refresh");
    try {
      await refreshTopics();
      setNotice({ message: "Knowledge topics and collection counts are up to date.", tone: "success" });
    } catch (error) {
      setNotice({ message: errorMessage(error), tone: "error" });
    } finally {
      setPendingAction(null);
    }
  }

  async function handleCreateTopic(event: FormEvent) {
    event.preventDefault();
    if (!newTopic.trim()) return;
    setPendingAction("create");
    try {
      const data = await createTopic(newTopic, topicHint);
      setTopics(data.topics);
      setUploadTopic(data.topic.name);
      setSearchTopic(data.topic.name);
      setNewTopic("");
      setTopicHint("");
      setCreateTopicOpen(false);
      setNotice({ message: `Topic "${data.topic.title}" created.`, tone: "success" });
    } catch (error) {
      setNotice({ message: errorMessage(error), tone: "error" });
    } finally {
      setPendingAction(null);
    }
  }

  async function handleUpload() {
    const files = fileRef.current?.files;
    if (!files?.length) return;
    setPendingAction("upload");
    try {
      const data = await uploadDocuments(uploadTopic, files, tenant || "default");
      setTopics(data.topics);
      setNotice({ message: `Added ${formatCount(data.added, "file")} to ${data.topic.collection}.`, tone: "success" });
      setSearchTopic(data.topic.name);
      if (data.failed.length) {
        setNotice({
          message: `Upload issues: ${data.failed.map((item) => `${item.name}: ${item.error}`).join("; ")}`,
          tone: "error"
        });
      }
      if (fileRef.current) fileRef.current.value = "";
      setSelectedFiles("No files selected");
    } catch (error) {
      setNotice({ message: errorMessage(error), tone: "error" });
    } finally {
      setPendingAction(null);
    }
  }

  async function handleClearTopic() {
    if (!currentUploadTopic) return;
    if (!clearArmed) {
      setClearArmed(true);
      setNotice({
        message: `Confirm once more to remove all documents from ${currentUploadTopic.collection}.`,
        tone: "info"
      });
      return;
    }
    setPendingAction("clear");
    try {
      const data = await clearTopic(uploadTopic);
      setTopics(data.topics);
      setNotice({ message: `Cleared documents from ${data.topic.collection}.`, tone: "success" });
    } catch (error) {
      setNotice({ message: errorMessage(error), tone: "error" });
    } finally {
      setClearArmed(false);
      setPendingAction(null);
    }
  }

  async function handleSend(event: FormEvent) {
    event.preventDefault();
    const message = input.trim();
    if (!message) return;
    const nextMessages: ChatMessage[] = [...messages, { role: "user", content: message }];
    setMessages(nextMessages);
    setInput("");
    setNotice(null);
    setPendingAction("chat");
    try {
      const result = await sendChat(message, searchTopic, messages, {
        mode,
        tenant: tenant || "default",
        user: user || "anonymous",
        acl: ["public"]
      });
      setLastResult(result);
      setFeedbackSent(null);
      setMessages([...nextMessages, { role: "assistant", content: result.answer }]);
      await refreshGovernance();
    } catch (error) {
      setMessages(nextMessages);
      setNotice({ message: errorMessage(error), tone: "error" });
    } finally {
      setPendingAction(null);
    }
  }

  async function handleForget(memoryId: string) {
    setPendingAction("memory");
    try {
      await forgetMemory(memoryId, tenant || "default", user || "anonymous");
      await refreshGovernance();
      setNotice({ message: "Memory tombstoned and its recoverable payload removed.", tone: "success" });
    } catch (error) {
      setNotice({ message: errorMessage(error), tone: "error" });
    } finally {
      setPendingAction(null);
    }
  }

  async function handleFeedback(kind: "thumbs_up" | "thumbs_down") {
    if (!lastResult || feedbackSent) return;
    try {
      await sendFeedback(
        kind,
        lastResult.run_id,
        tenant || "default",
        user || "anonymous",
        { question: messages.filter((item) => item.role === "user").at(-1)?.content, policy_path: lastResult.policy_path }
      );
      setFeedbackSent(kind === "thumbs_up" ? "up" : "down");
      setNotice({
        message: "Feedback recorded as an observation. It will not change production directly.",
        tone: "success"
      });
    } catch (error) {
      setNotice({ message: errorMessage(error), tone: "error" });
    }
  }

  return (
    <Tooltip.Provider>
      <div className="app-shell">
        <header className="topbar">
          <div className="brand-lockup">
            <div className="brand-mark">
              <span>LI</span>
            </div>
            <div>
              <h1>Linki Workspace</h1>
              <p>Ground answers in your sources, then inspect every decision.</p>
            </div>
          </div>
          <div className="status-group">
            <div className="status-pill data">
              <Database size={15} />
              {formatCount(topics.length, "topic")}
            </div>
            <div className="status-pill quiet">{formatCount(totalSources, "source")}</div>
            <div className="status-pill quiet">{formatCount(totalVectors, "vector")}</div>
          </div>
        </header>

        {notice ? <NoticeBanner notice={notice} onClose={() => setNotice(null)} /> : null}

        <Tabs.Root value={activeView} onValueChange={setActiveView} className="workspace-tabs">
          <Tabs.List className="workspace-tab-list" aria-label="Workspace sections">
            <Tabs.Trigger value="rag">
              <MessageSquare size={15} />
              <span>
                <strong>Ask</strong>
                <small>Query and inspect</small>
              </span>
            </Tabs.Trigger>
            <Tabs.Trigger value="vector">
              <Library size={15} />
              <span>
                <strong>Knowledge</strong>
                <small>Topics and sources</small>
              </span>
            </Tabs.Trigger>
          </Tabs.List>

          <Tabs.Content value="vector" className="workspace-tab-panel">
            <section className="panel knowledge-panel" aria-labelledby="knowledge-heading">
              <div className="panel-header">
                <div>
                  <h2 id="knowledge-heading">Knowledge</h2>
                  <p>Organize source sets into focused topics and keep their indexes current.</p>
                </div>
                <div className="panel-actions">
                  <button
                    type="button"
                    className="button secondary"
                    onClick={() => {
                      if (currentUploadTopic) setSearchTopic(currentUploadTopic.name);
                      setActiveView("rag");
                    }}
                  >
                    Ask this topic
                    <ArrowRight size={15} />
                  </button>
                  <IconTip label="Refresh topics and collection counts">
                    <button type="button" className="icon-button" onClick={handleRefreshTopics} disabled={busy} aria-label="Refresh topics">
                      <RefreshCw className={pendingAction === "refresh" ? "spin" : undefined} size={16} />
                    </button>
                  </IconTip>
                </div>
              </div>

              <div className="knowledge-layout">
                <div className="topic-column">
                  <div className="column-heading">
                    <h3>Topics</h3>
                    <span>{formatCount(topics.length, "collection")}</span>
                  </div>
                  <div className="topic-list">
                    {topics.length ? (
                      topics.map((topic) => (
                        <TopicItem
                          key={topic.name}
                          topic={topic}
                          activeUpload={topic.name === uploadTopic}
                          activeSearch={topic.name === searchTopic}
                          onChoose={() => {
                            setUploadTopic(topic.name);
                            setSearchTopic(topic.name);
                          }}
                        />
                      ))
                    ) : (
                      <div className="empty compact">Create a topic to start building a searchable collection.</div>
                    )}
                  </div>
                </div>

                <div className="workflow-stack">
                  <div className="form-section primary-workflow">
                    <div>
                      <h3>Add documents</h3>
                      <p>Upload PDF or Markdown files into the selected topic collection.</p>
                    </div>
                    <div className="upload-row">
                      <SelectBox
                        label="Destination topic"
                        value={uploadTopic}
                        onChange={setUploadTopic}
                        topics={topics}
                        hint={currentUploadTopic ? `Collection: ${currentUploadTopic.collection}` : undefined}
                      />
                      <label className="file-picker">
                        <FileUp size={17} />
                        <span>Choose files</span>
                        <strong>{selectedFiles}</strong>
                        <input
                          ref={fileRef}
                          type="file"
                          accept=".pdf,.md,.markdown"
                          multiple
                          disabled={busy || !currentUploadTopic}
                          onChange={(event) => {
                            const files = event.currentTarget.files;
                            if (!files?.length) {
                              setSelectedFiles("No files selected");
                              return;
                            }
                            setSelectedFiles(files.length === 1 ? files[0].name : `${files.length} files selected`);
                          }}
                        />
                      </label>
                      <button
                        type="button"
                        className="button primary"
                        onClick={handleUpload}
                        disabled={busy || !currentUploadTopic || selectedFiles === "No files selected"}
                      >
                        {pendingAction === "upload" ? <Loader2 className="spin" size={16} /> : <Upload size={16} />}
                        {pendingAction === "upload" ? "Uploading" : "Upload files"}
                      </button>
                    </div>
                  </div>

                  <details
                    className="workflow-disclosure"
                    open={createTopicOpen}
                    onToggle={(event) => setCreateTopicOpen(event.currentTarget.open)}
                  >
                    <summary>
                      <span>
                        <strong>Create a new topic</strong>
                        <small>Separate unrelated sources and retrieval routes.</small>
                      </span>
                      <ChevronDown size={16} aria-hidden="true" />
                    </summary>
                    <form className="topic-form" onSubmit={handleCreateTopic}>
                      <label className="field">
                        <span>Topic name</span>
                        <input value={newTopic} onChange={(event) => setNewTopic(event.target.value)} placeholder="Product API docs" />
                      </label>
                      <label className="field">
                        <span>Routing hint</span>
                        <input
                          value={topicHint}
                          onChange={(event) => setTopicHint(event.target.value)}
                          placeholder="Use for questions about API behavior"
                        />
                      </label>
                      <button className="button secondary" disabled={busy || !newTopic.trim()}>
                        {pendingAction === "create" ? <Loader2 className="spin" size={16} /> : <Plus size={16} />}
                        {pendingAction === "create" ? "Creating" : "Create topic"}
                      </button>
                    </form>
                  </details>

                  <div className="maintenance-row danger-zone">
                    <div className="danger-copy">
                      <ShieldAlert size={17} />
                      <div>
                        <h3>Clear selected topic</h3>
                        <p>Remove its indexed documents; other topics stay untouched.</p>
                      </div>
                    </div>
                    <button
                      type="button"
                      className={clearArmed ? "button danger confirming" : "button danger"}
                      onClick={handleClearTopic}
                      disabled={busy || !currentUploadTopic}
                    >
                      {pendingAction === "clear" ? <Loader2 className="spin" size={16} /> : <Trash2 size={16} />}
                      {pendingAction === "clear" ? "Clearing" : clearArmed ? "Confirm clear" : "Clear topic"}
                    </button>
                  </div>
                </div>
              </div>
            </section>
          </Tabs.Content>

          <Tabs.Content value="rag" className="workspace-tab-panel">
            <main className="workspace">
              <section className="panel chat-panel" aria-labelledby="ask-heading">
                <div className="panel-header">
                  <div>
                    <h2 id="ask-heading">Ask your knowledge</h2>
                    <p>
                      {currentSearchTopic
                        ? `Ground every answer in ${currentSearchTopic.title}, with evidence you can inspect.`
                        : "Set up a knowledge topic before asking your first question."}
                    </p>
                  </div>
                  {messages.length ? (
                    <button
                      type="button"
                      className="button quiet"
                      onClick={() => {
                        setMessages([]);
                        setLastResult(null);
                        setFeedbackSent(null);
                      }}
                      disabled={busy}
                    >
                      Clear chat
                    </button>
                  ) : null}
                </div>

                <TopicContext topic={currentSearchTopic} onManage={() => setActiveView("vector")} />

                <div className="chat-controls">
                  <SelectBox
                    label="Search topic"
                    value={searchTopic}
                    onChange={setSearchTopic}
                    topics={topics}
                    hint={currentSearchTopic ? `Tool: ${currentSearchTopic.tool_name}` : undefined}
                  />
                  <ModeSelect value={mode} onChange={setMode} />
                </div>

                <details className="scope-disclosure">
                  <summary>
                    <span className="scope-summary">
                      <Settings2 size={15} aria-hidden="true" />
                      <span>
                        <strong>Query scope</strong>
                        <small>{tenant || "default"} / {user || "anonymous"}</small>
                      </span>
                    </span>
                    <ChevronDown size={16} aria-hidden="true" />
                  </summary>
                  <div className="scope-fields">
                    <label className="field">
                      <span>Tenant</span>
                      <input
                        value={tenant}
                        onChange={(event) => setTenant(event.target.value)}
                        placeholder="default"
                        disabled={busy}
                      />
                      <small>Isolation and snapshot namespace</small>
                    </label>
                    <label className="field">
                      <span>User</span>
                      <input
                        value={user}
                        onChange={(event) => setUser(event.target.value)}
                        placeholder="anonymous"
                        disabled={busy}
                      />
                      <small>Personal memory and cache scope</small>
                    </label>
                  </div>
                </details>

                <div className="messages" ref={messagesRef} role="log" aria-live="polite" aria-busy={chatBusy}>
                  {messages.length === 0 ? (
                    <div className="welcome">
                      {currentSearchTopic?.document_count ? <MessageSquare size={26} /> : <Library size={26} />}
                      <strong>
                        {!currentSearchTopic
                          ? "Build your first knowledge topic"
                          : !currentSearchTopic.document_count
                            ? `Add sources to ${currentSearchTopic.title}`
                            : "Ask a question you can verify"}
                      </strong>
                      <span>
                        {currentSearchTopic?.document_count
                          ? "Linki can plan retrieval, refine weak searches, and verify the final answer."
                          : "Upload PDF or Markdown sources so answers can be grounded and cited."}
                      </span>
                      {currentSearchTopic?.document_count ? (
                        <div className="prompt-row" aria-label="Suggested prompts">
                          {suggestedPrompts.map((prompt) => (
                            <button type="button" key={prompt} onClick={() => setInput(prompt)}>
                              {prompt}
                            </button>
                          ))}
                        </div>
                      ) : (
                        <button type="button" className="button secondary welcome-action" onClick={() => setActiveView("vector")}>
                          Open Knowledge
                          <ArrowRight size={15} />
                        </button>
                      )}
                    </div>
                  ) : (
                    messages.map((message, index) => (
                      <div className={`message ${message.role}`} key={`${message.role}-${index}`} aria-label={`${message.role} message`}>
                        <div className="avatar">{message.role === "assistant" ? <Bot size={15} /> : "U"}</div>
                        <p>{message.content}</p>
                      </div>
                    ))
                  )}
                  {chatBusy ? (
                    <div className="message assistant thinking" role="status" aria-label="Linki is preparing an answer">
                      <div className="avatar"><Bot size={15} /></div>
                      <div className="thinking-copy">
                        <strong>Planning and checking evidence…</strong>
                        <span className="skeleton-line" />
                        <span className="skeleton-line short" />
                      </div>
                    </div>
                  ) : null}
                </div>

                <form className="composer" onSubmit={handleSend}>
                  <div className="composer-input">
                    <label className="sr-only" htmlFor="rag-question">Question</label>
                    <textarea
                      id="rag-question"
                      value={input}
                      rows={2}
                      onChange={(event) => setInput(event.target.value)}
                      onKeyDown={(event) => {
                        if (event.key === "Enter" && !event.shiftKey) {
                          event.preventDefault();
                          event.currentTarget.form?.requestSubmit();
                        }
                      }}
                      placeholder={currentSearchTopic?.document_count ? "Ask about the selected topic" : "Add source documents to start asking"}
                      disabled={busy || !currentSearchTopic?.document_count}
                      aria-describedby="composer-hint"
                    />
                    <small id="composer-hint">Enter to ask · Shift + Enter for a new line</small>
                  </div>
                  <button className="button primary send-button" disabled={busy || !input.trim() || !currentSearchTopic?.document_count}>
                    {chatBusy ? <Loader2 className="spin" size={16} /> : <SendHorizontal size={16} />}
                    {chatBusy ? "Working" : "Ask"}
                  </button>
                </form>
                {lastResult ? (
                  <div className="answer-feedback">
                    <div>
                      <strong>{lastResult.policy_path.toUpperCase()}</strong>
                      <span>{lastResult.policy.reason || "Adaptive policy decision"}</span>
                      {lastResult.shadow_policy?.path ? <small>Shadow: {lastResult.shadow_policy.path.toUpperCase()}</small> : null}
                    </div>
                    <div className="feedback-actions" aria-label="Rate this answer">
                      <span>{feedbackSent ? "Feedback recorded" : "Was this useful?"}</span>
                      <button
                        type="button"
                        className={feedbackSent === "up" ? "icon-button compact selected" : "icon-button compact"}
                        onClick={() => handleFeedback("thumbs_up")}
                        disabled={Boolean(feedbackSent)}
                        aria-label="Helpful answer"
                      ><ThumbsUp size={14} /></button>
                      <button
                        type="button"
                        className={feedbackSent === "down" ? "icon-button compact selected" : "icon-button compact"}
                        onClick={() => handleFeedback("thumbs_down")}
                        disabled={Boolean(feedbackSent)}
                        aria-label="Answer needs work"
                      ><ThumbsDown size={14} /></button>
                    </div>
                  </div>
                ) : null}
              </section>

              <aside className="panel inspect-panel" aria-labelledby="inspect-heading">
                <div className="panel-header compact">
                  <div>
                    <h2 id="inspect-heading">Inspect</h2>
                    <p>Inspect policy, evidence, cost, governed memory, and published knowledge.</p>
                  </div>
                </div>
                <RunSummary result={lastResult} />
                <Tabs.Root defaultValue="run" className="tabs">
                  <Tabs.List className="tabs-list">
                    <Tabs.Trigger value="run">Run</Tabs.Trigger>
                    <Tabs.Trigger value="grounding">Grounding</Tabs.Trigger>
                    <Tabs.Trigger value="context">Context</Tabs.Trigger>
                  </Tabs.List>
                  <Tabs.Content value="run" className="tab-panel inspector-stack">
                    <InspectorSection title="Execution trace" description="Routing, retrieval, grading, and verification" defaultOpen>
                      <TracePanel result={lastResult} />
                    </InspectorSection>
                    <InspectorSection title="Query plan" description="Sub-queries and their retrieval targets">
                      <PlanRows result={lastResult} />
                    </InspectorSection>
                    <InspectorSection title="Cost and versions" description="Model calls, tokens, latency, and prompt versions">
                      <CostPanel result={lastResult} />
                    </InspectorSection>
                  </Tabs.Content>
                  <Tabs.Content value="grounding" className="tab-panel inspector-stack">
                    <InspectorSection title="Evidence" description="Passages selected to support the answer" defaultOpen>
                      <EvidencePanel result={lastResult} />
                    </InspectorSection>
                    <InspectorSection title="Citations" description="Source labels exposed in the answer">
                      <SourcesPanel result={lastResult} />
                    </InspectorSection>
                  </Tabs.Content>
                  <Tabs.Content value="context" className="tab-panel inspector-stack">
                    <InspectorSection title="Memory" description="Auditable preferences in the active user scope" defaultOpen>
                      <MemoryPanel items={memoryItems} busy={busy} onForget={handleForget} />
                    </InspectorSection>
                    <InspectorSection title="Wiki" description="Published tenant knowledge snapshots">
                      <WikiPanel pages={wikiPages} snapshotId={wikiSnapshot} />
                    </InspectorSection>
                  </Tabs.Content>
                </Tabs.Root>
              </aside>
            </main>
          </Tabs.Content>
        </Tabs.Root>
      </div>
    </Tooltip.Provider>
  );
}
