import * as Select from "@radix-ui/react-select";
import * as Tabs from "@radix-ui/react-tabs";
import * as Tooltip from "@radix-ui/react-tooltip";
import {
  ArrowRight,
  Bot,
  Check,
  ChevronDown,
  Compass,
  Database,
  FileUp,
  Loader2,
  MessageSquare,
  Plus,
  RefreshCw,
  Search,
  ShieldAlert,
  Trash2,
  Upload
} from "lucide-react";
import { FormEvent, ReactNode, useEffect, useMemo, useRef, useState } from "react";
import { clearTopic, createTopic, getTopics, sendChat, uploadDocuments } from "./api";
import type { ChatMessage, ChatResponse, Topic } from "./types";

function formatCount(value: number, singular: string, plural = `${singular}s`) {
  return `${value} ${value === 1 ? singular : plural}`;
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

function TracePanel({ result }: { result: ChatResponse | null }) {
  if (!result?.trace.length) {
    return <div className="empty">Run a question to see routing, retrieval, grading, and verification steps.</div>;
  }

  return (
    <div className="trace-list">
      {result.trace.map((step, index) => (
        <details className={`trace-step ${step.kind}`} key={`${step.title}-${index}`} open={index === result.trace.length - 1}>
          <summary>
            <span className="dot">
              <Check size={11} />
            </span>
            <span>{step.title}</span>
            <small>{step.status}</small>
          </summary>
          <pre>{step.detail}</pre>
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
      ))}
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
        </div>
      ))}
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
          <span>Create or select a vector collection before running RAG.</span>
        </div>
        <button type="button" className="button secondary" onClick={onManage}>
          Vector Database
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
          <span>Active RAG topic</span>
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
  const [activeView, setActiveView] = useState("vector");
  const [topics, setTopics] = useState<Topic[]>([]);
  const [uploadTopic, setUploadTopic] = useState("default");
  const [searchTopic, setSearchTopic] = useState("default");
  const [newTopic, setNewTopic] = useState("");
  const [topicHint, setTopicHint] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [selectedFiles, setSelectedFiles] = useState("No files selected");
  const [lastResult, setLastResult] = useState<ChatResponse | null>(null);
  const [notice, setNotice] = useState("");
  const [clearArmed, setClearArmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);

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
    if (!currentSearchTopic) {
      return ["Create a topic first", "Upload source documents", "Return to Vector Database"];
    }
    if (!currentSearchTopic.document_count) {
      return ["What should I upload first?", "Create a source checklist", "Explain this topic setup"];
    }
    return [
      "Summarize the source set",
      "List key facts with citations",
      currentSearchTopic.usage_hint || "What should I read first?"
    ];
  }, [currentSearchTopic]);

  async function refreshTopics(nextTopic?: string) {
    const data = await getTopics();
    setTopics(data.topics);
    const fallback = data.topics[0]?.name ?? "default";
    const chosen = nextTopic && data.topics.some((topic) => topic.name === nextTopic) ? nextTopic : fallback;
    setUploadTopic((current) => (data.topics.some((topic) => topic.name === current) ? current : chosen));
    setSearchTopic((current) => (data.topics.some((topic) => topic.name === current) ? current : chosen));
  }

  useEffect(() => {
    refreshTopics().catch((error) => setNotice(error.message));
  }, []);

  useEffect(() => {
    setClearArmed(false);
  }, [uploadTopic]);

  async function handleCreateTopic(event: FormEvent) {
    event.preventDefault();
    if (!newTopic.trim()) return;
    setBusy(true);
    try {
      const data = await createTopic(newTopic, topicHint);
      setTopics(data.topics);
      setUploadTopic(data.topic.name);
      setSearchTopic(data.topic.name);
      setNewTopic("");
      setTopicHint("");
      setNotice(`Topic "${data.topic.title}" created.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleUpload() {
    const files = fileRef.current?.files;
    if (!files?.length) return;
    setBusy(true);
    try {
      const data = await uploadDocuments(uploadTopic, files);
      setTopics(data.topics);
      setNotice(`Added ${formatCount(data.added, "file")} to ${data.topic.collection}.`);
      setSearchTopic(data.topic.name);
      if (data.failed.length) {
        setNotice(`Upload issues: ${data.failed.map((item) => `${item.name}: ${item.error}`).join("; ")}`);
      }
      if (fileRef.current) fileRef.current.value = "";
      setSelectedFiles("No files selected");
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setBusy(false);
    }
  }

  async function handleClearTopic() {
    if (!currentUploadTopic) return;
    if (!clearArmed) {
      setClearArmed(true);
      setNotice(`Press Confirm clear to remove documents from ${currentUploadTopic.collection}.`);
      return;
    }
    setBusy(true);
    try {
      const data = await clearTopic(uploadTopic);
      setTopics(data.topics);
      setNotice(`Cleared documents from ${data.topic.collection}.`);
    } catch (error) {
      setNotice(error instanceof Error ? error.message : String(error));
    } finally {
      setClearArmed(false);
      setBusy(false);
    }
  }

  async function handleSend(event: FormEvent) {
    event.preventDefault();
    const message = input.trim();
    if (!message) return;
    const nextMessages: ChatMessage[] = [...messages, { role: "user", content: message }];
    setMessages(nextMessages);
    setInput("");
    setBusy(true);
    try {
      const result = await sendChat(message, searchTopic, messages);
      setLastResult(result);
      setMessages([...nextMessages, { role: "assistant", content: result.answer }]);
    } catch (error) {
      const text = error instanceof Error ? error.message : String(error);
      setMessages([...nextMessages, { role: "assistant", content: `Error: ${text}` }]);
      setNotice(text);
    } finally {
      setBusy(false);
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
              <p>Manage vector collections and run grounded RAG workflows.</p>
            </div>
          </div>
          <div className="status-group">
            <div className="status-pill">
              <Database size={15} />
              {formatCount(topics.length, "topic")}
            </div>
            <div className="status-pill quiet">{formatCount(totalSources, "source")}</div>
            <div className="status-pill quiet">{formatCount(totalVectors, "vector")}</div>
          </div>
        </header>

        {notice ? (
          <button type="button" className="notice" onClick={() => setNotice("")}>
            {notice}
          </button>
        ) : null}

        <Tabs.Root value={activeView} onValueChange={setActiveView} className="workspace-tabs">
          <Tabs.List className="workspace-tab-list" aria-label="Workspace sections">
            <Tabs.Trigger value="vector">
              <Database size={15} />
              Vector Database
            </Tabs.Trigger>
            <Tabs.Trigger value="rag">
              <MessageSquare size={15} />
              RAG
            </Tabs.Trigger>
          </Tabs.List>

          <Tabs.Content value="vector" className="workspace-tab-panel">
            <section className="panel knowledge-panel" aria-labelledby="knowledge-heading">
              <div className="panel-header">
                <div>
                  <h2 id="knowledge-heading">Vector database</h2>
                  <p>Create isolated topics, upload source files, and maintain collection contents.</p>
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
                    Ask in RAG
                    <ArrowRight size={15} />
                  </button>
                  <IconTip label="Refresh topics and collection counts">
                    <button type="button" className="icon-button" onClick={() => refreshTopics()} disabled={busy} aria-label="Refresh topics">
                      <RefreshCw size={16} />
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
                  <div className="form-section">
                    <div>
                      <h3>Create a topic</h3>
                      <p>Use topics to keep unrelated document sets and retrieval tools separate.</p>
                    </div>
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
                        <Plus size={16} />
                        Create topic
                      </button>
                    </form>
                  </div>

                  <div className="form-section">
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
                          disabled={busy}
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
                      <button type="button" className="button primary" onClick={handleUpload} disabled={busy || !currentUploadTopic}>
                        <Upload size={16} />
                        Upload files
                      </button>
                    </div>
                  </div>

                  <div className="maintenance-row danger-zone">
                    <div className="danger-copy">
                      <ShieldAlert size={17} />
                      <div>
                        <h3>Danger zone</h3>
                        <p>Clear documents only from the selected upload topic. Other topics stay untouched.</p>
                      </div>
                    </div>
                    <button
                      type="button"
                      className={clearArmed ? "button danger confirming" : "button danger"}
                      onClick={handleClearTopic}
                      disabled={busy || !currentUploadTopic}
                    >
                      <Trash2 size={16} />
                      {clearArmed ? "Confirm clear" : "Clear topic"}
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
                    <h2 id="ask-heading">RAG</h2>
                    <p>
                      {currentSearchTopic
                        ? `Searching ${currentSearchTopic.title} for grounded answers.`
                        : "Choose a topic before asking a question."}
                    </p>
                  </div>
                  <button type="button" className="button secondary" onClick={() => setMessages([])} disabled={busy || !messages.length}>
                    Clear chat
                  </button>
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
                </div>

                <div className="messages" aria-live="polite">
                  {messages.length === 0 ? (
                    <div className="welcome">
                      <MessageSquare size={28} />
                      <strong>Ask about your documents</strong>
                      <span>Answers are constrained to the selected topic collection.</span>
                      <div className="prompt-row" aria-label="Suggested prompts">
                        {suggestedPrompts.map((prompt) => (
                          <button
                            type="button"
                            key={prompt}
                            onClick={() => {
                              if (!currentSearchTopic || prompt === "Return to Vector Database") {
                                setActiveView("vector");
                                return;
                              }
                              setInput(prompt);
                            }}
                          >
                            {prompt}
                          </button>
                        ))}
                      </div>
                    </div>
                  ) : (
                    messages.map((message, index) => (
                      <div className={`message ${message.role}`} key={`${message.role}-${index}`}>
                        <div className="avatar">{message.role === "assistant" ? <Bot size={15} /> : "U"}</div>
                        <p>{message.content}</p>
                      </div>
                    ))
                  )}
                </div>

                <form className="composer" onSubmit={handleSend}>
                  <Search size={18} />
                  <input
                    value={input}
                    onChange={(event) => setInput(event.target.value)}
                    placeholder="Ask about the selected topic"
                    disabled={busy}
                  />
                  <button className="button primary" disabled={busy || !input.trim()}>
                    {busy ? <Loader2 className="spin" size={16} /> : null}
                    {busy ? "Sending" : "Send"}
                  </button>
                </form>
              </section>

              <aside className="panel inspect-panel" aria-labelledby="inspect-heading">
                <div className="panel-header compact">
                  <div>
                    <h2 id="inspect-heading">Inspect</h2>
                    <p>Review the path from routing to evidence.</p>
                  </div>
                </div>
                <Tabs.Root defaultValue="trace" className="tabs">
                  <Tabs.List className="tabs-list">
                    <Tabs.Trigger value="trace">Trace</Tabs.Trigger>
                    <Tabs.Trigger value="evidence">Evidence</Tabs.Trigger>
                    <Tabs.Trigger value="sources">Sources</Tabs.Trigger>
                  </Tabs.List>
                  <Tabs.Content value="trace" className="tab-panel">
                    <TracePanel result={lastResult} />
                  </Tabs.Content>
                  <Tabs.Content value="evidence" className="tab-panel">
                    <EvidencePanel result={lastResult} />
                  </Tabs.Content>
                  <Tabs.Content value="sources" className="tab-panel">
                    {lastResult?.citations.length ? (
                      <div className="source-list">
                        {lastResult.citations.map((source) => (
                          <div className="source-item" key={`${source.index}-${source.chunk_id}`}>
                            <code>[{source.index}]</code>
                            <span>{source.label}</span>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="empty">No citations yet. Sources appear after an answer is generated.</div>
                    )}
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
