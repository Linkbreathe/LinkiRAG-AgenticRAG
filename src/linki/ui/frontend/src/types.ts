export type Role = "user" | "assistant";
export type ExecutionMode = "auto" | "fast" | "balanced" | "deep";

export interface Topic {
  name: string;
  title: string;
  usage_hint: string;
  tool_name: string;
  collection: string;
  documents: string[];
  document_count: number;
  vector_count: number | null;
}

export interface ChatMessage {
  role: Role;
  content: string;
}

export interface TraceHit {
  chunk_id?: string;
  source?: string;
  heading_path?: string;
  score?: number;
}

export interface TraceSubQuery {
  id?: string;
  query: string;
  target_kb?: string;
  reason?: string;
}

export interface TraceIssue {
  claim?: string;
  problem?: string;
  fix_instruction?: string;
}

export interface TraceStep {
  kind: string;
  title: string;
  status: string;
  detail: string;
  hits?: TraceHit[];
  sub_queries?: TraceSubQuery[];
  round?: number;
  query?: string;
  kb?: string;
  sufficient?: boolean;
  kept?: number;
  missing?: string;
  refined_query?: string;
  verified?: boolean;
  issues?: TraceIssue[];
}

export interface Evidence {
  evidence_id?: string;
  chunk_id?: string;
  parent_id?: string;
  kb?: string;
  source?: string;
  heading_path?: string;
  text?: string;
  score?: number;
  retrieval_score?: number;
  rerank_score?: number | null;
  rerank_backend?: string;
  quote?: string;
  char_start?: number;
  char_end?: number;
  token_count?: number;
  supports?: string[];
}

export interface Citation {
  index: number;
  source: string;
  heading_path?: string;
  parent_id?: string;
  chunk_id?: string;
  evidence_id?: string;
  quote?: string;
  char_start?: number;
  char_end?: number;
  label: string;
}

export interface PolicyDecision {
  path?: string;
  mode?: ExecutionMode;
  reason?: string;
  signals?: string[];
  confidence?: number;
  budget?: {
    max_model_calls?: number;
    max_input_tokens?: number;
    evidence_tokens?: number;
    deadline_ms?: number;
  };
}

export interface NodeCost {
  node: string;
  model: string;
  prompt_version: string;
  input_tokens: number;
  output_tokens: number;
  total_ms: number;
  usage_source: "provider" | "estimated";
}

export interface RunCost {
  llm_calls: number;
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
  latency_ms: number;
  node_costs: NodeCost[];
}

export interface MemoryItem {
  memory_id: string;
  type: string;
  scope: string;
  status: string;
  content: Record<string, unknown>;
  confidence: number;
  version: number;
  source_episode_ids: string[];
  recall_score?: number;
}

export interface WikiPage {
  page_id: string;
  version: number;
  slug: string;
  title: string;
  domain: string;
  topic: string;
  snapshot_id: string;
  status: string;
  markdown: string;
  claim_ids: string[];
  source_spans: Record<string, unknown>[];
}

export interface ChatResponse {
  answer: string;
  trace: TraceStep[];
  evidence: Evidence[];
  citations: Citation[];
  topic: Topic;
  run_id: string;
  policy: PolicyDecision;
  policy_path: string;
  cost: RunCost;
  cache: { hit?: boolean; coalesced?: boolean; source?: string | null };
  snapshots: Record<string, string>;
  versions: Record<string, unknown>;
  evidence_pack_id?: string;
  evidence_pack?: {
    token_budget: number;
    token_count: number;
    units: Evidence[];
  };
  memory: {
    snapshot_id?: string;
    recalled: MemoryItem[];
    changes: MemoryItem[];
  };
}
