export type Role = "user" | "assistant";

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
  chunk_id?: string;
  parent_id?: string;
  kb?: string;
  source?: string;
  heading_path?: string;
  text?: string;
  score?: number;
}

export interface Citation {
  index: number;
  source: string;
  heading_path?: string;
  parent_id?: string;
  chunk_id?: string;
  label: string;
}

export interface ChatResponse {
  answer: string;
  trace: TraceStep[];
  evidence: Evidence[];
  citations: Citation[];
  topic: Topic;
}
