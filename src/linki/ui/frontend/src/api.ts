import type { ChatMessage, ChatResponse, ExecutionMode, MemoryItem, Topic, WikiPage } from "./types";

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init);
  if (!response.ok) {
    let detail =
      response.status === 404 && url.startsWith("/api/")
        ? "Linki API is not available. Start the backend server and reload the workspace."
        : response.statusText;
    try {
      const data = await response.json();
      if (!(response.status === 404 && data.detail === "Not Found" && url.startsWith("/api/"))) {
        detail = data.detail ?? detail;
      }
    } catch {
      // Keep the HTTP status text when the body is not JSON.
    }
    throw new Error(detail);
  }
  return response.json() as Promise<T>;
}

export function getTopics() {
  return request<{ topics: Topic[] }>("/api/topics");
}

export function createTopic(title: string, usage_hint: string) {
  return request<{ topic: Topic; topics: Topic[] }>("/api/topics", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ title, usage_hint })
  });
}

export function clearTopic(topic: string) {
  return request<{ topic: Topic; topics: Topic[] }>(`/api/topics/${encodeURIComponent(topic)}/documents`, {
    method: "DELETE"
  });
}

export function uploadDocuments(topic: string, files: FileList, tenant = "default", acl = "public") {
  const body = new FormData();
  body.set("topic", topic);
  body.set("tenant", tenant);
  body.set("acl", acl);
  Array.from(files).forEach((file) => body.append("files", file));
  return request<{ topic: Topic; added: number; failed: { name: string; error: string }[]; topics: Topic[] }>(
    "/api/documents",
    { method: "POST", body }
  );
}

export function sendChat(
  message: string,
  topic: string,
  history: ChatMessage[],
  options: { mode: ExecutionMode; tenant: string; user: string; acl?: string[]; deadline_ms?: number }
) {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, topic, history, ...options })
  });
}

export function getMemory(tenant: string, user: string) {
  const query = new URLSearchParams({ tenant, user });
  return request<{ items: MemoryItem[]; snapshot_id: string }>(`/api/memory?${query}`);
}

export function forgetMemory(memoryId: string, tenant: string, user: string) {
  const query = new URLSearchParams({ tenant, user });
  return request<{ deleted: string[]; snapshot_id: string }>(
    `/api/memory/${encodeURIComponent(memoryId)}?${query}`,
    { method: "DELETE" }
  );
}

export function getWiki(tenant: string, acl = "public") {
  const query = new URLSearchParams({ tenant, acl });
  return request<{ pages: WikiPage[]; snapshot_id: string | null }>(`/api/wiki?${query}`);
}

export function sendFeedback(
  kind: "thumbs_up" | "thumbs_down",
  runId: string,
  tenant: string,
  user: string,
  payload: Record<string, unknown>
) {
  return request<{ feedback: { feedback_id: string; status: string } }>("/api/feedback", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ kind, run_id: runId, tenant, user, acl: ["public"], payload })
  });
}
