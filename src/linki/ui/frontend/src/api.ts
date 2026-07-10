import type { ChatMessage, ChatResponse, Topic } from "./types";

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

export function uploadDocuments(topic: string, files: FileList) {
  const body = new FormData();
  body.set("topic", topic);
  Array.from(files).forEach((file) => body.append("files", file));
  return request<{ topic: Topic; added: number; failed: { name: string; error: string }[]; topics: Topic[] }>(
    "/api/documents",
    { method: "POST", body }
  );
}

export function sendChat(message: string, topic: string, history: ChatMessage[]) {
  return request<ChatResponse>("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ message, topic, history })
  });
}
