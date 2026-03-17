export const API_BASE =
  process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export interface Citation {
  title: string;
  section: string | null;
}

export interface ChunkData {
  title: string;
  section: string | null;
  text: string;
}

export interface AskResponse {
  query: string;
  answer: string;
  citations: Citation[];
  chunks: ChunkData[];
  metrics: { retrieval_ms: number; generation_ms: number };
}

export interface IngestResponse {
  indexed_docs: number;
  indexed_chunks: number;
}

export interface MetricsResponse {
  total_docs: number;
  total_chunks: number;
  avg_retrieval_latency_ms: number;
  avg_generation_latency_ms: number;
  embedding_model: string;
  llm_model: string;
}

export async function apiAsk(
  query: string,
  k: number = 8
): Promise<AskResponse> {
  const r = await fetch(`${API_BASE}/api/ask`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k }),
  });
  if (!r.ok) throw new Error("Ask failed");
  return r.json();
}

export async function apiIngest(): Promise<IngestResponse> {
  const r = await fetch(`${API_BASE}/api/ingest`, { method: "POST" });
  if (!r.ok) throw new Error("Ingest failed");
  return r.json();
}

export async function apiMetrics(): Promise<MetricsResponse> {
  const r = await fetch(`${API_BASE}/api/metrics`);
  if (!r.ok) throw new Error("Metrics failed");
  return r.json();
}
