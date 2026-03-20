"use client";
import React from "react";
import { apiIngest, apiMetrics, MetricsResponse } from "../lib/api";

export default function AdminPanel() {
  const [metrics, setMetrics] = React.useState<MetricsResponse | null>(null);
  const [busy, setBusy] = React.useState(false);
  const [error, setError] = React.useState<string | null>(null);

  const refresh = async () => {
    try {
      setError(null);
      const data = await apiMetrics();
      setMetrics(data);
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Failed to load metrics";
      setError(message);
    }
  };

  const ingest = async () => {
    setBusy(true);
    setError(null);
    try {
      await apiIngest();
      await refresh();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : "Ingestion failed";
      setError(message);
    } finally {
      setBusy(false);
    }
  };

  React.useEffect(() => {
    refresh();
  }, []);

  return (
    <section className="card" aria-label="Admin panel">
      <h2>Admin</h2>
      <div style={{ display: "flex", gap: 8, marginBottom: 8 }}>
        <button
          onClick={ingest}
          disabled={busy}
          aria-label="Ingest sample documents"
          style={{
            padding: "8px 12px",
            borderRadius: 8,
            border: "1px solid #111",
            background: "#fff",
            cursor: busy ? "not-allowed" : "pointer",
            opacity: busy ? 0.6 : 1,
          }}
        >
          {busy ? "Indexing..." : "Ingest sample docs"}
        </button>
        <button
          onClick={refresh}
          disabled={busy}
          aria-label="Refresh metrics"
          style={{
            padding: "8px 12px",
            borderRadius: 8,
            border: "1px solid #111",
            background: "#fff",
            cursor: "pointer",
          }}
        >
          Refresh metrics
        </button>
      </div>

      {error && (
        <div
          role="alert"
          style={{
            padding: 8,
            marginBottom: 8,
            background: "#fef2f2",
            border: "1px solid #fca5a5",
            borderRadius: 8,
            color: "#b91c1c",
            fontSize: 13,
          }}
        >
          {error}
        </div>
      )}

      {metrics && (
        <div className="code">
          <pre>{JSON.stringify(metrics, null, 2)}</pre>
        </div>
      )}
    </section>
  );
}
