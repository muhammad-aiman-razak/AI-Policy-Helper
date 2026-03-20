"use client";
import React from "react";
import { apiAsk, Citation, ChunkData } from "../lib/api";

/** Replace underscores with spaces for human-readable display. */
function formatTitle(title: string): string {
  return title.replace(/_/g, " ");
}

/** Strip leading markdown heading markers (e.g. "## ") from each line. */
function stripMarkdownHeading(text: string): string {
  return text.replace(/^\s*#{1,6}\s+/gm, "");
}

/** Render inline **bold** markers as <strong> elements. */
function renderInlineBold(text: string): React.ReactNode[] {
  return text.split(/(\*\*[^*]+\*\*)/g).map((seg, i) => {
    if (seg.startsWith("**") && seg.endsWith("**")) {
      return <strong key={i}>{seg.slice(2, -2)}</strong>;
    }
    return seg;
  });
}

/** Render text with basic markdown: headings stripped, bold, and list items. */
function renderChunkMarkdown(raw: string): React.ReactNode {
  const text = stripMarkdownHeading(raw);
  return text.split("\n").map((line, i) => {
    const trimmed = line.trimStart();
    if (trimmed.startsWith("- ")) {
      return (
        <div key={i} style={{ paddingLeft: 12 }}>
          {"• "}
          {renderInlineBold(trimmed.slice(2))}
        </div>
      );
    }
    return (
      <React.Fragment key={i}>
        {i > 0 && "\n"}
        {renderInlineBold(line)}
      </React.Fragment>
    );
  });
}

/** Show section only when it differs from title to avoid redundancy. */
function formatSection(
  title: string,
  section: string | null | undefined
): string {
  if (!section || section === title) return "";
  return ` \u2014 ${section}`;
}

/** Strip "Document:" prefix the LLM sometimes echoes from the context block. */
function stripDocumentPrefix(text: string): string {
  return text.replace(/\bDocument:\s*/g, "");
}

/**
 * Render answer text with bolded cited titles and inline **bold** markdown.
 * Also strips any "Document:" prefix the LLM may echo.
 */
function renderAnswerText(text: string, citations?: Citation[]): React.ReactNode {
  const cleaned = stripDocumentPrefix(text);

  if (!citations || citations.length === 0) {
    return renderInlineBold(cleaned);
  }

  // Build unique display titles to search for in the answer
  const titles = Array.from(
    new Set(citations.map((c) => formatTitle(c.title)))
  ).sort((a, b) => b.length - a.length); // longest first to avoid partial matches

  if (titles.length === 0) return renderInlineBold(cleaned);

  // Escape regex special chars and build alternation pattern
  const escaped = titles.map((t) => t.replace(/[.*+?^${}()|[\]\\]/g, "\\$&"));
  const pattern = new RegExp(`(${escaped.join("|")})`, "g");

  return cleaned.split(pattern).map((segment, i) => {
    if (titles.includes(segment)) {
      return <strong key={i}>{segment}</strong>;
    }
    // Process **bold** markdown in non-title segments
    return <React.Fragment key={i}>{renderInlineBold(segment)}</React.Fragment>;
  });
}

interface Message {
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  chunks?: ChunkData[];
}

export default function Chat() {
  const [messages, setMessages] = React.useState<Message[]>([]);
  const [query, setQuery] = React.useState("");
  const [loading, setLoading] = React.useState(false);
  const [expandedChunks, setExpandedChunks] = React.useState<
    Record<string, boolean>
  >({});

  const toggleChunk = (messageIndex: number, chunkIndex: number) => {
    const key = `${messageIndex}-${chunkIndex}`;
    setExpandedChunks((prev) => ({ ...prev, [key]: !prev[key] }));
  };

  const send = async () => {
    if (!query.trim() || loading) return;
    const userMessage: Message = { role: "user", content: query };
    setMessages((prev) => [...prev, userMessage]);
    setQuery("");
    setLoading(true);
    try {
      const res = await apiAsk(query);
      const assistantMessage: Message = {
        role: "assistant",
        content: res.answer,
        citations: res.citations,
        chunks: res.chunks,
      };
      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err: unknown) {
      const errorMessage =
        err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) => [
        ...prev,
        { role: "assistant", content: `Error: ${errorMessage}` },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  };

  return (
    <section className="card" aria-label="Chat">
      <h2>Chat</h2>
      <div
        role="log"
        aria-label="Conversation history"
        aria-live="polite"
        style={{
          maxHeight: 480,
          overflowY: "auto",
          padding: 12,
          border: "1px solid #eee",
          borderRadius: 8,
          marginBottom: 12,
        }}
      >
        {messages.length === 0 && (
          <p style={{ color: "#999", textAlign: "center", margin: "24px 0" }}>
            Ask a question about company policies or products.
          </p>
        )}
        {messages.map((msg, msgIdx) => (
          <article
            key={msgIdx}
            style={{
              margin: 0,
              padding: "12px 0",
              borderBottom: "1px solid #f0f0f0",
            }}
          >
            <div
              style={{
                fontSize: 12,
                fontWeight: 600,
                color: msg.role === "user" ? "#2563eb" : "#059669",
                marginBottom: 4,
              }}
            >
              {msg.role === "user" ? "You" : "Assistant"}
            </div>
            <div style={{ whiteSpace: "pre-wrap", lineHeight: 1.6 }}>
              {msg.role === "assistant"
                ? renderAnswerText(msg.content, msg.citations)
                : msg.content}
            </div>

            {msg.citations && msg.citations.length > 0 && (
              <nav aria-label="Citations" style={{ marginTop: 8 }}>
                {msg.citations.map((citation, citIdx) => {
                  const matchingChunk = msg.chunks?.find(
                    (ch) =>
                      ch.title === citation.title &&
                      ch.section === citation.section
                  );
                  const chunkKey = `${msgIdx}-${citIdx}`;
                  const isExpanded = expandedChunks[chunkKey] || false;
                  return (
                    <div key={citIdx} style={{ display: "inline" }}>
                      <button
                        className="badge"
                        onClick={() => toggleChunk(msgIdx, citIdx)}
                        aria-expanded={isExpanded}
                        aria-label={`Source: ${formatTitle(citation.title)}, ${citation.section || "General"}`}
                        title={citation.section || "General"}
                        style={{
                          cursor: "pointer",
                          border: isExpanded
                            ? "1px solid #6366f1"
                            : "1px solid transparent",
                          background: isExpanded ? "#e0e7ff" : "#eef2ff",
                        }}
                      >
                        {formatTitle(citation.title)}
                        {formatSection(citation.title, citation.section)}
                      </button>
                      {isExpanded && matchingChunk && (
                        <div
                          style={{
                            margin: "6px 0 8px",
                            padding: "10px 12px",
                            background: "#f8fafc",
                            borderLeft: "3px solid #6366f1",
                            borderRadius: 4,
                            fontSize: 13,
                            lineHeight: 1.5,
                          }}
                        >
                          <div style={{ fontWeight: 600, marginBottom: 4 }}>
                            {formatTitle(matchingChunk.title)}
                            {formatSection(
                              matchingChunk.title,
                              matchingChunk.section
                            )}
                          </div>
                          <div style={{ whiteSpace: "pre-wrap" }}>
                            {renderChunkMarkdown(matchingChunk.text)}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </nav>
            )}
          </article>
        ))}
        {loading && (
          <div style={{ padding: "12px 0", color: "#888" }} aria-busy="true">
            Thinking...
          </div>
        )}
      </div>

      <form
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
        style={{ display: "flex", gap: 8 }}
      >
        <input
          placeholder="Ask about policy or products..."
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          disabled={loading}
          aria-label="Question input"
          style={{
            flex: 1,
            padding: 10,
            borderRadius: 8,
            border: "1px solid #ddd",
            opacity: loading ? 0.6 : 1,
          }}
        />
        <button
          type="submit"
          disabled={loading || !query.trim()}
          aria-label="Send question"
          style={{
            padding: "10px 14px",
            borderRadius: 8,
            border: "1px solid #111",
            background: loading ? "#666" : "#111",
            color: "#fff",
            cursor: loading ? "not-allowed" : "pointer",
          }}
        >
          {loading ? "Thinking..." : "Send"}
        </button>
      </form>
    </section>
  );
}
