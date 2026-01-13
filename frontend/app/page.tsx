"use client";

import React, { useEffect, useMemo, useRef, useState } from "react";

/**
 * Updated for built-in dropdown docs (no file paths).
 * - Loads docs from GET /api/docs
 * - Streams from GET /api/ask/stream?question=...&doc_id=...&top_k_sources=...
 */

type AgentName = "relevance" | "retrieval" | "research" | "verify";

type AgentEvent = {
  agent: AgentName;
  status: "idle" | "running" | "done" | "error";
  summary?: string;
  ms?: number;
  details?: any;
};

type SourceItem = {
  content: string;
  metadata: Record<string, any>;
};

type FinalPayload = {
  question: string;
  is_relevant?: boolean | null;
  draft_answer?: string | null;
  verification_report?: string | null;
  sources?: SourceItem[];
  error?: string;
};

const AGENTS: { key: AgentName; label: string; description: string }[] = [
  {
    key: "relevance",
    label: "Relevance Agent",
    description: "Checks whether the documents contain material related to the question.",
  },
  {
    key: "retrieval",
    label: "Retriever",
    description: "Fetches the most relevant chunks from the vector/hybrid retriever.",
  },
  {
    key: "research",
    label: "Research Agent",
    description: "Drafts an answer grounded in retrieved sources.",
  },
  {
    key: "verify",
    label: "Verification Agent",
    description: "Checks support, contradictions, and relevance against context.",
  },
];

function badgeClass(status: AgentEvent["status"]) {
  switch (status) {
    case "running":
      return "bg-blue-50 text-blue-700 ring-blue-200";
    case "done":
      return "bg-green-50 text-green-700 ring-green-200";
    case "error":
      return "bg-red-50 text-red-700 ring-red-200";
    default:
      return "bg-gray-50 text-gray-700 ring-gray-200";
  }
}

function Badge({ status }: { status: AgentEvent["status"] }) {
  return (
    <span
      className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ring-1 ring-inset ${badgeClass(
        status
      )}`}
    >
      {status.toUpperCase()}
    </span>
  );
}

function Spinner() {
  return (
    <span className="inline-block h-4 w-4 animate-spin rounded-full border-2 border-gray-300 border-t-transparent" />
  );
}

function safeJsonParse(s: string) {
  try {
    return JSON.parse(s);
  } catch {
    return null;
  }
}

export default function Page() {
  const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

  const [question, setQuestion] = useState<string>("");
  const [topKSources, setTopKSources] = useState<number>(5);

  // NEW: built-in docs dropdown
  const [docs, setDocs] = useState<string[]>([]);
  const [docId, setDocId] = useState<string>("");

  const [events, setEvents] = useState<Record<AgentName, AgentEvent>>(() => ({
    relevance: { agent: "relevance", status: "idle" },
    retrieval: { agent: "retrieval", status: "idle" },
    research: { agent: "research", status: "idle" },
    verify: { agent: "verify", status: "idle" },
  }));

  const [finalResult, setFinalResult] = useState<FinalPayload | null>(null);
  const [errorMsg, setErrorMsg] = useState<string>("");
  const [isStreaming, setIsStreaming] = useState<boolean>(false);

  const eventSourceRef = useRef<EventSource | null>(null);

  // load docs on mount
  useEffect(() => {
    let mounted = true;

    (async () => {
      try {
        const res = await fetch(`${API_BASE}/api/docs`);
        if (!res.ok) throw new Error(`Failed to load docs: HTTP ${res.status}`);
        const data = (await res.json()) as { docs?: string[] };
        const list = Array.isArray(data?.docs) ? data.docs : [];
        if (!mounted) return;
        setDocs(list);
        // Auto-select first doc if none chosen
        if (!docId && list.length > 0) setDocId(list[0]);
      } catch (e: any) {
        if (!mounted) return;
        setErrorMsg(e?.message || "Failed to load built-in documents.");
      }
    })();

    return () => {
      mounted = false;
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [API_BASE]);

  function resetRun() {
    setErrorMsg("");
    setFinalResult(null);
    setEvents({
      relevance: { agent: "relevance", status: "idle" },
      retrieval: { agent: "retrieval", status: "idle" },
      research: { agent: "research", status: "idle" },
      verify: { agent: "verify", status: "idle" },
    });
  }

  function stopStream() {
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
      eventSourceRef.current = null;
    }
    setIsStreaming(false);
  }

  async function copyToClipboard(text: string) {
    try {
      await navigator.clipboard.writeText(text);
    } catch {
      // no-op
    }
  }

  function startStream() {
    resetRun();

    const q = question.trim();
    if (!q) {
      setErrorMsg("Please enter a question.");
      return;
    }
    if (!docId) {
      setErrorMsg("Please select a document.");
      return;
    }

    // Mark first step as running
    setEvents((prev) => ({
      ...prev,
      relevance: { ...prev.relevance, status: "running" },
    }));
    setIsStreaming(true);

    // UPDATED: doc_id instead of file_paths
    const params = new URLSearchParams({
      question: q,
      doc_id: docId,
      top_k_sources: String(topKSources),
    });

    const url = `${API_BASE}/api/ask/stream?${params.toString()}`;

    const es = new EventSource(url);
    eventSourceRef.current = es;

    es.addEventListener("agent", (evt: MessageEvent) => {
      const data = safeJsonParse(evt.data);
      if (!data?.agent) return;

      const agent = data.agent as AgentName;
      const status = (data.status || "running") as AgentEvent["status"];

      setEvents((prev) => ({
        ...prev,
        [agent]: {
          agent,
          status,
          summary: data.summary,
          ms: data.ms,
          details: data,
        },
      }));
    });

    es.addEventListener("final", (evt: MessageEvent) => {
      const data = safeJsonParse(evt.data) as FinalPayload | null;
      if (data?.error) {
        setErrorMsg(data.error);
      } else if (data) {
        setFinalResult(data);
      }
      stopStream();
    });

    es.onerror = () => {
      stopStream();
      setErrorMsg("Stream error (backend unreachable or crashed). Check FastAPI logs.");
      setEvents((prev) => ({
        ...prev,
        verify: { ...prev.verify, status: prev.verify.status === "done" ? "done" : "error" },
      }));
    };
  }

  const overallStatus = useMemo(() => {
    if (isStreaming) return "RUNNING";
    if (errorMsg) return "ERROR";
    if (finalResult) return "DONE";
    return "IDLE";
  }, [isStreaming, errorMsg, finalResult]);

  return (
    <main className="min-h-screen bg-[rgb(var(--bg))] text-[rgb(var(--fg))]">
      <div className="mx-auto max-w-6xl px-4 py-10">
        {/* Header */}
        <header className="flex flex-col gap-2 md:flex-row md:items-end md:justify-between">
          <div>
            <h1 className="text-2xl font-semibold tracking-tight">Agentic DocChat</h1>
            <p className="mt-1 text-sm text-[rgb(var(--fg)/0.65)]">
              A multi-agent RAG demo (relevance → retrieval → research → verification)
            </p>
          </div>

          <div className="flex items-center gap-3">
            <span className="text-sm text-[rgb(var(--fg)/0.65)]">Status:</span>

            <span className="inline-flex items-center gap-2 rounded-full bg-[rgb(var(--card-muted))] px-3 py-1 text-sm font-medium ring-1 ring-inset ring-[rgb(var(--border))]">
              {isStreaming && <Spinner />}
              {overallStatus}
            </span>

            {isStreaming && (
              <button
                onClick={stopStream}
                className="rounded-md bg-[rgb(var(--card))] px-3 py-2 text-sm font-medium ring-1 ring-inset ring-[rgb(var(--border))] hover:bg-[rgb(var(--card-muted))]"
              >
                Stop
              </button>
            )}
          </div>
        </header>

        {/* Grid */}
        <div className="mt-8 grid gap-6 lg:grid-cols-12">
          {/* Left: Inputs */}
          <section className="lg:col-span-5">
            <div className="rounded-xl bg-[rgb(var(--card))] p-5 ring-1 ring-inset ring-[rgb(var(--border))]">
              <h2 className="text-base font-semibold">Ask a question</h2>

              {/* NEW: Document dropdown */}
              <label className="mt-4 block text-sm font-medium text-[rgb(var(--fg)/0.7)]">
                Document
              </label>
              <select
                value={docId}
                onChange={(e) => setDocId(e.target.value)}
                className="mt-2 w-full rounded-lg bg-[rgb(var(--card))] p-2 text-sm ring-1 ring-inset ring-[rgb(var(--border))] focus:outline-none focus:ring-2 focus:ring-[rgb(var(--accent)/0.35)]"
              >
                {docs.length === 0 ? (
                  <option value="">No documents found</option>
                ) : (
                  docs.map((d) => (
                    <option key={d} value={d}>
                      {d}
                    </option>
                  ))
                )}
              </select>

              <label className="mt-4 block text-sm font-medium text-[rgb(var(--fg)/0.7)]">
                Question
              </label>
              <textarea
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                rows={5}
                placeholder="Ask something about the document..."
                className="mt-2 w-full rounded-lg bg-[rgb(var(--card))] p-3 text-sm ring-1 ring-inset ring-[rgb(var(--border))] focus:outline-none focus:ring-2 focus:ring-[rgb(var(--accent)/0.35)]"
              />

              <div className="mt-4 grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-[rgb(var(--fg)/0.7)]">
                    Top K sources
                  </label>
                  <input
                    type="number"
                    min={0}
                    max={50}
                    value={topKSources}
                    onChange={(e) => setTopKSources(parseInt(e.target.value || "5", 10))}
                    className="mt-2 w-full rounded-lg bg-[rgb(var(--card))] p-2 text-sm ring-1 ring-inset ring-[rgb(var(--border))] focus:outline-none focus:ring-2 focus:ring-[rgb(var(--accent)/0.35)]"
                  />
                </div>

                <div className="flex items-end">
                  <button
                    onClick={startStream}
                    disabled={isStreaming || !question.trim() || !docId}
                    className="w-full rounded-lg bg-[rgb(var(--accent))] px-4 py-2 text-sm font-semibold text-white hover:opacity-95 disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    {isStreaming ? "Running..." : "Run agents"}
                  </button>
                </div>
              </div>

              {errorMsg && (
                <p className="mt-4 rounded-lg bg-[rgb(var(--error)/0.12)] p-3 text-sm text-[rgb(var(--error))] ring-1 ring-inset ring-[rgb(var(--error)/0.25)]">
                  {errorMsg}
                </p>
              )}

              <div className="mt-4 text-xs text-[rgb(var(--fg)/0.65)]">
                Backend playground:{" "}
                <a
                  className="underline decoration-[rgb(var(--accent)/0.5)] underline-offset-2 hover:text-[rgb(var(--accent))]"
                  href={`${API_BASE}/docs`}
                  target="_blank"
                  rel="noreferrer"
                >
                  {API_BASE}/docs
                </a>
              </div>
            </div>

            {/* Agent Trace */}
            <div className="mt-6 rounded-xl bg-[rgb(var(--card))] p-5 ring-1 ring-inset ring-[rgb(var(--border))]">
              <div className="flex items-center justify-between">
                <h2 className="text-base font-semibold">Agent Trace</h2>
                <button
                  onClick={resetRun}
                  className="rounded-md bg-[rgb(var(--card))] px-3 py-2 text-xs font-medium ring-1 ring-inset ring-[rgb(var(--border))] hover:bg-[rgb(var(--card-muted))]"
                >
                  Reset
                </button>
              </div>

              <ol className="mt-4 space-y-3">
                {AGENTS.map((a) => {
                  const ev = events[a.key];
                  return (
                    <li
                      key={a.key}
                      className="rounded-lg bg-[rgb(var(--card))] p-3 ring-1 ring-inset ring-[rgb(var(--border))]"
                    >
                      <div className="flex items-center justify-between gap-3">
                        <div>
                          <div className="flex items-center gap-2">
                            <span className="text-sm font-semibold">{a.label}</span>
                            <Badge status={ev.status} />
                            {ev.status === "running" && <Spinner />}
                          </div>
                          <p className="mt-1 text-xs text-[rgb(var(--fg)/0.65)]">{a.description}</p>
                        </div>

                        <div className="text-right">
                          {typeof ev.ms === "number" && (
                            <div className="text-xs text-[rgb(var(--fg)/0.65)]">{ev.ms} ms</div>
                          )}
                        </div>
                      </div>

                      {ev.summary && (
                        <div className="mt-2 text-sm">
                          <span className="font-medium text-[rgb(var(--fg)/0.8)]">Summary:</span>{" "}
                          <span className="text-[rgb(var(--fg)/0.85)]">{ev.summary}</span>
                        </div>
                      )}

                      {ev.details && (
                        <details className="mt-2">
                          <summary className="cursor-pointer text-xs text-[rgb(var(--fg)/0.65)] hover:text-[rgb(var(--fg)/0.85)]">
                            View event payload
                          </summary>
                          <pre className="mt-2 overflow-auto rounded-md bg-[rgb(var(--card-muted))] p-2 text-xs text-[rgb(var(--fg)/0.85)] ring-1 ring-inset ring-[rgb(var(--border))]">
                            {JSON.stringify(ev.details, null, 2)}
                          </pre>
                        </details>
                      )}
                    </li>
                  );
                })}
              </ol>

              <p className="mt-4 text-xs text-[rgb(var(--fg)/0.65)]">
                Note: This shows an auditable trace (tools/sources/steps), not raw chain-of-thought.
              </p>
            </div>
          </section>

          {/* Right: Results */}
          <section className="lg:col-span-7">
            <div className="rounded-xl bg-[rgb(var(--card))] p-5 ring-1 ring-inset ring-[rgb(var(--border))]">
              <div className="flex items-start justify-between gap-3">
                <div>
                  <h2 className="text-base font-semibold">Results</h2>
                  <p className="mt-1 text-xs text-[rgb(var(--fg)/0.65)]">
                    Draft answer + verification report + top sources
                  </p>
                </div>

                {finalResult?.draft_answer && (
                  <button
                    onClick={() => copyToClipboard(finalResult.draft_answer || "")}
                    className="rounded-md bg-[rgb(var(--card))] px-3 py-2 text-xs font-medium ring-1 ring-inset ring-[rgb(var(--border))] hover:bg-[rgb(var(--card-muted))]"
                  >
                    Copy answer
                  </button>
                )}
              </div>

              {/* Draft Answer */}
              <div className="mt-4 rounded-lg bg-[rgb(var(--card))] p-4 ring-1 ring-inset ring-[rgb(var(--border))]">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold">Draft Answer</h3>
                  <div className="text-xs text-[rgb(var(--fg)/0.65)]">
                    Relevant:{" "}
                    <span className="font-medium text-[rgb(var(--fg)/0.85)]">
                      {finalResult?.is_relevant === undefined || finalResult?.is_relevant === null
                        ? "—"
                        : finalResult.is_relevant
                        ? "YES"
                        : "NO"}
                    </span>
                  </div>
                </div>

                <pre className="mt-2 whitespace-pre-wrap text-sm text-[rgb(var(--fg)/0.9)]">
                  {finalResult?.draft_answer || (isStreaming ? "Working..." : "No answer yet.")}
                </pre>
              </div>

              {/* Verification */}
              <div className="mt-4 rounded-lg bg-[rgb(var(--card))] p-4 ring-1 ring-inset ring-[rgb(var(--border))]">
                <div className="flex items-center justify-between">
                  <h3 className="text-sm font-semibold">Verification Report</h3>
                  {finalResult?.verification_report && (
                    <button
                      onClick={() => copyToClipboard(finalResult.verification_report || "")}
                      className="rounded-md bg-[rgb(var(--card-muted))] px-3 py-1.5 text-xs font-medium ring-1 ring-inset ring-[rgb(var(--border))] hover:opacity-90"
                    >
                      Copy
                    </button>
                  )}
                </div>
                <pre className="mt-2 whitespace-pre-wrap text-sm text-[rgb(var(--fg)/0.9)]">
                  {finalResult?.verification_report || (isStreaming ? "Verifying..." : "No report yet.")}
                </pre>
              </div>

              {/* Sources */}
              <div className="mt-4 rounded-lg bg-[rgb(var(--card))] p-4 ring-1 ring-inset ring-[rgb(var(--border))]">
                <h3 className="text-sm font-semibold">Sources</h3>
                <p className="mt-1 text-xs text-[rgb(var(--fg)/0.65)]">
                  Top {topKSources} retrieved chunks (expand to view metadata)
                </p>

                <ol className="mt-3 space-y-3">
                  {(finalResult?.sources || []).length === 0 ? (
                    <li className="text-sm text-[rgb(var(--fg)/0.75)]">
                      {isStreaming ? "Retrieving sources..." : "No sources returned."}
                    </li>
                  ) : (
                    (finalResult?.sources || []).map((s, i) => (
                      <li
                        key={i}
                        className="rounded-lg bg-[rgb(var(--card-muted))] p-3 ring-1 ring-inset ring-[rgb(var(--border))]"
                      >
                        <div className="flex items-center justify-between">
                          <span className="text-xs font-semibold text-[rgb(var(--fg)/0.85)]">
                            Source {i + 1}
                          </span>
                          <details>
                            <summary className="cursor-pointer text-xs text-[rgb(var(--fg)/0.65)] hover:text-[rgb(var(--fg)/0.85)]">
                              metadata
                            </summary>
                            <pre className="mt-2 max-h-48 overflow-auto rounded-md bg-[rgb(var(--card))] p-2 text-xs text-[rgb(var(--fg)/0.9)] ring-1 ring-inset ring-[rgb(var(--border))]">
                              {JSON.stringify(s.metadata, null, 2)}
                            </pre>
                          </details>
                        </div>
                        <pre className="mt-2 whitespace-pre-wrap text-sm text-[rgb(var(--fg)/0.9)]">
                          {s.content}
                        </pre>
                      </li>
                    ))
                  )}
                </ol>
              </div>
            </div>
          </section>
        </div>
      </div>
    </main>
  );
}

