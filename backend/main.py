import os
import json
import time
import asyncio
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Any, Dict
from contextlib import asynccontextmanager

from sse_starlette.sse import EventSourceResponse
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from document_processor.file_handler import DocumentProcessor
from retriever.builder import RetrieverBuilder
from agents.workflow import AgentWorkflow


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- startup ---
    EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    _load_builtin_docs()
    yield
    # --- shutdown ---
    # optional: cleanup if you want (not required)
    # _RETRIEVER_BY_DOC.clear()

# ----------------------------
# App + CORS
# ----------------------------
app = FastAPI(title="Agentic DocChat API", version="0.2.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # dev-friendly (tighten in prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Pipeline singletons
# ----------------------------
processor = DocumentProcessor()
retriever_builder = RetrieverBuilder()
workflow = AgentWorkflow()

# ----------------------------
# Built-in docs (dropdown-only mode)
# ----------------------------
EXAMPLES_DIR = Path("examples")

_DOC_IDS: List[str] = []
_DOC_PATHS: Dict[str, str] = {}              # doc_id -> absolute path
_RETRIEVER_BY_DOC: Dict[str, Any] = {}       # doc_id -> retriever
_DOC_FP: Dict[str, str] = {}                 # doc_id -> fingerprint (mtime+size path)


# ----------------------------
# Models (FastAPI uses these to render Swagger UI)
# ----------------------------
class AskRequest(BaseModel):
    question: str = Field(..., min_length=1, description="User question")
    doc_id: str = Field(..., min_length=1, description="Selected built-in PDF (filename from /api/docs)")
    top_k_sources: int = Field(default=5, ge=0, le=50, description="How many source chunks to return")


class SourceItem(BaseModel):
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AskResponse(BaseModel):
    question: str
    is_relevant: Optional[bool] = None
    draft_answer: Optional[str] = None
    verification_report: Optional[str] = None
    sources: List[SourceItem] = Field(default_factory=list)


# ----------------------------
# Helpers
# ----------------------------
@dataclass
class LocalFile:
    name: str


def _fingerprint_path(path: str) -> str:
    """
    Fast fingerprint (safe enough for demo):
    changes when file changes on disk (mtime/size), and includes path.
    """
    ap = os.path.abspath(path)
    st = os.stat(ap)
    h = hashlib.sha256()
    h.update(ap.encode("utf-8"))
    h.update(str(st.st_mtime_ns).encode("utf-8"))
    h.update(str(st.st_size).encode("utf-8"))
    return h.hexdigest()


def _load_builtin_docs() -> None:
    """
    Build the dropdown list from examples/*.pdf.
    This does NOT build retrievers yet.
    """
    global _DOC_IDS, _DOC_PATHS, _DOC_FP

    pdfs = sorted(EXAMPLES_DIR.glob("*.pdf"))
    _DOC_IDS = [p.name for p in pdfs]
    _DOC_PATHS = {p.name: str(p.resolve()) for p in pdfs}
    _DOC_FP = {p.name: _fingerprint_path(str(p.resolve())) for p in pdfs}


def _ensure_doc_retriever(doc_id: str):
    """
    Build or reuse a retriever for a built-in doc.
    Rebuild automatically if the underlying file changed.
    Uses your DocumentProcessor caching (pkl) and should not interfere with it.
    """
    if doc_id not in _DOC_PATHS:
        raise FileNotFoundError(f"Unknown doc_id: {doc_id}")

    path = _DOC_PATHS[doc_id]
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found on server: {path}")

    fp = _fingerprint_path(path)
    cached = _RETRIEVER_BY_DOC.get(doc_id)

    # Reuse only if file hasn't changed
    if cached is not None and _DOC_FP.get(doc_id) == fp:
        return cached

    # Rebuild
    files = [LocalFile(name=path)]
    chunks = processor.process(files)  # uses your file_handler caching internally

    # Tag metadata (helps your Sources panel + filtering)
    for c in chunks:
        c.metadata = c.metadata or {}
        c.metadata.update({"doc_id": doc_id, "source": doc_id})

    # retriever = retriever_builder.build_hybrid_retriever(chunks)
    retriever = retriever_builder.build_hybrid_retriever(chunks, collection_name=doc_id)

    _RETRIEVER_BY_DOC[doc_id] = retriever
    _DOC_FP[doc_id] = fp
    return retriever



# ----------------------------
# Routes
# ----------------------------
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/api/docs")
def list_docs():
    """
    Returns the list of built-in PDFs (dropdown choices).
    Frontend should call this and allow user to pick doc_id.
    """
    # Refresh list in case you changed files while server is running
    _load_builtin_docs()
    return {"docs": _DOC_IDS}


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest):
    question = payload.question.strip()
    doc_id = payload.doc_id.strip()
    top_k_sources = payload.top_k_sources

    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")
    if not doc_id:
        raise HTTPException(status_code=400, detail="Missing 'doc_id'")

    try:
        retriever = _ensure_doc_retriever(doc_id)
        state = workflow.full_pipeline(question=question, retriever=retriever)

        docs = state.get("documents") or []
        sources = [
            SourceItem(
                content=d.page_content,
                metadata=getattr(d, "metadata", {}) or {},
            )
            for d in docs[:top_k_sources]
        ]

        return AskResponse(
            question=state.get("question", question),
            is_relevant=state.get("is_relevant"),
            draft_answer=state.get("draft_answer"),
            verification_report=state.get("verification_report"),
            sources=sources,
        )

    except FileNotFoundError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ask/stream")
async def ask_stream(question: str, doc_id: str, top_k_sources: int = 5):
    """
    SSE endpoint for the Next.js "Agent Trace" UI.

    Frontend should call:
      GET /api/ask/stream?question=...&doc_id=<filename.pdf>&top_k_sources=5
    """
    question = (question or "").strip()
    doc_id = (doc_id or "").strip()

    if not question:
        raise HTTPException(status_code=400, detail="Missing 'question'")
    if not doc_id:
        raise HTTPException(status_code=400, detail="Missing 'doc_id'")

    async def emit(agent: str, status: str, **extra):
        payload = {"agent": agent, "status": status, **extra}
        return {"event": "agent", "data": json.dumps(payload)}

    async def event_gen():
        t0 = time.perf_counter()

        # 1) relevance (placeholder event; your workflow does its own relevance step)
        yield await emit("relevance", "running")
        await asyncio.sleep(0.05)
        yield await emit(
            "relevance",
            "done",
            summary="Relevance check complete",
            ms=int((time.perf_counter() - t0) * 1000),
        )

        # 2) retrieval (build or reuse doc retriever)
        yield await emit("retrieval", "running")
        t_retr = time.perf_counter()
        try:
            retriever = await asyncio.to_thread(_ensure_doc_retriever, doc_id)
        except Exception as e:
            yield await emit("retrieval", "error", summary=str(e))
            return

        yield await emit(
            "retrieval",
            "done",
            summary=f"Retriever ready for {doc_id}",
            ms=int((time.perf_counter() - t_retr) * 1000),
        )

        # 3) research + 4) verify (inside full_pipeline)
        yield await emit("research", "running")
        yield await emit("verify", "running")

        t_pipe = time.perf_counter()
        try:
            state = await asyncio.to_thread(workflow.full_pipeline, question, retriever)
        except Exception as e:
            yield await emit("research", "error", summary="Pipeline failed")
            yield {"event": "final", "data": json.dumps({"error": str(e)})}
            return

        docs = state.get("documents") or []
        sources = [
            {"content": d.page_content, "metadata": getattr(d, "metadata", {}) or {}}
            for d in docs[:top_k_sources]
        ]

        draft = state.get("draft_answer")
        verification = state.get("verification_report")

        yield await emit(
            "research",
            "done",
            summary="Draft created",
            ms=int((time.perf_counter() - t_pipe) * 1000),
            preview=(draft[:220] + "…") if isinstance(draft, str) and len(draft) > 220 else draft,
        )

        yield await emit(
            "verify",
            "done",
            summary="Verification complete",
            ms=int((time.perf_counter() - t_pipe) * 1000),
        )

        yield {
            "event": "final",
            "data": json.dumps(
                {
                    "question": state.get("question", question),
                    "draft_answer": draft,
                    "verification_report": verification,
                    "is_relevant": state.get("is_relevant"),
                    "sources": sources,
                }
            ),
        }

    return EventSourceResponse(
        event_gen(),
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )

