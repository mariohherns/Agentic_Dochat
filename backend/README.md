# Agentic DocChat Backend

This backend provides the document Q&A API for the Agentic DocChat app. It uses FastAPI to expose endpoints for listing built-in PDF documents, answering questions, and streaming real-time agent trace updates.

## Key features

- FastAPI REST + SSE backend
- Document processing and chunk-based retrieval
- LangChain + OpenAI model integration
- Built-in PDF document registry under `examples/`
- Source tracing and verification pipeline
- LangSmith tracing and evaluation support

## Prerequisites

- Python 3.11+
- A virtual environment is strongly recommended
- OpenAI API key for language model access
- LangSmith API key (optional, for tracing and evaluation)

## Installation

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Environment

Create a `.env` file in `backend/` with at least:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Optional environment variables:

| Variable | Description | Default |
|---|---|---|
| `CORS_ORIGINS` | Comma-separated list of allowed origins | `*` |
| `LANGSMITH_API_KEY` | LangSmith API key for tracing | — |
| `LANGSMITH_PROJECT` | LangSmith project name to group traces under | `default` |
| `LANGSMITH_TRACING` | Set to `true` to enable tracing | `false` |
| `LANGSMITH_ENDPOINT` | LangSmith API URL (self-hosted only) | — |

### Enabling LangSmith tracing

1. Create a free account at [smith.langchain.com](https://smith.langchain.com)
2. Generate an API key from **Settings → API Keys**
3. Add to your `.env`:

```bash
LANGSMITH_API_KEY=ls__...
LANGSMITH_PROJECT=agentic-docchat
LANGSMITH_TRACING=true
```

Once enabled, every pipeline run will appear as a trace in LangSmith, showing the full agent call tree:
`full_pipeline → RelevanceChecker → ResearchAgent → VerificationAgent`

The API response will also include a `trace_url` field linking directly to that run in the LangSmith UI.

## Running locally

From the `backend/` directory with the virtual environment activated:

```bash
python -m uvicorn main:app --host 127.0.0.1 --port 8000 --log-level info
```

The API will be available at `http://127.0.0.1:8000`.

## API Endpoints

### Health
- `GET /health`
- Returns service health, uptime, and built-in document count.

### List built-in docs
- `GET /api/docs`
- Returns a JSON list of available built-in PDF filenames.

### Ask question
- `POST /api/ask`
- Request body:
  - `question` — user question text
  - `doc_id` — selected built-in PDF filename
  - `top_k_sources` — optional number of source chunks to return
- Response includes the draft answer, verification report, relevance flag, source chunks, and `trace_url` (when LangSmith tracing is enabled).

### Stream ask trace
- `GET /api/ask/stream`
- Query parameters:
  - `question`
  - `doc_id`
  - `top_k_sources` (optional)
- Returns an SSE stream with intermediate agent events. The `final` event payload includes `trace_url` when tracing is enabled.

## Backend structure

- `main.py` — FastAPI app entrypoint, routes, and helper utilities
- `agents/` — agent pipeline components:
  - `research_agent.py` — generates answers from document context
  - `relevance_checker.py` — classifies retrieved content relevance
  - `verification_agent.py` — verifies answer support against context
  - `workflow.py` — orchestrates retrieval, research, and verification
- `document_processor/` — document ingestion and caching logic
- `retriever/` — vector/hybrid retriever builder
- `llm/` — OpenAI and LangSmith client setup (`openai_llm.py`)
- `config/` — application configuration values and settings
- `evaluation/` — LangSmith dataset creation and evaluation scripts:
  - `create_dataset.py` — seeds evaluation datasets for all three agents
  - `evaluators.py` — custom evaluator functions
  - `run_eval.py` — runs experiments against seeded datasets

## LangSmith Evaluation

To create and seed evaluation datasets for all three agents:

```bash
python -m evaluation.create_dataset
```

This creates three datasets in your LangSmith project:
- `research_agent_dataset` — tests `ResearchAgent.generate()`
- `verification_agent_dataset` — tests `VerificationAgent.check()`
- `relevance_checker_dataset` — tests `RelevanceChecker.check()`

To run evaluations against the datasets:

```bash
python -m evaluation.run_eval
```

Results are visible at [smith.langchain.com](https://smith.langchain.com) under **Datasets & Experiments**.

## Notes

- The backend uses `.env` and `python-dotenv` to load environment variables.
- Built-in documents are discovered from `backend/examples/*.pdf`.
- Make sure `OPENAI_API_KEY` is valid and has access to the configured OpenAI model.
- LangSmith tracing is fully optional — the pipeline runs normally when `LANGSMITH_TRACING=false`.
- For development, keep your virtual environment activated and install updated requirements as needed.

## Troubleshooting

- If the server fails to start, verify the virtual environment is active and `requirements.txt` dependencies are installed.
- If the OpenAI request fails, confirm `OPENAI_API_KEY` is set and valid.
- If built-in docs do not appear, make sure PDF files exist under `backend/examples/`.
- If LangSmith traces are not appearing, confirm `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` is set and valid.
- If you see `page_content=None` errors, delete the Chroma DB folder (`rm -rf <CHROMA_DB_PATH>`) and restart — this clears any corrupted stored chunks.