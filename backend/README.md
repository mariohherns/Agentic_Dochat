# Agentic DocChat Backend

This backend provides the document Q&A API for the Agentic DocChat app. It uses FastAPI to expose endpoints for listing built-in PDF documents, answering questions, and streaming real-time agent trace updates.

## Key features

- FastAPI REST + SSE backend
- Document processing and chunk-based retrieval
- LangChain + OpenAI model integration
- Built-in PDF document registry under `examples/`
- Source tracing and verification pipeline

## Prerequisites

- Python 3.11+
- A virtual environment is strongly recommended
- OpenAI API key for language model access

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

- `CORS_ORIGINS` — comma-separated list of allowed origins (defaults to `*`)

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
- Response includes the draft answer, verification report, relevance flag, and source chunks.

### Stream ask trace

- `GET /api/ask/stream`
- Query parameters:
  - `question`
  - `doc_id`
  - `top_k_sources` (optional)
- Returns an SSE stream with intermediate agent events.

## Backend structure

- `main.py` — FastAPI app entrypoint, routes, and helper utilities
- `agents/` — agent pipeline components:
  - `research_agent.py` — generates answers from document context
  - `relevance_checker.py` — classifies retrieved content relevance
  - `verification_agent.py` — verifies answer support against context
  - `workflow.py` — orchestrates retrieval, research, and verification
- `document_processor/` — document ingestion and caching logic
- `retriever/` — vector/hybrid retriever builder
- `llm/` — OpenAI API key loading and model setup
- `config/` — application configuration values and settings

## Notes

- The backend uses `.env` and `python-dotenv` to load environment variables.
- Built-in documents are discovered from `backend/examples/*.pdf`.
- Make sure `OPENAI_API_KEY` is valid and has access to the configured OpenAI model.
- For development, keep your virtual environment activated and install updated requirements as needed.

## Troubleshooting

- If the server fails to start, verify the virtual environment is active and `requirements.txt` dependencies are installed.
- If the OpenAI request fails, confirm `OPENAI_API_KEY` is set and valid.
- If built-in docs do not appear, make sure PDF files exist under `backend/examples/`.
