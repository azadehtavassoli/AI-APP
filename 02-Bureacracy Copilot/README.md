# Bureaucracy Copilot (German Bureaucracy Assistant)

AI assistant for German bureaucracy support, built with Streamlit + FastAPI + LangChain/LangGraph + ChromaDB.

It lets users:

- ingest sources (text, PDF, URL, visual documents),
- ask questions,
- receive grounded answers with citations,
- use an optional agent path with tool calling + MCP integration.

---

## Architecture overview

This repo is a small multi-service system:

- **Frontend (Streamlit):** `frontend/app.py`
- **Backend (FastAPI):** `backend/main.py`
- **Knowledge MCP server:** `knowledge_mcp/server.py`

Two query paths exist:

- **Classic RAG**: `POST /api/query` (retrieval + answer with citations)
- **Agentic**: `POST /api/agent/query` (tool calling; may use MCP tools)

---

## Key features

- Ingestion: text / PDF / URL / visual documents
- Vector store: ChromaDB persistent storage
- Retrieval: multilingual query variants (EN/DE/FA) + dedupe/rerank
- Answers: citations + token usage + estimated cost
- Agent runtime: tool calling + MCP-backed retrieval/ingestion tools

---

## Repository structure

```text
backend/
   main.py
   src/
      agent/        # agent runtime + tools + MCP bridge
      api/          # REST + agent routes
      services/     # ingest, retrieval, RAG, vector store
      core/         # settings, logging, rate limiting
      models/       # request/response schemas
frontend/
   app.py
   app/
      client/       # REST API client
      ui/           # Streamlit UI panels + shared styles
knowledge_mcp/
   server.py       # MCP tool server
scripts/
   run_stack.ps1
   stop_stack.ps1
tests/
docs/
```

---

## Setup & run (Windows)

### Prerequisites

- Python 3.11+
- OpenAI API key

### Install

```powershell
python -m venv .venv
\.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `.env` at repo root and set at least:

```env
OPENAI_API_KEY=your_key_here
OPENAI_MODEL=gpt-4.1-mini
```

### Run full stack

```powershell
\.\scripts\run_stack.ps1
```

Services:

- Frontend: `http://127.0.0.1:8501`
- Backend: `http://127.0.0.1:8000`
- MCP endpoint: `http://127.0.0.1:8765/mcp`

Stop services:

```powershell
\.\scripts\stop_stack.ps1
```

---

## Testing

Run targeted tests:

```powershell
python -m pytest tests/test_backend_smoke.py -q
python -m pytest tests/test_agent_mcp_integration.py -q
```

Run all tests:

```powershell
python -m pytest tests -q
```

---

## Evaluation (optional)

This repo includes a reproducible evaluation script for a public Berlin “Wohnsitzanmeldung” resource:

- website: https://service.berlin.de/dienstleistung/120686/de_plain/

Artifacts:

- report: `docs/EVALUATION_RESULTS.md`
- compact run summary: `docs/EVALUATION_WOHNSITZ.md`
- raw output: `docs/evaluation_wohnsitz_results.json`
- script: `scripts/run_wohnsitz_eval.py`

Run evaluation (requires providing the PDF path explicitly):

```powershell
python scripts/run_wohnsitz_eval.py --pdf-path <path-to-info_wohnsitz_-_wohnung_anmelden.pdf>
```

---

## Security and engineering notes

- Input validation on ingestion/query endpoints
- Rate limiting for agent endpoints
- Structured logging and error handling
- Frontend API client kept stable (no signature changes)
