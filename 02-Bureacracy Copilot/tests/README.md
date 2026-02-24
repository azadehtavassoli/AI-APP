# Tests

This directory contains backend and CLI integration tests for Bureaucracy Copilot.

## Available Tests

### Agent CLI Functional Tests

`test_agent_cli_functional.py` validates the LangChain agent via `cli.py` end-to-end.

Coverage:
- agent ingest via CLI: text, URL, file
- agent retrieval and answering from ingested context
- agent web search tool call
- agent visual ingestion path (`ingest-visual`)
- agent memory/session continuity

Important behavior:
- Each test clears the knowledge base **before and after** execution.

### Backend API Smoke Tests

`test_backend_smoke.py` validates core REST endpoints (`/api/health`, ingestion, query, sources).

## Running Tests

### Prerequisites
```powershell
pip install pytest
cd backend
python -m uvicorn main:app --reload
```

### Run Agent CLI Functional Tests
```powershell
pytest tests/test_agent_cli_functional.py -v
```

### Run Backend Smoke Tests
```powershell
python tests/test_backend_smoke.py
```

### Run Full Suite
```powershell
pytest tests/ -v
```

## Notes

- Tests are integration-heavy and require a running backend.
- Network-dependent tests (URL ingestion/web search) may be slower.
