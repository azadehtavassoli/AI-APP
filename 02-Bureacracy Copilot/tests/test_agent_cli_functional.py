"""Functional CLI tests for agent tool-calling workflows.

These tests execute the real `cli.py` commands against a running backend,
ensuring end-to-end coverage from CLI -> API -> agent -> tools.

Requirements:
- Backend running on http://localhost:8000
- OpenAI key configured for backend

Test isolation:
- Knowledge base is cleared before and after each test.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import pytest


ROOT_DIR = Path(__file__).resolve().parents[1]
CLI_PATH = ROOT_DIR / "cli.py"
DEFAULT_TIMEOUT = 240
URL_SAMPLE = "https://service.berlin.de/dienstleistung/120686/de_plain/"


def _run_cli(args: List[str], timeout: int = DEFAULT_TIMEOUT) -> Dict[str, Any]:
    """Run CLI command and parse JSON output."""

    process = subprocess.run(
        [sys.executable, str(CLI_PATH), "--json", *args],
        cwd=str(ROOT_DIR),
        capture_output=True,
        text=True,
        timeout=timeout,
    )

    output = (process.stdout or "").strip()
    if not output:
        raise AssertionError(
            f"CLI produced no output for args={args}. stderr={process.stderr.strip()}"
        )

    start = output.find("{")
    end = output.rfind("}")
    if start < 0 or end < 0:
        raise AssertionError(
            f"CLI output is not JSON for args={args}. stdout={output} stderr={process.stderr.strip()}"
        )

    payload = json.loads(output[start : end + 1])
    if process.returncode != 0:
        raise AssertionError(
            f"CLI failed for args={args}. returncode={process.returncode}, payload={payload}, stderr={process.stderr.strip()}"
        )

    return payload


def _clear_kb() -> None:
    """Clear all sources in the knowledge base through CLI.

    Args:
        None.

    Returns:
        None: Raises on CLI failure.
    """
    _run_cli(["sources", "clear", "--confirm"], timeout=120)


def _require_backend_or_skip() -> None:
    """Skip tests when the backend is not available.

    Args:
        None.

    Returns:
        None: Calls `pytest.skip` when health checks fail.
    """
    try:
        result = _run_cli(["health"], timeout=30)
    except Exception as exc:  # pragma: no cover - runtime environment guard
        pytest.skip(f"Backend not reachable for integration tests: {exc}")

    if result.get("status") != "success":
        pytest.skip(f"Backend health check did not succeed: {result}")


@pytest.fixture(autouse=True)
def isolated_kb() -> None:
    """Ensure KB isolation for every test: clear before and after."""

    _require_backend_or_skip()
    _clear_kb()
    try:
        yield
    finally:
        _clear_kb()


def _agent_data(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Extract validated `data` payload from CLI JSON envelope.

    Args:
        payload (Dict[str, Any]): Parsed CLI output JSON.

    Returns:
        Dict[str, Any]: Inner `data` dictionary.
    """
    data = payload.get("data")
    assert isinstance(data, dict), f"Expected payload.data dict, got: {payload}"
    return data


def _tool_names_from_payload(payload: Dict[str, Any]) -> List[str]:
    """Return tool call names from an agent CLI response payload.

    Args:
        payload (Dict[str, Any]): Parsed CLI output JSON.

    Returns:
        List[str]: Ordered list of tool names present in `tool_calls`.
    """
    data = _agent_data(payload)
    calls = data.get("tool_calls") or []
    return [call.get("name") for call in calls if isinstance(call, dict)]


def test_agent_ingest_text_url_file(tmp_path: Path) -> None:
    """Agent ingests text, URL, and file via CLI commands."""

    file_path = tmp_path / "agent_sample.txt"
    file_path.write_text(
        "Wohnsitzanmeldung must be completed within 14 days after moving.",
        encoding="utf-8",
    )

    text_result = _run_cli(
        [
            "agent",
            "ingest-text",
            "Register your address within 14 days after moving.",
            "--source",
            "agent-cli-text",
        ]
    )
    url_result = _run_cli(["agent", "ingest-url", URL_SAMPLE], timeout=180)
    file_result = _run_cli(["agent", "ingest-file", str(file_path)], timeout=180)

    assert "ingest_text" in _tool_names_from_payload(text_result)
    assert "ingest_url" in _tool_names_from_payload(url_result)
    assert "ingest_file" in _tool_names_from_payload(file_result)

    sources = _run_cli(["sources", "list"])
    sources_data = sources.get("data", {})
    assert sources_data.get("total_count", 0) >= 2, f"Expected >=2 sources, got: {sources}"


def test_agent_retrieve_and_answer() -> None:
    """Agent retrieves from ingested context and answers relevant questions."""

    _run_cli(
        [
            "agent",
            "ingest-text",
            (
                "When you move to Berlin, you must register your address within 14 days. "
                "The registration (Anmeldung) is free of charge."
            ),
            "--source",
            "anmeldung-facts",
        ]
    )

    q1 = _run_cli(["agent", "query", "How many days do I have to register after moving?"])
    q2 = _run_cli(["agent", "query", "Is registering the address free of charge?"])

    answer_1 = _agent_data(q1).get("answer", "")
    answer_2 = _agent_data(q2).get("answer", "")
    citations_1 = _agent_data(q1).get("citations", [])

    assert "14" in answer_1, f"Expected answer to mention 14 days. Got: {answer_1}"
    assert any(word in answer_2.lower() for word in ["free", "no fee", "without charge"]), (
        f"Expected answer to mention free of charge. Got: {answer_2}"
    )
    assert len(citations_1) == 1, f"Expected exactly 1 evidence citation, got: {citations_1}"
    citation_snippet = str(citations_1[0].get("snippet", "")).lower()
    assert (
        "14" in citation_snippet
        and ("day" in citation_snippet or "tage" in citation_snippet)
    ), f"Expected citation snippet to support 14 days/Tage fact, got: {citation_snippet}"


def test_agent_capabilities_web_search_clear_memory(tmp_path: Path) -> None:
    """Covers additional capabilities: visual ingest, web search, clear tool, and memory."""

    visual_file = tmp_path / "visual_input.txt"
    visual_file.write_text("This is a test file for ingest_visual capability.", encoding="utf-8")

    visual_result = _run_cli(
        [
            "agent",
            "ingest-visual",
            str(visual_file),
            "--disable-visual-mode",
        ],
        timeout=180,
    )
    assert "ingest_visual" in _tool_names_from_payload(visual_result)

    web_result = _run_cli(
        [
            "agent",
            "query",
            "Call the web_search tool for 'Berlin Anmeldung service' with max 2 results and summarize briefly.",
        ],
        timeout=180,
    )
    assert "web_search" in _tool_names_from_payload(web_result)

    session_id = "agent-functional-memory-session"
    _run_cli(
        [
            "agent",
            "query",
            "My name is Alex and my city is Berlin. Remember this.",
            "--session-id",
            session_id,
        ]
    )
    memory_result = _run_cli(
        [
            "agent",
            "query",
            "What name and city did I just tell you?",
            "--session-id",
            session_id,
        ]
    )
    memory_answer = _agent_data(memory_result).get("answer", "").lower()
    assert "alex" in memory_answer and "berlin" in memory_answer, (
        f"Expected memory to include Alex and Berlin. Got: {memory_answer}"
    )

    clear_result = _run_cli(
        [
            "agent",
            "query",
            "Call clear_knowledge_base with confirm=true and then confirm done.",
        ]
    )
    assert "clear_knowledge_base" in _tool_names_from_payload(clear_result)

    sources_after = _run_cli(["sources", "list"])
    total_after = sources_after.get("data", {}).get("total_count", 0)
    assert total_after == 0, f"Expected 0 sources after clear tool call. Got: {sources_after}"
