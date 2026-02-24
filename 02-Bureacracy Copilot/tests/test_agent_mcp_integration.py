"""Tests for agent MCP-over-HTTP tooling integration."""

from __future__ import annotations

from typing import Any, Dict
from pathlib import Path
import sys
import types


PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKEND_DIR = PROJECT_ROOT / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from src.agent import mcp_tools
from src.agent import tools as agent_tools


class _FakeMCPCallResult:
    """Minimal call result stub exposing model_dump for parser tests."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        """Store fake MCP payload for parser-focused tests.

        Args:
            payload (Dict[str, Any]): Raw payload returned by `model_dump`.

        Returns:
            None.
        """
        self._payload = payload

    def model_dump(self, mode: str = "json", exclude_none: bool = True) -> Dict[str, Any]:
        """Return stored payload to emulate MCP SDK call result behavior.

        Args:
            mode (str): Serialization mode (unused in stub).
            exclude_none (bool): Exclude-none flag (unused in stub).

        Returns:
            Dict[str, Any]: Stored fake payload.
        """
        del mode, exclude_none
        return self._payload


def test_retrieve_context_tool_aggregates_search_and_fetch(monkeypatch):
    """`retrieve_context` should compose search + fetch responses into context/citations."""

    def fake_translate(question: str, target_language: str) -> Any:
        """Disable translation variants for this focused aggregation test.

        Args:
            question (str): Original question text.
            target_language (str): Target language code.

        Returns:
            Any: `None` to keep only the original query variant.
        """
        del question, target_language
        return None

    def fake_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Return deterministic tool payloads for search/fetch composition tests.

        Args:
            tool_name (str): MCP tool name being invoked.
            arguments (Dict[str, Any]): Tool argument payload.

        Returns:
            Dict[str, Any]: Fake tool response payload.
        """
        if tool_name == "search":
            assert arguments["query"] == "registration deadline"
            return {
                "total": 1,
                "hits": [
                    {
                        "id": "chunk-abc",
                        "snippet": "Anmeldung within 14 days",
                        "score": 0.1,
                        "metadata": {"source_id": "src-1"},
                    }
                ],
            }
        if tool_name == "fetch":
            assert arguments["id"] == "chunk-abc"
            return {
                "found": True,
                "id": "chunk-abc",
                "text": "Anmeldung should be completed within 14 days after moving.",
                "metadata": {
                    "source_id": "src-1",
                    "source_type": "text",
                    "source_name": "Berlin Guide",
                    "city": "Berlin",
                },
            }
        raise AssertionError(f"Unexpected tool call: {tool_name}")

    monkeypatch.setattr(mcp_tools, "_translate_query_with_llm", fake_translate)
    monkeypatch.setattr(mcp_tools, "_call_mcp_tool", fake_call)

    result = mcp_tools.retrieve_context_tool.invoke(
        {
            "question": "registration deadline",
            "top_k": 3,
            "city": "Berlin",
        }
    )

    assert result["retrieved_count"] == 1
    assert result["citations"][0]["chunk_id"] == "chunk-abc"
    assert "14 days" in result["context"]


def test_get_default_tools_includes_mcp_tools(monkeypatch):
    """Agent toolset should include MCP-backed KB tools by default."""

    fake_module = types.ModuleType("src.agent.mcp_tools")
    fake_module.get_mcp_tools = lambda: ["mcp_ingest", "mcp_retrieve"]
    monkeypatch.setitem(sys.modules, "src.agent.mcp_tools", fake_module)

    selected_tools = agent_tools.get_default_tools()

    assert selected_tools[0] == "mcp_ingest"
    assert selected_tools[1] == "mcp_retrieve"


def test_extract_transport_streams_supports_three_item_tuple():
    """MCP transport context may include an extra third value in newer SDKs."""

    read_stream = object()
    write_stream = object()
    extra_value = object()

    extracted_read, extracted_write = mcp_tools._extract_transport_streams(
        (read_stream, write_stream, extra_value)
    )

    assert extracted_read is read_stream
    assert extracted_write is write_stream


def test_extract_result_payload_unwraps_result_wrapper():
    """Parser should unwrap FastMCP JSON payloads shaped as {'result': {...}}."""

    payload = {
        "isError": False,
        "structuredContent": {
            "result": {
                "total": 1,
                "hits": [
                    {
                        "id": "chunk-1",
                        "snippet": "deadline is 14 days",
                    }
                ],
            }
        },
    }

    parsed = mcp_tools._extract_result_payload(_FakeMCPCallResult(payload))

    assert parsed["total"] == 1
    assert parsed["hits"][0]["id"] == "chunk-1"


def test_retrieve_context_tool_searches_multiple_query_variants(monkeypatch):
    """`retrieve_context` should call MCP search once per generated query variant."""

    call_log = []

    def fake_translate(question: str, target_language: str) -> str:
        """Return deterministic translations for variant-building tests.

        Args:
            question (str): Original question text.
            target_language (str): Target language code.

        Returns:
            str: Deterministic translated query variant.
        """
        assert question == "What is the registration deadline?"
        if target_language == "de":
            return "Was ist die Anmeldefrist?"
        raise AssertionError(f"Unexpected target language: {target_language}")

    def fake_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Capture tool invocations and return deterministic MCP payloads.

        Args:
            tool_name (str): MCP tool name.
            arguments (Dict[str, Any]): Tool call arguments.

        Returns:
            Dict[str, Any]: Fake MCP response payload.
        """
        call_log.append((tool_name, arguments))
        if tool_name == "search":
            query = arguments["query"]
            if query == "What is the registration deadline?":
                return {"total": 1, "hits": [{"id": "chunk-en", "score": 0.40}]}
            if query == "Was ist die Anmeldefrist?":
                return {"total": 1, "hits": [{"id": "chunk-de", "score": 0.20}]}
            raise AssertionError(f"Unexpected query: {query}")
        if tool_name == "fetch":
            chunk_id = arguments["id"]
            return {
                "found": True,
                "id": chunk_id,
                "text": f"text for {chunk_id}",
                "metadata": {
                    "source_id": f"src-{chunk_id}",
                    "source_type": "text",
                    "source_name": f"Source {chunk_id}",
                },
            }
        raise AssertionError(f"Unexpected tool call: {tool_name}")

    monkeypatch.setattr(mcp_tools, "_translate_query_with_llm", fake_translate)
    monkeypatch.setattr(mcp_tools, "_call_mcp_tool", fake_call)

    result = mcp_tools.retrieve_context_tool.invoke(
        {
            "question": "What is the registration deadline?",
            "top_k": 5,
        }
    )

    search_queries = [arguments["query"] for name, arguments in call_log if name == "search"]
    assert search_queries == [
        "What is the registration deadline?",
        "Was ist die Anmeldefrist?",
    ]
    assert result["retrieved_count"] == 2


def test_retrieve_context_tool_dedupes_by_chunk_id_using_best_score(monkeypatch):
    """`retrieve_context` should dedupe duplicate ids and keep lower-distance score hit."""

    def fake_translate(question: str, target_language: str) -> str:
        """Return deterministic German translation for dedupe scenario.

        Args:
            question (str): Original question text.
            target_language (str): Target language code.

        Returns:
            str: Deterministic translation.
        """
        assert question == "Where is the office?"
        assert target_language == "de"
        return "Wo ist das Amt?"

    fetch_calls = []

    def fake_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Return duplicate hits across variants and capture fetch ids.

        Args:
            tool_name (str): MCP tool name.
            arguments (Dict[str, Any]): MCP call arguments.

        Returns:
            Dict[str, Any]: Fake MCP response payload.
        """
        if tool_name == "search":
            query = arguments["query"]
            if query == "Where is the office?":
                return {
                    "hits": [
                        {"id": "chunk-1", "score": 0.50},
                        {"id": "chunk-2", "score": 0.30},
                    ]
                }
            if query == "Wo ist das Amt?":
                return {
                    "hits": [
                        {"id": "chunk-1", "score": 0.10},
                    ]
                }
            raise AssertionError(f"Unexpected query: {query}")

        if tool_name == "fetch":
            fetch_calls.append(arguments["id"])
            return {
                "found": True,
                "id": arguments["id"],
                "text": f"context {arguments['id']}",
                "metadata": {
                    "source_id": f"src-{arguments['id']}",
                    "source_type": "text",
                    "source_name": f"Source {arguments['id']}",
                },
            }

        raise AssertionError(f"Unexpected tool call: {tool_name}")

    monkeypatch.setattr(mcp_tools, "_translate_query_with_llm", fake_translate)
    monkeypatch.setattr(mcp_tools, "_call_mcp_tool", fake_call)

    result = mcp_tools.retrieve_context_tool.invoke(
        {
            "question": "Where is the office?",
            "top_k": 2,
        }
    )

    assert fetch_calls == ["chunk-1", "chunk-2"]
    assert result["retrieved_count"] == 2
    first_citation = result["citations"][0]
    assert first_citation["chunk_id"] == "chunk-1"
    assert first_citation["score"] == 0.10


def test_retrieve_context_tool_falls_back_when_translation_raises(monkeypatch):
    """`retrieve_context` should continue with original query when translation fails."""

    search_calls = []

    def raising_translate(question: str, target_language: str) -> str:
        """Raise deterministic error to validate translation fallback logic.

        Args:
            question (str): Original question text.
            target_language (str): Target language code.

        Returns:
            str: Never returns because it always raises.

        Raises:
            RuntimeError: Always raised for deterministic fallback testing.
        """
        raise RuntimeError(f"translation unavailable for {target_language}")

    def fake_call(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Return deterministic MCP payload and capture search invocations.

        Args:
            tool_name (str): MCP tool name.
            arguments (Dict[str, Any]): MCP call arguments.

        Returns:
            Dict[str, Any]: Fake MCP response payload.
        """
        if tool_name == "search":
            search_calls.append(arguments["query"])
            return {"hits": [{"id": "chunk-only", "score": 0.42}]}
        if tool_name == "fetch":
            return {
                "found": True,
                "id": "chunk-only",
                "text": "single hit context",
                "metadata": {
                    "source_id": "src-only",
                    "source_type": "text",
                    "source_name": "Single Source",
                },
            }
        raise AssertionError(f"Unexpected tool call: {tool_name}")

    monkeypatch.setattr(mcp_tools, "_translate_query_with_llm", raising_translate)
    monkeypatch.setattr(mcp_tools, "_call_mcp_tool", fake_call)

    result = mcp_tools.retrieve_context_tool.invoke(
        {
            "question": "What is required?",
            "top_k": 3,
        }
    )

    assert search_calls == ["What is required?"]
    assert result["retrieved_count"] == 1
