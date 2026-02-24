"""
CLI Usage Examples for AI Agents

This module demonstrates how AI agents can use the CLI to test and validate
backend features programmatically.

Key Capabilities:
- Test backend health
- Ingest documents
- Validate data
- Parse JSON output
- Handle errors

Usage:
    python examples/agent_cli_usage.py
"""

import subprocess
import json
import sys
from typing import Tuple, Dict, Any


class CLIAgent:
    """
    AI Agent Interface to CLI
    
    This class provides a Python interface for AI agents to interact
    with the Bureaucracy Copilot CLI.
    """
    
    def __init__(self, cli_path: str = "cli.py"):
        """
        Initialize CLI Agent
        
        Args:
            cli_path: Path to cli.py file
        """
        self.cli_path = cli_path
        self.python_cmd = sys.executable
    
    def _run_command(self, args: list) -> Tuple[bool, Any]:
        """
        Run CLI command and return result
        
        Args:
            args: CLI arguments
        
        Returns:
            Tuple of (success, data/error)
        """
        cmd = [self.python_cmd, self.cli_path, "--json"] + args
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                # Parse JSON output
                output = result.stdout.strip()
                if output:
                    data = json.loads(output)
                    return True, data.get("data", data)  # Extract data field if present
                return True, None
            else:
                error_output = result.stderr or result.stdout
                try:
                    error_data = json.loads(error_output)
                    return False, error_data.get("message", "Unknown error")
                except:
                    return False, error_output.strip() or "Command failed"
        
        except subprocess.TimeoutExpired:
            return False, "Command timeout"
        except json.JSONDecodeError as e:
            return False, f"JSON parse error: {e}"
        except Exception as e:
            return False, str(e)
    
    def check_health(self) -> Tuple[bool, Dict]:
        """
        Check backend health
        
        Returns:
            Tuple of (success, health_data)
        """
        success, data = self._run_command(["health"])
        if success:
            return True, {
                "status": data.get("status"),
                "version": data.get("version"),
                "openai_configured": data.get("openai_configured")
            }
        return False, {"error": data}
    
    def ingest_text(self, text: str, source_name: str) -> Tuple[bool, Dict]:
        """
        Ingest text document
        
        Args:
            text: Document text
            source_name: Source name
        
        Returns:
            Tuple of (success, result_data)
        """
        success, data = self._run_command([
            "ingest", "text", text, "--source", source_name
        ])
        
        if success:
            return True, {
                "source_id": data.get("source", {}).get("id"),
                "chunks": data.get("source", {}).get("chunks"),
                "characters": data.get("source", {}).get("characters")
            }
        return False, {"error": data}
    
    def ingest_pdf(self, pdf_path: str) -> Tuple[bool, Dict]:
        """
        Ingest PDF document
        
        Args:
            pdf_path: Path to PDF file
        
        Returns:
            Tuple of (success, result_data)
        """
        success, data = self._run_command(["ingest", "pdf", pdf_path])
        
        if success:
            return True, {
                "source_id": data.get("source", {}).get("id"),
                "chunks": data.get("source", {}).get("chunks"),
                "characters": data.get("source", {}).get("characters")
            }
        return False, {"error": data}
    
    def ingest_url(self, url: str) -> Tuple[bool, Dict]:
        """
        Ingest URL
        
        Args:
            url: Web page URL
        
        Returns:
            Tuple of (success, result_data)
        """
        success, data = self._run_command(["ingest", "url", url])
        
        if success:
            return True, {
                "source_id": data.get("source", {}).get("id"),
                "chunks": data.get("source", {}).get("chunks"),
                "characters": data.get("source", {}).get("characters"),
                "truncated": data.get("source", {}).get("metadata", {}).get("truncated", False)
            }
        return False, {"error": data}
    
    def list_sources(self) -> Tuple[bool, list]:
        """
        List all sources
        
        Returns:
            Tuple of (success, sources_list)
        """
        success, data = self._run_command(["sources", "list"])
        
        if success:
            # Data is wrapped in success response, extract sources list
            if isinstance(data, dict) and "sources" in data:
                return True, data["sources"]
            elif isinstance(data, list):
                # Backwards compatibility if data is directly a list
                return True, data
            return True, []
        return False, []
    
    def clear_knowledge_base(self) -> Tuple[bool, str]:
        """
        Clear all documents
        
        Returns:
            Tuple of (success, message)
        """
        success, data = self._run_command(["sources", "clear", "--confirm"])
        
        if success:
            return True, data.get("message", "Cleared")
        return False, data


# Example usage functions for agents

def example_health_check():
    """Example: Check backend health"""
    print("=== Health Check Example ===")
    
    agent = CLIAgent()
    success, data = agent.check_health()
    
    if success:
        print(f"✓ Backend is healthy")
        print(f"  Version: {data['version']}")
        print(f"  OpenAI: {'✓' if data['openai_configured'] else '✗'}")
        return True
    else:
        print(f"✗ Health check failed: {data['error']}")
        return False


def example_ingest_and_validate():
    """Example: Ingest document and validate"""
    print("\n=== Ingest and Validation Example ===")
    
    agent = CLIAgent()
    
    # Ingest test document
    text = "This is a test document for AI agent validation."
    source = "Agent Test Document"
    
    print(f"Ingesting: {source}")
    success, data = agent.ingest_text(text, source)
    
    if not success:
        print(f"✗ Ingestion failed: {data['error']}")
        return False
    
    print(f"✓ Ingested successfully")
    print(f"  Source ID: {data['source_id']}")
    print(f"  Chunks: {data['chunks']}")
    print(f"  Characters: {data['characters']}")
    
    # Validate by listing sources
    print("\nValidating...")
    success, sources = agent.list_sources()
    
    if not success:
        print("✗ Failed to list sources")
        return False
    
    # Check if our document is in the list
    found = any(s.get("source_name") == source for s in sources)
    
    if found:
        print(f"✓ Document found in knowledge base")
        print(f"  Total sources: {len(sources)}")
        return True
    else:
        print(f"✗ Document not found in knowledge base")
        return False


def example_bulk_operation():
    """Example: Bulk ingest multiple documents"""
    print("\n=== Bulk Operation Example ===")
    
    agent = CLIAgent()
    
    documents = [
        ("Test document 1", "Test 1"),
        ("Test document 2", "Test 2"),
        ("Test document 3", "Test 3"),
    ]
    
    ingested = 0
    failed = 0
    
    for text, source in documents:
        success, data = agent.ingest_text(text, source)
        if success:
            ingested += 1
            print(f"✓ Ingested: {source}")
        else:
            failed += 1
            print(f"✗ Failed: {source} - {data['error']}")
    
    print(f"\nResults: {ingested} success, {failed} failed")
    return failed == 0


def example_comprehensive_test():
    """Example: Comprehensive backend test suite"""
    print("\n=== Comprehensive Test Suite ===")
    
    agent = CLIAgent()
    tests_passed = 0
    tests_failed = 0
    
    # Test 1: Health check
    print("Test 1: Health check")
    success, _ = agent.check_health()
    if success:
        print("  ✓ PASS")
        tests_passed += 1
    else:
        print("  ✗ FAIL")
        tests_failed += 1
    
    # Test 2: Text ingestion
    print("Test 2: Text ingestion")
    success, _ = agent.ingest_text("Test content", "Comprehensive Test")
    if success:
        print("  ✓ PASS")
        tests_passed += 1
    else:
        print("  ✗ FAIL")
        tests_failed += 1
    
    # Test 3: List sources
    print("Test 3: List sources")
    success, sources = agent.list_sources()
    if success and len(sources) > 0:
        print("  ✓ PASS")
        tests_passed += 1
    else:
        print("  ✗ FAIL")
        tests_failed += 1
    
    # Test 4: Validate source exists
    print("Test 4: Validate ingested document")
    if sources and any(s.get("source_name") == "Comprehensive Test" for s in sources):
        print("  ✓ PASS")
        tests_passed += 1
    else:
        print("  ✗ FAIL")
        tests_failed += 1
    
    print(f"\nResults: {tests_passed}/{tests_passed + tests_failed} tests passed")
    return tests_failed == 0


def example_error_handling():
    """Example: Error handling"""
    print("\n=== Error Handling Example ===")
    
    agent = CLIAgent()
    
    # Test invalid URL
    print("Test: Invalid URL ingestion")
    success, data = agent.ingest_url("not-a-valid-url")
    if not success:
        print(f"✓ Error caught correctly: {data['error']}")
    else:
        print(f"✗ Should have failed but succeeded")
    
    # Test empty text
    print("\nTest: Empty text ingestion")
    success, data = agent.ingest_text("", "Empty Test")
    if not success:
        print(f"✓ Error caught correctly: {data['error']}")
    else:
        print(f"✗ Should have failed but succeeded")
    
    return True


if __name__ == "__main__":
    """
    Run all examples
    
    This demonstrates how AI agents can use the CLI to:
    1. Test backend features
    2. Validate functionality
    3. Automate workflows
    4. Handle errors
    """
    
    print("=" * 60)
    print("CLI Agent Usage Examples")
    print("=" * 60)
    
    # Run examples
    results = []
    
    results.append(("Health Check", example_health_check()))
    results.append(("Ingest & Validate", example_ingest_and_validate()))
    results.append(("Bulk Operation", example_bulk_operation()))
    results.append(("Comprehensive Test", example_comprehensive_test()))
    results.append(("Error Handling", example_error_handling()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{name:.<40} {status}")
    
    total = len(results)
    passed = sum(1 for _, p in results if p)
    
    print(f"\nTotal: {passed}/{total} examples passed")
    
    # Exit with appropriate code
    sys.exit(0 if passed == total else 1)
