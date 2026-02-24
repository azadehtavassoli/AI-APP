"""
Bureaucracy Copilot CLI - Backend API Client

This CLI tool provides a command-line interface to interact with the backend API,
bypassing the UI for faster development and testing workflows.

Purpose:
- Speed up backend development by providing direct API access
- Enable automation and scripting for testing
- Allow agents to interact with the system programmatically
- Provide comprehensive feature parity with the UI

Architecture:
CLI (this file) → HTTP → Backend API (FastAPI) → Services

New Development Workflow:
1. Implement backend feature
2. Test via CLI (this tool) - FAST
3. Update UI to expose feature - SLOW
4. Full integration testing

Usage Examples:
    # Health check
    python cli.py health
    
    # Ingest text
    python cli.py ingest text "Your document text here" --source "Test Doc"
    
    # Ingest PDF
    python cli.py ingest pdf path/to/document.pdf
    
    # Ingest URL
    python cli.py ingest url "https://example.com"
    
    # List all sources
    python cli.py sources list
    
    # Clear knowledge base
    python cli.py sources clear
    
    # JSON output for agents/scripts
    python cli.py sources list --json
    
    # Interactive mode
    python cli.py interactive

Author: Development Team
Created: 2026-02-10
Last Modified: 2026-02-10
"""

import sys
import os
import base64
import json
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from datetime import datetime

# Add project root to path for imports
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

import requests
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import print as rprint
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn


# ============================================================================
# CONFIGURATION
# ============================================================================

DEFAULT_BACKEND_URL = "http://localhost:8000"
DEFAULT_TIMEOUT = 30


# ============================================================================
# API CLIENT
# ============================================================================

class CLIAPIClient:
    """
    Enhanced API Client for CLI Use
    
    Similar to frontend api_client.py but optimized for CLI:
    - Better error messages
    - Request/response logging
    - JSON output support
    - Timeout configuration
    """
    
    def __init__(
        self,
        base_url: str = DEFAULT_BACKEND_URL,
        timeout: int = DEFAULT_TIMEOUT,
        verbose: bool = False
    ):
        """
        Initialize CLI API Client
        
        Args:
            base_url: Backend API base URL
            timeout: Request timeout in seconds
            verbose: Whether to show detailed request/response info
        """
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.verbose = verbose
        self.console = Console()
    
    def _log_request(self, method: str, url: str, data: Optional[Dict] = None):
        """Log outgoing request if verbose mode enabled"""
        if self.verbose:
            self.console.print(f"\n[dim]→ {method} {url}[/dim]")
            if data:
                self.console.print(f"[dim]  Payload: {json.dumps(data, indent=2)[:200]}...[/dim]")
    
    def _log_response(self, status_code: int, response_time: float):
        """Log incoming response if verbose mode enabled"""
        if self.verbose:
            status_color = "green" if 200 <= status_code < 300 else "red"
            self.console.print(f"[dim]← Status: [{status_color}]{status_code}[/{status_color}] ({response_time:.2f}s)[/dim]")
    
    def _make_request(
        self,
        method: str,
        endpoint: str,
        json_data: Optional[Dict] = None,
        timeout: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Make HTTP Request to Backend
        
        Args:
            method: HTTP method
            endpoint: API endpoint
            json_data: Request body
            timeout: Request timeout (uses instance timeout if None)
        
        Returns:
            Dict with 'success', 'data', 'error', 'status_code'
        
        Raises:
            Exception: For connection errors or unexpected failures
        """
        url = f"{self.base_url}{endpoint}"
        request_timeout = timeout or self.timeout
        
        self._log_request(method, url, json_data)
        
        try:
            start_time = datetime.now()
            
            response = requests.request(
                method=method.upper(),
                url=url,
                json=json_data,
                timeout=request_timeout,
                headers={"Content-Type": "application/json"}
            )
            
            response_time = (datetime.now() - start_time).total_seconds()
            self._log_response(response.status_code, response_time)
            
            # Parse response
            if response.ok:
                try:
                    data = response.json()
                    return {
                        "success": True,
                        "data": data,
                        "status_code": response.status_code,
                        "response_time": response_time
                    }
                except ValueError:
                    return {
                        "success": False,
                        "error": "Invalid JSON response from server",
                        "status_code": response.status_code
                    }
            else:
                # Extract error message
                try:
                    error_data = response.json()
                    error_message = error_data.get("detail") or error_data.get("error") or "Unknown error"
                except ValueError:
                    error_message = response.text or f"HTTP {response.status_code}"
                
                return {
                    "success": False,
                    "error": error_message,
                    "status_code": response.status_code
                }
        
        except requests.exceptions.ConnectionError:
            raise Exception(
                f"Cannot connect to backend at {self.base_url}\n"
                f"Is the backend running? Try: cd backend && uvicorn main:app --reload"
            )
        
        except requests.exceptions.Timeout:
            raise Exception(f"Request timeout after {request_timeout} seconds")
        
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request failed: {str(e)}")
    
    # API Methods
    
    def health_check(self) -> Dict[str, Any]:
        """Check backend health status"""
        return self._make_request("GET", "/api/health")
    
    def ingest_text(self, text: str, source_name: str) -> Dict[str, Any]:
        """Ingest text document"""
        return self._make_request(
            "POST",
            "/api/ingest/text",
            json_data={"text": text, "source_name": source_name}
        )
    
    def ingest_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """Ingest PDF document"""
        # Read and encode PDF
        try:
            with open(pdf_path, "rb") as f:
                pdf_bytes = f.read()
            
            pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
            filename = Path(pdf_path).name
            
            return self._make_request(
                "POST",
                "/api/ingest/pdf",
                json_data={"pdf_base64": pdf_base64, "filename": filename},
                timeout=60
            )
        except FileNotFoundError:
            raise Exception(f"PDF file not found: {pdf_path}")
        except Exception as e:
            raise Exception(f"Failed to read PDF: {str(e)}")
    
    def ingest_url(self, url: str) -> Dict[str, Any]:
        """Ingest web page from URL"""
        return self._make_request(
            "POST",
            "/api/ingest/url",
            json_data={"url": url},
            timeout=60
        )

    def ingest_visual(
        self,
        file_path: str,
        enable_visual_mode: bool = True,
        llm_model: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Ingest a document using MarkItDown with optional visual/LLM descriptions."""

        path = Path(file_path)
        if not path.exists():
            raise Exception(f"File not found: {file_path}")

        with open(path, "rb") as f:
            file_bytes = f.read()

        encoded = base64.b64encode(file_bytes).decode("utf-8")

        payload = {
            "file_base64": encoded,
            "filename": path.name,
            "enable_visual_mode": enable_visual_mode,
            "llm_model": llm_model,
        }

        return self._make_request(
            "POST",
            "/api/ingest/visual",
            json_data=payload,
            timeout=120,
        )
    
    def list_sources(self) -> Dict[str, Any]:
        """List all ingested sources"""
        return self._make_request("GET", "/api/sources")
    
    def clear_knowledge_base(self) -> Dict[str, Any]:
        """Clear all documents from knowledge base"""
        return self._make_request("POST", "/api/clear")

    # === Agent endpoints ===

    def agent_query(
        self,
        query: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Send a free-form agent query request to the backend.

        Args:
            query (str): User prompt or question.
            session_id (Optional[str]): Conversation/session identifier.
            metadata (Optional[Dict[str, Any]]): Optional metadata forwarded to backend.

        Returns:
            Dict[str, Any]: Backend response envelope.
        """
        return self._make_request(
            "POST",
            "/api/agent/query",
            json_data={"query": query, "session_id": session_id, "metadata": metadata or {}},
        )

    def agent_ingest_text(
        self,
        text: str,
        source_name: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Request agent-driven text ingestion.

        Args:
            text (str): Raw text content to ingest.
            source_name (str): Source label used by ingestion tool.
            session_id (Optional[str]): Conversation/session identifier.
            metadata (Optional[Dict[str, Any]]): Optional metadata forwarded to backend.

        Returns:
            Dict[str, Any]: Backend response envelope.
        """
        return self._make_request(
            "POST",
            "/api/agent/ingest/text",
            json_data={
                "text": text,
                "source_name": source_name,
                "session_id": session_id,
                "metadata": metadata or {},
            },
            timeout=60,
        )

    def agent_ingest_file(
        self,
        file_path: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Request agent-driven file ingestion using base64 payloads.

        Args:
            file_path (str): Local file path to ingest.
            session_id (Optional[str]): Conversation/session identifier.
            metadata (Optional[Dict[str, Any]]): Optional metadata forwarded to backend.

        Returns:
            Dict[str, Any]: Backend response envelope.

        Raises:
            Exception: If the input file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise Exception(f"File not found: {file_path}")

        with open(path, "rb") as f:
            file_bytes = f.read()

        encoded = base64.b64encode(file_bytes).decode("utf-8")

        return self._make_request(
            "POST",
            "/api/agent/ingest/file",
            json_data={
                "file_base64": encoded,
                "filename": path.name,
                "session_id": session_id,
                "metadata": metadata or {},
            },
            timeout=90,
        )

    def agent_ingest_url(
        self,
        url: str,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Request agent-driven URL ingestion.

        Args:
            url (str): URL to ingest.
            session_id (Optional[str]): Conversation/session identifier.
            metadata (Optional[Dict[str, Any]]): Optional metadata forwarded to backend.

        Returns:
            Dict[str, Any]: Backend response envelope.
        """
        return self._make_request(
            "POST",
            "/api/agent/ingest/url",
            json_data={"url": url, "session_id": session_id, "metadata": metadata or {}},
            timeout=60,
        )

    def agent_ingest_visual(
        self,
        file_path: str,
        enable_visual_mode: bool = True,
        llm_model: Optional[str] = None,
        session_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Request agent-driven visual ingestion via MarkItDown mode.

        Args:
            file_path (str): Local file path to ingest.
            enable_visual_mode (bool): Whether visual/LLM mode is enabled.
            llm_model (Optional[str]): Optional LLM model override.
            session_id (Optional[str]): Conversation/session identifier.
            metadata (Optional[Dict[str, Any]]): Optional metadata forwarded to backend.

        Returns:
            Dict[str, Any]: Backend response envelope.

        Raises:
            Exception: If the input file does not exist.
        """
        path = Path(file_path)
        if not path.exists():
            raise Exception(f"File not found: {file_path}")

        with open(path, "rb") as f:
            file_bytes = f.read()

        encoded = base64.b64encode(file_bytes).decode("utf-8")

        return self._make_request(
            "POST",
            "/api/agent/ingest/visual",
            json_data={
                "file_base64": encoded,
                "filename": path.name,
                "enable_visual_mode": enable_visual_mode,
                "llm_model": llm_model,
                "session_id": session_id,
                "metadata": metadata or {},
            },
            timeout=150,
        )


# ============================================================================
# CLI OUTPUT FORMATTING
# ============================================================================

class CLIFormatter:
    """
    Rich Console Output Formatter
    
    Provides beautiful, structured output for CLI commands.
    Supports both human-readable and machine-readable (JSON) formats.
    """
    
    def __init__(self, json_output: bool = False):
        """
        Initialize formatter
        
        Args:
            json_output: If True, output JSON instead of formatted text
        """
        self.json_output = json_output
        self.console = Console()
    
    def success(self, message: str, data: Optional[Dict] = None):
        """Display success message"""
        if self.json_output:
            output = {"status": "success", "message": message}
            if data:
                output["data"] = data
            print(json.dumps(output, indent=2))
        else:
            self.console.print(f"[green]✓[/green] {message}")
            if data:
                # For non-JSON mode, show formatted data
                for key, value in data.items():
                    if isinstance(value, (dict, list)):
                        continue  # Skip complex structures in simple output
                    self.console.print(f"  {key}: [cyan]{value}[/cyan]")
    
    def error(self, message: str, details: Optional[str] = None):
        """Display error message"""
        if self.json_output:
            print(json.dumps({"status": "error", "message": message, "details": details}, indent=2))
        else:
            self.console.print(f"[red]✗[/red] {message}", style="bold red")
            if details:
                self.console.print(f"[dim]{details}[/dim]")
    
    def info(self, message: str):
        """Display info message"""
        if not self.json_output:
            self.console.print(f"[blue]ℹ[/blue] {message}")
    
    def warning(self, message: str):
        """Display warning message"""
        if not self.json_output:
            self.console.print(f"[yellow]⚠[/yellow] {message}")
    
    def header(self, title: str):
        """Display section header"""
        if not self.json_output:
            self.console.print(f"\n[bold cyan]{title}[/bold cyan]")
    
    def table(self, title: str, columns: List[str], rows: List[List[str]]):
        """Display data as table"""
        if self.json_output:
            # Convert to list of dicts for JSON
            data = [dict(zip(columns, row)) for row in rows]
            print(json.dumps(data, indent=2))
        else:
            table = Table(title=title, show_header=True, header_style="bold magenta")
            for col in columns:
                table.add_column(col)
            for row in rows:
                table.add_row(*[str(cell) for cell in row])
            self.console.print(table)
    
    def panel(self, content: str, title: str = ""):
        """Display content in a panel"""
        if not self.json_output:
            self.console.print(Panel(content, title=title, expand=False))


# ============================================================================
# CLI COMMANDS
# ============================================================================

class CLICommands:
    """
    CLI Command Implementations
    
    Each method corresponds to a CLI command and handles:
    - API calls
    - Output formatting
    - Error handling
    """
    
    def __init__(self, client: CLIAPIClient, formatter: CLIFormatter):
        """
        Initialize command handler
        
        Args:
            client: API client instance
            formatter: Output formatter instance
        """
        self.client = client
        self.fmt = formatter
    
    def cmd_health(self):
        """Check backend health"""
        self.fmt.header("Backend Health Check")
        
        try:
            result = self.client.health_check()
            
            if result["success"]:
                data = result["data"]
                
                if self.fmt.json_output:
                    # For JSON output, pass data directly
                    self.fmt.success("Backend is healthy", data)
                else:
                    # For human output, format nicely
                    self.fmt.success("Backend is healthy")
                    self.fmt.console.print(f"  Version: [cyan]{data.get('version', 'Unknown')}[/cyan]")
                    self.fmt.console.print(f"  Status: [green]{data.get('status', 'Unknown')}[/green]")
                    
                    openai_status = "✓ Configured" if data.get('openai_configured') else "✗ Not configured"
                    openai_color = "green" if data.get('openai_configured') else "red"
                    self.fmt.console.print(f"  OpenAI API: [{openai_color}]{openai_status}[/{openai_color}]")
                
                return 0
            else:
                self.fmt.error("Health check failed", result.get("error"))
                return 1
        
        except Exception as e:
            self.fmt.error("Connection failed", str(e))
            return 1
    
    def cmd_ingest_text(self, text: str, source_name: str):
        """Ingest text document"""
        self.fmt.header("Ingesting Text Document")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.fmt.console,
                transient=True
            ) as progress:
                if not self.fmt.json_output:
                    progress.add_task(description="Processing...", total=None)
                
                result = self.client.ingest_text(text, source_name)
            
            if result["success"]:
                data = result["data"]
                source = data.get("source", {})
                
                if self.fmt.json_output:
                    self.fmt.success(f"Successfully ingested: {source_name}", data)
                else:
                    self.fmt.success(f"Successfully ingested: {source_name}")
                    self.fmt.console.print(f"  Source ID: [cyan]{source.get('id', 'N/A')}[/cyan]")
                    self.fmt.console.print(f"  Chunks: [yellow]{source.get('chunks', 0)}[/yellow]")
                    self.fmt.console.print(f"  Characters: [yellow]{source.get('characters', 0)}[/yellow]")
                    self.fmt.console.print(f"  Type: [blue]{source.get('type', 'N/A')}[/blue]")
                
                return 0
            else:
                self.fmt.error("Ingestion failed", result.get("error"))
                return 1
        
        except Exception as e:
            self.fmt.error("Operation failed", str(e))
            return 1
    
    def cmd_ingest_pdf(self, pdf_path: str):
        """Ingest PDF document"""
        self.fmt.header("Ingesting PDF Document")
        
        # Validate path
        path = Path(pdf_path)
        if not path.exists():
            self.fmt.error(f"File not found: {pdf_path}")
            return 1
        
        if not path.suffix.lower() == '.pdf':
            self.fmt.error(f"Not a PDF file: {pdf_path}")
            return 1
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.fmt.console,
                transient=True
            ) as progress:
                if not self.fmt.json_output:
                    task = progress.add_task(description="Processing PDF...", total=None)
                
                result = self.client.ingest_pdf(str(path))
            
            if result["success"]:
                data = result["data"]
                source = data.get("source", {})
                
                if self.fmt.json_output:
                    self.fmt.success(f"Successfully ingested: {path.name}", data)
                else:
                    self.fmt.success(f"Successfully ingested: {path.name}")
                    self.fmt.console.print(f"  Source ID: [cyan]{source.get('id', 'N/A')}[/cyan]")
                    self.fmt.console.print(f"  Chunks: [yellow]{source.get('chunks', 0)}[/yellow]")
                    self.fmt.console.print(f"  Characters: [yellow]{source.get('characters', 0)}[/yellow]")
                    
                    metadata = source.get('metadata', {})
                    if metadata.get('extraction_method'):
                        self.fmt.console.print(f"  Extraction: [blue]{metadata['extraction_method']}[/blue]")
                
                return 0
            else:
                self.fmt.error("Ingestion failed", result.get("error"))
                return 1
        
        except Exception as e:
            self.fmt.error("Operation failed", str(e))
            return 1
    
    def cmd_ingest_url(self, url: str):
        """Ingest web page from URL"""
        self.fmt.header("Ingesting URL")
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.fmt.console,
                transient=True
            ) as progress:
                if not self.fmt.json_output:
                    progress.add_task(description="Fetching and processing...", total=None)
                
                result = self.client.ingest_url(url)
            
            if result["success"]:
                data = result["data"]
                source = data.get("source", {})
                
                if self.fmt.json_output:
                    self.fmt.success(f"Successfully ingested: {url}", data)
                else:
                    self.fmt.success(f"Successfully ingested: {url}")
                    self.fmt.console.print(f"  Source ID: [cyan]{source.get('id', 'N/A')}[/cyan]")
                    self.fmt.console.print(f"  Chunks: [yellow]{source.get('chunks', 0)}[/yellow]")
                    self.fmt.console.print(f"  Characters: [yellow]{source.get('characters', 0)}[/yellow]")
                    
                    metadata = source.get('metadata', {})
                    if metadata.get('truncated'):
                        self.fmt.warning(
                            f"Content was truncated: {metadata.get('original_length', 0)} → "
                            f"{metadata.get('used_length', 0)} chars"
                        )
                
                return 0
            else:
                self.fmt.error("Ingestion failed", result.get("error"))
                return 1
        
        except Exception as e:
            self.fmt.error("Operation failed", str(e))
            return 1

    def cmd_ingest_visual(self, file_path: str, enable_visual_mode: bool, llm_model: Optional[str]):
        """Ingest a document using MarkItDown (visual/LLM mode by default)."""

        self.fmt.header("Ingesting Visual Document")

        path = Path(file_path)
        if not path.exists():
            self.fmt.error(f"File not found: {file_path}")
            return 1

        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.fmt.console,
                transient=True,
            ) as progress:
                if not self.fmt.json_output:
                    progress.add_task(description="Converting with MarkItDown...", total=None)

                result = self.client.ingest_visual(
                    str(path),
                    enable_visual_mode=enable_visual_mode,
                    llm_model=llm_model,
                )

            if result["success"]:
                data = result.get("data", {})
                source = data.get("source", {})
                metadata = source.get("metadata", {})

                if self.fmt.json_output:
                    self.fmt.success(f"Successfully ingested: {path.name}", data)
                else:
                    self.fmt.success(f"Successfully ingested: {path.name}")
                    self.fmt.console.print(f"  Source ID: [cyan]{source.get('id', 'N/A')}[/cyan]")
                    self.fmt.console.print(f"  Chunks: [yellow]{source.get('chunks', 0)}[/yellow]")
                    self.fmt.console.print(f"  Characters: [yellow]{source.get('characters', 0)}[/yellow]")
                    self.fmt.console.print(f"  Visual mode: [blue]{metadata.get('visual_mode')}")
                    if metadata.get("llm_model"):
                        self.fmt.console.print(f"  LLM model: [blue]{metadata.get('llm_model')}[/blue]")
                return 0

            self.fmt.error("Ingestion failed", result.get("error"))
            return 1

        except Exception as e:
            self.fmt.error("Operation failed", str(e))
            return 1
    
    def cmd_list_sources(self):
        """List all sources"""
        self.fmt.header("Knowledge Base Sources")
        
        try:
            result = self.client.list_sources()
            
            if result["success"]:
                data = result["data"]
                sources = data.get("sources", [])
                total = data.get("total_count", 0)
                
                if total == 0:
                    if self.fmt.json_output:
                        self.fmt.success("Knowledge base is empty", {"sources": [], "total_count": 0})
                    else:
                        self.fmt.info("Knowledge base is empty")
                    return 0
                
                if self.fmt.json_output:
                    self.fmt.success(f"Found {total} sources", {"sources": sources, "total_count": total})
                else:
                    # Create table
                    columns = ["Type", "Name/URL", "Chunks", "Chars", "Added"]
                    rows = []
                    
                    for source in sources:
                        # Truncate long URIs
                        uri = source.get("uri", "N/A")
                        if len(uri) > 50:
                            uri = uri[:47] + "..."
                        
                        # Format timestamp
                        timestamp = source.get("timestamp", "N/A")
                        if timestamp != "N/A":
                            try:
                                dt = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                                timestamp = dt.strftime("%Y-%m-%d %H:%M")
                            except:
                                pass
                        
                        rows.append([
                            source.get("source_type", "N/A"),
                            uri,
                            source.get("chunk_count", 0),
                            source.get("char_count", 0),
                            timestamp
                        ])
                    
                    self.fmt.table(f"Sources ({total} total)", columns, rows)
                
                return 0
            else:
                self.fmt.error("Failed to list sources", result.get("error"))
                return 1
        
        except Exception as e:
            self.fmt.error("Operation failed", str(e))
            return 1
    
    def cmd_clear_knowledge_base(self, confirm: bool = False):
        """Clear knowledge base"""
        self.fmt.header("Clear Knowledge Base")
        
        if not confirm:
            self.fmt.warning("This will delete ALL documents from the knowledge base!")
            if not self.fmt.json_output:
                response = input("Type 'yes' to confirm: ")
                if response.lower() != "yes":
                    self.fmt.info("Operation cancelled")
                    return 0
        
        try:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=self.fmt.console,
                transient=True
            ) as progress:
                if not self.fmt.json_output:
                    progress.add_task(description="Clearing...", total=None)
                
                result = self.client.clear_knowledge_base()
            
            if result["success"]:
                data = result["data"]
                message = data.get("message", "Knowledge base cleared")
                self.fmt.success(message)
                return 0
            else:
                self.fmt.error("Clear operation failed", result.get("error"))
                return 1
        
        except Exception as e:
            self.fmt.error("Operation failed", str(e))
            return 1

    # === Agent commands ===

    @staticmethod
    def _parse_metadata(metadata_str: Optional[str]) -> Dict[str, Any]:
        """Parse optional metadata JSON into a dictionary.

        Args:
            metadata_str (Optional[str]): Raw JSON string from CLI flag.

        Returns:
            Dict[str, Any]: Parsed metadata dictionary, or empty dict.

        Raises:
            Exception: If provided JSON is malformed.
        """
        if not metadata_str:
            return {}
        try:
            return json.loads(metadata_str)
        except json.JSONDecodeError as exc:
            raise Exception(f"Invalid metadata JSON: {exc}")

    def _render_agent_response(self, result: Dict[str, Any]) -> int:
        """Render a normalized agent response in JSON or rich text mode.

        Args:
            result (Dict[str, Any]): Backend response envelope.

        Returns:
            int: Process-style status code (`0` on success, `1` on failure).
        """
        if not result["success"]:
            self.fmt.error("Agent request failed", result.get("error"))
            return 1

        data = result.get("data", {})
        answer = data.get("answer", "")
        citations = data.get("citations", []) or []
        tool_calls = data.get("tool_calls", []) or []
        session_id = data.get("session_id", "n/a")
        run_id = data.get("run_id", "n/a")

        if self.fmt.json_output:
            self.fmt.success("Agent response", data)
            return 0

        # Render a concise human-readable response with structured sections.
        self.fmt.header("Agent Response")
        self.fmt.console.print(f"[cyan]Session:[/cyan] {session_id}")
        self.fmt.console.print(f"[cyan]Run ID:[/cyan] {run_id}")
        self.fmt.panel(answer, title="Answer")

        if citations:
            rows = []
            for c in citations:
                rows.append([
                    c.get("source_type", ""),
                    c.get("source_uri", ""),
                    c.get("chunk_id", ""),
                ])
            self.fmt.table("Citations", ["Type", "URI", "Chunk"], rows)

        if tool_calls:
            self.fmt.header("Tool Calls")
            for call in tool_calls:
                pretty = json.dumps(call, indent=2, ensure_ascii=False)
                self.fmt.console.print(Syntax(pretty, "json", theme="ansi_dark"))

        return 0

    def cmd_agent_query(self, query: str, session_id: Optional[str], metadata_str: Optional[str]):
        """Run the `agent query` CLI command.

        Args:
            query (str): User question or instruction.
            session_id (Optional[str]): Optional conversation identifier.
            metadata_str (Optional[str]): Optional metadata JSON string.

        Returns:
            int: Command exit code.
        """
        self.fmt.header("Agent Query")
        metadata = self._parse_metadata(metadata_str)
        try:
            result = self.client.agent_query(query=query, session_id=session_id, metadata=metadata)
            return self._render_agent_response(result)
        except Exception as e:
            self.fmt.error("Operation failed", str(e))
            return 1

    def cmd_agent_ingest_text(
        self,
        text: str,
        source_name: str,
        session_id: Optional[str],
        metadata_str: Optional[str],
    ):
        """Run the `agent ingest-text` CLI command.

        Args:
            text (str): Text content to ingest.
            source_name (str): Source label used by ingestion tool.
            session_id (Optional[str]): Optional conversation identifier.
            metadata_str (Optional[str]): Optional metadata JSON string.

        Returns:
            int: Command exit code.
        """
        self.fmt.header("Agent Ingest Text")
        metadata = self._parse_metadata(metadata_str)
        try:
            result = self.client.agent_ingest_text(
                text=text,
                source_name=source_name,
                session_id=session_id,
                metadata=metadata,
            )
            return self._render_agent_response(result)
        except Exception as e:
            self.fmt.error("Operation failed", str(e))
            return 1

    def cmd_agent_ingest_file(
        self,
        file_path: str,
        session_id: Optional[str],
        metadata_str: Optional[str],
    ):
        """Run the `agent ingest-file` CLI command.

        Args:
            file_path (str): Local file path to ingest.
            session_id (Optional[str]): Optional conversation identifier.
            metadata_str (Optional[str]): Optional metadata JSON string.

        Returns:
            int: Command exit code.
        """
        self.fmt.header("Agent Ingest File")
        metadata = self._parse_metadata(metadata_str)
        try:
            result = self.client.agent_ingest_file(
                file_path=file_path,
                session_id=session_id,
                metadata=metadata,
            )
            return self._render_agent_response(result)
        except Exception as e:
            self.fmt.error("Operation failed", str(e))
            return 1

    def cmd_agent_ingest_url(
        self,
        url: str,
        session_id: Optional[str],
        metadata_str: Optional[str],
    ):
        """Run the `agent ingest-url` CLI command.

        Args:
            url (str): URL to ingest via agent tool invocation.
            session_id (Optional[str]): Optional conversation identifier.
            metadata_str (Optional[str]): Optional metadata JSON string.

        Returns:
            int: Command exit code.
        """
        self.fmt.header("Agent Ingest URL")
        metadata = self._parse_metadata(metadata_str)
        try:
            result = self.client.agent_ingest_url(
                url=url,
                session_id=session_id,
                metadata=metadata,
            )
            return self._render_agent_response(result)
        except Exception as e:
            self.fmt.error("Operation failed", str(e))
            return 1

    def cmd_agent_ingest_visual(
        self,
        file_path: str,
        enable_visual_mode: bool,
        llm_model: Optional[str],
        session_id: Optional[str],
        metadata_str: Optional[str],
    ):
        """Run the `agent ingest-visual` CLI command.

        Args:
            file_path (str): Local file path to ingest.
            enable_visual_mode (bool): Whether visual/LLM mode is enabled.
            llm_model (Optional[str]): Optional LLM model override.
            session_id (Optional[str]): Optional conversation identifier.
            metadata_str (Optional[str]): Optional metadata JSON string.

        Returns:
            int: Command exit code.
        """
        self.fmt.header("Agent Ingest Visual Document")
        metadata = self._parse_metadata(metadata_str)
        try:
            result = self.client.agent_ingest_visual(
                file_path=file_path,
                enable_visual_mode=enable_visual_mode,
                llm_model=llm_model,
                session_id=session_id,
                metadata=metadata,
            )
            return self._render_agent_response(result)
        except Exception as e:
            self.fmt.error("Operation failed", str(e))
            return 1
    
    def cmd_interactive(self):
        """Interactive mode"""
        self.fmt.console.print("\n[bold cyan]Bureaucracy Copilot CLI - Interactive Mode[/bold cyan]")
        self.fmt.console.print("Type 'help' for available commands, 'exit' to quit\n")
        
        while True:
            try:
                # Prompt
                user_input = input("\n[copilot]> ").strip()
                
                if not user_input:
                    continue
                
                # Parse command
                parts = user_input.split()
                cmd = parts[0].lower()
                args = parts[1:]
                
                # Handle commands
                if cmd in ["exit", "quit", "q"]:
                    self.fmt.console.print("[dim]Goodbye![/dim]")
                    break
                
                elif cmd == "help":
                    self._show_interactive_help()
                
                elif cmd == "health":
                    self.cmd_health()
                
                elif cmd == "sources":
                    self.cmd_list_sources()
                
                elif cmd == "clear":
                    self.cmd_clear_knowledge_base(confirm=False)
                
                elif cmd == "text":
                    if len(args) < 1:
                        self.fmt.error("Usage: text <text> [source_name]")
                    else:
                        text = " ".join(args[:-1]) if len(args) > 1 else args[0]
                        source = args[-1] if len(args) > 1 else "Interactive input"
                        self.cmd_ingest_text(text, source)
                
                elif cmd == "pdf":
                    if len(args) < 1:
                        self.fmt.error("Usage: pdf <path>")
                    else:
                        self.cmd_ingest_pdf(args[0])
                
                elif cmd == "url":
                    if len(args) < 1:
                        self.fmt.error("Usage: url <url>")
                    else:
                        self.cmd_ingest_url(args[0])
                
                else:
                    self.fmt.error(f"Unknown command: {cmd}", "Type 'help' for available commands")
            
            except KeyboardInterrupt:
                self.fmt.console.print("\n[dim]Use 'exit' to quit[/dim]")
            except EOFError:
                break
            except Exception as e:
                self.fmt.error("Command error", str(e))
        
        return 0
    
    def _show_interactive_help(self):
        """Show help for interactive mode"""
        help_text = """
[bold]Available Commands:[/bold]

  [cyan]health[/cyan]              Check backend health
  [cyan]sources[/cyan]             List all sources
  [cyan]clear[/cyan]               Clear knowledge base
  [cyan]text <text> [name][/cyan]  Ingest text
  [cyan]pdf <path>[/cyan]          Ingest PDF file
  [cyan]url <url>[/cyan]           Ingest web page
  [cyan]help[/cyan]                Show this help
  [cyan]exit[/cyan]                Quit interactive mode

[bold]Examples:[/bold]
  text "Hello world" "Test doc"
  pdf document.pdf
  url https://example.com
        """
        self.fmt.panel(help_text.strip(), title="Interactive Mode Help")


# ============================================================================
# MAIN CLI INTERFACE
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """
    Create CLI Argument Parser
    
    Defines all CLI commands and their arguments.
    
    Returns:
        Configured ArgumentParser instance
    """
    parser = argparse.ArgumentParser(
        prog="cli",
        description="Bureaucracy Copilot CLI - Backend API Client",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check backend health
  python cli.py health
  
  # Ingest text
  python cli.py ingest text "Your document text" --source "My Document"
  
  # Ingest PDF
  python cli.py ingest pdf documents/visa.pdf
  
  # Ingest URL
  python cli.py ingest url "https://example.com"
  
  # List sources
  python cli.py sources list
  
  # Clear knowledge base
  python cli.py sources clear --confirm
  
  # Interactive mode
  python cli.py interactive
  
  # JSON output for scripting
  python cli.py sources list --json
  
  # Verbose mode for debugging
  python cli.py health --verbose

For more information, see docs/CLI.md
        """
    )
    
    # Global options
    parser.add_argument(
        "--backend-url",
        default=os.getenv("BACKEND_URL", DEFAULT_BACKEND_URL),
        help=f"Backend API URL (default: {DEFAULT_BACKEND_URL})"
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Request timeout in seconds (default: {DEFAULT_TIMEOUT})"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output in JSON format (for scripting)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Show detailed request/response information"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Health command
    subparsers.add_parser(
        "health",
        help="Check backend health status"
    )
    
    # Ingest commands
    ingest_parser = subparsers.add_parser(
        "ingest",
        help="Ingest documents into knowledge base"
    )
    ingest_subparsers = ingest_parser.add_subparsers(dest="ingest_type", help="Ingestion type")
    
    # Ingest text
    text_parser = ingest_subparsers.add_parser("text", help="Ingest text document")
    text_parser.add_argument("text", help="Text content to ingest")
    text_parser.add_argument(
        "--source",
        default="CLI input",
        help="Source name for this document"
    )
    
    # Ingest PDF
    pdf_parser = ingest_subparsers.add_parser("pdf", help="Ingest PDF document")
    pdf_parser.add_argument("path", help="Path to PDF file")
    
    # Ingest URL
    url_parser = ingest_subparsers.add_parser("url", help="Ingest web page from URL")
    url_parser.add_argument("url", help="URL to fetch and ingest")

    # Ingest via MarkItDown with visuals
    visual_parser = ingest_subparsers.add_parser(
        "visual", help="Ingest document using MarkItDown (with visual descriptions by default)"
    )
    visual_parser.add_argument("path", help="Path to the document (pdf/txt/etc.)")
    visual_parser.add_argument(
        "--disable-visual-mode",
        action="store_false",
        dest="visual_mode",
        help="Disable LLM visual mode and use deterministic conversion",
    )
    visual_parser.add_argument(
        "--llm-model",
        default=None,
        help="Optional LLM model override when visual mode is enabled",
    )
    visual_parser.set_defaults(visual_mode=True)
    
    # Sources commands
    sources_parser = subparsers.add_parser(
        "sources",
        help="Manage knowledge base sources"
    )
    sources_subparsers = sources_parser.add_subparsers(dest="sources_action", help="Sources action")
    
    # List sources
    sources_subparsers.add_parser("list", help="List all sources")
    
    # Clear knowledge base
    clear_parser = sources_subparsers.add_parser("clear", help="Clear all sources")
    clear_parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip confirmation prompt"
    )
    
    # Interactive mode
    subparsers.add_parser(
        "interactive",
        help="Start interactive CLI mode"
    )

    # Agent commands
    agent_parser = subparsers.add_parser("agent", help="Agent-powered actions")
    agent_subparsers = agent_parser.add_subparsers(dest="agent_action", help="Agent action")

    agent_query = agent_subparsers.add_parser("query", help="Run an agent query")
    agent_query.add_argument("query", help="User question or instruction")
    agent_query.add_argument("--session-id", default=None, help="Optional session id")
    agent_query.add_argument("--metadata", default=None, help="Optional metadata JSON")

    agent_ingest_text = agent_subparsers.add_parser("ingest-text", help="Agent ingest plain text")
    agent_ingest_text.add_argument("text", help="Text content to ingest")
    agent_ingest_text.add_argument("--source", default="CLI agent input", help="Source name")
    agent_ingest_text.add_argument("--session-id", default=None, help="Optional session id")
    agent_ingest_text.add_argument("--metadata", default=None, help="Optional metadata JSON")

    agent_ingest_file = agent_subparsers.add_parser("ingest-file", help="Agent ingest file (base64 inside)")
    agent_ingest_file.add_argument("path", help="Path to the file")
    agent_ingest_file.add_argument("--session-id", default=None, help="Optional session id")
    agent_ingest_file.add_argument("--metadata", default=None, help="Optional metadata JSON")

    agent_ingest_url = agent_subparsers.add_parser("ingest-url", help="Agent ingest URL")
    agent_ingest_url.add_argument("url", help="URL to fetch and ingest")
    agent_ingest_url.add_argument("--session-id", default=None, help="Optional session id")
    agent_ingest_url.add_argument("--metadata", default=None, help="Optional metadata JSON")

    agent_ingest_visual = agent_subparsers.add_parser(
        "ingest-visual", help="Agent ingest document via MarkItDown"
    )
    agent_ingest_visual.add_argument("path", help="Path to the document")
    agent_ingest_visual.add_argument(
        "--disable-visual-mode",
        action="store_false",
        dest="visual_mode",
        help="Disable LLM visual mode and use deterministic conversion",
    )
    agent_ingest_visual.add_argument(
        "--llm-model",
        default=None,
        help="Optional LLM model override when visual mode is enabled",
    )
    agent_ingest_visual.add_argument("--session-id", default=None, help="Optional session id")
    agent_ingest_visual.add_argument("--metadata", default=None, help="Optional metadata JSON")
    agent_ingest_visual.set_defaults(visual_mode=True)
    
    return parser


def main():
    """
    Main CLI Entry Point
    
    Parses arguments and dispatches to appropriate command handler.
    
    Returns:
        Exit code (0 for success, non-zero for error)
    """
    parser = create_parser()
    args = parser.parse_args()
    
    # Show help if no command
    if not args.command:
        parser.print_help()
        return 0
    
    # Initialize client and formatter
    try:
        client = CLIAPIClient(
            base_url=args.backend_url,
            timeout=args.timeout,
            verbose=args.verbose
        )
        formatter = CLIFormatter(json_output=args.json)
        commands = CLICommands(client, formatter)
    except Exception as e:
        print(f"Initialization error: {e}", file=sys.stderr)
        return 1
    
    # Dispatch command
    try:
        if args.command == "health":
            return commands.cmd_health()
        
        elif args.command == "ingest":
            if args.ingest_type == "text":
                return commands.cmd_ingest_text(args.text, args.source)
            elif args.ingest_type == "pdf":
                return commands.cmd_ingest_pdf(args.path)
            elif args.ingest_type == "url":
                return commands.cmd_ingest_url(args.url)
            elif args.ingest_type == "visual":
                return commands.cmd_ingest_visual(args.path, args.visual_mode, args.llm_model)
            else:
                parser.parse_args(["ingest", "--help"])
                return 1
        
        elif args.command == "sources":
            if args.sources_action == "list":
                return commands.cmd_list_sources()
            elif args.sources_action == "clear":
                return commands.cmd_clear_knowledge_base(confirm=args.confirm)
            else:
                parser.parse_args(["sources", "--help"])
                return 1
        
        elif args.command == "interactive":
            return commands.cmd_interactive()

        elif args.command == "agent":
            if args.agent_action == "query":
                return commands.cmd_agent_query(args.query, args.session_id, args.metadata)
            elif args.agent_action == "ingest-text":
                return commands.cmd_agent_ingest_text(args.text, args.source, args.session_id, args.metadata)
            elif args.agent_action == "ingest-file":
                return commands.cmd_agent_ingest_file(args.path, args.session_id, args.metadata)
            elif args.agent_action == "ingest-url":
                return commands.cmd_agent_ingest_url(args.url, args.session_id, args.metadata)
            elif args.agent_action == "ingest-visual":
                return commands.cmd_agent_ingest_visual(
                    args.path,
                    args.visual_mode,
                    args.llm_model,
                    args.session_id,
                    args.metadata,
                )
            else:
                parser.parse_args(["agent", "--help"])
                return 1
        
        else:
            parser.print_help()
            return 1
    
    except KeyboardInterrupt:
        if not args.json:
            print("\n\nInterrupted by user", file=sys.stderr)
        return 130
    except Exception as e:
        if args.json:
            print(json.dumps({"status": "error", "message": str(e)}, indent=2))
        else:
            print(f"Error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
