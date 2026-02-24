"""
Backend API Client

This module provides a Python client for communication with the FastAPI backend.
It wraps all HTTP requests in easy-to-use functions.

Why separate the API client?
- Reusability: Same client can be used across multiple UI components
- Maintainability: All API calls in one place
- Error handling: Centralized error processing
- Testing: Easier to mock for tests
- Type safety: Provides typed responses

Architecture:
Frontend (Streamlit) → API Client → HTTP → Backend (FastAPI) → Services
"""

import requests
import base64
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import os


@dataclass
class APIResponse:
    """
    Standard API Response Wrapper
    
    This wraps all API responses in a consistent format.
    Makes error handling easier in the UI.
    
    Attributes:
        success: Whether the request succeeded
        data: Response data (if success)
        error: Error message (if not success)
        status_code: HTTP status code
    """
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    status_code: Optional[int] = None


class BackendAPIClient:
    """
    Client for FastAPI Backend Communication
    
    This class handles all communication with the backend service.
    It provides methods for each API endpoint with proper error handling.
    
    Usage:
        client = BackendAPIClient()
        response = client.ingest_text("Your document text", "My Document")
        if response.success:
            print(f"Success! Source ID: {response.data['source_id']}")
        else:
            print(f"Error: {response.error}")
    """
    
    def __init__(self, base_url: Optional[str] = None):
        """
        Initialize the API Client
        
        Args:
            base_url: Base URL of the backend API
                     If None, reads from environment variable BACKEND_URL
                     Defaults to http://localhost:8000
        
        Example:
            # Use default (localhost:8000)
            client = BackendAPIClient()
            
            # Use custom URL
            client = BackendAPIClient("http://api.example.com")
        """
        # Set base URL from parameter, environment variable, or default
        self.base_url = base_url or os.getenv("BACKEND_URL", "http://localhost:8000")
        
        # Ensure no trailing slash (for consistent URL building)
        self.base_url = self.base_url.rstrip("/")
        
        # Default timeout for requests (seconds)
        self.timeout = 30
    
    def _make_request(
        self, 
        method: str, 
        endpoint: str, 
        json_data: Optional[Dict] = None,
        timeout: Optional[int] = None
    ) -> APIResponse:
        """
        Make HTTP Request to Backend
        
        This is an internal helper method that handles:
        - Building full URL
        - Making HTTP request
        - Processing response
        - Error handling
        
        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g., "/api/health")
            json_data: Request body data (for POST requests)
            timeout: Request timeout (uses self.timeout if None)
        
        Returns:
            APIResponse with success status and data/error
        
        Error handling:
        - Network errors: Connection refused, timeout, etc.
        - HTTP errors: 4xx, 5xx status codes
        - JSON parsing errors: Invalid response format
        """
        # Build full URL
        url = f"{self.base_url}{endpoint}"
        
        # Use provided timeout or default
        request_timeout = timeout or self.timeout
        
        try:
            # Make HTTP request
            # requests.request() is generic method that accepts any HTTP verb
            response = requests.request(
                method=method.upper(),
                url=url,
                json=json_data,  # Automatically serialized to JSON
                timeout=request_timeout,
                headers={"Content-Type": "application/json"}
            )
            
            # Check if request was successful (status 200-299)
            if response.ok:
                # Parse JSON response
                try:
                    data = response.json()
                    return APIResponse(
                        success=True,
                        data=data,
                        status_code=response.status_code
                    )
                except ValueError:
                    # Response wasn't valid JSON
                    return APIResponse(
                        success=False,
                        error="Invalid JSON response from server",
                        status_code=response.status_code
                    )
            else:
                # HTTP error (4xx, 5xx)
                # Try to extract error message from response
                try:
                    error_data = response.json()
                    error_message = error_data.get("detail") or error_data.get("error") or "Unknown error"
                except ValueError:
                    error_message = response.text or f"HTTP {response.status_code}"
                
                return APIResponse(
                    success=False,
                    error=f"HTTP {response.status_code}: {error_message}",
                    status_code=response.status_code
                )
        
        except requests.exceptions.ConnectionError:
            # Backend is not reachable
            return APIResponse(
                success=False,
                error=f"Cannot connect to backend at {self.base_url}. Is the backend running?"
            )
        
        except requests.exceptions.Timeout:
            # Request took too long
            return APIResponse(
                success=False,
                error=f"Request timeout after {request_timeout} seconds"
            )
        
        except requests.exceptions.RequestException as e:
            # Other request errors
            return APIResponse(
                success=False,
                error=f"Request failed: {str(e)}"
            )
        
        except Exception as e:
            # Unexpected error
            return APIResponse(
                success=False,
                error=f"Unexpected error: {str(e)}"
            )
    
    def health_check(self) -> APIResponse:
        """
        Check Backend Health
        
        Verifies the backend service is running and responding.
        
        Returns:
            APIResponse with health status
        
        Example:
            response = client.health_check()
            if response.success:
                print(f"Backend version: {response.data['version']}")
                print(f"OpenAI configured: {response.data['openai_configured']}")
        """
        return self._make_request("GET", "/api/health")
    
    def ingest_text(self, text: str, source_name: str = "Direct text input") -> APIResponse:
        """
        Ingest Text Document
        
        Sends text to backend for processing through RAG pipeline.
        
        Args:
            text: Document text content
            source_name: Friendly name for this source
        
        Returns:
            APIResponse with ingestion results
        
        Example:
            response = client.ingest_text(
                text="Your document content...",
                source_name="Immigration Letter"
            )
            if response.success:
                print(f"Created {response.data['chunk_count']} chunks")
                print(f"Source ID: {response.data['source_id']}")
        """
        return self._make_request(
            "POST",
            "/api/agent/ingest/text",
            json_data={
                "text": text,
                "source_name": source_name
            }
        )
    
    def ingest_pdf(self, pdf_bytes: bytes, filename: str) -> APIResponse:
        """
        Ingest PDF Document
        
        Sends PDF file to backend for text extraction and processing.
        
        Process:
        1. Encode PDF bytes to base64 (JSON-compatible format)
        2. Send to backend
        3. Backend decodes, extracts text, processes
        
        Args:
            pdf_bytes: PDF file content as bytes
            filename: Original filename
        
        Returns:
            APIResponse with ingestion results
        
        Example:
            with open("visa.pdf", "rb") as f:
                pdf_bytes = f.read()
            
            response = client.ingest_pdf(pdf_bytes, "visa.pdf")
            if response.success:
                print(f"Extracted from {response.data['metadata']['pages']} pages")
        """
        # Encode PDF bytes to base64 string
        # This allows sending binary data in JSON
        file_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        
        return self._make_request(
            "POST",
            "/api/ingest/pdf",
            json_data={
                "pdf_base64": file_base64,
                "filename": filename,
            },
            timeout=60  # PDFs may take longer to process
        )
    
    def ingest_url(self, url: str) -> APIResponse:
        """
        Ingest Web Page URL
        
        Sends URL to backend for fetching and processing.
        
        Args:
            url: Web page URL to fetch and ingest
        
        Returns:
            APIResponse with ingestion results
        
        Example:
            response = client.ingest_url("https://example.gov.de/immigration")
            if response.success:
                print(f"Fetched: {response.data['metadata']['title']}")
        """
        return self._make_request(
            "POST",
            "/api/agent/ingest/url",
            json_data={"url": url},
            timeout=60  # URL fetching may take time
        )
    
    def get_sources(self) -> APIResponse:
        """
        Get List of All Ingested Sources
        
        Retrieves list of all documents in the knowledge base.
        
        Returns:
            APIResponse with list of sources
        
        Example:
            response = client.get_sources()
            if response.success:
                for source in response.data['sources']:
                    print(f"{source['source_name']} - {source['chunk_count']} chunks")
                print(f"Total sources: {response.data['total_count']}")
        """
        return self._make_request("GET", "/api/sources")
    
    def clear_knowledge_base(self) -> APIResponse:
        """
        Clear All Documents from Knowledge Base
        
        Deletes all ingested documents. This cannot be undone.
        
        Returns:
            APIResponse with success/error message
        
        Example:
            response = client.clear_knowledge_base()
            if response.success:
                print("Knowledge base cleared successfully")
        """
        return self._make_request("POST", "/api/clear")
    
    def query(
        self, 
        question: str, 
        top_k: int = 5,
        chat_history: Optional[List[Dict]] = None
    ) -> APIResponse:
        """
        Query Documents with RAG
        
        Sends a question to the backend for RAG-based answering.
        Returns generated answer with source citations.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve (1-20, default: 5)
            chat_history: Previous conversation messages (optional)
                Format: [{"role": "user"/"assistant", "content": "..."}]
        
        Returns:
            APIResponse with answer and citations
        
        Example:
            response = client.query(
                question="What is the visa deadline?",
                top_k=5,
                chat_history=[
                    {"role": "user", "content": "Tell me about my visa"},
                    {"role": "assistant", "content": "..."}
                ]
            )
            if response.success:
                print(f"Answer: {response.data['answer']}")
                for citation in response.data['citations']:
                    print(f"Source: {citation['source_uri']}")
        """
        return self._make_request(
            "POST",
            "/api/query",
            json_data={
                "question": question,
                "top_k": top_k,
                "chat_history": chat_history or []
            },
            timeout=60  # Query with LLM may take time
        )


# Singleton instance for easy import
# This allows: from app.client.api_client import api_client
_default_client = None

def get_client() -> BackendAPIClient:
    """
    Get or Create Default API Client Instance
    
    This provides a singleton client instance that can be reused
    throughout the application.
    
    Returns:
        BackendAPIClient: The default client instance
    
    Example:
        from app.client.api_client import get_client
        
        client = get_client()
        response = client.health_check()
    """
    global _default_client
    if _default_client is None:
        _default_client = BackendAPIClient()
    return _default_client


# Usage examples in UI code:
#
# Example 1: Check backend health
# ----------------------------
# from app.client.api_client import get_client
# 
# client = get_client()
# response = client.health_check()
# if not response.success:
#     st.error(f"Backend not available: {response.error}")
#     return
#
# Example 2: Ingest text
# ----------------------------
# response = client.ingest_text(
#     text=user_input,
#     source_name="My Document"
# )
# if response.success:
#     st.success(f"Ingested! Chunks: {response.data['chunk_count']}")
# else:
#     st.error(f"Ingestion failed: {response.error}")
#
# Example 3: Upload PDF
# ----------------------------
# uploaded_file = st.file_uploader("Upload PDF", type=["pdf"])
# if uploaded_file:
#     pdf_bytes = uploaded_file.read()
#     response = client.ingest_pdf(pdf_bytes, uploaded_file.name)
#     if response.success:
#         st.success("PDF processed!")
#
# Example 4: Display sources
# ----------------------------
# response = client.get_sources()
# if response.success:
#     for source in response.data['sources']:
#         st.write(f"- {source['source_name']}")
