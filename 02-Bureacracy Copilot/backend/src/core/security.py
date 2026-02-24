"""
Security utilities for input validation and sanitization

Provides functions to validate and sanitize user input to prevent security issues.
"""

import re
import ipaddress
import socket
from urllib.parse import urlparse
from typing import Optional


def validate_text_length(text: str, max_length: int) -> tuple[bool, Optional[str]]:
    """
    Validate Text Length
    
    Checks if text exceeds maximum allowed length.
    Prevents processing of excessively large documents.
    
    Args:
        text: Input text to validate
        max_length: Maximum allowed length in characters
    
    Returns:
        tuple: (is_valid, error_message)
            - (True, None) if valid
            - (False, error_message) if invalid
    """
    if len(text) > max_length:
        return False, f"Text exceeds maximum length of {max_length} characters (got {len(text)})"
    return True, None


def sanitize_filename(filename: str) -> str:
    """
    Sanitize Filename
    
    Removes potentially dangerous characters from filenames.
    Prevents path traversal attacks (e.g., ../../../etc/passwd)
    
    Args:
        filename: Original filename from user
    
    Returns:
        str: Sanitized filename safe for storage
    
    Security considerations:
    - Removes directory separators (/ \\)
    - Removes null bytes
    - Limits to alphanumeric, dash, underscore, dot
    """
    # Remove path components
    filename = filename.replace("\\", "_").replace("/", "_")
    
    # Remove null bytes (security issue)
    filename = filename.replace("\x00", "")
    
    # Allow only safe characters
    filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    
    # Prevent hidden files on Unix
    if filename.startswith('.'):
        filename = '_' + filename
    
    return filename


def validate_url(url: str) -> tuple[bool, Optional[str]]:
    """
    Validate URL Format and Protocol
    
    Ensures URL is well-formed and uses allowed protocols.
    Prevents SSRF (Server-Side Request Forgery) attacks.
    
    Args:
        url: URL to validate
    
    Returns:
        tuple: (is_valid, error_message)
    
    Security checks:
    - Must be valid URL format
    - Only http:// and https:// allowed
    - Has a valid hostname
    """
    try:
        parsed = urlparse(url)
        
        # Check scheme
        if parsed.scheme not in ['http', 'https']:
            return False, f"Invalid URL scheme: {parsed.scheme}. Only http and https are allowed."
        
        # Check hostname exists
        if not parsed.netloc:
            return False, "URL must have a valid hostname"

        hostname = parsed.hostname
        if not hostname:
            return False, "URL must include a hostname"

        hostname_lower = hostname.lower()
        blocked_hostnames = {"localhost", "localhost.localdomain"}
        if hostname_lower in blocked_hostnames:
            return False, f"Blocked URL target: {hostname_lower}"

        try:
            resolved = socket.getaddrinfo(hostname, None)
        except socket.gaierror:
            return False, f"Could not resolve hostname: {hostname}"

        for entry in resolved:
            ip_value = entry[4][0]
            try:
                ip_obj = ipaddress.ip_address(ip_value)
            except ValueError:
                continue

            if (
                ip_obj.is_private
                or ip_obj.is_loopback
                or ip_obj.is_link_local
                or ip_obj.is_reserved
                or ip_obj.is_multicast
                or ip_obj.is_unspecified
            ):
                return False, f"Blocked URL target: {hostname} resolved to restricted IP {ip_value}"
        
        return True, None
    
    except Exception as e:
        return False, f"Invalid URL format: {str(e)}"


def sanitize_source_name(name: str, max_length: int = 200) -> str:
    """
    Sanitize Source Name
    
    Cleans up user-provided source names for display and storage.
    
    Args:
        name: Original source name
        max_length: Maximum length for source name
    
    Returns:
        str: Sanitized source name
    """
    # Strip whitespace
    name = name.strip()
    
    # Truncate if too long
    if len(name) > max_length:
        name = name[:max_length] + "..."
    
    # Remove control characters
    name = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', name)
    
    return name if name else "Untitled Document"


# Security warning message for PII
PII_WARNING = """
⚠️ **Privacy Notice**: This application sends your documents to OpenAI for processing.
Do not upload documents containing sensitive personal information (SSN, passwords, etc.)
unless you understand and accept the risks.
"""
