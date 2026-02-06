"""Tools for each specialized agent in the multi-agent system."""

import asyncio
import os
import subprocess
import sys
import tempfile
from typing import Any

import httpx
from langchain_core.tools import tool
from tavily import TavilyClient

from src.config import config


# =============================================================================
# Research Agent Tools
# =============================================================================

@tool
def tavily_search(query: str, max_results: int = 5) -> str:
    """Search the web using Tavily API.
    
    Args:
        query: The search query to look up
        max_results: Maximum number of results to return (default 5)
        
    Returns:
        Search results as formatted string
    """
    if not config.tavily_api_key:
        return "Error: TAVILY_API_KEY not configured"
    
    try:
        client = TavilyClient(api_key=config.tavily_api_key)
        response = client.search(query=query, max_results=max_results)
        
        results = []
        for item in response.get("results", []):
            results.append(f"**{item.get('title', 'No title')}**\n{item.get('url', '')}\n{item.get('content', '')}\n")
        
        return "\n---\n".join(results) if results else "No results found."
    except Exception as e:
        return f"Search error: {str(e)}"


# =============================================================================
# Code Agent Tools
# =============================================================================

@tool
def execute_python(code: str) -> str:
    """Execute Python code in a sandboxed subprocess.
    
    Args:
        code: Python code to execute
        
    Returns:
        Output from code execution (stdout + stderr)
    """
    try:
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            result = subprocess.run(
                [sys.executable, temp_path],
                capture_output=True,
                text=True,
                timeout=config.code_execution_timeout,
                cwd=tempfile.gettempdir()
            )
            
            output = ""
            if result.stdout:
                output += f"STDOUT:\n{result.stdout}\n"
            if result.stderr:
                output += f"STDERR:\n{result.stderr}\n"
            if result.returncode != 0:
                output += f"Exit code: {result.returncode}"
            
            return output.strip() or "Code executed successfully (no output)"
        finally:
            os.unlink(temp_path)
            
    except subprocess.TimeoutExpired:
        return f"Error: Code execution timed out after {config.code_execution_timeout} seconds"
    except Exception as e:
        return f"Execution error: {str(e)}"


# =============================================================================
# File Agent Tools
# =============================================================================

def _validate_path(path: str) -> str | None:
    """Validate that path is within allowed directory. Returns error message if invalid."""
    try:
        abs_path = os.path.abspath(path)
        allowed = os.path.abspath(config.allowed_file_path)
        if not abs_path.startswith(allowed):
            return f"Error: Path '{path}' is outside allowed directory '{config.allowed_file_path}'"
        return None
    except Exception as e:
        return f"Path validation error: {str(e)}"


@tool
def read_file(file_path: str) -> str:
    """Read contents of a file.
    
    Args:
        file_path: Path to the file to read
        
    Returns:
        File contents or error message
    """
    if error := _validate_path(file_path):
        return error
    
    try:
        with open(file_path, 'r') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {file_path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"


@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file.
    
    Args:
        file_path: Path to the file to write
        content: Content to write to the file
        
    Returns:
        Success message or error
    """
    if error := _validate_path(file_path):
        return error
    
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w') as f:
            f.write(content)
        return f"Successfully wrote {len(content)} characters to {file_path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"


@tool
def list_directory(directory_path: str) -> str:
    """List contents of a directory.
    
    Args:
        directory_path: Path to the directory to list
        
    Returns:
        Directory listing or error message
    """
    if error := _validate_path(directory_path):
        return error
    
    try:
        entries = []
        for entry in os.scandir(directory_path):
            entry_type = "DIR " if entry.is_dir() else "FILE"
            size = "" if entry.is_dir() else f" ({entry.stat().st_size} bytes)"
            entries.append(f"{entry_type} {entry.name}{size}")
        
        return "\n".join(sorted(entries)) if entries else "Directory is empty"
    except FileNotFoundError:
        return f"Error: Directory not found: {directory_path}"
    except Exception as e:
        return f"Error listing directory: {str(e)}"


# =============================================================================
# Database Agent Tools
# =============================================================================

@tool
def execute_sql(query: str, database_path: str = "") -> str:
    """Execute SQL query against SQLite database.
    
    Args:
        query: SQL query to execute
        database_path: Path to SQLite database (uses default if not provided)
        
    Returns:
        Query results or error message
    """
    import sqlite3
    
    db_path = database_path or config.database_url.replace("sqlite:///", "")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute(query)
        
        if query.strip().upper().startswith("SELECT"):
            rows = cursor.fetchall()
            columns = [desc[0] for desc in cursor.description] if cursor.description else []
            
            if not rows:
                return "Query returned no results"
            
            result = " | ".join(columns) + "\n" + "-" * 40 + "\n"
            for row in rows:
                result += " | ".join(str(val) for val in row) + "\n"
            return result
        else:
            conn.commit()
            return f"Query executed successfully. Rows affected: {cursor.rowcount}"
            
    except Exception as e:
        return f"SQL error: {str(e)}"
    finally:
        if 'conn' in locals():
            conn.close()


# =============================================================================
# API Agent Tools
# =============================================================================

@tool
def http_request(
    url: str,
    method: str = "GET",
    headers: dict[str, str] | None = None,
    body: str | None = None
) -> str:
    """Make an HTTP request to an external API.
    
    Args:
        url: URL to request
        method: HTTP method (GET, POST, PUT, DELETE)
        headers: Optional request headers
        body: Optional request body (for POST/PUT)
        
    Returns:
        Response body or error message
    """
    try:
        with httpx.Client(timeout=30.0) as client:
            response = client.request(
                method=method.upper(),
                url=url,
                headers=headers or {},
                content=body
            )
            
            result = f"Status: {response.status_code}\n"
            result += f"Headers: {dict(response.headers)}\n"
            result += f"Body:\n{response.text[:2000]}"
            
            if len(response.text) > 2000:
                result += f"\n... (truncated, {len(response.text)} total chars)"
            
            return result
    except httpx.TimeoutException:
        return f"Error: Request to {url} timed out"
    except Exception as e:
        return f"HTTP error: {str(e)}"


# =============================================================================
# Tool Collections for Each Agent
# =============================================================================

RESEARCH_TOOLS = [tavily_search]
CODE_TOOLS = [execute_python]
FILE_TOOLS = [read_file, write_file, list_directory]
DATABASE_TOOLS = [execute_sql]
API_TOOLS = [http_request]

AGENT_TOOLS = {
    "research": RESEARCH_TOOLS,
    "code": CODE_TOOLS,
    "files": FILE_TOOLS,
    "database": DATABASE_TOOLS,
    "api": API_TOOLS,
}
