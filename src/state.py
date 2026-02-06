"""Shared state definitions for the multi-agent system."""

from typing import Annotated, Literal, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    """Shared state passed between all agents in the graph.
    
    Attributes:
        messages: Conversation history with message accumulation
        next_agent: The next agent to route to (set by supervisor)
        task_context: Additional context about the current task
        intermediate_results: Results from agent executions
    """
    messages: Annotated[list[BaseMessage], add_messages]
    next_agent: str
    task_context: dict
    intermediate_results: list[dict]


# Agent identifiers
AGENTS = Literal["research", "code", "files", "database", "api", "FINISH"]

# Agent descriptions for supervisor routing
AGENT_DESCRIPTIONS = {
    "research": "Web search and information retrieval from the internet",
    "code": "Python code generation and execution",
    "files": "File system operations (read, write, list, delete files)",
    "database": "Database queries and operations (SQL)",
    "api": "External API requests and integrations",
    "FINISH": "Task is complete, return final response to user",
}
