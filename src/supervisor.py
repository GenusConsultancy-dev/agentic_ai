"""Supervisor agent for routing between specialized agents."""

from typing import Literal

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from src.config import config
from src.state import AGENT_DESCRIPTIONS, AgentState


class RouterDecision(BaseModel):
    """Structured output for supervisor routing decisions."""
    
    next_agent: Literal["research", "code", "files", "database", "api", "FINISH"] = Field(
        description="The next agent to route to, or FINISH if task is complete"
    )
    reasoning: str = Field(
        description="Brief explanation of why this agent was chosen"
    )


async def supervisor_node(state: AgentState) -> AgentState:
    """Supervisor agent that decides which agent to route to next.
    
    The supervisor analyzes:
    1. The original user request
    2. Conversation history and intermediate results
    3. What work remains to be done
    
    Then routes to the most appropriate agent or FINISH.
    """
    messages = state["messages"]
    intermediate_results = state.get("intermediate_results", [])
    task_context = state.get("task_context", {})
    
    # Build agent descriptions for the prompt
    agent_desc = "\n".join([
        f"- {name}: {desc}" 
        for name, desc in AGENT_DESCRIPTIONS.items()
    ])
    
    # Build summary of work done so far
    work_done = ""
    if intermediate_results:
        work_done = "\n\nWork completed so far:\n"
        for result in intermediate_results:
            agent = result.get("agent", "unknown")
            tools_used = [tr.get("tool", "?") for tr in result.get("tool_results", [])]
            work_done += f"- {agent} agent used: {', '.join(tools_used)}\n"
    
    system_prompt = f"""You are a supervisor agent that routes tasks to specialized agents.

Available agents:
{agent_desc}

Your job is to:
1. Analyze the user's request and conversation history
2. Determine which agent should handle the next step
3. Route to FINISH when the task is fully complete
{work_done}
Guidelines:
- Only route to agents that can make progress on the task
- Use research agent for web searches and information lookup
- Use code agent for calculations, data processing, or code generation
- Use files agent for reading/writing files
- Use database agent for SQL queries
- Use api agent for HTTP requests to external services
- Route to FINISH only when the user's request is fully addressed

Current task context: {task_context}"""

    # Create LLM with structured output
    llm = ChatOpenAI(
        model=config.model_name,
        temperature=0,  # Deterministic routing
        api_key=config.openai_api_key
    )
    
    structured_llm = llm.with_structured_output(RouterDecision)
    
    # Prepare messages
    supervisor_messages = [HumanMessage(content=system_prompt)] + list(messages)
    
    # Get routing decision
    decision: RouterDecision = await structured_llm.ainvoke(supervisor_messages)
    
    # Update task context with routing info
    new_context = dict(task_context)
    new_context["last_routing_reason"] = decision.reasoning
    
    return {
        "next_agent": decision.next_agent,
        "task_context": new_context
    }
