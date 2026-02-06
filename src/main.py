"""Main entry point for the multi-agent system."""

import asyncio
import sys

from langchain_core.messages import HumanMessage

from src.config import config
from src.graph import workflow
from src.state import AgentState


async def run_agent(query: str, verbose: bool = True) -> str:
    """Run the multi-agent system with a user query.
    
    Args:
        query: The user's question or task
        verbose: Whether to print intermediate steps
        
    Returns:
        Final response from the agent system
    """
    # Validate configuration
    missing = config.validate()
    if missing:
        return f"Error: Missing required configuration: {', '.join(missing)}"
    
    # Create initial state
    initial_state: AgentState = {
        "messages": [HumanMessage(content=query)],
        "next_agent": "",
        "task_context": {"original_query": query},
        "intermediate_results": []
    }
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Query: {query}")
        print(f"{'='*60}\n")
    
    # Run the workflow
    final_state = None
    async for state in workflow.astream(initial_state):
        if verbose:
            # Print which node just executed
            for node_name, node_output in state.items():
                if node_name == "supervisor":
                    next_agent = node_output.get("next_agent", "?")
                    reason = node_output.get("task_context", {}).get("last_routing_reason", "")
                    print(f"[SUPERVISOR] Routing to: {next_agent}")
                    if reason:
                        print(f"  Reason: {reason}")
                else:
                    messages = node_output.get("messages", [])
                    if messages:
                        print(f"[{node_name.upper()}] {messages[-1].content[:200]}...")
                print()
        
        final_state = state
    
    # Extract final response
    if final_state:
        # Get the last set of messages from any node
        for node_output in final_state.values():
            messages = node_output.get("messages", [])
            if messages:
                return messages[-1].content
    
    return "No response generated"


async def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python -m src.main '<your query>'")
        print("\nExample:")
        print("  python -m src.main 'What is the capital of France?'")
        print("  python -m src.main 'Calculate the factorial of 10'")
        sys.exit(1)
    
    query = " ".join(sys.argv[1:])
    result = await run_agent(query)
    
    print(f"\n{'='*60}")
    print("FINAL RESULT:")
    print(f"{'='*60}")
    print(result)


if __name__ == "__main__":
    asyncio.run(main())
