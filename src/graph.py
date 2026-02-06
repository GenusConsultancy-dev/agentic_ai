"""LangGraph workflow definition for the multi-agent system."""

from langgraph.graph import END, StateGraph

from src.agents import AGENT_NODES
from src.state import AgentState
from src.supervisor import supervisor_node


def create_graph() -> StateGraph:
    """Create and configure the multi-agent workflow graph.
    
    The graph follows a hub-and-spoke pattern:
    - Supervisor is the central hub that routes to agents
    - Agents execute their tasks and return to supervisor
    - Supervisor decides next step until task is complete
    
    Returns:
        Compiled StateGraph ready for execution
    """
    # Create the state graph
    graph = StateGraph(AgentState)
    
    # Add supervisor node (entry point)
    graph.add_node("supervisor", supervisor_node)
    
    # Add all agent nodes
    for agent_name, agent_func in AGENT_NODES.items():
        graph.add_node(agent_name, agent_func)
    
    # Define routing function based on supervisor's decision
    def route_to_agent(state: AgentState) -> str:
        """Route to the next agent based on supervisor decision."""
        next_agent = state.get("next_agent", "FINISH")
        if next_agent == "FINISH":
            return END
        return next_agent
    
    # Add conditional edges from supervisor to agents or END
    graph.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "research": "research",
            "code": "code",
            "files": "files",
            "database": "database",
            "api": "api",
            END: END
        }
    )
    
    # Add edges from all agents back to supervisor
    for agent_name in AGENT_NODES.keys():
        graph.add_edge(agent_name, "supervisor")
    
    # Set entry point
    graph.set_entry_point("supervisor")
    
    return graph.compile()


# Create the compiled graph
workflow = create_graph()
