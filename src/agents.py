"""Agent nodes for the multi-agent system."""

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI

from src.config import config
from src.state import AgentState
from src.tools import AGENT_TOOLS


def create_agent_node(agent_name: str):
    """Factory function to create an agent node.
    
    Args:
        agent_name: Name of the agent (research, code, files, database, api)
        
    Returns:
        Agent node function that processes state and returns updated state
    """
    tools = AGENT_TOOLS.get(agent_name, [])
    
    llm = ChatOpenAI(
        model=config.model_name,
        temperature=config.temperature,
        api_key=config.openai_api_key
    )
    
    if tools:
        llm_with_tools = llm.bind_tools(tools)
    else:
        llm_with_tools = llm
    
    async def agent_node(state: AgentState) -> AgentState:
        """Process the current state and execute agent logic.
        
        The agent will:
        1. Analyze the messages and task context
        2. Decide whether to use tools
        3. Execute tools if needed
        4. Return results to the conversation
        """
        messages = state["messages"]
        task_context = state.get("task_context", {})
        
        # Create system message for this agent
        system_prompt = f"""You are a specialized {agent_name} agent.
Your role: {_get_agent_role(agent_name)}

Guidelines:
- Use your tools to complete the assigned task
- Be concise and focused on the specific task
- Report results clearly
- If you cannot complete the task, explain why

Current task context: {task_context}"""
        
        # Prepare messages with system prompt
        agent_messages = [HumanMessage(content=system_prompt)] + list(messages)
        
        # Get response from LLM
        response = await llm_with_tools.ainvoke(agent_messages)
        
        # If the model wants to use tools, execute them
        if hasattr(response, 'tool_calls') and response.tool_calls:
            tool_results = []
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                
                # Find and execute the tool
                for tool in tools:
                    if tool.name == tool_name:
                        try:
                            result = tool.invoke(tool_args)
                            tool_results.append({
                                "tool": tool_name,
                                "args": tool_args,
                                "result": result
                            })
                        except Exception as e:
                            tool_results.append({
                                "tool": tool_name,
                                "args": tool_args,
                                "error": str(e)
                            })
                        break
            
            # Create summary message with tool results
            result_summary = f"[{agent_name.upper()} AGENT RESULTS]\n"
            for tr in tool_results:
                if "error" in tr:
                    result_summary += f"Tool {tr['tool']} failed: {tr['error']}\n"
                else:
                    result_summary += f"Tool {tr['tool']}: {tr['result']}\n"
            
            final_message = AIMessage(content=result_summary)
            
            # Update intermediate results
            new_results = list(state.get("intermediate_results", []))
            new_results.append({
                "agent": agent_name,
                "tool_results": tool_results
            })
            
            return {
                "messages": [final_message],
                "intermediate_results": new_results
            }
        else:
            # No tool calls, just return the response
            return {
                "messages": [response]
            }
    
    return agent_node


def _get_agent_role(agent_name: str) -> str:
    """Get a detailed role description for each agent."""
    roles = {
        "research": "Search the web for information using Tavily. Find relevant data, articles, and facts to answer questions.",
        "code": "Write and execute Python code to solve problems, perform calculations, and process data.",
        "files": "Read, write, and manage files on the file system. Handle file operations safely.",
        "database": "Execute SQL queries against databases. Create tables, insert data, and run queries.",
        "api": "Make HTTP requests to external APIs. Handle REST endpoints, fetch data, and interact with web services."
    }
    return roles.get(agent_name, "Assist with the task at hand.")


# Create agent nodes
research_agent = create_agent_node("research")
code_agent = create_agent_node("code")
files_agent = create_agent_node("files")
database_agent = create_agent_node("database")
api_agent = create_agent_node("api")

AGENT_NODES = {
    "research": research_agent,
    "code": code_agent,
    "files": files_agent,
    "database": database_agent,
    "api": api_agent,
}
