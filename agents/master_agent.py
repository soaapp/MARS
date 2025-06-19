import os
import sys
from typing import Dict, List, Annotated, TypedDict, Union, Any, Literal
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the SQL and Qdrant agents
from agents.sql_agent import run_sql_agent
from agents.qdrant_agent import run_qdrant_agent

# Define the state schema for our multi-agent system
class AgentState(TypedDict):
    messages: List[BaseMessage]  # Chat history
    query: str                   # User's query
    source: str                  # Which data source to use ("qdrant" or "sql")
    result: str                  # Final result to return to user

# Initialize the LLM
def get_llm(temperature=0):
    return ChatOllama(model="llama4", temperature=temperature)

# Create the router agent that will decide which data source to use
def create_router_agent():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are the Router Agent in a Multi-Agent RAG System. 
        Your job is to analyze the user's query and determine which data source would be best to answer it: 
        either 'qdrant' for document retrieval or 'sql' for structured database queries.
        
        Respond with ONLY 'qdrant' or 'sql' based on the nature of the query.
        
        For example:
        - If the query is about documents, reports, or unstructured information, respond with 'qdrant'
        - If the query is about specific records, people, or structured data, respond with 'sql'
        """),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="{query}"),
    ])
    
    return prompt | get_llm() | StrOutputParser()

# Function to route to the appropriate agent
def route(state: AgentState) -> Dict[str, Any]:
    router = create_router_agent()
    source = router.invoke({"query": state["query"], "messages": state.get("messages", [])})
    
    # Ensure the source is either 'qdrant' or 'sql'
    if source.lower().strip() not in ["qdrant", "sql"]:
        source = "qdrant"  # Default to qdrant if the response is unexpected
    
    return {"source": source.lower().strip()}

# Function to call the SQL agent
def call_sql_agent(state: AgentState) -> Dict[str, Any]:
    result = run_sql_agent(state["query"], state.get("messages", []))
    return {"result": result}

# Function to call the Qdrant agent
def call_qdrant_agent(state: AgentState) -> Dict[str, Any]:
    result = run_qdrant_agent(state["query"], state.get("messages", []))
    return {"result": result}

# Function to decide which agent to call next
def decide_next(state: AgentState) -> Literal["sql_agent", "qdrant_agent"]:
    if state["source"] == "sql":
        return "sql_agent"
    else:
        return "qdrant_agent"

# Create the graph for our multi-agent system
def create_agent_graph():
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("router", route)
    workflow.add_node("sql_agent", call_sql_agent)
    workflow.add_node("qdrant_agent", call_qdrant_agent)
    
    # Define edges
    workflow.add_conditional_edges(
        "router",
        decide_next,
        {
            "sql_agent": "sql_agent",
            "qdrant_agent": "qdrant_agent"
        }
    )
    workflow.add_edge("sql_agent", END)
    workflow.add_edge("qdrant_agent", END)
    
    # Set the entry point
    workflow.set_entry_point("router")
    
    # Compile the graph
    return workflow.compile()

# Function to run the agent graph
def run_master_agent(query: str, messages: List[BaseMessage] = None) -> Dict[str, Any]:
    if messages is None:
        messages = []
    
    graph = create_agent_graph()
    result = graph.invoke({"query": query, "messages": messages})
    
    return result
