import time
import logging
from typing import Dict, List, Annotated, TypedDict, Union
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, BaseMessage
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MARS')

# Define the state schema for our multi-agent system
class AgentState(TypedDict):
    messages: List[BaseMessage]  # Chat history
    query: str                   # User's query
    source: str                  # Which data source to use ("qdrant" or "sql")
    result: str                  # Final result to return to user

# Initialize the LLM
def get_llm(temperature=0):
    logger.info(f"Initializing Llama3.2 with temperature {temperature}")
    return ChatOllama(
        model="llama3.2", 
        temperature=temperature,
        timeout=30,  # 30 second timeout
        stop=["\n\n"]  # Stop on double newline to encourage brevity
    )

# Create the master agent that will decide which data source to use
def create_master_agent():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a Router Agent. Analyze the query and respond ONLY with 'sql' or 'qdrant'.
        - 'sql': For queries about people, employees, records, database information, or any structured data
        - 'qdrant': ONLY for queries about documents or knowledge base information
        
        IMPORTANT: If the query mentions employees, people, or database records, ALWAYS use 'sql'."""),
        HumanMessage(content="{query}"),
    ])
    
    return prompt | get_llm() | StrOutputParser()

# Create the graph for our multi-agent system
def create_agent_graph():
    # Define the nodes
    master_agent = create_master_agent()
    
    # Define the graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("master", master_agent)
    
    # Define edges
    workflow.add_edge("master", END)
    
    # Set the entry point
    workflow.set_entry_point("master")
    
    # Compile the graph
    return workflow.compile()

# Function to run the agent graph
def run_agent_graph(query: str, messages: List[BaseMessage] = None):
    if messages is None:
        messages = []
    
    graph = create_agent_graph()
    result = graph.invoke({"query": query, "messages": messages})
    
    return result
