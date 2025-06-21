import os
import sys
import time
import logging
from typing import Dict, Any, List, TypedDict, Annotated, Sequence, Tuple
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END
from agents.sql_agent import run_sql_agent
from agents.qdrant_agent import run_qdrant_agent
from agents.azure_doc_agent import run_azure_doc_agent

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MARS')

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the SQL and Qdrant agents
from agents.sql_agent import run_sql_agent
from agents.qdrant_agent import run_qdrant_agent

# Define the state schema for our multi-agent system
class AgentState(TypedDict):
    messages: List[BaseMessage]  # Chat history
    query: str                   # User's query
    source: str                  # Which data source to use ("qdrant", "sql", or "azure_doc")
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

# Create the router agent that will decide which data source to use
def create_router_agent():
    # Rule-based approach for routing queries
    def router(query):
        query_lower = query.lower()
        
        # If query contains document analysis terms, route to Azure Doc agent
        document_terms = ['document', 'pdf', 'scan', 'image', 'ocr', 'analyze document', 'extract text', 'extract table', 'document intelligence']
        for term in document_terms:
            if term in query_lower:
                logger.info(f"Rule-based routing detected document term '{term}' - routing to Azure Doc agent")
                return "azure_doc"
        
        # If query contains employee-related terms, route to SQL
        employee_terms = ['employee', 'employees', 'people', 'person', 'staff', 'worker', 'workers', 'team', 'member', 'members', 'list']
        for term in employee_terms:
            if term in query_lower:
                logger.info(f"Rule-based routing detected employee term '{term}' - routing to SQL")
                return "sql"
        
        # Check for URLs which might indicate document analysis
        if "http://" in query_lower or "https://" in query_lower:
            if any(ext in query_lower for ext in ['.pdf', '.jpg', '.jpeg', '.png', '.tiff', '.docx']):
                logger.info(f"Rule-based routing detected document URL - routing to Azure Doc agent")
                return "azure_doc"
        
        # Otherwise use the LLM for routing
        prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Router Agent. Analyze the query and respond ONLY with 'sql', 'qdrant', or 'azure_doc'.
            - 'sql': For queries about people, employees, records, database information, or any structured data
            - 'qdrant': For queries about general knowledge base information or documents without specific analysis needs
            - 'azure_doc': For queries about document analysis, text extraction, OCR, or table extraction from documents"""),
            HumanMessage(content="{query}"),
        ])
        
        llm = get_llm()
        chain = prompt | llm | StrOutputParser()
        result = chain.invoke({"query": query})
        return result.strip().lower()
    
    return router


# Function to route the query to the appropriate agent
def route(state: AgentState) -> Dict[str, Any]:
    query = state["query"]
    
    logger.info(f"Routing query: {query}")
    start_time = time.time()
    
    # Use the rule-based router to decide which data source to use
    router_func = create_router_agent()
    source = router_func(query)
    
    end_time = time.time()
    logger.info(f"Router decided on source: {source} (took {end_time - start_time:.2f} seconds)")
    
    return {"source": source}

# Function to call the SQL agent
def call_sql_agent(state: AgentState) -> Dict[str, Any]:
    result = run_sql_agent(state["query"], state.get("messages", []))
    return {"result": result}

# Function to call the Qdrant agent
def call_qdrant_agent(state: AgentState) -> Dict[str, Any]:
    result = run_qdrant_agent(state["query"], state.get("messages", []))
    return {"result": result}

# Function to call the Azure Document Intelligence agent
def call_azure_doc_agent(state: AgentState) -> Dict[str, Any]:
    result = run_azure_doc_agent(state["query"], state.get("messages", []))
    return {"result": result}

# Function to decide which agent to call next
def decide_next(state: AgentState) -> str:
    if state["source"] == "sql":
        return "sql_agent"
    elif state["source"] == "azure_doc":
        return "azure_doc_agent"
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
    workflow.add_node("azure_doc_agent", call_azure_doc_agent)
    
    # Define edges
    workflow.add_conditional_edges(
        "router",
        decide_next,
        {
            "sql_agent": "sql_agent",
            "qdrant_agent": "qdrant_agent",
            "azure_doc_agent": "azure_doc_agent"
        }
    )
    workflow.add_edge("sql_agent", END)
    workflow.add_edge("qdrant_agent", END)
    workflow.add_edge("azure_doc_agent", END)
    
    # Set the entry point
    workflow.set_entry_point("router")
    
    # Compile the graph
    return workflow.compile()

# Function to run the master agent
def run_master_agent(query: str, messages: List[BaseMessage] = None) -> Dict[str, Any]:
    if messages is None:
        messages = []
    
    logger.info(f"Master agent received query: {query}")
    start_time = time.time()
    
    # Create the agent graph
    graph = create_agent_graph()
    
    # Run the graph
    result = graph.invoke({"query": query, "messages": messages})
    
    end_time = time.time()
    logger.info(f"Master agent completed processing in {end_time - start_time:.2f} seconds")
    logger.info(f"Result source: {result['source']}")
    
    return result
