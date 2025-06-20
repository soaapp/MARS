import os
import sys
import time
import logging
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser
from langchain_community.embeddings import OllamaEmbeddings
from qdrant_client.models import Distance, VectorParams
from qdrant.client_setup import client, collection_name

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MARS')

# Add the parent directory to sys.path to import from qdrant module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdrant.client_setup import client, collection_name

# Initialize embeddings
embeddings = OllamaEmbeddings(model="llama3.2")

# Function to search documents in Qdrant
def search_documents(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    logger.info(f"Searching documents for query: {query}")
    start_time = time.time()
    
    # Get embeddings for the query
    embedding_start = time.time()
    query_vector = embeddings.embed_query(query)
    embedding_end = time.time()
    logger.info(f"Generated embeddings in {embedding_end - embedding_start:.2f} seconds")
    
    # Search in Qdrant
    search_start = time.time()
    search_results = client.search(
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit
    )
    search_end = time.time()
    logger.info(f"Searched Qdrant in {search_end - search_start:.2f} seconds, found {len(search_results)} results")
    
    # Format results
    results = []
    for result in search_results:
        payload = result.payload if hasattr(result, 'payload') else {}
        score = result.score if hasattr(result, 'score') else 0.0
        results.append({
            "id": result.id,
            "score": score,
            "content": payload.get("content", ""),
            "metadata": {
                "title": payload.get("metadata", {}).get("title", "Untitled"),
                "source": payload.get("metadata", {}).get("source", "Unknown")
            }
        })
    
    end_time = time.time()
    logger.info(f"Document search completed in {end_time - start_time:.2f} seconds")
    return results

# Create the Qdrant agent
def create_qdrant_agent():
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a Document Retrieval Agent that helps find relevant information from a document store.
        Given a user query, you will search for relevant documents and provide a helpful response based on the retrieved information.
        If no relevant documents are found, acknowledge that and suggest alternative approaches.
        Always cite your sources by mentioning document IDs.
        """),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="{query}"),
    ])
    
    # Create a prompt template for generating a response based on retrieved documents
    response_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="""You are a Document Assistant. Answer based only on the retrieved documents. Be concise."""),
        HumanMessage(content="""Query: {query}
        
        Documents:
        {documents}
        """),
    ])
    
    # Chain for generating response
    response_chain = response_prompt | ChatOllama(
        model="llama3.2", 
        temperature=0,
        timeout=30,  # 30 second timeout
        stop=["\n\n"]  # Stop on double newline to encourage brevity
    ) | StrOutputParser()
    
    def qdrant_agent(query: str, messages: List[BaseMessage] = None) -> str:
        if messages is None:
            messages = []
        
        logger.info(f"Qdrant Agent processing query: {query}")
        start_time = time.time()
        
        # Search for relevant documents
        search_results = search_documents(query)
        
        # If no results, return a message
        if not search_results:
            logger.info("No relevant documents found")
            return "I couldn't find any relevant documents to answer your query."
        
        # Format the documents for the prompt
        documents = []
        for i, doc in enumerate(search_results):
            documents.append(f"Document {i+1}:\nTitle: {doc['metadata']['title']}\nSource: {doc['metadata']['source']}\nContent: {doc['content']}\nRelevance Score: {doc['score']}")
        
        documents_str = "\n\n".join(documents)
        logger.info(f"Formatted {len(search_results)} documents for response generation")
        
        # Generate response using the LLM
        response_start = time.time()
        response = response_chain.invoke({"query": query, "documents": documents_str})
        response_end = time.time()
        logger.info(f"Generated response in {response_end - response_start:.2f} seconds")
        
        end_time = time.time()
        logger.info(f"Qdrant Agent completed processing in {end_time - start_time:.2f} seconds")
        
        return response
    
    return qdrant_agent

# Function to run the Qdrant agent
def run_qdrant_agent(query: str, messages: List[BaseMessage] = None) -> str:
    agent = create_qdrant_agent()
    return agent(query, messages)

# Function to add a document to Qdrant
def add_document(content: str, metadata: Dict[str, Any] = None) -> str:
    try:
        if metadata is None:
            metadata = {}
        
        # Generate embeddings for the document
        doc_vector = embeddings.embed_query(content)
        
        # Create a unique ID for the document
        import uuid
        doc_id = str(uuid.uuid4())
        
        # Add the document to Qdrant
        client.upsert(
            collection_name=collection_name,
            points=[
                {
                    "id": doc_id,
                    "vector": doc_vector,
                    "payload": {"content": content, **metadata}
                }
            ]
        )
        
        return f"Document added successfully with ID: {doc_id}"
    except Exception as e:
        return f"Error adding document: {str(e)}"
