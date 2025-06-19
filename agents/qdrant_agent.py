import os
import sys
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_community.embeddings import OllamaEmbeddings
from langchain_core.output_parsers import StrOutputParser

# Add the parent directory to sys.path to import from qdrant module
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qdrant.client_setup import client, collection_name

# Initialize embeddings model
embeddings = OllamaEmbeddings(model="llama4")

# Function to search documents in Qdrant
def search_documents(query: str, limit: int = 3) -> List[Dict[str, Any]]:
    try:
        # Generate embeddings for the query
        query_vector = embeddings.embed_query(query)
        
        # Search for similar documents in Qdrant
        search_results = client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=limit
        )
        
        # Extract and return the results
        results = []
        for result in search_results:
            payload = result.payload if hasattr(result, 'payload') else {}
            score = result.score if hasattr(result, 'score') else 0.0
            results.append({
                "id": result.id,
                "score": score,
                "content": payload.get("content", ""),
                "metadata": {k: v for k, v in payload.items() if k != "content"}
            })
        
        return results
    except Exception as e:
        return [{"error": str(e)}]

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
        SystemMessage(content="You are a helpful assistant that answers questions based on the provided documents."),
        MessagesPlaceholder(variable_name="messages"),
        ("human", "Based on the following documents, answer this question: {query}\n\nDocuments: {documents}"),
    ])
    
    # Chain for generating response
    response_chain = response_prompt | ChatOllama(model="llama4", temperature=0) | StrOutputParser()
    
    def qdrant_agent(query: str, messages: List[BaseMessage] = None) -> str:
        if messages is None:
            messages = []
        
        # Search for relevant documents
        documents = search_documents(query)
        
        # Check for errors
        if documents and "error" in documents[0]:
            return f"Error searching documents: {documents[0]['error']}"
        
        # If no documents found
        if not documents:
            return "No relevant documents found for your query. Please try a different search term or approach."
        
        # Prepare context from retrieved documents
        context = "\n\n".join([f"Document {i+1} (ID: {doc['id']}, Score: {doc['score']:.2f}):\n{doc['content']}" 
                             for i, doc in enumerate(documents)])
        
        # Generate response using the retrieved documents
        response = response_chain.invoke({
            "query": f"Based on the following documents, answer this question: {query}\n\nDocuments:\n{context}",
            "messages": messages
        })
        
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
