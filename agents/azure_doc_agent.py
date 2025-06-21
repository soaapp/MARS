import os
import time
import logging
import json
from typing import List, Dict, Any, Optional
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser

# Load environment variables from .env file
load_dotenv()

# Azure Document Intelligence imports
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeResult
from azure.ai.documentintelligence.models import AnalyzeDocumentRequest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MARS')

# Azure Document Intelligence configuration
# These should be set as environment variables
AZURE_ENDPOINT = os.environ.get("AZURE_DOC_ENDPOINT")
AZURE_KEY = os.environ.get("AZURE_DOC_KEY")

# Check if credentials are available
if not AZURE_ENDPOINT or not AZURE_KEY:
    logger.warning("Azure Document Intelligence credentials not found in environment variables.")
    logger.warning("Please set AZURE_DOC_ENDPOINT and AZURE_DOC_KEY in your .env file.")
    logger.warning("Using .env.example as a template.")
    # Don't set default values for security reasons

# Helper functions for document analysis
def get_words(page, line):
    """Extract words from a line in a document page."""
    result = []
    for word in page.words:
        if _in_span(word, line.spans):
            result.append(word)
    return result

def _in_span(word, spans):
    """Check if a word is within a span."""
    for span in spans:
        if word.span.offset >= span.offset and (
            word.span.offset + word.span.length
        ) <= (span.offset + span.length):
            return True
    return False

def analyze_document(document_url: str, model_id: str = "prebuilt-layout") -> Dict[str, Any]:
    """
    Analyze a document using Azure Document Intelligence.
    
    Args:
        document_url: URL or local path to the document
        model_id: The model ID to use for analysis (default: prebuilt-layout)
        
    Returns:
        Dict containing the analysis results
    """
    try:
        start_time = time.time()
        logger.info(f"Starting document analysis with model: {model_id}")
        
        # Initialize the Document Intelligence client
        document_intelligence_client = DocumentIntelligenceClient(
            endpoint=AZURE_ENDPOINT, 
            credential=AzureKeyCredential(AZURE_KEY)
        )
        
        # Determine if the document is a URL or local file
        if document_url.startswith(('http://', 'https://')):
            logger.info(f"Analyzing document from URL: {document_url}")
            poller = document_intelligence_client.begin_analyze_document(
                model_id, AnalyzeDocumentRequest(url_source=document_url)
            )
        else:
            # For local files, we need to read the file and send its content
            logger.info(f"Analyzing local document: {document_url}")
            with open(document_url, "rb") as f:
                document_content = f.read()
            poller = document_intelligence_client.begin_analyze_document(
                model_id, document_content
            )
        
        # Get the result
        result: AnalyzeResult = poller.result()
        
        # Convert the result to a dictionary for easier handling
        analysis_result = {
            "document_type": model_id,
            "pages": [],
            "tables": [],
            "contains_handwritten": False,
            "processing_time_seconds": time.time() - start_time
        }
        
        # Check for handwritten content
        if result.styles and any([style.is_handwritten for style in result.styles]):
            analysis_result["contains_handwritten"] = True
        
        # Process pages
        for page in result.pages:
            page_data = {
                "page_number": page.page_number,
                "width": page.width,
                "height": page.height,
                "unit": page.unit,
                "lines": []
            }
            
            # Process lines and words
            if page.lines:
                for line_idx, line in enumerate(page.lines):
                    words = get_words(page, line)
                    line_data = {
                        "line_number": line_idx,
                        "content": line.content,
                        "polygon": line.polygon,
                        "words": []
                    }
                    
                    # Process words
                    for word in words:
                        line_data["words"].append({
                            "content": word.content,
                            "confidence": word.confidence
                        })
                    
                    page_data["lines"].append(line_data)
            
            # Process selection marks
            if page.selection_marks:
                page_data["selection_marks"] = []
                for mark in page.selection_marks:
                    page_data["selection_marks"].append({
                        "state": mark.state,
                        "polygon": mark.polygon,
                        "confidence": mark.confidence
                    })
            
            analysis_result["pages"].append(page_data)
        
        # Process tables
        if result.tables:
            for table_idx, table in enumerate(result.tables):
                table_data = {
                    "table_number": table_idx,
                    "row_count": table.row_count,
                    "column_count": table.column_count,
                    "regions": [],
                    "cells": []
                }
                
                # Process table regions
                if table.bounding_regions:
                    for region in table.bounding_regions:
                        table_data["regions"].append({
                            "page_number": region.page_number,
                            "polygon": region.polygon
                        })
                
                # Process cells
                for cell in table.cells:
                    cell_data = {
                        "row_index": cell.row_index,
                        "column_index": cell.column_index,
                        "content": cell.content,
                        "regions": []
                    }
                    
                    # Process cell regions
                    if cell.bounding_regions:
                        for region in cell.bounding_regions:
                            cell_data["regions"].append({
                                "page_number": region.page_number,
                                "polygon": region.polygon
                            })
                    
                    table_data["cells"].append(cell_data)
                
                analysis_result["tables"].append(table_data)
        
        logger.info(f"Document analysis completed in {analysis_result['processing_time_seconds']:.2f} seconds")
        return analysis_result
    
    except Exception as e:
        logger.error(f"Error analyzing document: {str(e)}")
        return {"error": str(e)}

def extract_text_from_document(document_url: str) -> str:
    """
    Extract all text content from a document.
    
    Args:
        document_url: URL or local path to the document
        
    Returns:
        String containing all text from the document
    """
    try:
        analysis_result = analyze_document(document_url)
        
        if "error" in analysis_result:
            return f"Error extracting text: {analysis_result['error']}"
        
        # Extract text from all pages
        all_text = []
        for page in analysis_result["pages"]:
            page_text = []
            for line in page["lines"]:
                page_text.append(line["content"])
            
            all_text.append("\n".join(page_text))
        
        return "\n\n".join(all_text)
    
    except Exception as e:
        logger.error(f"Error extracting text from document: {str(e)}")
        return f"Error extracting text: {str(e)}"

def extract_tables_from_document(document_url: str) -> List[Dict[str, Any]]:
    """
    Extract tables from a document.
    
    Args:
        document_url: URL or local path to the document
        
    Returns:
        List of tables with their content
    """
    try:
        analysis_result = analyze_document(document_url)
        
        if "error" in analysis_result:
            return [{"error": analysis_result["error"]}]
        
        return analysis_result["tables"]
    
    except Exception as e:
        logger.error(f"Error extracting tables from document: {str(e)}")
        return [{"error": str(e)}]

# Create the Azure Document Intelligence agent
def create_azure_doc_agent():
    # Get the schema for document analysis
    schema_str = json.dumps({
        "analyze_document": {
            "description": "Analyze a document using Azure Document Intelligence",
            "parameters": {
                "document_url": "URL or local path to the document",
                "model_id": "The model ID to use (default: prebuilt-layout)"
            },
            "returns": "Document analysis results"
        },
        "extract_text": {
            "description": "Extract all text from a document",
            "parameters": {
                "document_url": "URL or local path to the document"
            },
            "returns": "Extracted text content"
        },
        "extract_tables": {
            "description": "Extract tables from a document",
            "parameters": {
                "document_url": "URL or local path to the document"
            },
            "returns": "List of tables with their content"
        }
    }, indent=2)
    
    # Create the prompt for document analysis
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are an Azure Document Intelligence expert. You can analyze documents and extract information from them.
        
        AVAILABLE OPERATIONS:
{schema_str}
        
        INSTRUCTIONS:
        1. You can analyze documents using Azure Document Intelligence.
        2. You can extract text and tables from documents.
        3. For document URLs, you can use public URLs or local file paths.
        4. Always provide clear explanations of the analysis results.
        5. If an error occurs, explain the possible reasons and suggest solutions.
        
        Return your response in a clear, structured format.
        """),
        MessagesPlaceholder(variable_name="messages"),
        HumanMessage(content="{query}")
    ])
    
    # Chain for document analysis
    llm = ChatOllama(
        model="llama3.2",
        temperature=0,
        timeout=30
    )
    
    doc_chain = prompt | llm | StrOutputParser()
    
    # Azure Document Intelligence agent function
    def azure_doc_agent(query: str, messages: List[BaseMessage] = None) -> str:
        if messages is None:
            messages = []
        
        logger.info(f"Azure Doc Agent processing query: {query}")
        start_time = time.time()
        
        # Check if the query is asking for document analysis
        query_lower = query.lower()
        
        # Extract document URL from the query if present
        document_url = None
        if "http://" in query or "https://" in query:
            # Simple extraction of URL - in production, use a more robust method
            words = query.split()
            for word in words:
                if word.startswith(("http://", "https://")):
                    document_url = word
                    break
        
        # Process based on query intent
        if document_url and ("analyze" in query_lower or "extract" in query_lower):
            if "extract text" in query_lower:
                result = extract_text_from_document(document_url)
                response = f"Extracted text from document {document_url}:\n\n{result}"
            elif "extract table" in query_lower:
                tables = extract_tables_from_document(document_url)
                response = f"Extracted {len(tables)} tables from document {document_url}:\n\n{json.dumps(tables, indent=2)}"
            else:
                # Default to full document analysis
                analysis_result = analyze_document(document_url)
                response = f"Analysis of document {document_url}:\n\n{json.dumps(analysis_result, indent=2)}"
        else:
            # Use the LLM for general queries about document analysis
            response = doc_chain.invoke({"query": query, "messages": messages})
        
        end_time = time.time()
        logger.info(f"Azure Doc Agent completed processing in {end_time - start_time:.2f} seconds")
        
        return response
    
    return azure_doc_agent

# Function to run the Azure Document Intelligence agent
def run_azure_doc_agent(query: str, messages: List[BaseMessage] = None) -> str:
    agent = create_azure_doc_agent()
    return agent(query, messages)

# For testing
if __name__ == "__main__":
    # Test with a sample document
    sample_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-layout.pdf"
    print(run_azure_doc_agent(f"Analyze this document: {sample_url}"))
