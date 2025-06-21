import os
import logging
from dotenv import load_dotenv
from agents.azure_doc_agent import run_azure_doc_agent
from agents.master_agent import run_master_agent

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MARS')

def test_local_document(file_path):
    """Test the Azure Document Intelligence agent with a local document"""
    print("\n=== Testing Azure Doc Agent with Local Document ===")
    
    # Create a query that includes the local file path
    query = f"Extract text and tables from this document: {file_path}"
    
    print(f"Query: {query}")
    
    # Check if credentials are available in environment variables
    if not os.environ.get("AZURE_DOC_ENDPOINT") or not os.environ.get("AZURE_DOC_KEY"):
        print("\nWARNING: Azure Document Intelligence credentials not found in environment variables.")
        print("Please create a .env file with your credentials using .env.example as a template.")
        print("You can set them manually for this test run, but this is not recommended for security.")
        
        # Ask if the user wants to proceed with manual credential entry
        proceed = input("\nDo you want to enter credentials manually for this test? (y/n): ")
        if proceed.lower() == 'y':
            endpoint = input("Enter your Azure Document Intelligence endpoint: ")
            key = input("Enter your Azure Document Intelligence key: ")
            os.environ["AZURE_DOC_ENDPOINT"] = endpoint
            os.environ["AZURE_DOC_KEY"] = key
        else:
            print("Test aborted. Please set up your .env file and try again.")
            exit(1)
    
    # Run the query through the Azure Doc agent
    result = run_azure_doc_agent(query)
    
    print("\n=== Results ===")
    print(f"{result[:1000]}...")  # Show first 1000 chars of result
    
    return result

if __name__ == "__main__":
    # Replace this with the path to your local document
    document_path = input("Enter the full path to your local document (PDF, JPG, PNG, etc.): ")
    
    if os.path.exists(document_path):
        test_local_document(document_path)
    else:
        print(f"Error: File not found at {document_path}")
        print("Please provide a valid file path.")
