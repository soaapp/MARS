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

def test_direct_azure_doc_agent():
    """Test the Azure Document Intelligence agent directly"""
    # Sample document URL from Azure samples
    sample_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-layout.pdf"
    
    print("\n=== Testing Azure Doc Agent Directly ===")
    query = f"Extract text from this document: {sample_url}"
    
    print(f"Query: {query}")
    result = run_azure_doc_agent(query)
    print(f"Result: {result[:500]}...")  # Show first 500 chars of result

def test_master_agent_routing():
    """Test the master agent routing to Azure Doc agent"""
    # Sample document URL from Azure samples
    sample_url = "https://raw.githubusercontent.com/Azure-Samples/cognitive-services-REST-api-samples/master/curl/form-recognizer/sample-layout.pdf"
    
    print("\n=== Testing Master Agent Routing to Azure Doc Agent ===")
    query = f"Analyze this document for me: {sample_url}"
    
    print(f"Query: {query}")
    result = run_master_agent(query)
    print(f"Source: {result['source']}")
    print(f"Result: {result['result'][:500]}...")  # Show first 500 chars of result

if __name__ == "__main__":
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
    
    # Run tests
    test_direct_azure_doc_agent()
    test_master_agent_routing()
