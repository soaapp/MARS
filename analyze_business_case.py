import os
import logging
from dotenv import load_dotenv
from agents.azure_doc_agent import run_azure_doc_agent

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MARS')

def analyze_business_case():
    """Analyze the Business Case.pdf document using Azure Document Intelligence"""
    # Path to the Business Case.pdf document
    file_path = os.path.join(os.path.dirname(__file__), "docs", "Business Case.pdf")
    
    # Check if the file exists
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        print("Please make sure the Business Case.pdf file is in the docs directory.")
        return
    
    print(f"\n=== Analyzing Business Case.pdf ===")
    print(f"File path: {file_path}")
    
    # Create a query for the Azure Doc agent
    query = f"Extract text and tables from this document: {file_path}"
    
    # Run the query through the Azure Doc agent
    result = run_azure_doc_agent(query)
    
    print("\n=== Analysis Results ===")
    print(result)
    
    # Save the results to a text file
    output_file = os.path.join(os.path.dirname(__file__), "docs", "Business_Case_Analysis.txt")
    with open(output_file, "w") as f:
        f.write(result)
    
    print(f"\nResults saved to: {output_file}")

if __name__ == "__main__":
    analyze_business_case()
