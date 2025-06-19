# MARS - Multi-Agent RAG System

MARS (Multi-Agent RAG System) is an application that aims to setup a multi-agent RAG system for many AI based scenarios. The system integrates both vector database (Qdrant) and structured database (SQLite) retrieval capabilities through a coordinated multi-agent architecture using LangChain and LangGraph.

## Architecture

The system consists of the following components:

1. **Master Agent**: Coordinates between sub-agents and determines which data source to use based on the query
2. **Qdrant Agent**: Handles document retrieval from the Qdrant vector database
3. **SQL Agent**: Handles structured data queries from the SQLite database
4. **Flask Frontend**: Provides a user-friendly chat interface to interact with the system

## Setup

### Prerequisites

- Python 3.8+
- OpenAI API key

### Installation

1. Clone the repository
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Set up your OpenAI API key:
   ```
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or create a `.env` file in the project root with:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

## Usage

### Running the Application

1. Start the Flask server:
   ```
   python frontend/app.py
   ```
2. Open your browser and navigate to `http://127.0.0.1:5000`

### Adding Documents to Qdrant

You can add documents to the Qdrant collection using the `/add_document` endpoint:

```
POST /add_document
Content-Type: application/json

{
    "content": "Your document content here",
    "metadata": {
        "title": "Document Title",
        "author": "Author Name",
        "date": "2023-06-19"
    }
}
```

## Project Structure

```
MARS/
├── agents/
│   ├── master_agent.py   # Master agent that coordinates between sub-agents
│   ├── qdrant_agent.py   # Agent for document retrieval from Qdrant
│   └── sql_agent.py      # Agent for structured data queries from SQLite
├── db/
│   ├── init.py           # SQLite database initialization
│   └── mock_data.db      # SQLite database file
├── frontend/
│   ├── app.py            # Flask application
│   └── templates/
│       └── index.html    # Chat interface
├── qdrant/
│   └── client_setup.py   # Qdrant client initialization
├── langgraph_setup.py    # LangGraph setup for the multi-agent system
└── requirements.txt      # Project dependencies
```

## Future Enhancements

- Add authentication
- Implement more sophisticated routing logic
- Add support for more data sources
- Improve the UI with additional features
- Add document upload functionality
- Implement conversation memory
