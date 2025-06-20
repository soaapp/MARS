import os
import sys
import time
import logging
import sqlite3
from typing import List, Dict, Any
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_ollama import ChatOllama
from langchain_core.output_parsers import StrOutputParser

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MARS')

# Path to the SQLite database
DB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'db', 'mock_data.db')

# Function to execute SQL queries
def execute_sql_query(query: str) -> List[Dict[str, Any]]:
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.row_factory = sqlite3.Row  # This enables column access by name
        cursor = conn.cursor()
        cursor.execute(query)
        results = [dict(row) for row in cursor.fetchall()]
        conn.close()
        return results
    except Exception as e:
        return [{"error": str(e)}]

# Function to get table schema
def get_table_schema() -> Dict[str, List[str]]:
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    
    # Get all tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    
    schema = {}
    for table in tables:
        table_name = table[0]
        cursor.execute(f"PRAGMA table_info({table_name})")
        columns = cursor.fetchall()
        schema[table_name] = [f"{col[1]} ({col[2]})" for col in columns]
    
    conn.close()
    return schema

# Create the SQL agent
def create_sql_agent():
    # Get the database schema
    schema = get_table_schema()
    schema_str = "\n".join([f"Table: {table}\nColumns: {', '.join(columns)}" for table, columns in schema.items()])
    
    # Create the prompt for SQL generation
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are an SQL expert. Write a single SQLite query to answer the question.
        Schema: 
{schema_str}
        
        IMPORTANT: The database contains employee information in the 'people' table.
        For queries about employees, ALWAYS use the people table.
        
        Return ONLY the SQL query with no explanations or formatting."""),
        HumanMessage(content="{query}"),
    ])
    
    # Chain for generating SQL
    llm = ChatOllama(
        model="llama3.2", 
        temperature=0,
        timeout=30,  # 30 second timeout
        stop=["\n\n"]  # Stop on double newline to encourage brevity
    )
    sql_chain = prompt | llm | StrOutputParser()
    
    def sql_agent(query: str, messages: List[BaseMessage] = None) -> str:
        if messages is None:
            messages = []
        
        logger.info(f"SQL Agent processing query: {query}")
        start_time = time.time()
        
        # Generate SQL query
        sql_generation_start = time.time()
        sql_query = sql_chain.invoke({"query": query, "messages": messages})
        
        # Extract the SQL query from the response (assuming it's wrapped in ```sql ... ```)
        if "```sql" in sql_query:
            sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
        elif "```" in sql_query:
            sql_query = sql_query.split("```")[1].split("```")[0].strip()
        
        sql_generation_end = time.time()
        logger.info(f"Generated SQL query in {sql_generation_end - sql_generation_start:.2f} seconds: {sql_query}")
        
        # Execute the SQL query
        try:
            execution_start = time.time()
            results = execute_sql_query(sql_query)
            execution_end = time.time()
            logger.info(f"Executed SQL query in {execution_end - execution_start:.2f} seconds")
            
            end_time = time.time()
            logger.info(f"SQL Agent completed processing in {end_time - start_time:.2f} seconds")
            
            # Format the results
            if "error" in results[0] if results else {}:
                return f"Error executing query: {results[0]['error']}\n\nThe query was: {sql_query}"
            
            # Format the results as a readable string
            if not results:
                return f"No results found for query: {sql_query}"
            
            result_str = "Results:\n"
            for i, row in enumerate(results):
                result_str += f"Row {i+1}: {row}\n"
            
            return f"Query executed: {sql_query}\n\n{result_str}"
        except Exception as e:
            logger.error(f"SQL execution error: {str(e)}")
            return f"SQL Query: {sql_query}\n\nError: {str(e)}"
    
    return sql_agent

# Function to run the SQL agent
def run_sql_agent(query: str, messages: List[BaseMessage] = None) -> str:
    agent = create_sql_agent()
    return agent(query, messages)
