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
    
    # Create the prompt for SQL generation with examples
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=f"""You are an SQL expert. Write a single SQLite query to answer the question.
        
        DATABASE SCHEMA:
{schema_str}
        
        CRITICAL INSTRUCTIONS:
        1. The database contains employee information in the 'people' table.
        2. The 'people' table has EXACTLY these columns and NOTHING else: id (INTEGER), name (TEXT), role (TEXT).
        3. There is NO department column, location column, or any other columns besides id, name, and role.
        4. For ANY queries about employees, people, staff, or team members, you MUST use the people table.
        5. NEVER reference columns that don't exist in the schema.
        6. Return ONLY the SQL query with no explanations, markdown formatting, or code blocks.
        7. End your query with a semicolon.
        8. Keep your query simple and focused on the available columns.
        9. IMPORTANT: Use appropriate WHERE clauses to filter results based on the query.
        10. If the query mentions specific roles like 'Engineer', 'Manager', 'Developer', etc., use a WHERE clause to filter by that role.
        
        EXAMPLES:
        
        Question: List all employees
        SQL: SELECT * FROM people;
        
        Question: Who works as an Engineer?
        SQL: SELECT * FROM people WHERE role = 'Engineer';
        
        Question: How many employees are there?
        SQL: SELECT COUNT(*) as count FROM people;
        
        Question: List employees in the Engineering department
        SQL: SELECT * FROM people WHERE role = 'Engineer';
        
        Question: Show me all staff members
        SQL: SELECT * FROM people;
        
        Question: Who is in the company?
        SQL: SELECT * FROM people;
        
        Question: List all engineers
        SQL: SELECT * FROM people WHERE role = 'Engineer';
        
        Question: How many managers do we have?
        SQL: SELECT COUNT(*) as count FROM people WHERE role = 'Manager';
        
        Question: Find employees with the name Alice
        SQL: SELECT * FROM people WHERE name = 'Alice';
        """),
        HumanMessage(content="{query}"),
    ])
    
    # Chain for generating SQL - using more precise parameters
    llm = ChatOllama(
        model="llama3.2", 
        temperature=0,  # Zero temperature for deterministic output
        timeout=30,     # 30 second timeout
        stop=["\n\n", ";"],  # Stop on double newline or semicolon to get just the query
        num_ctx=2048    # Ensure enough context for the schema
    )
    sql_chain = prompt | llm | StrOutputParser()
    
    def sql_agent(query: str, messages: List[BaseMessage] = None) -> str:
        if messages is None:
            messages = []
        
        logger.info(f"SQL Agent processing query: {query}")
        start_time = time.time()
        
        # Start timing for SQL generation
        sql_generation_start = time.time()
        
        # Direct parsing approach for common queries
        query_lower = query.lower()
        
        # Check if this is a query about specific roles
        roles = {
            'engineer': 'Engineer',
            'manager': 'Manager',
            'developer': 'Developer',
            'designer': 'Designer'
        }
        
        # Check if this is a query about counting employees
        is_count_query = any(term in query_lower for term in ['how many', 'count', 'number of'])
        
        # Check if this is a query about specific roles
        role_filter = None
        for role_key, role_value in roles.items():
            if role_key in query_lower:
                role_filter = role_value
                break
        
        # Generate appropriate SQL based on query type
        if is_count_query and role_filter:
            # Count of specific role
            sql_query = f"SELECT COUNT(*) as count FROM people WHERE role = '{role_filter}';"
            logger.info(f"Generated count query for role '{role_filter}'")
        elif is_count_query:
            # Count of all employees
            sql_query = "SELECT COUNT(*) as count FROM people;"
            logger.info("Generated count query for all employees")
        elif role_filter:
            # Filter by specific role
            sql_query = f"SELECT * FROM people WHERE role = '{role_filter}';"
            logger.info(f"Generated filter query for role '{role_filter}'")
        else:
            # Use the LLM for more complex queries
            sql_query = sql_chain.invoke({"query": query, "messages": messages})
            
            # Extract the SQL query from the response
            # First, try to extract from code blocks if present
            if "```sql" in sql_query:
                sql_query = sql_query.split("```sql")[1].split("```")[0].strip()
            elif "```" in sql_query:
                sql_query = sql_query.split("```")[1].split("```")[0].strip()
            
            # Clean up any remaining explanation text
            sql_query = sql_query.strip()
            
            # Remove any text before SELECT/WITH/etc. and after the semicolon
            sql_keywords = ["SELECT", "WITH", "INSERT", "UPDATE", "DELETE", "CREATE", "DROP", "ALTER"]
            for keyword in sql_keywords:
                if keyword in sql_query.upper():
                    sql_query = sql_query[sql_query.upper().find(keyword):]
                    break
            
            # If there's a semicolon in the middle, keep only up to the first semicolon
            if ';' in sql_query:
                sql_query = sql_query.split(';')[0] + ';'
            # Ensure the query ends with a semicolon
            elif not sql_query.endswith(';'):
                sql_query += ';'
                
            logger.info(f"LLM generated SQL query: {sql_query}")
        
        # If we still don't have a WHERE clause but should have one based on the query
        if role_filter and 'WHERE' not in sql_query.upper():
            if 'FROM people' in sql_query:
                parts = sql_query.split('FROM people')
                sql_query = f"{parts[0]} FROM people WHERE role = '{role_filter}';"
                logger.info(f"Added missing WHERE clause for role '{role_filter}'")
        
        logger.info(f"Final SQL query: {sql_query}")
        
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
