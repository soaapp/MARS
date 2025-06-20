from flask import Flask, render_template, request, jsonify
import os
import sys
import time
import logging
from langchain_core.messages import HumanMessage, AIMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('MARS')

# Add the parent directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the master agent
from agents.master_agent import run_master_agent

app = Flask(__name__)

# Create templates directory if it doesn't exist
os.makedirs(os.path.join(app.root_path, 'templates'), exist_ok=True)
os.makedirs(os.path.join(app.root_path, 'static'), exist_ok=True)

# Store conversation history
conversation_history = []

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    start_time = time.time()
    logger.info("Received chat request")
    
    user_message = request.json.get('message', '')
    logger.info(f"User message: {user_message}")
    
    if not user_message:
        logger.warning("Empty message received")
        return jsonify({'response': 'Please enter a message.'})
    
    # Add user message to history
    conversation_history.append({'role': 'user', 'content': user_message})
    
    # Convert conversation history to LangChain message format
    langchain_messages = []
    for message in conversation_history:
        if message['role'] == 'user':
            langchain_messages.append(HumanMessage(content=message['content']))
        elif message['role'] == 'assistant':
            langchain_messages.append(AIMessage(content=message['content']))
    
    logger.info(f"Conversation history length: {len(langchain_messages)} messages")
    
    try:
        # Call the master agent
        agent_start_time = time.time()
        logger.info("Calling master agent")
        result = run_master_agent(user_message, langchain_messages)
        agent_end_time = time.time()
        logger.info(f"Master agent processing completed in {agent_end_time - agent_start_time:.2f} seconds")
        
        # Extract the response
        if 'result' in result:
            response = result['result']
            source = result.get('source', 'unknown')
            response_with_source = f"[Source: {source}] {response}"
            logger.info(f"Response generated from source: {source}")
        else:
            response_with_source = "Sorry, I couldn't process your request."
            logger.warning("No result in master agent response")
    except Exception as e:
        logger.error(f"Error in chat processing: {str(e)}")
        response_with_source = f"Error: {str(e)}"
    
    # Add response to history
    conversation_history.append({'role': 'assistant', 'content': response_with_source})
    
    end_time = time.time()
    total_time = end_time - start_time
    logger.info(f"Total chat request processing time: {total_time:.2f} seconds")
    
    return jsonify({'response': response_with_source})

@app.route('/history')
def history():
    return jsonify(conversation_history)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    conversation_history.clear()
    return jsonify({'status': 'success'})

# Route to add a document to Qdrant
@app.route('/add_document', methods=['POST'])
def add_document():
    start_time = time.time()
    logger.info("Received document addition request")
    
    content = request.json.get('content', '')
    metadata = request.json.get('metadata', {})
    
    if not content:
        logger.warning("Empty document content received")
        return jsonify({'error': 'Document content is required'})
    
    try:
        from agents.qdrant_agent import add_document as qdrant_add_document
        logger.info(f"Adding document with metadata: {metadata}")
        
        add_start_time = time.time()
        result = qdrant_add_document(content, metadata)
        add_end_time = time.time()
        
        logger.info(f"Document added successfully in {add_end_time - add_start_time:.2f} seconds")
        
        end_time = time.time()
        logger.info(f"Total document addition time: {end_time - start_time:.2f} seconds")
        
        return jsonify({'result': result})
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    logger.info("Starting MARS Flask application")
    app.run(debug=True)
