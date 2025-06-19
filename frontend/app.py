from flask import Flask, render_template, request, jsonify
import os
import sys
from langchain_core.messages import HumanMessage, AIMessage

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
    user_message = request.json.get('message', '')
    
    if not user_message:
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
    
    try:
        # Call the master agent
        result = run_master_agent(user_message, langchain_messages)
        
        # Extract the response
        if 'result' in result:
            response = result['result']
            source = result.get('source', 'unknown')
            response_with_source = f"[Source: {source}] {response}"
        else:
            response_with_source = "Sorry, I couldn't process your request."
    except Exception as e:
        response_with_source = f"Error: {str(e)}"
    
    # Add response to history
    conversation_history.append({'role': 'assistant', 'content': response_with_source})
    
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
    content = request.json.get('content', '')
    metadata = request.json.get('metadata', {})
    
    if not content:
        return jsonify({'error': 'Document content is required'})
    
    try:
        from agents.qdrant_agent import add_document as qdrant_add_document
        result = qdrant_add_document(content, metadata)
        return jsonify({'result': result})
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)
