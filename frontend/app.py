from flask import Flask, render_template, request, jsonify
import os

# This will later be replaced with actual agent imports
# from agents.master_agent import MasterAgent

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
    
    # TODO: Replace with actual agent call
    # For now, just return a mock response
    response = f"This is a mock response to: {user_message}. In the future, this will be handled by the MARS multi-agent system."
    
    # Add response to history
    conversation_history.append({'role': 'assistant', 'content': response})
    
    return jsonify({'response': response})

@app.route('/history')
def history():
    return jsonify(conversation_history)

@app.route('/clear_history', methods=['POST'])
def clear_history():
    conversation_history.clear()
    return jsonify({'status': 'success'})

if __name__ == '__main__':
    app.run(debug=True)
