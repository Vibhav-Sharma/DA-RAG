from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO
import sys
import os
import json
from datetime import datetime

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.pipeline import DynamicRAGPipeline
from config import DEFAULT_TOPICS

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'  # Change this in production
socketio = SocketIO(app)

# Initialize the RAG pipeline
rag_pipeline = DynamicRAGPipeline()

# Store conversation history
conversation_history = []

@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def process_query():
    """Process a user query and return the response."""
    data = request.json
    query = data.get('query', '')
    
    if not query:
        return jsonify({'error': 'No query provided'}), 400
    
    try:
        # Process the query through the RAG pipeline
        result = rag_pipeline.process_query(query)
        
        # Add to conversation history
        conversation_entry = {
            'timestamp': datetime.now().isoformat(),
            'query': query,
            'response': result.get('response', ''),
            'context': result.get('context', []),
            'metrics': result.get('metrics', {})
        }
        conversation_history.append(conversation_entry)
        
        # Emit the result through WebSocket for real-time updates
        socketio.emit('query_response', conversation_entry)
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/history', methods=['GET'])
def get_history():
    """Get the conversation history."""
    return jsonify(conversation_history)

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get the current system metrics."""
    metrics = rag_pipeline.get_performance_metrics()
    return jsonify(metrics)

@app.route('/api/topics', methods=['GET'])
def get_topics():
    """Get available topics."""
    return jsonify(DEFAULT_TOPICS)

if __name__ == '__main__':
    socketio.run(app, debug=True, port=5000) 