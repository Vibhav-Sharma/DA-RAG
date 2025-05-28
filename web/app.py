import os
import sys
import logging
import time
from datetime import datetime
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import json
from src.pipeline import DynamicRAGPipeline

# Add project root to Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize rate limiter
limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["200 per day", "50 per hour"]
)

# Initialize pipeline
try:
    pipeline = DynamicRAGPipeline()
    logger.info("Pipeline initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize pipeline: {str(e)}")
    raise

def create_response(data, status_code=200):
    """Create a standardized API response with security headers."""
    response = jsonify(data)
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['Content-Security-Policy'] = "default-src 'self'"
    return response

@app.route('/')
def root():
    """Root endpoint that returns available API endpoints."""
    logger.info("Root endpoint accessed")
    return create_response({
        'message': 'Welcome to the Dynamic RAG API',
        'endpoints': {
            'GET /api/health': 'Check API health',
            'POST /api/query': 'Process a query',
            'GET /api/history': 'Get conversation history',
            'GET /api/metrics': 'Get performance metrics',
            'GET /api/topics': 'Get available topics'
        }
    })

@app.route('/api/health')
@limiter.limit("30 per minute")
def health_check():
    """Health check endpoint that verifies pipeline functionality."""
    logger.info("Health check requested")
    start_time = time.time()
    
    try:
        # Test pipeline with a simple query
        test_query = "What is machine learning?"
        result = pipeline.process_query(test_query)
        
        response_time = time.time() - start_time
        logger.info(f"Health check successful, response time: {response_time:.2f}s")
        
        return create_response({
            'status': 'healthy',
            'pipeline_status': 'operational',
            'response_time': response_time,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return create_response({
            'status': 'unhealthy',
            'error': str(e),
            'timestamp': datetime.now().isoformat()
        }, 500)

@app.route('/api/query', methods=['POST'])
@limiter.limit("60 per minute")
def process_query():
    """Process a user query through the pipeline."""
    logger.info("Query endpoint accessed")
    
    try:
        # Validate request
        if not request.is_json:
            raise ValueError("Request must be JSON")
        
        data = request.get_json()
        if not data or 'query' not in data:
            raise ValueError("Request must include 'query' field")
        
        query = data['query'].strip()
        if not query:
            raise ValueError("Query cannot be empty")
        
        # Process query
        logger.info(f"Processing query: {query}")
        result = pipeline.process_query(query)
        
        # Log metrics
        if 'metrics' in result:
            logger.info(f"Query processed in {result['metrics']['response_time']:.2f}s")
        
        return create_response(result)
        
    except ValueError as e:
        logger.warning(f"Invalid request: {str(e)}")
        return create_response({
            'error': str(e)
        }, 400)
        
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        return create_response({
            'error': "Internal server error",
            'details': str(e)
        }, 500)

@app.route('/api/history')
@limiter.limit("30 per minute")
def get_history():
    """Get conversation history with optional limit."""
    logger.info("History endpoint accessed")
    
    try:
        limit = request.args.get('limit', type=int)
        history = pipeline.get_conversation_history(limit)
        
        return create_response({
            'history': history,
            'total_entries': len(history)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving history: {str(e)}")
        return create_response({
            'error': "Failed to retrieve history",
            'details': str(e)
        }, 500)

@app.route('/api/metrics')
@limiter.limit("30 per minute")
def get_metrics():
    """Get current performance metrics."""
    logger.info("Metrics endpoint accessed")
    
    try:
        metrics = pipeline.get_metrics()
        return create_response(metrics)
        
    except Exception as e:
        logger.error(f"Error retrieving metrics: {str(e)}")
        return create_response({
            'error': "Failed to retrieve metrics",
            'details': str(e)
        }, 500)

@app.route('/api/topics')
@limiter.limit("30 per minute")
def get_topics():
    """Get available topics and hierarchies."""
    logger.info("Topics endpoint accessed")
    
    try:
        topics = pipeline.retriever.topic_hierarchy
        return create_response({
            'topics': topics,
            'total_topics': len(topics)
        })
        
    except Exception as e:
        logger.error(f"Error retrieving topics: {str(e)}")
        return create_response({
            'error': "Failed to retrieve topics",
            'details': str(e)
        }, 500)

if __name__ == '__main__':
    try:
        logger.info("Starting Flask application")
        app.run(host='0.0.0.0', port=5000, debug=True)
    except Exception as e:
        logger.error(f"Failed to start Flask application: {str(e)}")
        raise 