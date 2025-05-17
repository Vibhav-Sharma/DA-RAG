"""
Configuration settings for the Dynamic Retrieval-Augmented Generation (DA-RAG) system.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Database settings
DATABASE_PATH = "data/knowledge_base.db"

# API Keys
NEWS_API_KEY = os.getenv('NEWS_API_KEY')

# Model settings
SBERT_MODEL = "facebook-dpr-ctx_encoder-single-nq-base"
BART_MODEL = "facebook/bart-large"

# Data loading settings
WIKIPEDIA_MAX_PAGES = 10
NEWS_DAYS_BACK = 7
MAX_DATABASE_ENTRIES = 1000

# RAG settings
TOP_K_RESULTS = 3
SIMILARITY_THRESHOLD = 0.7
MAX_CONTEXT_LENGTH = 1024

# Update intervals (in hours)
KNOWLEDGE_BASE_UPDATE_INTERVAL = 24
NEWS_UPDATE_INTERVAL = 6

# Default topics for knowledge base
DEFAULT_TOPICS = [
    "Artificial Intelligence",
    "Machine Learning",
    "Natural Language Processing",
    "Deep Learning",
    "Computer Science",
    "Data Science",
    "Neural Networks",
    "Transformer Models",
    "Large Language Models",
    "Information Retrieval"
]

# Logging settings
LOG_LEVEL = "INFO"
LOG_FILE = "logs/rag_system.log"

# Model configurations
MODELS = {
    'retriever': {
        'model_name': SBERT_MODEL,
        'embedding_dim': 768
    },
    'generator': {
        'model_name': BART_MODEL, 
        'max_length': 256,
        'min_length': 50,
        'num_beams': 4,
        'length_penalty': 2.0
    }
}

# Retrieval settings
RETRIEVAL = {
    'top_k_documents': TOP_K_RESULTS,  # Number of documents to retrieve
    'min_similarity_score': SIMILARITY_THRESHOLD,  # Minimum similarity score for relevance
    'reranking_enabled': True  # Whether to enable reranking of retrieved documents
}

# Topic narrowing settings
TOPIC_NARROWING = {
    'clarification_threshold': 0.7,  # Confidence threshold for requiring clarification
    'max_clarification_questions': 3,  # Maximum number of clarification questions to ask
    'min_topics_for_clarification': 2  # Minimum number of potential topics to trigger clarification
}

# Sample topic hierarchies
TOPIC_HIERARCHIES = {
    'python': ['python programming', 'python language', 'python libraries', 'python frameworks'],
    'machine learning': ['supervised learning', 'unsupervised learning', 'reinforcement learning', 'neural networks'],
    'climate change': ['global warming', 'carbon emissions', 'sea level rise', 'climate policy'],
    'artificial intelligence': ['machine learning', 'natural language processing', 'computer vision', 'robotics'],
    'database': ['SQL', 'NoSQL', 'data modeling', 'query optimization'],
    'web development': ['frontend', 'backend', 'full-stack', 'web frameworks', 'web security']
}

# Logging configuration
LOGGING = {
    'level': LOG_LEVEL,
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': LOG_FILE
}

# Performance monitoring settings
PERFORMANCE = {
    'track_metrics': True,
    'log_interval': 10,  # Log metrics every N queries
    'save_metrics': True
}

# Knowledge base settings
KNOWLEDGE_BASE = {
    'index_save_path': 'data/faiss_index',
    'texts_save_path': 'data/texts.pkl',
    'metadata_save_path': 'data/metadata.pkl'
}