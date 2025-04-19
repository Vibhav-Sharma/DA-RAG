"""
Configuration settings for the Dynamic Retrieval-Augmented Generation (DA-RAG) system.
"""

# Model configurations
MODELS = {
    'retriever': {
        'model_name': 'facebook-dpr-ctx_encoder-single-nq-base',
        'embedding_dim': 768
    },
    'generator': {
        'model_name': 'facebook/bart-large', 
        'max_length': 256,
        'min_length': 50,
        'num_beams': 4,
        'length_penalty': 2.0
    }
}

# Retrieval settings
RETRIEVAL = {
    'top_k_documents': 5,  # Number of documents to retrieve
    'min_similarity_score': 0.6,  # Minimum similarity score for relevance
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
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    'file': 'da_rag.log'
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