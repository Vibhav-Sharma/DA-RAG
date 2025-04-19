# DA-RAG: Dynamic Retrieval-Augmented Generation

DA-RAG is a system that improves response accuracy and efficiency by dynamically narrowing down topics after a user submits a query. The system first identifies broader topics, asks clarifying questions, and then retrieves and generates targeted responses based on the narrowed-down topic.

## Key Features

- **Dynamic Topic Narrowing**: Automatically narrows down broad topics through clarification questions
- **Targeted Knowledge Retrieval**: Retrieves information relevant only to the specified topic
- **Semantic Search**: Uses FAISS and sentence embeddings for efficient similarity search
- **Conversational Flow**: Maintains coherent conversation while refining topics
- **Performance Monitoring**: Tracks retrieval and generation performance metrics

## System Components

### 1. Retriever (`retriever.py`)

The DynamicRetriever is responsible for:
- Identifying broad topics from user queries
- Organizing knowledge by topic hierarchies
- Retrieving relevant documents using FAISS similarity search
- Generating clarification questions to narrow down topics

### 2. Generator (`generator.py`)

The ResponseGenerator handles:
- Generating responses based on retrieved context
- Creating clarification prompts to narrow down topics
- Producing focused responses for specific topics
- Tracking generation metrics

### 3. Pipeline (`pipeline.py`)

The DynamicRAGPipeline orchestrates the entire process:
- Processing initial queries and identifying topics
- Deciding when clarification is needed
- Managing the conversation flow
- Integrating retrieval and generation components

## Requirements

- Python 3.8+
- PyTorch
- Transformers
- FAISS
- Sentence-Transformers

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/DA-RAG.git
cd DA-RAG

# Install dependencies
pip install -r requirements.txt
```

## Usage Example

```python
from src.pipeline import DynamicRAGPipeline
from config import TOPIC_HIERARCHIES

# Initialize the pipeline
rag_pipeline = DynamicRAGPipeline()

# Add knowledge with topic labels
documents = [
    "Python is a high-level programming language often used for web development.",
    "Python snakes are non-venomous constrictors found in Africa and Asia.",
    "Machine learning models require large datasets for training."
]
topics = [
    ["python programming", "programming languages"],
    ["python snake", "reptiles", "animals"],
    ["machine learning", "artificial intelligence", "data science"]
]
rag_pipeline.add_knowledge(documents, topics)

# Add topic hierarchies for narrowing
rag_pipeline.add_topic_hierarchies(TOPIC_HIERARCHIES)

# Process an initial query
result = rag_pipeline.process_query("Tell me about Python")
print(result)
# This will generate clarification questions to determine if the user
#