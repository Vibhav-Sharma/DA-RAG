# DA-RAG: Dynamic Retrieval-Augmented Generation

DA-RAG is a system that improves response accuracy and efficiency by dynamically narrowing down topics after a user submits a query. The system first identifies broader topics, asks clarifying questions, and then retrieves and generates targeted responses based on the narrowed-down topic.

## Key Features

- **Dynamic Topic Narrowing**: Automatically narrows down broad topics through clarification questions
- **Targeted Knowledge Retrieval**: Retrieves information relevant only to the specified topic
- **Semantic Search**: Uses FAISS and sentence embeddings for efficient similarity search
- **Conversational Flow**: Maintains coherent conversation while refining topics
- **Performance Monitoring**: Tracks retrieval and generation performance metrics
- **Web Interface**: Interactive Flask-based web application for easy interaction
- **Real-time Updates**: WebSocket support for dynamic conversation updates

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

### 4. Web Application (`app.py`)

The Flask web application provides:
- Interactive web interface for querying the system
- Real-time conversation updates via WebSocket
- Dynamic topic visualization
- User-friendly response presentation

## Requirements

### Core Dependencies
- Python 3.8+
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- FAISS-CPU >= 1.7.4
- Sentence-Transformers >= 2.2.2
- NumPy >= 1.24.0
- Pandas >= 2.0.0
- Scikit-learn >= 1.6.1

### Web Application Dependencies
- Flask == 3.0.2
- Flask-WTF == 1.2.1
- Flask-SocketIO == 5.3.6
- python-socketio == 5.11.1
- python-engineio == 4.9.1
- Werkzeug == 3.0.1

### Additional Dependencies
- python-dotenv >= 1.0.0
- SQLAlchemy >= 2.0.0
- requests >= 2.31.0
- wikipedia >= 1.4.0
- newsapi-python >= 0.2.7
- arxiv >= 1.4.8
- scholarly >= 1.7.11

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/DA-RAG.git
cd DA-RAG
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/MacOS
python -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables:
Create a `.env` file in the project root with the following variables:
```
OPENAI_API_KEY=your_api_key_here
NEWS_API_KEY=your_news_api_key_here
DATABASE_URL=your_database_url_here
```

## Running the Application

1. Start the Flask web application:
```bash
python app.py
```

2. Access the web interface at `http://localhost:5000`

## Usage

### Web Interface
1. Open your web browser and navigate to `http://localhost:5000`
2. Enter your query in the search box
3. The system will:
   - Process your initial query
   - Ask clarification questions if needed
   - Provide focused responses based on the narrowed topic
4. View the conversation history and topic hierarchy in real-time

### Python API
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

# Process a query
result = rag_pipeline.process_query("Tell me about Python")
print(result)
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
