from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
import numpy as np
from typing import List, Dict, Tuple
import torch
from collections import defaultdict
import faiss
import pickle

class DynamicRAG:
    def __init__(self):
        # Initialize SBERT model
        self.sbert_model = SentenceTransformer("facebook-dpr-ctx_encoder-single-nq-base")
        
        # Initialize BART model
        self.bart_model_name = "facebook/bart-large"
        self.bart_model = BartForConditionalGeneration.from_pretrained(self.bart_model_name)
        self.tokenizer = BartTokenizer.from_pretrained(self.bart_model_name)
        
        # Initialize FAISS index
        self.dimension = 768  # SBERT embedding dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Initialize knowledge base
        self.texts = []  # List of texts
        self.query_history = []   # Store past queries
        
    def add_to_knowledge_base(self, text: str) -> None:
        """Add new text to the knowledge base with its embedding"""
        embedding = self.sbert_model.encode(text)
        embedding = embedding.reshape(1, -1)  # Reshape for FAISS
        self.index.add(embedding)
        self.texts.append(text)
        
    def update_embeddings(self) -> None:
        """Update embeddings for all texts in knowledge base using FAISS"""
        # Get all current texts
        texts = self.texts.copy()
        
        # Clear the current index
        self.index = faiss.IndexFlatL2(self.dimension)
        self.texts = []
        
        # Re-add all texts with updated embeddings
        for text in texts:
            self.add_to_knowledge_base(text)
        
    def find_relevant_context(self, query: str, top_k: int = 3) -> List[str]:
        """Find most relevant context for a query using FAISS"""
        query_embedding = self.sbert_model.encode(query)
        query_embedding = query_embedding.reshape(1, -1)  # Reshape for FAISS
        
        # Search in FAISS index
        distances, indices = self.index.search(query_embedding, top_k)
        
        # Get the relevant texts
        relevant_texts = [self.texts[idx] for idx in indices[0]]
        return relevant_texts
    
    def generate_response(self, query: str, context: List[str]) -> str:
        """Generate response using BART"""
        # Combine context and query
        prompt = f"Context: {' '.join(context)}\nQuery: {query}\nAnswer:"
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.bart_model.generate(**inputs)
        response = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        return response
    
    def process_query(self, query: str) -> Tuple[str, List[str]]:
        """Process a user query and update the system"""
        # Store query in history
        self.query_history.append(query)
        
        # Find relevant context
        context = self.find_relevant_context(query)
        
        # Generate response
        response = self.generate_response(query, context)
        
        # Update embeddings based on new query
        self.update_embeddings()
        
        return response, context
    
    def save_state(self, filepath: str) -> None:
        """Save the current state of the RAG system"""
        state = {
            'texts': self.texts,
            'query_history': self.query_history,
            'index': faiss.serialize_index(self.index)
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str) -> None:
        """Load a saved state of the RAG system"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.texts = state['texts']
        self.query_history = state['query_history']
        self.index = faiss.deserialize_index(state['index'])

# Example usage
if __name__ == "__main__":
    # Initialize the system
    rag = DynamicRAG()
    
    # Add some initial knowledge
    initial_knowledge = [
        "Large language models are powerful for text generation.",
        "Retrieval-augmented generation combines retrieval and generation.",
        "SBERT is used for semantic search and similarity.",
        "FAISS enables efficient similarity search in high-dimensional spaces."
    ]
    
    for text in initial_knowledge:
        rag.add_to_knowledge_base(text)
    
    # Example query
    query = "How does RAG work with language models?"
    response, context = rag.process_query(query)
    
    print("Query:", query)
    print("Context used:", context)
    print("Response:", response)
    
    # Save the state
    rag.save_state("rag_state.pkl") 