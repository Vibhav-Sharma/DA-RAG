from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
import numpy as np
import faiss
import pickle
from typing import List, Dict, Tuple, Optional
import torch
from collections import defaultdict
import time
import psutil
import logging

class AdvancedRAG:
    def __init__(self):
        # Initialize models
        self.sbert_model = SentenceTransformer("facebook-dpr-ctx_encoder-single-nq-base")
        self.bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        self.tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        
        # Initialize FAISS
        self.dimension = 768
        self.index = faiss.IndexFlatL2(self.dimension)
        
        # Knowledge base
        self.texts = []
        self.metadata = []  # Store additional information about texts
        
        # Performance metrics
        self.metrics = {
            'retrieval_time': [],
            'generation_time': [],
            'memory_usage': [],
            'accuracy_scores': []
        }
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def detect_ambiguity(self, query: str) -> Tuple[bool, List[str]]:
        """Detect if a query is ambiguous and return potential interpretations"""
        # Generate embedding for the query
        query_embedding = self.sbert_model.encode(query)
        
        # Find similar queries from history
        if len(self.texts) > 0:
            similarities = []
            for text in self.texts:
                text_embedding = self.sbert_model.encode(text)
                similarity = np.dot(query_embedding, text_embedding) / (
                    np.linalg.norm(query_embedding) * np.linalg.norm(text_embedding)
                )
                similarities.append(similarity)
            
            # If multiple high-similarity matches exist, query might be ambiguous
            high_similarity_count = sum(1 for s in similarities if s > 0.7)
            is_ambiguous = high_similarity_count > 1
            
            # Get potential interpretations
            interpretations = []
            if is_ambiguous:
                top_indices = np.argsort(similarities)[-3:]
                interpretations = [self.texts[i] for i in top_indices]
            
            return is_ambiguous, interpretations
        return False, []
    
    def refine_query(self, query: str) -> Tuple[str, List[str]]:
        """Refine ambiguous queries and break down complex ones"""
        is_ambiguous, interpretations = self.detect_ambiguity(query)
        
        if is_ambiguous:
            self.logger.info(f"Query is ambiguous. Potential interpretations: {interpretations}")
            # Generate clarifying questions
            prompt = f"Given these interpretations: {interpretations}\nGenerate clarifying questions for: {query}"
            inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
            clarifying_questions = self.bart_model.generate(**inputs)
            questions = self.tokenizer.decode(clarifying_questions[0], skip_special_tokens=True)
            return query, questions.split("?")
        
        # Break down complex queries
        prompt = f"Break down this complex query into simpler sub-queries: {query}"
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        sub_queries = self.bart_model.generate(**inputs)
        sub_queries = self.tokenizer.decode(sub_queries[0], skip_special_tokens=True)
        
        return query, [q.strip() for q in sub_queries.split("\n") if q.strip()]
    
    def add_to_knowledge_base(self, text: str, metadata: Optional[Dict] = None) -> None:
        """Add new text to the knowledge base with metadata"""
        start_time = time.time()
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        embedding = self.sbert_model.encode(text)
        embedding = embedding.reshape(1, -1)
        self.index.add(embedding)
        self.texts.append(text)
        self.metadata.append(metadata or {})
        
        # Update metrics
        self.metrics['memory_usage'].append(psutil.Process().memory_info().rss / 1024 / 1024 - memory_before)
        self.logger.info(f"Added text to knowledge base. Memory usage: {self.metrics['memory_usage'][-1]:.2f}MB")
    
    def retrieve_context(self, query: str, top_k: int = 3) -> Tuple[List[str], List[Dict]]:
        """Retrieve relevant context using FAISS"""
        start_time = time.time()
        
        query_embedding = self.sbert_model.encode(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        distances, indices = self.index.search(query_embedding, top_k)
        
        relevant_texts = [self.texts[idx] for idx in indices[0]]
        relevant_metadata = [self.metadata[idx] for idx in indices[0]]
        
        # Update metrics
        self.metrics['retrieval_time'].append(time.time() - start_time)
        self.logger.info(f"Retrieval time: {self.metrics['retrieval_time'][-1]:.2f}s")
        
        return relevant_texts, relevant_metadata
    
    def generate_response(self, query: str, context: List[str], metadata: List[Dict]) -> str:
        """Generate response using BART with contextual augmentation"""
        start_time = time.time()
        
        # Build enriched prompt
        context_str = "\n".join([f"Source {i+1}: {text}" for i, text in enumerate(context)])
        prompt = f"""Context:
{context_str}

Query: {query}

Based on the above context, please provide a detailed answer:"""
        
        # Generate response
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = self.bart_model.generate(**inputs)
        response = self.tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        
        # Update metrics
        self.metrics['generation_time'].append(time.time() - start_time)
        self.logger.info(f"Generation time: {self.metrics['generation_time'][-1]:.2f}s")
        
        return response
    
    def process_query(self, query: str) -> Tuple[str, List[str], Dict]:
        """Process a user query through the complete pipeline"""
        # Refine query
        refined_query, sub_queries = self.refine_query(query)
        
        # Retrieve context
        context, metadata = self.retrieve_context(refined_query)
        
        # Generate response
        response = self.generate_response(refined_query, context, metadata)
        
        # Return results and metrics
        metrics_summary = {
            'retrieval_time': np.mean(self.metrics['retrieval_time'][-5:]),
            'generation_time': np.mean(self.metrics['generation_time'][-5:]),
            'memory_usage': np.mean(self.metrics['memory_usage'][-5:])
        }
        
        return response, context, metrics_summary
    
    def save_state(self, filepath: str) -> None:
        """Save the current state of the RAG system"""
        state = {
            'texts': self.texts,
            'metadata': self.metadata,
            'index': faiss.serialize_index(self.index),
            'metrics': self.metrics
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
    
    def load_state(self, filepath: str) -> None:
        """Load a saved state of the RAG system"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        
        self.texts = state['texts']
        self.metadata = state['metadata']
        self.index = faiss.deserialize_index(state['index'])
        self.metrics = state['metrics']

# Example usage
if __name__ == "__main__":
    # Initialize the system
    rag = AdvancedRAG()
    
    # Add some initial knowledge with metadata
    knowledge = [
        ("Python is a high-level programming language.", {"type": "programming", "topic": "python"}),
        ("Python is a non-venomous snake found in Asia.", {"type": "animal", "topic": "python"}),
        ("Machine learning is a subset of artificial intelligence.", {"type": "AI", "topic": "ML"})
    ]
    
    for text, metadata in knowledge:
        rag.add_to_knowledge_base(text, metadata)
    
    # Example ambiguous query
    query = "Tell me about Python"
    response, context, metrics = rag.process_query(query)
    
    print("\nQuery:", query)
    print("\nContext used:")
    for i, text in enumerate(context):
        print(f"{i+1}. {text}")
    print("\nResponse:", response)
    print("\nMetrics:", metrics)
    
    # Save the state
    rag.save_state("advanced_rag_state.pkl") 