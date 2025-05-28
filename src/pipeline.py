from typing import List, Dict, Tuple, Optional, Any
import logging
import time
from .retriever import DynamicRetriever
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch
import numpy as np

logger = logging.getLogger(__name__)

class DynamicRAGPipeline:
    """
    Complete pipeline for Dynamic Retrieval-Augmented Generation,
    combining Wikipedia-based retrieval, topic narrowing, and response generation.
    """
    
    def __init__(self):
        """Initialize the pipeline with retriever and generator models."""
        logger.info("Initializing DynamicRAGPipeline")
        
        # Initialize retriever
        self.retriever = DynamicRetriever()
        
        # Initialize generator model
        model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Move model to GPU if available
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        
        # Initialize conversation history
        self.conversation_history = []
        
        # Initialize metrics
        self.metrics = {
            'total_queries': 0,
            'successful_queries': 0,
            'failed_queries': 0,
            'average_response_time': 0,
            'wikipedia_fetch_times': [],
            'generation_times': []
        }
        
        logger.info(f"DynamicRAGPipeline initialized successfully on {self.device}")
    
    def process_query(self, query: str, max_retries: int = 2) -> Dict:
        """Process a user query with improved error handling and retries."""
        start_time = time.time()
        self.metrics['total_queries'] += 1
        
        try:
            # Validate input
            if not isinstance(query, str) or not query.strip():
                raise ValueError("Query must be a non-empty string")
            
            # Clean and normalize query
            query = query.strip()
            logger.info(f"Processing query: {query}")
            
            # Try to identify broad topic
            broad_topics = self.retriever.identify_broad_topic(query)
            if not broad_topics:
                logger.warning("Could not identify broad topic")
            
            # Fetch relevant documents with retries
            documents = []
            for attempt in range(max_retries):
                try:
                    documents = self.retriever.fetch_wikipedia_data(query)
                    if documents:
                        break
                    logger.warning(f"Attempt {attempt + 1}: No documents found")
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1} failed: {str(e)}")
                    if attempt == max_retries - 1:
                        raise
            
            if not documents:
                raise ValueError("No relevant documents found")
            
            # Generate response
            generation_start = time.time()
            context = "\n".join([doc['summary'] for doc in documents])
            response = self._generate_response(query, context)
            generation_time = time.time() - generation_start
            self.metrics['generation_times'].append(generation_time)
            
            # Prepare response with sources
            sources = [{
                'title': doc['title'],
                'url': doc['url'],
                'relevance_score': doc.get('relevance_score', 0.0)
            } for doc in documents]
            
            # Update conversation history
            self.conversation_history.append({
                'query': query,
                'response': response,
                'sources': sources,
                'timestamp': time.time()
            })
            
            # Update metrics
            self.metrics['successful_queries'] += 1
            total_time = time.time() - start_time
            self.metrics['average_response_time'] = (
                (self.metrics['average_response_time'] * (self.metrics['successful_queries'] - 1) + total_time)
                / self.metrics['successful_queries']
            )
            
            return {
                'response': response,
                'sources': sources,
                'metrics': {
                    'response_time': total_time,
                    'generation_time': generation_time,
                    'wikipedia_fetch_time': self.retriever.metrics['wikipedia_fetch_time'][-1]
                }
            }
            
        except Exception as e:
            self.metrics['failed_queries'] += 1
            logger.error(f"Error processing query: {str(e)}")
            return {
                'error': str(e),
                'response': "I apologize, but I encountered an error processing your query. Please try rephrasing it or ask about a different topic.",
                'sources': [],
                'metrics': {
                    'response_time': time.time() - start_time,
                    'error_type': type(e).__name__
                }
            }

    def _generate_response(self, query: str, context: str) -> str:
        """Generate a response using the BART model with improved context handling."""
        try:
            # Prepare input
            input_text = f"question: {query} context: {context}"
            inputs = self.tokenizer(input_text, max_length=1024, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Generate response
            outputs = self.generator.generate(
                **inputs,
                max_length=150,
                min_length=50,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True
            )
            
            # Decode and clean response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            response = response.strip()
            
            # Add source attribution
            response += "\n\nThis information is from Wikipedia."
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            raise

    def get_conversation_history(self, limit: Optional[int] = None) -> List[Dict]:
        """Get conversation history with optional limit."""
        if limit is None:
            return self.conversation_history
        return self.conversation_history[-limit:]

    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        return {
            'total_queries': self.metrics['total_queries'],
            'successful_queries': self.metrics['successful_queries'],
            'failed_queries': self.metrics['failed_queries'],
            'success_rate': (self.metrics['successful_queries'] / self.metrics['total_queries'] * 100 
                           if self.metrics['total_queries'] > 0 else 0),
            'average_response_time': self.metrics['average_response_time'],
            'average_generation_time': (sum(self.metrics['generation_times']) / len(self.metrics['generation_times'])
                                      if self.metrics['generation_times'] else 0),
            'average_wikipedia_fetch_time': (sum(self.retriever.metrics['wikipedia_fetch_time']) 
                                           / len(self.retriever.metrics['wikipedia_fetch_time'])
                                           if self.retriever.metrics['wikipedia_fetch_time'] else 0)
        }