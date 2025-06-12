from typing import List, Dict, Optional
import logging
import time
from retriever import DynamicRetriever
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import torch

logger = logging.getLogger(__name__)

class DynamicRAGPipeline:
    def __init__(self):
        logger.info("Initializing DynamicRAGPipeline")
        self.retriever = DynamicRetriever()
        model_name = "facebook/bart-large-cnn"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.generator = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.generator.to(self.device)
        self.conversation_history = []
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
        start_time = time.time()
        self.metrics['total_queries'] += 1

        try:
            query = query.strip()
            documents = self.retriever.fetch_wikipedia_data(query)
            if not documents:
                raise ValueError("No relevant documents found")

            topics = documents[0].get("topics", [])
            if topics:
                clarification_questions = self.retriever.generate_clarification_questions(topics[0])
                return {
                    'response': f"Can you clarify what exactly you're asking about in: {topics[0]}?",
                    'clarification_needed': True,
                    'topics': topics,
                    'questions': clarification_questions
                }

            generation_start = time.time()
            context = "\n".join([doc.get('content', '') or doc.get('summary', '') for doc in documents])
            response = self._generate_response(query, context)
            generation_time = time.time() - generation_start
            self.metrics['generation_times'].append(generation_time)

            sources = [{
                'title': doc.get('title', ''),
                'url': doc.get('url', ''),
                'relevance_score': doc.get('relevance_score', 0.0)
            } for doc in documents]

            self.conversation_history.append({
                'query': query,
                'response': response,
                'sources': sources,
                'timestamp': time.time()
            })

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
        input_text = f"question: {query} context: {context}"
        inputs = self.tokenizer(input_text, max_length=1024, truncation=True, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.generator.generate(
            **inputs,
            max_length=150,
            min_length=50,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        return response + "\n\nThis information is from Wikipedia."
