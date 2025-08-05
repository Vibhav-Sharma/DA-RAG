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
        logger.info(f"DynamicRAGPipeline initialized on {self.device}")

    def process_query(self, query: str, clarification_topic: Optional[str] = None) -> Dict:
        start_time = time.time()
        self.metrics['total_queries'] += 1

        try:
            query = query.strip()

            documents = self.retriever.fetch_wikipedia_data(
                query,
                filter_topic=clarification_topic
            )

            if not documents:
                raise ValueError("No relevant documents found")

            top_doc = documents[0]
            topics = [t for t in top_doc.get("topics", []) if t.lower() not in
                      ['references', 'see also', 'notes', 'sources', 'further reading', 'external links']]

            if topics and clarification_topic is None:
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

            sources = [{'title': doc['title'], 'url': doc['url']} for doc in documents]

            self.conversation_history.append({
                'query': query,
                'response': response,
                'sources': sources,
                'timestamp': time.time()
            })

            total_time = time.time() - start_time
            self.metrics['successful_queries'] += 1
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
                'response': "Sorry, I couldn't process your query.",
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
