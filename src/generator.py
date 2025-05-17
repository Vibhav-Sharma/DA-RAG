from transformers import BartForConditionalGeneration, BartTokenizer
from typing import List, Dict, Tuple, Optional
import torch
import logging
import time

class ResponseGenerator:
    """
    Generates responses based on retrieved information and user queries,
    using BART or other compatible models.
    """
    
    def __init__(self, model_name: str = "facebook/bart-large"):
        """
        Initialize the response generator with a transformer model.
        
        Args:
            model_name: Name of the pre-trained model to use
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing ResponseGenerator with model: {model_name}")
        
        # Initialize model and tokenizer
        self.model = BartForConditionalGeneration.from_pretrained(model_name)
        self.tokenizer = BartTokenizer.from_pretrained(model_name)
        
        # Performance metrics
        self.metrics = {
            'generation_time': [],
            'prompt_length': [],
            'response_length': []
        }
    
    def generate_clarification_prompt(self, query: str, topics: List[Tuple[str, float]], 
                                      questions: List[str]) -> str:
        """
        Generate a clarification prompt to narrow down the topic.
        
        Args:
            query: The original user query
            topics: List of potential topics with confidence scores
            questions: List of clarification questions
            
        Returns:
            A formatted clarification prompt
        """
        # Take top 3 topics
        top_topics = topics[:min(3, len(topics))]
        
        # Create the prompt
        prompt = f"I see your question is about: '{query}'\n\n"
        
        if top_topics:
            prompt += "Based on your query, I think you might be interested in:\n"
            for i, (topic, score) in enumerate(top_topics):
                prompt += f"- {topic}\n"
            prompt += "\n"
        
        prompt += "To help me provide a more specific answer:\n"
        for i, question in enumerate(questions[:3]):  # Limit to top 3 questions
            prompt += f"{i+1}. {question}\n"
        
        return prompt
    
    def generate_response(self, query: str, context: List[str], 
                         narrow_topic: Optional[str] = None) -> str:
        """
        Generate a response based on the query and retrieved context.
        
        Args:
            query: The user query
            context: List of retrieved documents providing context
            narrow_topic: Optional narrowed-down topic for more focused response
            
        Returns:
            Generated response
        """
        start_time = time.time()
        
        # Validate and clean context
        cleaned_context = []
        for ctx in context:
            if isinstance(ctx, str):
                # Remove any non-printable characters and normalize whitespace
                cleaned = ''.join(char for char in ctx if char.isprintable())
                cleaned = ' '.join(cleaned.split())
                if cleaned:  # Only add non-empty strings
                    cleaned_context.append(cleaned)
        
        if not cleaned_context:
            self.logger.warning("No valid context found after cleaning")
            return "I apologize, but I couldn't find any valid context to answer your question. Could you please rephrase your query or provide more specific information?"
        
        # Build prompt with cleaned context
        context_text = " ".join(cleaned_context)
        
        # Add topic focus if provided
        topic_text = f" focusing specifically on {narrow_topic}" if narrow_topic else ""
        
        prompt = f"""Context information:
{context_text}

Based on the above context, please answer this query{topic_text}:
{query}

Answer:"""

        # Track prompt length
        self.metrics['prompt_length'].append(len(prompt.split()))
        
        try:
            # Tokenize input with explicit encoding
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                max_length=1024,
                truncation=True,
                padding=True,
                add_special_tokens=True
            )
            
            # Generate response with more controlled parameters
            output_ids = self.model.generate(
                inputs["input_ids"],
                max_length=256,
                min_length=50,
                num_beams=4,
                length_penalty=2.0,
                early_stopping=True,
                no_repeat_ngram_size=3,
                temperature=0.7,
                top_k=50,
                top_p=0.95
            )
            
            # Decode with explicit handling of special tokens
            response = self.tokenizer.decode(
                output_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            
            # Clean up the response
            response = ' '.join(response.split())
            
            # Track metrics
            gen_time = time.time() - start_time
            self.metrics['generation_time'].append(gen_time)
            self.metrics['response_length'].append(len(response.split()))
            
            self.logger.info(f"Response generated in {gen_time:.2f}s")
            return response
            
        except Exception as e:
            self.logger.error(f"Error generating response: {str(e)}")
            return "I apologize, but I encountered an error while generating the response. Please try rephrasing your query."
    
    def generate_topic_focused_response(self, query: str, original_context: List[str], 
                                       focused_context: List[str], topic: str) -> str:
        """
        Generate a response that combines both broad and focused context.
        
        Args:
            query: The user query
            original_context: Context from the initial broad query
            focused_context: Context specific to the narrowed topic
            topic: The narrowed topic
            
        Returns:
            Generated response
        """
        # Combine contexts, prioritizing focused context
        # Use set operations to remove duplicates
        combined_context = list(set(focused_context + original_context))
        
        # Limit context length
        combined_context = combined_context[:min(5, len(combined_context))]
        
        prompt = f"""Context information:
{' '.join(combined_context)}

The user asked about "{query}" with specific interest in the topic of "{topic}".
Based on the above context, provide a focused answer addressing the topic of {topic}:

Answer:"""

        # Track prompt length
        self.metrics['prompt_length'].append(len(prompt.split()))
        
        # Tokenize input
        inputs = self.tokenizer(prompt, return_tensors="pt", max_length=1024, truncation=True)
        
        # Generate response
        start_time = time.time()
        output_ids = self.model.generate(
            inputs["input_ids"],
            max_length=256,
            min_length=50,
            num_beams=4,
            length_penalty=2.0,
            early_stopping=True
        )
        
        response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)
        
        # Track metrics
        gen_time = time.time() - start_time
        self.metrics['generation_time'].append(gen_time)
        self.metrics['response_length'].append(len(response.split()))
        
        return response
    
    def get_performance_metrics(self) -> Dict:
        """Get the average performance metrics"""
        metrics = {}
        for key, values in self.metrics.items():
            if values:
                metrics[key] = {
                    'avg': sum(values) / len(values),
                    'max': max(values),
                    'min': min(values),
                    'count': len(values)
                }
            else:
                metrics[key] = {'avg': 0, 'max': 0, 'min': 0, 'count': 0}
        return metrics