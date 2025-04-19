from typing import List, Dict, Tuple, Optional, Any
import logging
import time
from .retriever import DynamicRetriever
from .generator import ResponseGenerator

class DynamicRAGPipeline:
    """
    Complete pipeline for Dynamic Retrieval-Augmented Generation,
    combining topic narrowing, dynamic retrieval, and response generation.
    """
    
    def __init__(self, 
                retriever_model: str = "facebook-dpr-ctx_encoder-single-nq-base",
                generator_model: str = "facebook/bart-large"):
        """
        Initialize the DA-RAG pipeline.
        
        Args:
            retriever_model: Model for the retriever
            generator_model: Model for the generator
        """
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.logger.info("Initializing Dynamic RAG Pipeline")
        
        # Initialize components
        self.retriever = DynamicRetriever(model_name=retriever_model)
        self.generator = ResponseGenerator(model_name=generator_model)
        
        # Conversation state
        self.conversation_history = []
        self.current_topic = None
        self.narrow_topic = None
        self.clarification_needed = False
        
        # Pipeline metrics
        self.metrics = {
            'total_time': [],
            'clarification_rounds': [],
            'topic_changes': []
        }
    
    def add_knowledge(self, documents: List[str], topics: List[List[str]]) -> None:
        """
        Add documents to the knowledge base with topic labels.
        
        Args:
            documents: List of document texts
            topics: List of topic lists, one per document
        """
        assert len(documents) == len(topics), "Documents and topics must have the same length"
        
        for doc, doc_topics in zip(documents, topics):
            self.retriever.add_document(doc, doc_topics)
        
        self.logger.info(f"Added {len(documents)} documents to knowledge base")
    
    def add_topic_hierarchies(self, hierarchies: Dict[str, List[str]]) -> None:
        """
        Add topic hierarchies for narrowing down topics.
        
        Args:
            hierarchies: Dictionary mapping parent topics to subtopics
        """
        for parent, subtopics in hierarchies.items():
            self.retriever.add_topic_hierarchy(parent, subtopics)
    
    def process_initial_query(self, query: str) -> Dict[str, Any]:
        """
        Process the initial user query, identify topics, and decide if clarification is needed.
        
        Args:
            query: The user query
            
        Returns:
            Dictionary with response type, potential topics, and clarification questions if needed
        """
        start_time = time.time()
        
        # Add query to conversation history
        self.conversation_history.append({"role": "user", "content": query})
        
        # Identify potential broad topics
        potential_topics = self.retriever.identify_broad_topic(query)
        
        # Retrieve initial context
        initial_context = self.retriever.retrieve(query, k=3)
        
        result = {
            "original_query": query,
            "potential_topics": potential_topics,
            "initial_context": initial_context
        }
        
        # Decide if clarification is needed
        if potential_topics and len(potential_topics) > 1:
            # Multiple potential topics - generate clarification questions
            self.clarification_needed = True
            self.current_topic = potential_topics[0][0]  # Set the most likely topic
            
            # Generate clarification questions for the most likely topic
            clarification_questions = self.retriever.generate_clarification_questions(self.current_topic)
            
            # Generate clarification prompt
            clarification_prompt = self.generator.generate_clarification_prompt(
                query, potential_topics, clarification_questions
            )
            
            result["response_type"] = "clarification"
            result["clarification_prompt"] = clarification_prompt
            result["clarification_questions"] = clarification_questions
            
            # Add to conversation history
            self.conversation_history.append({"role": "system", "content": clarification_prompt})
        else:
            # Clear topic or only one potential topic - generate response directly
            self.clarification_needed = False
            self.current_topic = potential_topics[0][0] if potential_topics else None
            
            # Generate initial response
            response = self.generator.generate_response(query, initial_context)
            
            result["response_type"] = "answer"
            result["response"] = response
            
            # Add to conversation history
            self.conversation_history.append({"role": "assistant", "content": response})
        
        # Track metrics
        self.metrics['total_time'].append(time.time() - start_time)
        
        return result
    
    def process_clarification_response(self, user_response: str) -> Dict[str, Any]:
        """
        Process user's response to clarification questions.
        
        Args:
            user_response: User's response to clarification
            
        Returns:
            Dictionary with response and context
        """
        start_time = time.time()
        
        # Add to conversation history
        self.conversation_history.append({"role": "user", "content": user_response})
        
        # Analyze user response to identify the narrowed topic
        original_query = self.conversation_history[-3]["content"]  # Get the original query
        
        # Combine original query with clarification response
        combined_query = f"{original_query} {user_response}"
        
        # Re-identify topics with the combined query
        topics = self.retriever.identify_broad_topic(combined_query)
        
        # Extract most likely narrow topic
        if topics:
            self.narrow_topic = topics[0][0]
        else:
            self.narrow_topic = self.current_topic  # Fallback to current topic
            
        self.logger.info(f"Narrowed topic: {self.narrow_topic}")
        
        # Retrieve focused context for the narrow topic
        focused_context = self.retriever.retrieve_for_narrow_topic(
            combined_query, self.narrow_topic, k=3
        )
        
        # Get initial context from the original query
        original_context = []
        for entry in self.conversation_history:
            if "initial_context" in entry:
                original_context = entry["initial_context"]
                break
        
        # Generate focused response
        response = self.generator.generate_topic_focused_response(
            original_query, original_context, focused_context, self.narrow_topic
        )
        
        # Reset clarification flag
        self.clarification_needed = False
        
        result = {
            "response_type": "answer",
            "response": response,
            "narrow_topic": self.narrow_topic,
            "focused_context": focused_context
        }
        
        # Add to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        
        # Track metrics
        self.metrics['total_time'].append(time.time() - start_time)
        self.metrics['clarification_rounds'].append(1)
        if self.narrow_topic != self.current_topic:
            self.metrics['topic_changes'].append(1)
        
        return result
    
    def process_query(self, query: str) -> Dict[str, Any]:
        """
        Process a user query through the complete pipeline.
        
        Args:
            query: The user query
            
        Returns:
            Dictionary with response and context
        """
        # Check if we need a clarification response
        if self.clarification_needed:
            return self.process_clarification_response(query)
        else:
            return self.process_initial_query(query)
    
    def get_performance_metrics(self) -> Dict:
        """Get combined performance metrics from all components"""
        retriever_metrics = self.retriever.get_performance_metrics()
        generator_metrics = self.generator.get_performance_metrics()
        
        # Calculate pipeline metrics
        for key, values in self.metrics.items():
            if values:
                retriever_metrics[f"pipeline_{key}"] = {
                    'avg': sum(values) / len(values),
                    'max': max(values),
                    'min': min(values),
                    'count': len(values)
                }
        
        # Combine all metrics
        metrics = {
            "retriever": retriever_metrics,
            "generator": generator_metrics,
            "conversation_turns": len(self.conversation_history)
        }
        
        return metrics