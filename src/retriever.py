import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import logging
import time
from collections import defaultdict

class DynamicRetriever:
    """
    Dynamic retrieval system that narrows down topics based on user queries
    and retrieves targeted information.
    """
    
    def __init__(self, model_name: str = "facebook-dpr-ctx_encoder-single-nq-base"):
        """
        Initialize the dynamic retriever with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"Initializing DynamicRetriever with model: {model_name}")
        
        # Initialize embedding model
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = 768  # Dimension for facebook-dpr model
        
        # Initialize FAISS indices - one main index and topic-specific indices
        self.main_index = faiss.IndexFlatL2(self.embedding_dim)
        self.topic_indices = {}  # Maps topic names to FAISS indices
        
        # Knowledge base storage
        self.documents = []  # All documents
        self.document_topics = []  # Topic labels for each document
        self.topic_documents = defaultdict(list)  # Maps topics to document indices
        
        # Topic hierarchy for narrowing down
        self.topic_hierarchy = {}  # Maps broader topics to specific sub-topics
        
        self.metrics = {
            'retrieval_time': [],
            'narrowing_time': [],
            'topic_accuracy': []
        }
    
    def add_document(self, document: str, topics: List[str]) -> int:
        """
        Add a document to the knowledge base with topic labels.
        
        Args:
            document: The document text
            topics: List of topic labels for the document
            
        Returns:
            Index of the added document
        """
        start_time = time.time()
        
        # Create embedding
        embedding = self.embedding_model.encode(document)
        embedding = embedding.reshape(1, -1)
        
        # Add to main index
        self.main_index.add(embedding)
        
        # Store document and its topics
        doc_id = len(self.documents)
        self.documents.append(document)
        self.document_topics.append(topics)
        
        # Add to topic-specific indices
        for topic in topics:
            if topic not in self.topic_indices:
                # Create new index for this topic
                self.topic_indices[topic] = faiss.IndexFlatL2(self.embedding_dim)
            
            # Add to topic index
            self.topic_indices[topic].add(embedding)
            
            # Track which documents belong to this topic
            self.topic_documents[topic].append(doc_id)
        
        self.metrics['retrieval_time'].append(time.time() - start_time)
        return doc_id
    
    def add_topic_hierarchy(self, parent_topic: str, subtopics: List[str]) -> None:
        """
        Add a topic hierarchy for narrowing down topics.
        
        Args:
            parent_topic: The broader parent topic
            subtopics: List of more specific subtopics
        """
        self.topic_hierarchy[parent_topic] = subtopics
        self.logger.info(f"Added topic hierarchy: {parent_topic} -> {subtopics}")
    
    def identify_broad_topic(self, query: str) -> List[Tuple[str, float]]:
        """
        Identify the broad topic of a user query.
        
        Args:
            query: The user query text
            
        Returns:
            List of potential topics with confidence scores
        """
        start_time = time.time()
        
        # Get query embedding
        query_embedding = self.embedding_model.encode(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Find similar documents in main index
        k = min(5, self.main_index.ntotal)  # Retrieve top-k documents
        if k == 0:
            return []
            
        distances, indices = self.main_index.search(query_embedding, k)
        
        # Count topic occurrences in retrieved documents
        topic_scores = defaultdict(float)
        for i, doc_idx in enumerate(indices[0]):
            similarity_score = 1.0 / (1.0 + distances[0][i])  # Convert distance to similarity
            
            for topic in self.document_topics[doc_idx]:
                topic_scores[topic] += similarity_score
        
        # Sort topics by score
        sorted_topics = sorted(
            [(topic, score) for topic, score in topic_scores.items()],
            key=lambda x: x[1],
            reverse=True
        )
        
        self.metrics['narrowing_time'].append(time.time() - start_time)
        return sorted_topics
    
    def generate_clarification_questions(self, broad_topic: str) -> List[str]:
        """
        Generate clarification questions to narrow down a broad topic.
        
        Args:
            broad_topic: The broad topic identified from the user query
            
        Returns:
            List of clarification questions
        """
        questions = []
        
        # Check if we have subtopics for this topic
        if broad_topic in self.topic_hierarchy:
            subtopics = self.topic_hierarchy[broad_topic]
            
            # Create general question
            general_question = f"Your query is about {broad_topic}. Could you specify which aspect you're interested in?"
            questions.append(general_question)
            
            # Create specific questions for each subtopic
            for subtopic in subtopics:
                question = f"Are you interested in {subtopic} specifically?"
                questions.append(question)
                
            # Add open-ended question
            questions.append(f"Is there a specific aspect of {broad_topic} you want to focus on?")
        else:
            # Generic questions if no subtopics are defined
            questions = [
                f"Could you provide more details about what aspect of {broad_topic} you're interested in?",
                f"What specific information about {broad_topic} are you looking for?",
                f"Are you looking for general information about {broad_topic} or something specific?"
            ]
            
        return questions
    
    def retrieve_for_narrow_topic(self, query: str, topic: str, k: int = 5) -> List[str]:
        """
        Retrieve documents for a narrowed-down topic.
        
        Args:
            query: The user query
            topic: The specific topic to search within
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        start_time = time.time()
        
        # Get query embedding
        query_embedding = self.embedding_model.encode(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        results = []
        
        # If we have a topic-specific index, use it
        if topic in self.topic_indices:
            topic_index = self.topic_indices[topic]
            
            # Retrieve from topic-specific index
            k = min(k, topic_index.ntotal)
            if k > 0:
                distances, indices = topic_index.search(query_embedding, k)
                
                # Get the actual document IDs from the topic's document list
                topic_doc_ids = self.topic_documents[topic]
                for idx in indices[0]:
                    results.append(self.documents[topic_doc_ids[idx]])
        else:
            # Fallback to main index if topic index doesn't exist
            # This is a simplified fallback; in a real system, you might use more sophisticated logic
            k = min(k, self.main_index.ntotal)
            if k > 0:
                distances, indices = self.main_index.search(query_embedding, k)
                for idx in indices[0]:
                    results.append(self.documents[idx])
        
        self.metrics['retrieval_time'].append(time.time() - start_time)
        return results
    
    def retrieve(self, query: str, topics: Optional[List[str]] = None, k: int = 5) -> List[str]:
        """
        Retrieve documents based on query, with optional topic filtering.
        
        Args:
            query: The user query
            topics: Optional list of topics to filter results
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        start_time = time.time()
        
        # Get query embedding
        query_embedding = self.embedding_model.encode(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Retrieve from main index
        k_main = min(k * 2, self.main_index.ntotal)  # Retrieve more, then filter
        if k_main == 0:
            return []
            
        distances, indices = self.main_index.search(query_embedding, k_main)
        
        # Filter by topics if specified
        results = []
        for i, doc_idx in enumerate(indices[0]):
            # If topics filter is provided, check if document has any of those topics
            if topics is None or any(topic in self.document_topics[doc_idx] for topic in topics):
                results.append((self.documents[doc_idx], distances[0][i]))
        
        # Sort by relevance and return top k
        results.sort(key=lambda x: x[1])
        documents = [doc for doc, _ in results[:k]]
        
        self.metrics['retrieval_time'].append(time.time() - start_time)
        return documents
    
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