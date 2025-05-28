import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import logging
import time
from collections import defaultdict
import wikipedia
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import os

logger = logging.getLogger(__name__)

class DynamicRetriever:
    """
    Dynamic retrieval system that fetches Wikipedia data on-demand and retrieves targeted information.
    """
    
    def __init__(self, model_name: str = "facebook-dpr-ctx_encoder-single-nq-base"):
        """
        Initialize the dynamic retriever with a sentence transformer model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        logger.info(f"Initializing DynamicRetriever with model: {model_name}")
        
        # Initialize models and storage
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.documents = []
        self.document_topics = []
        self.topic_hierarchy = {}
        
        # Initialize metrics
        self.metrics = {
            'retrieval_time': [],
            'narrowing_time': [],
            'topic_accuracy': [],
            'wikipedia_fetch_time': []
        }
        
        # Initialize Wikipedia session with retry logic
        self.wiki_session = requests.Session()
        retries = Retry(
            total=3,
            backoff_factor=0.5,
            status_forcelist=[500, 502, 503, 504]
        )
        self.wiki_session.mount('https://', HTTPAdapter(max_retries=retries))
        
        # Set up Wikipedia API
        wikipedia.set_lang("en")
        wikipedia.set_rate_limiting(True, min_wait=0.5)
        
        # Initialize cache directory
        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)
        
        logger.info("DynamicRetriever initialized successfully")
    
    @lru_cache(maxsize=1000)
    def _get_cached_page(self, title: str) -> Optional[Dict]:
        """Get a Wikipedia page from cache if available."""
        cache_file = os.path.join(self.cache_dir, f"{title.lower().replace(' ', '_')}.json")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Error reading cache for {title}: {str(e)}")
        return None

    def _cache_page(self, title: str, content: Dict):
        """Cache a Wikipedia page."""
        cache_file = os.path.join(self.cache_dir, f"{title.lower().replace(' ', '_')}.json")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(content, f, ensure_ascii=False, indent=2)
        except Exception as e:
            logger.warning(f"Error caching {title}: {str(e)}")

    def _handle_disambiguation(self, title: str, options: List[str]) -> Optional[str]:
        """Handle Wikipedia disambiguation pages by selecting the most relevant option."""
        if not options:
            return None
            
        # Get embeddings for all options
        option_embeddings = self.embedding_model.encode(options)
        title_embedding = self.embedding_model.encode([title])[0]
        
        # Calculate similarities
        similarities = np.dot(option_embeddings, title_embedding)
        best_idx = np.argmax(similarities)
        
        if similarities[best_idx] > 0.5:  # Only return if confidence is high enough
            return options[best_idx]
        return None

    def fetch_wikipedia_data(self, query: str, max_pages: int = 3) -> List[Dict]:
        """Fetch relevant Wikipedia pages for a query with improved error handling and caching."""
        start_time = time.time()
        results = []
        
        try:
            # First try to get from cache
            cached_result = self._get_cached_page(query)
            if cached_result:
                logger.info(f"Retrieved {query} from cache")
                return cached_result

            # Search Wikipedia
            search_results = wikipedia.search(query, results=max_pages)
            
            if not search_results:
                logger.warning(f"No Wikipedia pages found for query: {query}")
                return results

            # Process each search result
            for title in search_results:
                try:
                    # Check if it's a disambiguation page
                    page = wikipedia.page(title, auto_suggest=False)
                    
                    if "may refer to:" in page.summary or "may mean:" in page.summary:
                        # Handle disambiguation
                        options = [link for link in page.links if not link.startswith("Wikipedia:")]
                        selected_title = self._handle_disambiguation(query, options)
                        
                        if selected_title:
                            try:
                                page = wikipedia.page(selected_title, auto_suggest=False)
                            except wikipedia.exceptions.DisambiguationError as e:
                                logger.warning(f"Disambiguation failed for {title}: {str(e)}")
                                continue
                        else:
                            logger.warning(f"Could not resolve disambiguation for {title}")
                            continue
                    
                    # Cache the successful result
                    page_data = {
                        'title': page.title,
                        'url': page.url,
                        'summary': page.summary,
                        'content': page.content,
                        'topics': self._extract_topics(page.content)
                    }
                    self._cache_page(page.title, page_data)
                    results.append(page_data)
                    
                except wikipedia.exceptions.DisambiguationError as e:
                    logger.warning(f"Disambiguation error for {title}: {str(e)}")
                    # Try to handle disambiguation
                    selected_title = self._handle_disambiguation(query, e.options)
                    if selected_title:
                        try:
                            page = wikipedia.page(selected_title, auto_suggest=False)
                            page_data = {
                                'title': page.title,
                                'url': page.url,
                                'summary': page.summary,
                                'content': page.content,
                                'topics': self._extract_topics(page.content)
                            }
                            self._cache_page(page.title, page_data)
                            results.append(page_data)
                        except Exception as e:
                            logger.warning(f"Error fetching disambiguated page {selected_title}: {str(e)}")
                    
                except Exception as e:
                    logger.warning(f"Error fetching page {title}: {str(e)}")
                    continue

        except Exception as e:
            logger.error(f"Error fetching Wikipedia data: {str(e)}")
        
        finally:
            fetch_time = time.time() - start_time
            self.metrics['wikipedia_fetch_time'].append(fetch_time)
            logger.info(f"Wikipedia fetch completed in {fetch_time:.2f}s")
        
        return results

    def _extract_topics(self, content: str) -> List[str]:
        """Extract potential topics from content using simple heuristics."""
        # Split content into sentences and look for topic indicators
        sentences = content.split('.')
        topics = []
        
        for sentence in sentences:
            # Look for common topic indicators
            if any(indicator in sentence.lower() for indicator in ['is a', 'refers to', 'deals with', 'about']):
                # Extract the subject
                words = sentence.split()
                if len(words) > 3:
                    topics.append(' '.join(words[:4]))
        
        return list(set(topics))  # Remove duplicates

    def add_topic_hierarchy(self, parent_topic: str, subtopics: List[str]) -> None:
        """
        Add a topic hierarchy for narrowing down topics.
        
        Args:
            parent_topic: The broader parent topic
            subtopics: List of more specific subtopics
        """
        self.topic_hierarchy[parent_topic] = subtopics
        logger.info(f"Added topic hierarchy: {parent_topic} -> {subtopics}")
    
    def identify_broad_topic(self, query: str) -> List[Tuple[str, float]]:
        """
        Identify the broad topic of a user query by fetching relevant Wikipedia pages.
        
        Args:
            query: The user query text
            
        Returns:
            List of potential topics with confidence scores
        """
        start_time = time.time()
        
        # Fetch Wikipedia data for the query
        wiki_data = self.fetch_wikipedia_data(query)
        
        if not wiki_data:
            return []
        
        # Clear previous session data
        self.documents = []
        self.document_topics = []
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add fetched documents to the index
        for doc in wiki_data:
            self.documents.append(doc['text'])
            self.document_topics.append(doc['topics'])
            
            # Create and add embedding
            embedding = self.embedding_model.encode(doc['text'])
            embedding = embedding.reshape(1, -1)
            self.index.add(embedding)
        
        # Get query embedding
        query_embedding = self.embedding_model.encode(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Find similar documents
        k = min(5, self.index.ntotal)
        if k == 0:
            return []
            
        distances, indices = self.index.search(query_embedding, k)
        
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
    
    def retrieve(self, query: str, topics: Optional[List[str]] = None, k: int = 5) -> List[str]:
        """
        Retrieve documents based on query by fetching from Wikipedia.
        
        Args:
            query: The user query
            topics: Optional list of topics to filter results
            k: Number of documents to retrieve
            
        Returns:
            List of relevant documents
        """
        start_time = time.time()
        
        # Fetch Wikipedia data for the query
        wiki_data = self.fetch_wikipedia_data(query, max_pages=k)
        
        if not wiki_data:
            return []
        
        # Clear previous session data
        self.documents = []
        self.document_topics = []
        self.index = faiss.IndexFlatL2(self.embedding_dim)
        
        # Add fetched documents to the index
        for doc in wiki_data:
            self.documents.append(doc['text'])
            self.document_topics.append(doc['topics'])
            
            # Create and add embedding
            embedding = self.embedding_model.encode(doc['text'])
            embedding = embedding.reshape(1, -1)
            self.index.add(embedding)
        
        # Get query embedding
        query_embedding = self.embedding_model.encode(query)
        query_embedding = query_embedding.reshape(1, -1)
        
        # Retrieve from index
        k = min(k, self.index.ntotal)
        if k == 0:
            return []
            
        distances, indices = self.index.search(query_embedding, k)
        
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