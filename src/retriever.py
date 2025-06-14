import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from sentence_transformers import SentenceTransformer
import logging
import time
from collections import defaultdict
import wikipedia
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import json
import os
from datetime import timedelta
from functools import lru_cache

logger = logging.getLogger(__name__)

class DynamicRetriever:
    def __init__(self, model_name: str = "facebook-dpr-ctx_encoder-single-nq-base"):
        logger.info(f"Initializing DynamicRetriever with model: {model_name}")

        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        self.documents = []
        self.document_topics = []
        self.topic_hierarchy = {}

        self.metrics = {
            'retrieval_time': [],
            'narrowing_time': [],
            'topic_accuracy': [],
            'wikipedia_fetch_time': []
        }

        self.wiki_session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        self.wiki_session.mount('https://', HTTPAdapter(max_retries=retries))

        wikipedia.set_lang("en")
        wikipedia.set_rate_limiting(True, min_wait=timedelta(seconds=0.5))

        self.cache_dir = os.path.join(os.path.dirname(__file__), '..', 'cache')
        os.makedirs(self.cache_dir, exist_ok=True)

        logger.info("DynamicRetriever initialized successfully")

    def fetch_wikipedia_data(self, query: str, max_pages: int = 3) -> List[Dict]:
        start_time = time.time()
        results = []

        try:
            search_results = wikipedia.search(query, results=max_pages)
            if not search_results:
                logger.warning(f"No Wikipedia pages found for query: {query}")
                return results

            for title in search_results:
                try:
                    page = wikipedia.page(title, auto_suggest=False)
                    page_data = {
                        'title': page.title,
                        'url': page.url,
                        'summary': page.summary,
                        'content': page.content,
                        'topics': self._extract_topics(page.content)
                    }
                    results.append(page_data)
                except Exception as e:
                    logger.warning(f"Error fetching page {title}: {str(e)}")

        except Exception as e:
            logger.error(f"Error fetching Wikipedia data: {str(e)}")

        finally:
            fetch_time = time.time() - start_time
            self.metrics['wikipedia_fetch_time'].append(fetch_time)
            logger.info(f"Wikipedia fetch completed in {fetch_time:.2f}s")

        return results

    def _extract_topics(self, content: str) -> List[str]:
        import re
        if not content:
            return []
        headings = re.findall(r'==+\s*(.*?)\s*==+', content)
        clean_headings = [h.strip() for h in headings if len(h.strip()) > 2]
        return clean_headings if clean_headings else ["Overview", "Introduction"]

    def generate_clarification_questions(self, broad_topic: str) -> List[str]:
        return [
            f"Could you provide more details about what aspect of {broad_topic} you're interested in?",
            f"What specific information about {broad_topic} are you looking for?",
            f"Are you looking for general information about {broad_topic} or something specific?"
        ]
