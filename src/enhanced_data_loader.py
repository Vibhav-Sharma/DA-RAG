from typing import List, Dict, Any, Optional
import requests
import json
import pandas as pd
import wikipedia
import newsapi
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sqlite3
import os
import arxiv
import scholarly
from bs4 import BeautifulSoup
import feedparser
import yaml
from concurrent.futures import ThreadPoolExecutor
import time
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class EnhancedDataLoader:
    def __init__(self, db_path: str = "data/knowledge_base.db"):
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Initialize database
        self.db_path = db_path
        self._init_database()
        
        # Initialize API clients
        self.newsapi_key = os.getenv('NEWS_API_KEY')
        if self.newsapi_key:
            self.newsapi_client = newsapi.NewsApiClient(api_key=self.newsapi_key)
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load configuration from YAML file or use defaults."""
        config_path = Path("config/data_sources.yaml")
        if config_path.exists():
            with open(config_path) as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = {
                "arxiv": {
                    "max_results": 10,
                    "categories": ["cs.AI", "cs.CL", "cs.LG"]
                },
                "rss_feeds": [
                    "https://arxiv.org/rss/cs.AI",
                    "https://arxiv.org/rss/cs.CL",
                    "https://arxiv.org/rss/cs.LG"
                ]
            }
            # Create config directory if it doesn't exist
            config_path.parent.mkdir(parents=True, exist_ok=True)
            with open(config_path, 'w') as f:
                yaml.dump(self.config, f)
    
    def _init_database(self):
        """Initialize SQLite database with required tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create knowledge table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            title TEXT NOT NULL,
            content TEXT NOT NULL,
            source TEXT NOT NULL,
            url TEXT,
            date TEXT,
            metadata TEXT,
            embedding BLOB,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        # Create sources table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE,
            last_updated TEXT,
            article_count INTEGER DEFAULT 0
        )
        ''')
        
        # Create update_log table
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS update_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            source TEXT NOT NULL,
            update_time TEXT DEFAULT CURRENT_TIMESTAMP,
            articles_added INTEGER,
            status TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_wikipedia_data(self, topics: List[str], max_pages: int = 5) -> List[Dict[str, Any]]:
        """Load articles from Wikipedia for given topics."""
        articles = []
        for topic in topics:
            try:
                # Search for pages
                search_results = wikipedia.search(topic, results=max_pages)
                for page_title in search_results:
                    try:
                        # Get page content
                        page = wikipedia.page(page_title, auto_suggest=False)
                        article = {
                            'title': page.title,
                            'content': page.content,
                            'source': 'wikipedia',
                            'url': page.url,
                            'date': datetime.now().isoformat(),
                            'metadata': json.dumps({
                                'summary': page.summary,
                                'categories': page.categories
                            })
                        }
                        articles.append(article)
                        self.logger.info(f"Loaded Wikipedia article: {page.title}")
                    except wikipedia.exceptions.DisambiguationError as e:
                        self.logger.warning(f"Disambiguation page for {page_title}: {str(e)}")
                    except wikipedia.exceptions.PageError as e:
                        self.logger.warning(f"Page not found: {page_title}")
                    except Exception as e:
                        self.logger.error(f"Error loading Wikipedia page {page_title}: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error searching Wikipedia for topic {topic}: {str(e)}")
        
        return articles
    
    def load_arxiv_papers(self, topics: List[str], max_results: int = 10) -> List[Dict[str, Any]]:
        """Load papers from arXiv for given topics."""
        articles = []
        for topic in topics:
            try:
                # Search arXiv
                search = arxiv.Search(
                    query=topic,
                    max_results=max_results,
                    sort_by=arxiv.SortCriterion.SubmittedDate
                )
                
                for result in search.results():
                    article = {
                        'title': result.title,
                        'content': result.summary,
                        'source': 'arxiv',
                        'url': result.entry_id,
                        'date': result.published.isoformat(),
                        'metadata': json.dumps({
                            'authors': [author.name for author in result.authors],
                            'categories': result.categories,
                            'pdf_url': result.pdf_url
                        })
                    }
                    articles.append(article)
                    self.logger.info(f"Loaded arXiv paper: {result.title}")
            except Exception as e:
                self.logger.error(f"Error loading arXiv papers for topic {topic}: {str(e)}")
        
        return articles
    
    def load_rss_feeds(self, feed_urls: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """Load articles from RSS feeds."""
        if feed_urls is None:
            feed_urls = self.config['rss_feeds']
        
        articles = []
        for url in feed_urls:
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries:
                    article = {
                        'title': entry.title,
                        'content': entry.get('summary', ''),
                        'source': 'rss',
                        'url': entry.link,
                        'date': entry.get('published', datetime.now().isoformat()),
                        'metadata': json.dumps({
                            'author': entry.get('author', ''),
                            'tags': [tag.term for tag in entry.get('tags', [])]
                        })
                    }
                    articles.append(article)
                    self.logger.info(f"Loaded RSS article: {entry.title}")
            except Exception as e:
                self.logger.error(f"Error loading RSS feed {url}: {str(e)}")
        
        return articles
    
    def save_articles(self, articles: List[Dict[str, Any]]) -> None:
        """Save articles to the database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for article in articles:
            try:
                cursor.execute('''
                INSERT INTO knowledge (title, content, source, url, date, metadata)
                VALUES (?, ?, ?, ?, ?, ?)
                ''', (
                    article['title'],
                    article['content'],
                    article['source'],
                    article.get('url', ''),
                    article.get('date', datetime.now().isoformat()),
                    article.get('metadata', '{}')
                ))
                
                # Update source statistics
                cursor.execute('''
                INSERT INTO sources (name, last_updated, article_count)
                VALUES (?, ?, 1)
                ON CONFLICT(name) DO UPDATE SET
                    last_updated = ?,
                    article_count = article_count + 1
                ''', (article['source'], datetime.now().isoformat(), datetime.now().isoformat()))
                
            except Exception as e:
                self.logger.error(f"Error saving article {article['title']}: {str(e)}")
        
        conn.commit()
        conn.close()
    
    def load_articles_from_db(self, limit: int = 10, source: Optional[str] = None) -> List[Dict[str, Any]]:
        """Load articles from the database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = '''
        SELECT * FROM knowledge
        WHERE 1=1
        '''
        params = []
        
        if source:
            query += ' AND source = ?'
            params.append(source)
        
        query += ' ORDER BY date DESC LIMIT ?'
        params.append(limit)
        
        cursor.execute(query, params)
        articles = [dict(row) for row in cursor.fetchall()]
        
        conn.close()
        return articles
    
    def update_knowledge_base_parallel(self, topics: List[str]) -> None:
        """Update knowledge base using parallel processing."""
        with ThreadPoolExecutor(max_workers=3) as executor:
            # Start Wikipedia data loading
            wiki_future = executor.submit(self.load_wikipedia_data, topics)
            
            # Start arXiv data loading
            arxiv_future = executor.submit(self.load_arxiv_papers, topics)
            
            # Start RSS feed loading
            rss_future = executor.submit(self.load_rss_feeds)
            
            # Collect results
            articles = []
            articles.extend(wiki_future.result())
            articles.extend(arxiv_future.result())
            articles.extend(rss_future.result())
            
            # Save all articles
            self.save_articles(articles)
    
    def get_source_statistics(self) -> Dict[str, int]:
        """Get statistics about the knowledge base."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT source, COUNT(*) as count
        FROM knowledge
        GROUP BY source
        ''')
        
        stats = {row[0]: row[1] for row in cursor.fetchall()}
        conn.close()
        return stats
    
    def cleanup_old_entries(self, days_to_keep: int = 30) -> None:
        """Remove entries older than specified days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff_date = (datetime.now() - timedelta(days=days_to_keep)).isoformat()
        
        cursor.execute('''
        DELETE FROM knowledge
        WHERE date < ?
        ''', (cutoff_date,))
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        self.logger.info(f"Cleaned up {deleted_count} old entries")

# Example usage
if __name__ == "__main__":
    # Initialize enhanced data loader
    loader = EnhancedDataLoader()
    
    # Update knowledge base with parallel processing
    topics = [
        "Artificial Intelligence",
        "Machine Learning",
        "Natural Language Processing"
    ]
    
    loader.update_knowledge_base_parallel(topics)
    
    # Get statistics
    stats = loader.get_source_statistics()
    print("\nKnowledge base statistics:")
    for source, count in stats.items():
        print(f"{source}: {count} articles")
    
    # Cleanup old entries
    loader.cleanup_old_entries(days_to_keep=30) 