import requests
import json
import pandas as pd
from typing import List, Dict, Optional
import wikipedia
import newsapi
from datetime import datetime, timedelta
import logging
from pathlib import Path
import sqlite3
import os

class DataLoader:
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
        
    def _init_database(self):
        """Initialize SQLite database for knowledge storage"""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS knowledge (
            id INTEGER PRIMARY KEY,
            text TEXT NOT NULL,
            source TEXT NOT NULL,
            metadata TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        ''')
        
        cursor.execute('''
        CREATE TABLE IF NOT EXISTS sources (
            id INTEGER PRIMARY KEY,
            name TEXT NOT NULL,
            type TEXT NOT NULL,
            last_updated DATETIME,
            config TEXT
        )
        ''')
        
        conn.commit()
        conn.close()
    
    def load_wikipedia_data(self, topics: List[str], max_pages: int = 10) -> List[Dict]:
        """Load data from Wikipedia articles"""
        articles = []
        for topic in topics:
            try:
                # Search for relevant pages
                search_results = wikipedia.search(topic, results=max_pages)
                for page_title in search_results:
                    try:
                        # Get page content
                        page = wikipedia.page(page_title)
                        articles.append({
                            'text': page.content,
                            'source': 'wikipedia',
                            'metadata': {
                                'title': page.title,
                                'url': page.url,
                                'categories': page.categories
                            }
                        })
                        self.logger.info(f"Loaded Wikipedia article: {page_title}")
                    except wikipedia.exceptions.DisambiguationError as e:
                        # Handle disambiguation pages
                        self.logger.warning(f"Disambiguation page found for {page_title}")
                    except Exception as e:
                        self.logger.error(f"Error loading Wikipedia page {page_title}: {str(e)}")
            except Exception as e:
                self.logger.error(f"Error searching Wikipedia for {topic}: {str(e)}")
        
        return articles
    
    def load_news_data(self, query: str, days: int = 7) -> List[Dict]:
        """Load recent news articles using NewsAPI"""
        if not self.newsapi_key:
            self.logger.error("NewsAPI key not found. Set NEWS_API_KEY environment variable.")
            return []
        
        try:
            # Calculate date range
            to_date = datetime.now()
            from_date = to_date - timedelta(days=days)
            
            # Get news articles
            response = self.newsapi_client.get_everything(
                q=query,
                from_param=from_date.strftime('%Y-%m-%d'),
                to=to_date.strftime('%Y-%m-%d'),
                language='en',
                sort_by='relevancy'
            )
            
            articles = []
            for article in response['articles']:
                articles.append({
                    'text': f"{article['title']}\n{article['description']}\n{article['content']}",
                    'source': 'newsapi',
                    'metadata': {
                        'title': article['title'],
                        'url': article['url'],
                        'published_at': article['publishedAt'],
                        'source': article['source']['name']
                    }
                })
            
            self.logger.info(f"Loaded {len(articles)} news articles for query: {query}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error loading news data: {str(e)}")
            return []
    
    def load_csv_data(self, file_path: str, text_column: str, metadata_columns: Optional[List[str]] = None) -> List[Dict]:
        """Load data from CSV files"""
        try:
            df = pd.read_csv(file_path)
            articles = []
            
            for _, row in df.iterrows():
                metadata = {}
                if metadata_columns:
                    metadata = {col: row[col] for col in metadata_columns if col in row}
                
                articles.append({
                    'text': str(row[text_column]),
                    'source': 'csv',
                    'metadata': metadata
                })
            
            self.logger.info(f"Loaded {len(articles)} entries from CSV: {file_path}")
            return articles
            
        except Exception as e:
            self.logger.error(f"Error loading CSV data from {file_path}: {str(e)}")
            return []
    
    def save_to_database(self, articles: List[Dict]) -> None:
        """Save articles to the SQLite database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for article in articles:
            cursor.execute('''
            INSERT INTO knowledge (text, source, metadata)
            VALUES (?, ?, ?)
            ''', (
                article['text'],
                article['source'],
                json.dumps(article['metadata'])
            ))
        
        conn.commit()
        conn.close()
        self.logger.info(f"Saved {len(articles)} articles to database")
    
    def load_from_database(self, limit: int = 1000) -> List[Dict]:
        """Load articles from the database"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT text, source, metadata, timestamp
        FROM knowledge
        ORDER BY timestamp DESC
        LIMIT ?
        ''', (limit,))
        
        articles = []
        for row in cursor.fetchall():
            articles.append({
                'text': row[0],
                'source': row[1],
                'metadata': json.loads(row[2]),
                'timestamp': row[3]
            })
        
        conn.close()
        self.logger.info(f"Loaded {len(articles)} articles from database")
        return articles
    
    def update_knowledge_base(self, topics: List[str]) -> None:
        """Update knowledge base with new data from various sources"""
        # Load Wikipedia data
        wiki_articles = self.load_wikipedia_data(topics)
        self.save_to_database(wiki_articles)
        
        # Load news data
        for topic in topics:
            news_articles = self.load_news_data(topic)
            self.save_to_database(news_articles)
        
        self.logger.info(f"Knowledge base updated with data for topics: {topics}")

# Example usage
if __name__ == "__main__":
    # Initialize data loader
    loader = DataLoader()
    
    # Update knowledge base with some topics
    topics = [
        "Artificial Intelligence",
        "Machine Learning",
        "Natural Language Processing",
        "Deep Learning"
    ]
    
    loader.update_knowledge_base(topics)
    
    # Load some data from the database
    articles = loader.load_from_database(limit=10)
    print(f"Loaded {len(articles)} articles from database") 