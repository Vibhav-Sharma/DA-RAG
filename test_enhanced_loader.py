import logging
from src.enhanced_data_loader import EnhancedDataLoader
from datetime import datetime, timedelta
from typing import Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_enhanced_loader():
    try:
        # Initialize the enhanced data loader
        logger.info("Initializing EnhancedDataLoader...")
        loader = EnhancedDataLoader()
        
        # Test topics for data loading
        test_topics = [
            "Artificial Intelligence",
            "Machine Learning",
            "Natural Language Processing"
        ]
        
        # Test loading data from multiple sources
        logger.info("Testing data loading from multiple sources...")
        
        # 1. Test Wikipedia data loading
        logger.info("Testing Wikipedia data loading...")
        wiki_articles = loader.load_wikipedia_data(test_topics, max_pages=2)
        logger.info(f"Loaded {len(wiki_articles)} Wikipedia articles")
        
        # 2. Test arXiv data loading
        logger.info("Testing arXiv data loading...")
        arxiv_papers = loader.load_arxiv_papers(test_topics, max_results=5)
        logger.info(f"Loaded {len(arxiv_papers)} arXiv papers")
        
        # 3. Test RSS feed loading
        logger.info("Testing RSS feed loading...")
        rss_articles = loader.load_rss_feeds([
            "https://arxiv.org/rss/cs.AI",
            "https://arxiv.org/rss/cs.CL"
        ])
        logger.info(f"Loaded {len(rss_articles)} RSS articles")
        
        # 4. Test parallel knowledge base update
        logger.info("Testing parallel knowledge base update...")
        loader.update_knowledge_base_parallel(test_topics)
        
        # 5. Get source statistics
        logger.info("Getting source statistics...")
        stats = loader.get_source_statistics()
        logger.info("Source statistics:")
        for source, count in stats.items():
            logger.info(f"{source}: {count} articles")
        
        # 6. Test cleanup of old entries
        logger.info("Testing cleanup of old entries...")
        days_to_keep = 30
        loader.cleanup_old_entries(days_to_keep)
        
        # 7. Load articles from database
        logger.info("Loading articles from database...")
        articles = loader.load_articles_from_db(limit=5)
        logger.info(f"Loaded {len(articles)} articles from database")
        
        # Print sample article
        if articles:
            logger.info("\nSample article from database:")
            sample = articles[0]
            logger.info(f"Title: {sample['title']}")
            logger.info(f"Source: {sample['source']}")
            logger.info(f"Date: {sample['date']}")
            logger.info(f"Content preview: {sample['content'][:200]}...")
        
        logger.info("All tests completed successfully!")
        
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    test_enhanced_loader() 