from src.data_loader import DataLoader
from config import DEFAULT_TOPICS
import logging
import os
from dotenv import load_dotenv

def test_data_loader():
    print("\n🧪 Testing Data Loading System...")
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize data loader
    print("\n🔹 Initializing DataLoader...")
    loader = DataLoader()
    print("✅ DataLoader Initialized Successfully.")
    
    # Test Wikipedia data loading
    print("\n🔹 Testing Wikipedia data loading...")
    test_topics = ["Python Programming", "Machine Learning"]
    wiki_articles = loader.load_wikipedia_data(test_topics, max_pages=2)
    print(f"✅ Loaded {len(wiki_articles)} Wikipedia articles")
    
    # Test database operations
    print("\n🔹 Testing database operations...")
    loader.save_to_database(wiki_articles)
    loaded_articles = loader.load_from_database(limit=5)
    print(f"✅ Successfully saved and loaded {len(loaded_articles)} articles from database")
    
    # Test CSV data loading
    print("\n🔹 Testing CSV data loading...")
    # Create a sample CSV file
    import pandas as pd
    sample_data = {
        'text': ['Sample text 1', 'Sample text 2'],
        'category': ['AI', 'ML'],
        'source': ['test', 'test']
    }
    pd.DataFrame(sample_data).to_csv('data/sample.csv', index=False)
    
    csv_articles = loader.load_csv_data(
        'data/sample.csv',
        text_column='text',
        metadata_columns=['category', 'source']
    )
    print(f"✅ Loaded {len(csv_articles)} entries from CSV")
    
    # Test knowledge base update
    print("\n🔹 Testing knowledge base update...")
    loader.update_knowledge_base(test_topics)
    print("✅ Knowledge base updated successfully")
    
    # Print sample of loaded data
    print("\n📚 Sample of loaded data:")
    for i, article in enumerate(loaded_articles[:2]):
        print(f"\nArticle {i+1}:")
        print(f"Source: {article['source']}")
        print(f"Text: {article['text'][:100]}...")
        print(f"Metadata: {article['metadata']}")
    
    print("\n🎉 Data loading system tests completed successfully! 🚀")

if __name__ == "__main__":
    # Load environment variables
    load_dotenv()
    
    # Run tests
    test_data_loader() 