from src.retriever import DynamicRetriever
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_retriever():
    # Initialize the retriever
    retriever = DynamicRetriever()
    
    # Add some topic hierarchies for better topic narrowing
    retriever.add_topic_hierarchy(
        "artificial intelligence",
        ["machine learning", "natural language processing", "computer vision", "robotics"]
    )
    retriever.add_topic_hierarchy(
        "machine learning",
        ["supervised learning", "unsupervised learning", "reinforcement learning", "deep learning"]
    )
    
    # Test queries
    test_queries = [
        "What is artificial intelligence?",
        "How does machine learning work?",
        "Tell me about neural networks",
        "What are the applications of deep learning?"
    ]
    
    print("\n🚀 Testing Wikipedia-based Retriever")
    print("===================================")
    
    for query in test_queries:
        print(f"\n📝 Query: {query}")
        print("-" * 50)
        
        # Identify broad topic
        print("\n🔍 Identifying broad topic...")
        topics = retriever.identify_broad_topic(query)
        if topics:
            print("Potential topics:")
            for topic, score in topics[:3]:  # Show top 3 topics
                print(f"- {topic} (confidence: {score:.2f})")
            
            # Generate clarification questions for the top topic
            if topics:
                print("\n❓ Clarification questions:")
                questions = retriever.generate_clarification_questions(topics[0][0])
                for q in questions[:2]:  # Show first 2 questions
                    print(f"- {q}")
        
        # Retrieve relevant documents
        print("\n📚 Retrieving relevant documents...")
        documents = retriever.retrieve(query, k=2)  # Get top 2 documents
        
        if documents:
            print("\nRetrieved documents:")
            for i, doc in enumerate(documents, 1):
                print(f"\n{i}. {doc[:200]}...")  # Show first 200 chars
        else:
            print("No relevant documents found.")
        
        # Show metrics
        metrics = retriever.get_performance_metrics()
        print("\n📊 Performance metrics:")
        print(f"- Wikipedia fetch time: {metrics['wikipedia_fetch_time']['avg']:.2f}s")
        print(f"- Retrieval time: {metrics['retrieval_time']['avg']:.2f}s")
        print(f"- Topic narrowing time: {metrics['narrowing_time']['avg']:.2f}s")
        
        print("\n" + "="*50)

if __name__ == "__main__":
    test_retriever() 