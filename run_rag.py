from src.pipeline import DynamicRAGPipeline
import logging
import sys
import os
from typing import List, Dict
import wikipedia
import json

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_initial_knowledge() -> List[Dict]:
    """Load initial knowledge from Wikipedia for default topics."""
    knowledge = []
    
    # Default topics from config
    topics = [
        "Artificial Intelligence",
        "Machine Learning",
        "Natural Language Processing",
        "Deep Learning",
        "Computer Science",
        "Data Science",
        "Neural Networks",
        "Transformer Models",
        "Large Language Models",
        "Information Retrieval",
        "Python Programming",
        "JavaScript",
        "Web Development",
        "Database Systems",
        "Cloud Computing",
        "Cybersecurity",
        "Blockchain",
        "Internet of Things",
        "Quantum Computing",
        "Robotics"
    ]
    
    logger.info("Loading initial knowledge from Wikipedia...")
    
    for topic in topics:
        try:
            # Get Wikipedia page
            page = wikipedia.page(topic, auto_suggest=True)
            
            # Add main content
            knowledge.append({
                "text": page.summary,
                "topics": [topic.lower(), "general knowledge"],
                "source": "wikipedia",
                "url": page.url
            })
            
            # Add some sections if available
            if hasattr(page, 'sections') and page.sections:
                for section in page.sections[:3]:  # Take first 3 sections
                    try:
                        section_content = page.section(section)
                        if section_content:
                            knowledge.append({
                                "text": section_content,
                                "topics": [topic.lower(), section.lower()],
                                "source": "wikipedia",
                                "url": page.url
                            })
                    except:
                        continue
                        
            logger.info(f"Loaded knowledge for: {topic}")
            
        except Exception as e:
            logger.warning(f"Could not load knowledge for {topic}: {str(e)}")
            continue
    
    return knowledge

def initialize_rag_system() -> DynamicRAGPipeline:
    """Initialize the RAG system with initial knowledge."""
    logger.info("Initializing RAG system...")
    
    # Initialize pipeline
    pipeline = DynamicRAGPipeline()
    
    # Load initial knowledge
    knowledge = load_initial_knowledge()
    
    # Add knowledge to the system
    documents = [k["text"] for k in knowledge]
    topics = [k["topics"] for k in knowledge]
    
    pipeline.add_knowledge(documents, topics)
    logger.info(f"Added {len(documents)} documents to knowledge base")
    
    return pipeline

def process_user_query(pipeline: DynamicRAGPipeline, query: str) -> None:
    """Process a user query and display the response."""
    try:
        result = pipeline.process_query(query)
        
        if result["response_type"] == "error":
            print(f"\nError: {result['error']}")
            return
            
        if result["response_type"] == "clarification":
            print("\nI need some clarification:")
            print(result["clarification_prompt"])
            print("\nPlease provide more details about what you'd like to know.")
            return
            
        if result["response_type"] == "answer":
            print("\nResponse:", result["response"])
            print("\nSources used:")
            for i, context in enumerate(result["context"], 1):
                print(f"{i}. {context[:200]}...")
                
    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        print("\nAn error occurred while processing your query. Please try again.")

def main():
    """Main function to run the RAG system."""
    print("\n🚀 Initializing RAG System...")
    pipeline = initialize_rag_system()
    print("\n✅ RAG System initialized and ready!")
    print("\nYou can now ask questions about various topics.")
    print("Type 'exit' or 'quit' to end the session.")
    
    while True:
        try:
            query = input("\nYour question: ").strip()
            
            if query.lower() in ['exit', 'quit']:
                print("\nGoodbye! 👋")
                break
                
            if not query:
                print("Please enter a question.")
                continue
                
            process_user_query(pipeline, query)
            
        except KeyboardInterrupt:
            print("\n\nGoodbye! 👋")
            break
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            print("\nAn unexpected error occurred. Please try again.")

if __name__ == "__main__":
    main() 