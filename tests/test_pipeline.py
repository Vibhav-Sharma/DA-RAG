import unittest
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from src.pipeline import DynamicRAGPipeline

class TestDynamicRAGPipeline(unittest.TestCase):
    def setUp(self):
        """Set up the pipeline instance."""
        self.pipeline = DynamicRAGPipeline()

    def test_pipeline_end_to_end(self):
        """Test the full pipeline from user-provided query to response."""
        # Prompt the user for input
        query = input("Enter your query: ")

        # Run the pipeline
        response = self.pipeline.process_query(query)

        # Validate the response
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)
        print("\nGenerated Response:")
        print(response)

if __name__ == "__main__":
    unittest.main()