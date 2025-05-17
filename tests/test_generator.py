import unittest
from src.generator import ResponseGenerator

class TestResponseGenerator(unittest.TestCase):
    def setUp(self):
        """Set up the ResponseGenerator instance."""
        self.generator = ResponseGenerator(model_name="facebook/bart-large")

    def test_generate_clarification_prompt(self):
        """Test the clarification prompt generation."""
        query = "climate change"
        topics = [("agriculture", 0.9), ("sea levels", 0.8), ("renewable energy", 0.7)]
        questions = [
            "Do you want to focus on agriculture?",
            "Are you interested in sea levels?",
            "Should we discuss renewable energy?"
        ]
        prompt = self.generator.generate_clarification_prompt(query, topics, questions)
        self.assertIn("climate change", prompt)
        self.assertIn("agriculture", prompt)
        self.assertIn("1. Do you want to focus on agriculture?", prompt)

    def test_generate_response(self):
        """Test the response generation."""
        query = "What are the effects of climate change on agriculture?"
        context = [
            "Climate change affects crop yields due to temperature changes.",
            "Droughts and floods are becoming more frequent."
        ]
        response = self.generator.generate_response(query, context)
        self.assertIsInstance(response, str)
        self.assertGreater(len(response), 0)

    def test_generate_topic_focused_response(self):
        """Test the topic-focused response generation."""
        query = "What are the effects of climate change?"
        original_context = [
            "Climate change affects various aspects of the environment.",
            "It leads to rising sea levels and extreme weather."
        ]
        focused_context = [
            "Agriculture is heavily impacted by climate change.",
            "Crop yields are declining due to temperature changes."
        ]
        topic = "agriculture"
        response = self.generator.generate_topic_focused_response(query, original_context, focused_context, topic)
        self.assertIsInstance(response, str)
        self.assertIn("agriculture", response)

if __name__ == "__main__":
    unittest.main()