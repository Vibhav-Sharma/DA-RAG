from transformers import BartForConditionalGeneration, BartTokenizer

# Load BART model and tokenizer
bart_model_name = "facebook/bart-large"
bart_model = BartForConditionalGeneration.from_pretrained(bart_model_name)
tokenizer = BartTokenizer.from_pretrained(bart_model_name)

# Example: Summarization
text = "Large language models are powerful for text generation and retrieval-augmented generation."
inputs = tokenizer(text, return_tensors="pt", max_length=1024, truncation=True)
summary_ids = bart_model.generate(**inputs)
summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Generated Output:", summary)
