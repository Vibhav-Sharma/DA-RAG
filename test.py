# import torch
# import transformers
# import faiss
# import sentence_transformers
# import datasets

# print("PyTorch Version:", torch.__version__)
# print("Transformers Version:", transformers.__version__)
# print("FAISS Version:", faiss.__version__)
# print("Sentence-Transformers Version:", sentence_transformers.__version__)
# print("Datasets Version:", datasets.__version__)

# print("\n✅ All libraries are installed correctly!")






# checking BART

# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# # Load BART model and tokenizer
# bart_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large")
# tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")

# # Example text for testing BART
# test_text = "What is the purpose of Retrieval-Augmented Generation?"

# # Encode input for BART
# inputs = tokenizer(test_text, return_tensors="pt", max_length=1024, truncation=True)

# # Generate response using BART
# summary_ids = bart_model.generate(**inputs)
# generated_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# print("Generated Output:", generated_text)

# # Check if output is non-empty
# assert len(generated_text.strip()) > 0, "❌ BART output is empty!"
# print("✅ BART is working correctly!")








# Checking for SBERT

# from sentence_transformers import SentenceTransformer

# # ✅ Load SBERT Model (Make sure this runs before calling `.encode()`)
# sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# # Example Query for Testing
# query = "What is Retrieval-Augmented Generation?"

# # ✅ Generate Embeddings (Only after model is loaded)
# query_embedding = sbert_model.encode(query)

# print("Query Embedding Shape:", query_embedding.shape)
# print("✅ SBERT Model Loaded and Working!")







# pipelining check

from sentence_transformers import SentenceTransformer
from transformers import BartForConditionalGeneration, BartTokenizer
import faiss
import numpy as np

def check_pipeline():
    print("✅ Checking SBERT, FAISS, and BART pipeline...\n")

    try:
        # ✅ Step 1: Load SBERT Model
        print("🔹 Loading SBERT Model...")
        sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("✅ SBERT Loaded Successfully.\n")

        # ✅ Step 2: Create Dummy Dataset for FAISS
        sentences = [
            "Retrieval-Augmented Generation improves NLP tasks.",
            "Machine learning models benefit from large datasets.",
            "BART is used for text summarization and generation."
        ]

        print("🔹 Encoding sentences with SBERT...")
        sentence_embeddings = sbert_model.encode(sentences)
        print(f"✅ Sentence Embeddings Shape: {sentence_embeddings.shape}\n")

        # ✅ Step 3: Load FAISS Index
        print("🔹 Creating FAISS Index...")
        d = sentence_embeddings.shape[1]
        faiss_index = faiss.IndexFlatL2(d)
        faiss_index.add(np.array(sentence_embeddings))
        print(f"✅ FAISS Index Created with {faiss_index.ntotal} entries.\n")

        # ✅ Step 4: Query Encoding & Retrieval
        query = "How does retrieval help text generation?"
        print(f"🔹 Encoding Query: {query}")
        query_embedding = sbert_model.encode(query).reshape(1, -1)
        
        # Retrieve Top-1 closest sentence
        D, I = faiss_index.search(query_embedding, 1)
        retrieved_text = sentences[I[0][0]]

        print(f"✅ Retrieved Text: {retrieved_text}\n")

        # ✅ Step 5: Load BART Model
        print("🔹 Loading BART Model...")
        bart_model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
        print("✅ BART Loaded Successfully.\n")

        # ✅ Step 6: Generate Output using BART
        print(f"🔹 Passing Retrieved Text to BART: {retrieved_text}")
        inputs = tokenizer(retrieved_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = bart_model.generate(**inputs)

        generated_text = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        print(f"✅ Generated Text: {generated_text}\n")

        print("🎉 Pipeline is working correctly! 🚀")
    
    except Exception as e:
        print(f"❌ Error: {e}")

# Run the pipeline check
check_pipeline()
