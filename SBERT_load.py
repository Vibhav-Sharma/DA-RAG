from sentence_transformers import SentenceTransformer

# Load Facebook AI SBERT model
sbert_model = SentenceTransformer("facebook-dpr-ctx_encoder-single-nq-base")

# Example sentence embedding
sentence = "This is a test sentence."
embedding = sbert_model.encode(sentence)

print("Sentence Embedding Shape:", embedding.shape)
