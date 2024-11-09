import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

# Load text data for embedding
data = pd.read_csv('../data/data_text_for_embedding.csv')

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')  # A lightweight model for embeddings

# Generate embeddings for each text entry
embeddings = model.encode(data['text'].tolist(), show_progress_bar=True)

# Initialize FAISS index for CPU
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)  # L2 distance is common for FAISS

# Add embeddings to the index
index.add(embeddings)

# Save the FAISS index for future retrieval
faiss.write_index(index, '../data/energy_usage_index.faiss')

print("Embeddings created and stored in FAISS index.")
