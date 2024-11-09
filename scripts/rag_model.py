import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load the FAISS index and data
index = faiss.read_index('../data/energy_usage_index.faiss')
data = pd.read_csv('../data/data_text_for_embedding.csv')

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the QA model
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def retrieve_similar_texts(query, k=3):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(query_embedding, k)
    return data.iloc[indices[0]]['text'].tolist()

def generate_answer(query):
    retrieved_texts = retrieve_similar_texts(query, k=2)
    context = " ".join(retrieved_texts)[:500]  # Limit context for QA clarity
    
    # Generate a concise answer using a QA pipeline
    response = qa_model(question=query, context=context)
    
    # Extract and return only the answer text
    return response['answer']

if __name__ == "__main__":
    question = input("Enter your question about energy usage: ")
    answer = generate_answer(question)
    print("Answer:", answer)
