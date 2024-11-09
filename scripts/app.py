import streamlit as st
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# Load the FAISS index and dataset
index = faiss.read_index('../data/energy_usage_index.faiss')
data = pd.read_csv('../data/data_text_for_embedding.csv')

# Initialize the embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Initialize the QA model
qa_model = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# Function to retrieve relevant context
def retrieve_similar_texts(query, k=3):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(query_embedding, k)
    return " ".join(data.iloc[indices[0]]['text'])

# Function to generate answers
def generate_answer(query):
    context = retrieve_similar_texts(query, k=2)
    context = context[:500]  # Limit context for QA clarity
    response = qa_model(question=query, context=context)
    return response['answer']

# Streamlit UI
def main():
    st.title("Energy Usage Question Answering")
    
    # Input text box for user query
    question = st.text_input("Enter your question about energy usage:")

    if question:
        # Display the answer
        answer = generate_answer(question)
        st.write("Answer:", answer)

if __name__ == "__main__":
    main()
