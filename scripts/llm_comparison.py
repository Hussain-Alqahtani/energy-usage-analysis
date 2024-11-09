import faiss
import pandas as pd
import torch
import time
from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
from sentence_transformers import SentenceTransformer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# Load the FAISS index and dataset
index = faiss.read_index('../data/energy_usage_index.faiss')
data = pd.read_csv('../data/data_text_for_embedding.csv')

# Initialize the sentence transformer for embedding
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Define models for comparison
models_info = {
    "distilbert-base-cased-distilled-squad": {
        "pipeline": pipeline("question-answering", model="distilbert-base-cased-distilled-squad"),
    },
    "bert-large-uncased-whole-word-masking-finetuned-squad": {
        "pipeline": pipeline("question-answering", model="bert-large-uncased-whole-word-masking-finetuned-squad"),
    },
    "roberta-large-mnli": {
        "pipeline": pipeline("question-answering", model="deepset/roberta-large-squad2"),
    },
    "electra-large-discriminator-finetuned-squad": {
        "pipeline": pipeline("question-answering", model="ahotrod/electra_large_discriminator_squad2_512"),
    },
    "gpt-neo-125M": {
        "pipeline": pipeline("text-generation", model="EleutherAI/gpt-neo-125M"),
    },
    "t5-large": {
        "pipeline": pipeline("text2text-generation", model="t5-large"),
    }
}

# Function to retrieve relevant context for a question
def retrieve_similar_texts(query, k=3):
    query_embedding = embedding_model.encode([query])
    _, indices = index.search(query_embedding, k)
    return " ".join(data.iloc[indices[0]]['text'])

# Function to evaluate BLEU and ROUGE scores
def evaluate_answer(reference_answer, generated_answer):
    bleu_score = sentence_bleu([reference_answer.split()], generated_answer.split())
    rouge = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_score = rouge.score(reference_answer, generated_answer)['rougeL'].fmeasure
    return bleu_score, rouge_score

# Test questions and their reference answers
test_data = [
    {"question": "What is the energy used for the fan in July?", "reference": "12.665533969907408 units"},
    # Add more question-answer pairs here if needed
]

# Run the evaluation
results = []
for model_name, model_info in models_info.items():
    print(f"Evaluating model: {model_name}")
    qa_pipeline = model_info["pipeline"]

    for item in test_data:
        question = item["question"]
        reference_answer = item["reference"]

        # Retrieve context and generate answer
        context = retrieve_similar_texts(question)
        start_time = time.time()
        
        # Special handling for text generation models (GPT-Neo, T5)
        if model_name in ["gpt-neo-125M", "t5-large"]:
            response = qa_pipeline(f"Answer this question: {question} Context: {context}", max_new_tokens=50)
            generated_answer = response[0]['generated_text']
        else:
            response = qa_pipeline(question=question, context=context)
            generated_answer = response['answer']

        end_time = time.time()
        
        # Calculate evaluation metrics
        bleu_score, rouge_score = evaluate_answer(reference_answer, generated_answer)
        response_time = end_time - start_time

        results.append({
            "Model": model_name,
            "Question": question,
            "Reference Answer": reference_answer,
            "Generated Answer": generated_answer,
            "BLEU Score": bleu_score,
            "ROUGE-L Score": rouge_score,
            "Response Time (s)": response_time
        })

# Display results
results_df = pd.DataFrame(results)
print(results_df)
