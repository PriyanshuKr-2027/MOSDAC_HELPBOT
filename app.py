from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
import requests

app = Flask(__name__)
import os
HF_API_KEY = os.getenv("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")

# === Load CSV and Embed FAQ Corpus ===
df = pd.read_csv("Model-train.csv")
df.columns = ['Question', 'Answer']
corpus = (df['Question'] + " " + df['Answer']).tolist()

embedder = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')
embeddings = embedder.encode(corpus)
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(np.array(embeddings))

# === Hugging Face Inference API Setup ===
API_URL = "https://api-inference.huggingface.co/models/HuggingFaceH4/zephyr-7b-beta"
HF_API_KEY = "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"  # Replace with your actual token
headers = {"Authorization": f"Bearer {HF_API_KEY}"}

# === Chat Memory ===
chat_memory = []  # holds last few question-answer pairs

# === Helper Functions ===
def get_faq_context(query, k=3):
    query_vec = embedder.encode([query])
    distances, indices = faiss_index.search(np.array(query_vec), k)
    return [(df.iloc[idx]['Question'], df.iloc[idx]['Answer']) for idx in indices[0]]

def query_huggingface_api(prompt):
    payload = {"inputs": prompt}
    response = requests.post(API_URL, headers=headers, json=payload)
    try:
        return response.json()[0]['generated_text']
    except Exception as e:
        print("HF API error:", e, response.text)
        return "I'm sorry, I couldn't generate a response right now."

def generate_answer_with_zephyr(user_query, context_answers):
    context_text = "\n".join([f"- {a}" for _, a in context_answers])
    prompt = (
        "You are an expert assistant answering questions strictly about ISRO satellites and the MOSDAC portal.\n\n"
        f"Context:\n{context_text}\n\n"
        f"Question: {user_query}\n"
        "Answer:"
    )
    raw_response = query_huggingface_api(prompt)
    # Extract only the first answer after 'Answer:' and before the next 'Question:'
    if "Answer:" in raw_response:
        answer = raw_response.split("Answer:", 1)[1]
        if "Question:" in answer:
            answer = answer.split("Question:", 1)[0]
        answer = answer.strip()
    else:
        answer = raw_response.strip()
    return answer

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.json.get('query') or request.form.get('query')
    if not user_query:
        return jsonify({'error': 'No query provided'}), 400

    context = get_faq_context(user_query)
    answer = generate_answer_with_zephyr(user_query, context)
    related_faqs = [q for q, _ in context]

    # Add to chat memory for follow-up tracking
    chat_memory.append({"question": user_query, "answer": answer})

    return jsonify({'answer': answer, 'related_faqs': related_faqs})

if __name__ == "__main__":
    app.run(debug=True)
