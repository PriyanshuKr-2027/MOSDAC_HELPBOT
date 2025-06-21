from flask import Flask, render_template, request, jsonify
import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline

app = Flask(__name__)

# Load CSV and embed FAQ corpus
df = pd.read_csv("Model-train.csv")
df.columns = ['Question', 'Answer']
corpus = (df['Question'] + " " + df['Answer']).tolist()

embedder = SentenceTransformer('sentence-transformers/msmarco-distilbert-base-v4')
embeddings = embedder.encode(corpus)
dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(np.array(embeddings))

# Load Hugging Face text generation pipeline
generator = pipeline('text2text-generation', model='google/flan-t5-base')

def get_faq_context(query, k=3):
    query_vec = embedder.encode([query])
    distances, indices = index.search(np.array(query_vec), k)
    return [(df.iloc[idx]['Question'], df.iloc[idx]['Answer']) for idx in indices[0]]

def generate_answer_with_hf(query, context_answers):
    context_text = "\n".join([f"- {a}" for _, a in context_answers])
    prompt = f"Answer the question based on the context:\nContext:\n{context_text}\nQuestion: {query}\nAnswer:"
    outputs = generator(prompt, max_length=150, do_sample=False)
    return outputs[0]['generated_text']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    user_query = request.form['query']
    context = get_faq_context(user_query)
    answer = generate_answer_with_hf(user_query, context)
    related_faqs = [q for q, _ in context]
    return jsonify({'answer': answer, 'related_faqs': related_faqs})

if __name__ == "__main__":
    app.run(debug=True)