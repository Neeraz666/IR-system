from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import math
import os
import logging

app = Flask(__name__)
CORS(app)

# Load text files from the specified folder
def load_text_files(folder_path):
    data = {}
    doc_id_to_filename = {}
    doc_id = 0
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                data[doc_id] = file.read()
                doc_id_to_filename[doc_id] = filename
                logging.info(f"Loaded file: {filename} with doc_id: {doc_id}")
            doc_id += 1
    return data, doc_id_to_filename

# Tokenize and preprocess text
def tokenize(text):
    return text.lower().split()

# Calculate term frequency (TF)
def term_frequency(term, document):
    return document.count(term) / len(document)

# Extracting description
def extract_description(document):
    lines = document.splitlines()
    if len(lines) > 2: 
        description = " ".join(lines[3:6])
        return description.strip()
    return document.strip()

# Extracting link
def extract_url(document):
    lines = document.splitlines()
    https_lines = [line for line in lines if line.startswith("https")]
    return https_lines

# Calculate inverse document frequency (IDF)
def inverse_document_frequency(term, all_documents):
    num_docs_containing_term = sum(1 for doc in all_documents if term in doc)
    return math.log(len(all_documents) / (1 + num_docs_containing_term))

# Compute TF-IDF vector for a document
def compute_tfidf(document, all_documents, vocab):
    tfidf_vector = []
    for term in vocab:
        tf = term_frequency(term, document)
        idf = inverse_document_frequency(term, all_documents)
        tfidf_vector.append(tf * idf)
    return np.array(tfidf_vector)

# Compute cosine similarity
def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)

# Rank documents by cosine similarity
def rank_documents_by_similarity(cosine_similarities, doc_id_to_filename, queries, docs):
    ranked_results = []
    for i, similarities in enumerate(cosine_similarities):
        doc_similarities = []
        for j in range(len(similarities)):
            filename = os.path.splitext(doc_id_to_filename[j])[0]  # Remove .txt extension
            description = extract_description(docs[j])  # Extract the description from the document
            url = extract_url(docs[j])  # Extract the first URL if present
            doc_similarities.append({
                "title": filename,
                "similarity": similarities[j],
                "description": description,
                "url": url
            })
        ranked_docs = sorted(doc_similarities, key=lambda x: x['similarity'], reverse=True)
        ranked_results.append({"query": queries[i], "results": ranked_docs})
    return ranked_results


# Load documents
path = 'projectdataset'  # Folder path for documents
docs, doc_id_to_filename = load_text_files(path)

# Tokenize documents
tokenized_docs = [tokenize(doc) for doc in docs.values()]

# Build vocabulary
vocab = set([word for doc in tokenized_docs for word in doc])
vocab = sorted(vocab)

# Compute TF-IDF vectors for all documents
doc_tfidf_vectors = [compute_tfidf(doc, tokenized_docs, vocab) for doc in tokenized_docs]

# API endpoint to search queries
@app.route('/search', methods=['POST'])
def search():
    data = request.json
    query = data.get('query')
    
    if not query:
        return jsonify({"error": "Query is required"}), 400

    # Tokenize and compute TF-IDF for the query
    tokenized_query = tokenize(query)
    query_tfidf_vector = compute_tfidf(tokenized_query, tokenized_docs, vocab)

    # Compute cosine similarities
    cosine_similarities = [cosine_similarity(query_tfidf_vector, doc_tfidf_vector) for doc_tfidf_vector in doc_tfidf_vectors]

    # Rank the documents
    ranked_results = rank_documents_by_similarity([cosine_similarities], doc_id_to_filename, [query], list(docs.values()))

    return jsonify(ranked_results[0])

if __name__ == '__main__':
    app.run(debug=True)
