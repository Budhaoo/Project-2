import os
import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def load_text_files(folder_path):
    texts = {}
    for file in os.listdir(folder_path):
        if file.endswith(".txt"):
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                texts[file] = f.read()
    return texts

def clean_text(text):
    text = re.sub(r'\W+', ' ', text)  # Remove special characters
    text = text.lower().strip()  # Convert to lowercase
    return text

def compute_similarity(texts):
    vectorizer = TfidfVectorizer()
    text_list = [clean_text(text) for text in texts.values()]
    tfidf_matrix = vectorizer.fit_transform(text_list)
    similarity_matrix = cosine_similarity(tfidf_matrix)
    return similarity_matrix

def display_results(files, similarity_matrix):
    file_names = list(files.keys())
    print("\n=== Plagiarism Report ===")
    
    for i in range(len(file_names)):
        for j in range(i + 1, len(file_names)):
            similarity_score = similarity_matrix[i, j] * 100
            if similarity_score > 30:  # Flagging plagiarism if above 30%
                print(f"⚠ {file_names[i]} & {file_names[j]}: {similarity_score:.2f}% similarity")

if __name__ == "__main__":
    folder_path = input("Enter the folder path containing text files: ")
    texts = load_text_files(folder_path)
    if len(texts) < 2:
        print("Need at least 2 text files for comparison.")
    else:
        similarity_matrix = compute_similarity(texts)
        display_results(texts, similarity_matrix)
