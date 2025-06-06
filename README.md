# Plagiarism Checker

This is a simple Python-based plagiarism checker that compares `.txt` files in a given folder using **TF-IDF vectorization** and **cosine similarity**.

---

## Features

- Loads and reads all `.txt` files in a folder
- Cleans and processes text data
- Calculates similarity between documents using **TF-IDF**
- Flags potential plagiarism if similarity exceeds 30%
- User-friendly terminal output

---

## 🛠How It Works

1. **Load Files**: Reads all `.txt` files in a specified directory.
2. **Preprocess Text**: Cleans the content by removing special characters and converting text to lowercase.
3. **Vectorization**: Converts text to TF-IDF vectors.
4. **Similarity Check**: Uses cosine similarity to find how similar the documents are.
5. **Report**: Displays a similarity score between each file pair, and flags pairs with >30% similarity.

---

##  Folder Structure

