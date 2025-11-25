# 🧠 RAG App with MongoDB Atlas Vector Search + Local LLM (Ollama)

This project demonstrates a complete **Retrieval-Augmented Generation (RAG)** pipeline using:

- **MongoDB Atlas Vector Search**
- **LangChain**
- **Local LLM via Ollama (e.g., Llama 3)**
- **HuggingFace Embeddings**
- **Gradio UI**
- **Python backend**

It includes:
1. An **embedding pipeline** that loads `.txt` files, embeds them using HuggingFace, and stores them in MongoDB Atlas.
2. A **question-answering application** that retrieves relevant documents from MongoDB Atlas and generates answers using a local LLM.

---

## 📁 Project Structure
├── embed.py # Script to embed documents into MongoDB
├── app.py # Main RAG application with Gradio UI
├── sample_files/ # Folder containing .txt files to embed
├── requirements.txt
└── README.md


---

## 🚀 Features

### ✔ Vector Search with MongoDB Atlas  
Uses **Atlas Vector Index** to perform similarity search on embeddings.

### ✔ HuggingFace Embeddings  
Uses: `sentence-transformers/all-MiniLM-L6-v2`

### ✔ Local LLM via Ollama  
Works with any Ollama model (llama3, mistral, phi3, etc.).

### ✔ RAG Pipeline  
Uses LangChain’s `RetrievalQA` for augmenting LLM responses with MongoDB results.

### ✔ Clean Gradio UI  
Simple interface for asking questions and visualizing:
- What MongoDB retrieved
- What the LLM answered

---

## 🛠 Setup Instructions

### 1. Install Python dependencies

```bash
pip install -r requirements.txt

Run the Gradio RAG App
http://localhost:7860