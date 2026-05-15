DocChat

DocChat is a Retrieval-Augmented Generation (RAG) chatbot designed to answer questions from custom PDF and TXT documents using modern AI technologies. The system combines semantic search with large language models to provide accurate, context-aware, and conversational responses grounded in the uploaded documents.

The project is built with LangChain, FAISS, HuggingFace Embeddings, and Groq’s LLaMA 3 models, offering a lightweight and efficient solution for document-based AI assistants.

Features
Support for PDF and TXT document ingestion
Automatic document chunking and preprocessing
Semantic similarity search using FAISS
Local embeddings with HuggingFace models
Conversational memory for multi-turn interactions
Context-aware responses powered by LLaMA 3
Source tracking for retrieved documents
Arabic and English language support
Persistent vector database storage
Technology Stack
Python
LangChain
FAISS
HuggingFace Embeddings
Groq API
LLaMA 3
Project Structure
DocChat/
│
├── docs/                # Input documents
├── faiss_index/         # Saved FAISS vector database
├── app.py               # Main application
└── requirements.txt
Installation

Clone the repository:

git clone https://github.com/yourusername/docchat.git
cd docchat

Install dependencies:

pip install -r requirements.txt

Configure your Groq API key inside app.py:

GROQ_API_KEY = "your_api_key"

Get a free API key from:

Groq Console

Usage

Place your PDF or TXT files inside the docs/ directory.

Run the application:

python app.py

Example interaction:

You: What is machine learning?

DocChat: Machine learning is a branch of artificial intelligence that enables systems to learn from data.
System Workflow
Load documents from the docs/ directory
Split documents into overlapping text chunks
Generate embeddings for each chunk
Store embeddings inside a FAISS vector database
Retrieve the most relevant chunks for user queries
Generate grounded responses using LLaMA 3
Future Enhancements
Web interface using Streamlit or Flask
Support for DOCX and CSV files
OCR integration for scanned documents
Voice interaction capabilities
Cloud deployment support
Environment variable management for API keys
