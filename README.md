# DocChat – AI Chatbot on Custom Documents
**Pharos University | AI405 Pattern Recognition | Spring 2025/2026**

---

## What is this project?

DocChat is an AI-powered chatbot that answers questions based on **your own documents** (PDFs and TXT files). It uses **LangChain** and a technique called **RAG (Retrieval-Augmented Generation)** to find the most relevant information from your documents and generate accurate answers.

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Add your OpenAI API key
```bash
cp .env.example .env
# Edit .env and paste your API key
```

### 3. Add your documents
Put your `.pdf` or `.txt` files inside the `docs/` folder.

### 4. Run the chatbot
```bash
python app.py
```

---

## Project Structure

```
doc_chatbot/
├── app.py               ← Main application (all pipeline steps)
├── requirements.txt     ← Python dependencies
├── .env.example         ← API key template
├── docs/                ← Put your documents here
│   └── sample_document.txt
└── faiss_index/         ← Auto-created after first run
```

---

## Pipeline Steps

| Step | Component | What it does |
|------|-----------|--------------|
| 1 | `DirectoryLoader` / `PyPDFLoader` | Loads documents from disk |
| 2 | `RecursiveCharacterTextSplitter` | Splits text into 1000-char chunks |
| 3 | `OpenAIEmbeddings` + `FAISS` | Converts chunks to vectors and stores them |
| 4 | `ConversationalRetrievalChain` | Retrieves relevant chunks and generates answer |
| 5 | `ConversationBufferMemory` | Remembers previous messages |
