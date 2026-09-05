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

<<<<<<< HEAD
### 2. Configure your Groq API key
```bash
set GROQ_API_KEY=your_groq_api_key       # Windows
export GROQ_API_KEY=your_groq_api_key    # macOS/Linux
```

You can also enter the key in the Streamlit sidebar for the current browser session.

### 3. Add your documents
Put your `.pdf` or `.txt` files inside the `docs/` folder.

### 4. Run the web interface
```bash
python -m streamlit run app.py
```

Open the local URL shown by Streamlit, then use **Rebuild index** in the sidebar after adding documents.

=======
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

>>>>>>> 47f4cba8d8682d2485352b71f9ef7a2a7e256c1a
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
<<<<<<< HEAD
| 3 | `HuggingFaceEmbeddings` + `FAISS` | Converts chunks to vectors and stores them |
=======
| 3 | `OpenAIEmbeddings` + `FAISS` | Converts chunks to vectors and stores them |
>>>>>>> 47f4cba8d8682d2485352b71f9ef7a2a7e256c1a
| 4 | `ConversationalRetrievalChain` | Retrieves relevant chunks and generates answer |
| 5 | `ConversationBufferMemory` | Remembers previous messages |
