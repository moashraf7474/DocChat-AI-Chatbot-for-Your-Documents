"""
============================================================
  DocChat - AI Chatbot on Custom Documents
  Pharos University | AI405 - Pattern Recognition
  Spring 2025/2026

  Tech Stack:
    - LangChain  : AI pipeline framework
    - Groq       : Free and fast LLM API (LLaMA 3)
    - FAISS      : Vector database for similarity search
    - HuggingFace: Free local embeddings (no API needed)
============================================================
"""

import os

# Document loaders - used to read PDF and TXT files
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    DirectoryLoader
)

# Text splitter - breaks long documents into smaller chunks
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Free embeddings model that runs locally on your machine
from langchain_community.embeddings import HuggingFaceEmbeddings

# FAISS - vector database that stores and searches embeddings
from langchain_community.vectorstores import FAISS

# The main chain that combines retriever + LLM + memory
from langchain_classic.chains import ConversationalRetrievalChain

# Memory - stores conversation history between turns
from langchain_classic.memory import ConversationBufferMemory

# Prompt template - defines how the LLM should behave
from langchain_core.prompts import PromptTemplate

# Groq - free and fast LLM provider (uses LLaMA 3 model)
from langchain_groq import ChatGroq


# ============================================================
#  YOUR GROQ API KEY
#  Get it for free from: https://console.groq.com
#  Sign up → API Keys → Create API Key → paste it below
# ============================================================
GROQ_API_KEY = "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"   # <-- Replace with your key


# ============================================================
#  STEP 1 - LOAD DOCUMENTS
# ============================================================
def load_documents(docs_path="docs/"):
    """
    Loads all PDF and TXT files from the docs/ folder.

    How it works:
        - DirectoryLoader scans the folder automatically
        - PyPDFLoader reads each page of a PDF file
        - TextLoader reads plain .txt files
        - Each file becomes a list of Document objects
        - Each Document has:
            .page_content -> the actual text of the page
            .metadata     -> file name, page number, etc.

    Example output:
        Document(page_content="Machine learning is...",
                 metadata={"source": "docs/ml.pdf", "page": 0})
    """
    print("[Step 1] Loading documents from:", docs_path)
    all_docs = []

    # Load all PDF files from the docs/ folder
    try:
        pdf_loader = DirectoryLoader(
            docs_path,
            glob="**/*.pdf",          # find all .pdf files
            loader_cls=PyPDFLoader,   # use PDF reader
            silent_errors=True        # skip files that fail to load
        )
        pdf_docs = pdf_loader.load()
        all_docs.extend(pdf_docs)
        print(f"  PDF pages loaded: {len(pdf_docs)}")
    except Exception as e:
        print(f"  PDF loader error: {e}")

    # Load all TXT files from the docs/ folder
    try:
        txt_loader = DirectoryLoader(
            docs_path,
            glob="**/*.txt",          # find all .txt files
            loader_cls=TextLoader,    # use text reader
            silent_errors=True
        )
        txt_docs = txt_loader.load()
        all_docs.extend(txt_docs)
        print(f"  TXT files loaded: {len(txt_docs)}")
    except Exception as e:
        print(f"  TXT loader error: {e}")

    print(f"  Total documents loaded: {len(all_docs)}\n")
    return all_docs


# ============================================================
#  STEP 2 - SPLIT DOCUMENTS INTO CHUNKS
# ============================================================
def split_documents(documents):
    """
    Splits long documents into smaller overlapping chunks.

    Why do we split?
        LLMs have a maximum input size (context window limit).
        We cannot feed an entire PDF at once.
        Smaller chunks also make searching more precise.

    Parameters explained:
        chunk_size    = 1000  -> each chunk is 1000 characters long
        chunk_overlap = 200   -> last 200 chars of chunk N are repeated
                                 at the start of chunk N+1
                                 (so no information is lost at boundaries)

    Example:
        Chunk 1: "Machine learning is a branch of AI that enables..."
        Chunk 2: "...that enables computers to learn from data..."
                  ^^^ this part is repeated (overlap)
    """
    print("[Step 2] Splitting documents into chunks...")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        # tries to split on paragraph breaks first, then lines, then words
        separators=["\n\n", "\n", " ", ""]
    )

    chunks = splitter.split_documents(documents)
    print(f"  Total chunks created: {len(chunks)}\n")
    return chunks


# ============================================================
#  STEP 3 - CREATE EMBEDDINGS AND VECTOR STORE
# ============================================================
def create_vector_store(chunks, index_path="faiss_index"):
    """
    Converts each text chunk into a numerical vector (embedding)
    and stores all vectors in a FAISS index on disk.

    What are embeddings?
        An embedding converts text into a list of numbers that
        represent the meaning of that text.
        Example:
            "What is AI?" -> [0.12, -0.84, 0.33, 0.91, ...]
        Similar sentences produce similar vectors (close numbers).

    What is FAISS?
        FAISS (Facebook AI Similarity Search) is a library that
        stores vectors and can quickly find the most similar ones
        to a given query vector.

    Why save to disk?
        So we do not re-embed all documents every time we run
        the program. On the second run, we just load the saved index.
    """
    print("[Step 3] Creating embeddings and saving vector store...")

    # HuggingFace embedding model - runs locally, completely free
    # Downloads the model on first run (~90 MB), then uses cached version
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}   # run on CPU (no GPU needed)
    )

    # Convert all chunks to vectors and build the FAISS index
    vector_store = FAISS.from_documents(chunks, embeddings)

    # Save the index to disk so we can reload it next time
    vector_store.save_local(index_path)
    print(f"  Vector store saved to '{index_path}/'\n")
    return vector_store


def load_vector_store(index_path="faiss_index"):
    """
    Loads a previously saved FAISS index from disk.
    Called on second and subsequent runs to avoid re-embedding.
    """
    # Must use the same embedding model that was used to create the index
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}
    )
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True  # needed to load saved FAISS index
    )


# ============================================================
#  STEP 4 - BUILD THE CONVERSATIONAL RAG CHAIN
# ============================================================
def build_chain(vector_store):
    """
    Assembles the complete RAG (Retrieval-Augmented Generation) pipeline.

    RAG works like this for every user message:
        1. The user's question is converted to a vector
        2. FAISS finds the 4 most relevant chunks from the documents
        3. The LLM receives: question + relevant chunks + chat history
        4. The LLM generates an answer grounded in the document content

    Components used:
        ChatGroq               -> the LLM that generates the answer
        ConversationBufferMemory -> remembers previous messages
        Retriever              -> fetches relevant chunks from FAISS
        PromptTemplate         -> tells the LLM how to behave
        ConversationalRetrievalChain -> ties everything together
    """
    print("[Step 4] Building the conversational RAG chain...")

    # ── LLM: Groq with LLaMA 3 ──────────────────────────────
    # temperature=0.2 means more factual and less creative answers
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",   # LLaMA 3 model with 8192 token context
        temperature=0.2,
        groq_api_key=GROQ_API_KEY
    )

    # ── Memory: stores the full conversation ────────────────
    # Allows the bot to understand follow-up questions like
    # "explain more" or "what did you mean by that?"
    memory = ConversationBufferMemory(
        memory_key="chat_history",   # name used in the prompt template
        return_messages=True,        # return as Message objects
        output_key="answer"
    )

    # ── Retriever: finds relevant chunks from FAISS ─────────
    # For every question, returns the top 4 most similar chunks
    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}   # return top 4 chunks
    )

    # ── Prompt Template: instructions for the LLM ───────────
    # {context}      -> the 4 retrieved chunks (filled automatically)
    # {chat_history} -> previous messages (filled by memory)
    # {question}     -> the current user question (filled automatically)
    custom_prompt = PromptTemplate.from_template("""
You are DocChat, an intelligent assistant that answers questions
based strictly on the provided document context.

Rules:
- Answer ONLY from the context provided below.
- If the answer is not in the context, say: "I could not find that in the documents."
- Be concise, clear, and well-structured.
- You may answer in Arabic or English based on the user's language.

Context from documents:
{context}

Conversation history:
{chat_history}

User question: {question}

Answer:""")

    # ── Chain: the complete pipeline ────────────────────────
    # ConversationalRetrievalChain automatically:
    #   1. Runs the retriever to get relevant chunks
    #   2. Formats the prompt with context + history + question
    #   3. Calls the LLM to generate the answer
    #   4. Saves the exchange to memory
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        return_source_documents=True,  # also return which chunks were used
        combine_docs_chain_kwargs={"prompt": custom_prompt}
    )

    print("  Chain is ready!\n")
    return chain


# ============================================================
#  STEP 5 - CHAT FUNCTION
# ============================================================
def chat(chain, question):
    """
    Sends the user's question through the chain and returns
    the answer along with the source document names.

    Args:
        chain    -> the built ConversationalRetrievalChain
        question -> the user's question as a string

    Returns:
        dict with:
            "answer"  -> the LLM's response
            "sources" -> list of document file names used
    """
    # Run the full RAG pipeline
    result = chain.invoke({"question": question})

    # Extract unique source file names from the retrieved chunks
    sources = list({
        doc.metadata.get("source", "Unknown")
        for doc in result.get("source_documents", [])
    })

    return {
        "answer": result["answer"],
        "sources": sources
    }


# ============================================================
#  MAIN - CLI Chat Interface
# ============================================================
def main():
    print("""
============================================
   DocChat - AI Chatbot on Your Documents
   Type 'exit' to quit
============================================
    """)

    INDEX_PATH = "faiss_index"
    DOCS_PATH  = "docs/"

    # If a saved index exists, load it (skip re-embedding)
    # Otherwise, process the documents from scratch
    if os.path.exists(INDEX_PATH):
        print("Found existing index - loading from disk...")
        vector_store = load_vector_store(INDEX_PATH)
        print("  Index loaded successfully!\n")
    else:
        # First run: load, split, embed, and save
        documents    = load_documents(DOCS_PATH)
        chunks       = split_documents(documents)
        vector_store = create_vector_store(chunks, INDEX_PATH)

    # Build the full RAG chain
    chain = build_chain(vector_store)

    # Start the conversation loop
    while True:
        user_input = input("You: ").strip()

        # Skip empty inputs
        if not user_input:
            continue

        # Exit on goodbye words
        if user_input.lower() in ("exit", "quit", "bye"):
            print("DocChat: Goodbye!")
            break

        # Get answer from the chain
        response = chat(chain, user_input)

        # Print answer and sources
        print(f"\nDocChat: {response['answer']}")
        if response["sources"]:
            print(f"Sources: {', '.join(response['sources'])}")
        print()


if __name__ == "__main__":
    main()