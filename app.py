<<<<<<< HEAD
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

try:
    import streamlit as st
    from streamlit.runtime.scriptrunner import get_script_run_ctx
except ImportError:
    st = None
    get_script_run_ctx = None

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
from groq import Groq


# ============================================================
#  YOUR GROQ API KEY
#  Get it for free from: https://console.groq.com
#  Sign up → API Keys → Create API Key → paste it below
# ============================================================
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")


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
def get_groq_models(api_key):
    """Return chat-capable models available to the supplied Groq key."""
    models = Groq(api_key=api_key).models.list().data
    return sorted(
        model.id for model in models
        if getattr(model, "active", True)
        and "embedding" not in model.id.lower()
        and "whisper" not in model.id.lower()
        and "guard" not in model.id.lower()
    )


def build_chain(vector_store, groq_api_key=None, groq_model=None):
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
        model=groq_model or GROQ_MODEL,
        temperature=0.2,
        groq_api_key=groq_api_key or GROQ_API_KEY
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


def streamlit_app():
    st.set_page_config(
        page_title="DocChat",
        page_icon="✦",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Manrope:wght@400;600;700;800&display=swap');
    :root { --ink: #15221f; --paper: #f5f0e8; --mint: #b9e5d3; --coral: #ed795f; --line: #c9c3b8; }
    .stApp { background: var(--paper); color: var(--ink); }
    [data-testid="stSidebar"] { background: #162b27; border-right: 1px solid #29433c; }
    [data-testid="stSidebar"] * { color: #eff5ed; }
    h1, h2, h3, p, label, .stMarkdown { font-family: 'Manrope', sans-serif; }
    h1 { font-size: clamp(2.6rem, 6vw, 5.8rem); letter-spacing: -0.06em; line-height: .92; margin: 0; }
    .eyebrow { font: 500 0.72rem 'DM Mono', monospace; letter-spacing: .12em; text-transform: uppercase; color: var(--coral); }
    .hero { padding: 2.2rem 0 2.5rem; border-bottom: 1px solid var(--line); margin-bottom: 1.5rem; }
    .hero-copy { max-width: 680px; font-size: 1.05rem; margin-top: 1.1rem; color: #56625d; }
    .stat { border-top: 2px solid var(--ink); padding-top: .7rem; }
    .stat-number { font: 500 1.8rem 'DM Mono', monospace; }
    .stat-label { color: #68736e; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; }
    .source-tag { display: inline-block; border: 1px solid var(--line); padding: .35rem .55rem; margin: .2rem .3rem 0 0; font: .72rem 'DM Mono', monospace; }
    [data-testid="stChatMessage"] { border-bottom: 1px solid var(--line); border-radius: 0; padding: 1.25rem 0; }
    [data-testid="stChatInput"] { padding-bottom: 1rem; }
    </style>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    with st.sidebar:
        st.markdown("<div class='eyebrow'>Knowledge workspace</div>", unsafe_allow_html=True)
        st.markdown("## DocChat")
        st.caption("Ask questions. Stay grounded in your documents.")
        st.divider()
        configured_key = st.text_input(
            "Groq API key",
            value=os.getenv("GROQ_API_KEY", ""),
            type="password",
            help="Stored only for this browser session. A sidebar value overrides the environment value.",
        ).strip()
        model_default = os.getenv("GROQ_MODEL", GROQ_MODEL)
        available_models = st.session_state.get("available_groq_models", [])
        if configured_key and st.button("Load available models", use_container_width=True):
            try:
                with st.spinner("Checking models enabled for this key..."):
                    available_models = get_groq_models(configured_key)
                st.session_state.available_groq_models = available_models
                if not available_models:
                    st.warning("No active chat models were returned for this key.")
            except Exception as error:
                st.error(f"Could not load Groq models: {error}")
        if available_models:
            selected_model = st.selectbox("Groq model", available_models, index=(available_models.index(model_default) if model_default in available_models else 0))
        else:
            selected_model = st.text_input("Groq model", value=model_default, help="Click 'Load available models' to choose from models enabled for your key.").strip()
        uploaded_files = st.file_uploader(
            "Add documents",
            type=["pdf", "txt"],
            accept_multiple_files=True,
            help="Upload files, then rebuild the document index.",
        )
        if uploaded_files and st.button("Rebuild index", use_container_width=True, type="primary"):
            os.makedirs("docs", exist_ok=True)
            for uploaded_file in uploaded_files:
                with open(os.path.join("docs", uploaded_file.name), "wb") as output_file:
                    output_file.write(uploaded_file.getbuffer())
            with st.spinner("Reading and indexing documents..."):
                documents = load_documents("docs/")
                chunks = split_documents(documents)
                create_vector_store(chunks, "faiss_index")
            st.cache_resource.clear()
            st.session_state.pop("chain", None)
            st.session_state.messages = []
            st.success("Index rebuilt.")
            st.rerun()
        if st.button("Clear conversation", use_container_width=True):
            st.session_state.messages = []
            st.session_state.pop("chain", None)
            st.rerun()
        st.divider()
        st.caption("Powered by Groq · Llama 3 · FAISS")

    st.markdown("<div class='hero'><div class='eyebrow'>Document intelligence / 01</div><h1>Your documents,<br>made conversational.</h1><p class='hero-copy'>A focused question-and-answer space for the knowledge already in your files. Every response is retrieved from your indexed documents.</p></div>", unsafe_allow_html=True)

    index_path = "faiss_index"
    docs_path = "docs/"
    document_names = sorted(
        file_name for file_name in os.listdir(docs_path) if file_name.lower().endswith((".pdf", ".txt"))
    ) if os.path.isdir(docs_path) else []
    stat_cols = st.columns(3)
    with stat_cols[0]:
        st.markdown(f"<div class='stat'><div class='stat-number'>{len(document_names):02d}</div><div class='stat-label'>Documents indexed</div></div>", unsafe_allow_html=True)
    with stat_cols[1]:
        st.markdown(f"<div class='stat'><div class='stat-number'>{len(st.session_state.messages) // 2:02d}</div><div class='stat-label'>Questions asked</div></div>", unsafe_allow_html=True)
    with stat_cols[2]:
        st.markdown("<div class='stat'><div class='stat-number'>RAG</div><div class='stat-label'>Answer mode</div></div>", unsafe_allow_html=True)

    if not configured_key:
        st.info("Add your Groq API key in the sidebar to start chatting.")
        return
    if not os.path.exists(index_path):
        st.warning("No document index found. Add a PDF or TXT file in the sidebar, then rebuild the index.")
        return

    try:
        if (
            "chain" not in st.session_state
            or st.session_state.get("chain_api_key") != configured_key
            or st.session_state.get("chain_model") != selected_model
        ):
            with st.spinner("Warming up the document index..."):
                st.session_state.chain = build_chain(load_vector_store(index_path), configured_key, selected_model)
                st.session_state.chain_api_key = configured_key
                st.session_state.chain_model = selected_model
    except Exception as error:
        st.error(f"Could not load DocChat: {error}")
        return

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Sources used"):
                    for source in message["sources"]:
                        st.markdown(f"<span class='source-tag'>{os.path.basename(source)}</span>", unsafe_allow_html=True)

    question = st.chat_input("Ask something about your documents...")
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching your documents..."):
                try:
                    response = chat(st.session_state.chain, question)
                except Exception as error:
                    error_text = str(error).lower()
                    if "401" in error_text or "invalid api key" in error_text:
                        st.error("Your Groq API key is invalid or expired. Enter a valid key in the sidebar and try again.")
                    elif "404" in error_text or "model_not_found" in error_text:
                        st.error(f"The Groq model '{selected_model}' is unavailable for this account. Click 'Load available models' in the sidebar and choose one of the returned models.")
                    else:
                        st.error(f"Could not answer your question: {error}")
                    st.session_state.pop("chain", None)
                    st.session_state.pop("chain_api_key", None)
                    st.session_state.pop("chain_model", None)
                    return
            st.markdown(response["answer"])
            if response["sources"]:
                with st.expander("Sources used"):
                    for source in response["sources"]:
                        st.markdown(f"<span class='source-tag'>{os.path.basename(source)}</span>", unsafe_allow_html=True)
        st.session_state.messages.append({"role": "assistant", "content": response["answer"], "sources": response["sources"]})


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
    if st is not None and get_script_run_ctx is not None and get_script_run_ctx() is not None:
        streamlit_app()
    else:
        main()
=======
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
>>>>>>> 47f4cba8d8682d2485352b71f9ef7a2a7e256c1a
