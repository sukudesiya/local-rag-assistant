import json
import os
import sqlite3
import tempfile
import uuid

import ollama
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="Local RAG Research Assistant", layout="wide")

# ---------- Settings ----------
# Local mode (default): Ollama on your own computer, storage in research.db.
# Hosted mode: turns on automatically when HOSTED_API_KEY is set (environment
# variable or Streamlit secret). Used for the public deployment.
EMBED_MODEL = "granite-embedding:30m"
CHAT_MODEL = "gemma3:1b"
CHUNK_SIZE = 900
CHUNK_OVERLAP = 150
TOP_K = 4
EMBED_BATCH_SIZE = 50
DEFAULT_QUESTION = "What helps distributed teams stay productive?"


def get_setting(name, default=""):
    value = os.environ.get(name)
    if value:
        return value
    try:
        return str(st.secrets.get(name, default))
    except Exception:
        return default


HOSTED_API_KEY = get_setting("HOSTED_API_KEY")
HOSTED_MODE = bool(HOSTED_API_KEY)
HOSTED_BASE_URL = get_setting("HOSTED_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/")
HOSTED_EMBED_MODEL = get_setting("HOSTED_EMBED_MODEL", "gemini-embedding-001")
HOSTED_CHAT_MODEL = get_setting("HOSTED_CHAT_MODEL", "gemini-2.5-flash")


def get_db_path():
    if not HOSTED_MODE:
        return "research.db"
    # Hosted mode: every browser session gets its own private database file,
    # so visitors never see each other's documents.
    if "db_path" not in st.session_state:
        st.session_state.db_path = os.path.join(tempfile.gettempdir(), f"research_{uuid.uuid4().hex}.db")
    return st.session_state.db_path


DB_PATH = get_db_path()
DB_LABEL = "this session's knowledge base" if HOSTED_MODE else "research.db"


# ---------- Model access (Ollama locally, hosted API when deployed) ----------
def get_hosted_client():
    from openai import OpenAI

    return OpenAI(api_key=HOSTED_API_KEY, base_url=HOSTED_BASE_URL)


def embed_texts(texts):
    if HOSTED_MODE:
        client = get_hosted_client()
        vectors = []
        for start in range(0, len(texts), EMBED_BATCH_SIZE):
            batch = texts[start:start + EMBED_BATCH_SIZE]
            response = client.embeddings.create(model=HOSTED_EMBED_MODEL, input=batch)
            vectors.extend(item.embedding for item in sorted(response.data, key=lambda item: item.index))
        return vectors
    return ollama.embed(model=EMBED_MODEL, input=texts).embeddings


def chat_completion(messages):
    if HOSTED_MODE:
        response = get_hosted_client().chat.completions.create(model=HOSTED_CHAT_MODEL, messages=messages)
        return response.choices[0].message.content or ""
    return ollama.chat(model=CHAT_MODEL, messages=messages).message.content


# ---------- Document processing ----------
def extract_documents(uploaded_files):
    pages = []
    for uploaded_file in uploaded_files:
        uploaded_file.seek(0)
        source = uploaded_file.name
        if source.lower().endswith(".pdf"):
            reader = PdfReader(uploaded_file)
            for page_number, page in enumerate(reader.pages, start=1):
                text = (page.extract_text() or "").strip()
                if text:
                    pages.append({"source": source, "page": page_number, "text": text})
        else:
            text = uploaded_file.getvalue().decode("utf-8", errors="replace").strip()
            if text:
                pages.append({"source": source, "page": 1, "text": text})
    return pages


def chunk_pages(pages):
    chunks = []
    for page in pages:
        text = page["text"]
        start = 0
        chunk_index = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            content = text[start:end].strip()
            if content:
                chunks.append({
                    "id": f"{page['source']}|{page['page']}|{chunk_index}",
                    "source": page["source"],
                    "page": page["page"],
                    "chunk_index": chunk_index,
                    "content": content,
                })
            if end == len(text):
                break
            start = end - CHUNK_OVERLAP
            chunk_index += 1
    return chunks


def literal_phrase_search(chunks, question):
    normalized_question = question.strip().lower()
    if not normalized_question:
        return []
    return [chunk for chunk in chunks if normalized_question in chunk["content"].lower()]


# ---------- Knowledge base ----------
def initialize_database():
    connection = sqlite3.connect(DB_PATH)
    try:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS chunks (
                id TEXT PRIMARY KEY,
                source TEXT NOT NULL,
                page INTEGER NOT NULL,
                chunk_index INTEGER NOT NULL,
                content TEXT NOT NULL,
                embedding TEXT NOT NULL
            )
            """
        )
        connection.commit()
    finally:
        connection.close()


def store_chunks(chunks):
    texts = [chunk["content"] for chunk in chunks]
    embeddings = embed_texts(texts)
    rows = [
        (chunk["id"], chunk["source"], chunk["page"], chunk["chunk_index"], chunk["content"], json.dumps(embedding))
        for chunk, embedding in zip(chunks, embeddings)
    ]
    sources = sorted({chunk["source"] for chunk in chunks})
    connection = sqlite3.connect(DB_PATH)
    try:
        connection.executemany("DELETE FROM chunks WHERE source = ?", [(source,) for source in sources])
        connection.executemany(
            "INSERT INTO chunks (id, source, page, chunk_index, content, embedding) VALUES (?, ?, ?, ?, ?, ?)",
            rows,
        )
        connection.commit()
    finally:
        connection.close()


def count_stored_chunks():
    connection = sqlite3.connect(DB_PATH)
    try:
        return connection.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
    finally:
        connection.close()


def load_sources():
    connection = sqlite3.connect(DB_PATH)
    try:
        return [row[0] for row in connection.execute("SELECT DISTINCT source FROM chunks ORDER BY source").fetchall()]
    finally:
        connection.close()


def load_rows(selected_sources=None):
    connection = sqlite3.connect(DB_PATH)
    try:
        if selected_sources:
            placeholders = ", ".join("?" for _ in selected_sources)
            sql = "SELECT source, page, chunk_index, content, embedding " + f"FROM chunks WHERE source IN ({placeholders})"
            return connection.execute(sql, tuple(selected_sources)).fetchall()
        return connection.execute("SELECT source, page, chunk_index, content, embedding FROM chunks").fetchall()
    finally:
        connection.close()


# ---------- Retrieval and answers ----------
def cosine_similarity(left, right):
    dot_product = sum(a * b for a, b in zip(left, right))
    left_magnitude = sum(value * value for value in left) ** 0.5
    right_magnitude = sum(value * value for value in right) ** 0.5
    if left_magnitude == 0 or right_magnitude == 0:
        return 0.0
    return dot_product / (left_magnitude * right_magnitude)


def retrieve(question, selected_sources=None):
    rows = load_rows(selected_sources)
    if not rows:
        return []
    query_vector = embed_texts([question])[0]
    scored = []
    for source, page, chunk_index, content, embedding_json in rows:
        scored.append({
            "source": source,
            "page": page,
            "chunk_index": chunk_index,
            "content": content,
            "score": cosine_similarity(query_vector, json.loads(embedding_json)),
        })
    scored.sort(key=lambda item: item["score"], reverse=True)
    return scored[:TOP_K]


def answer_question(question, evidence):
    evidence_text = "\n\n".join(
        f"[Source {index}: {item['source']}, page {item['page']}]\n{item['content']}"
        for index, item in enumerate(evidence, start=1)
    )
    return chat_completion([
        {"role": "system", "content": "You are a careful research assistant. Answer only from the provided evidence. Cite claims with the exact bracketed source labels. If the evidence is insufficient, say so clearly."},
        {"role": "user", "content": f"Question:\n{question}\n\nEvidence:\n{evidence_text}"},
    ])


def render_evidence(evidence):
    for index, item in enumerate(evidence, start=1):
        with st.expander(f"Source {index}: {item['source']}, page {item['page']}"):
            st.write(f"Cosine similarity: {item['score']:.3f}")
            st.write(item["content"])


initialize_database()

# ===== UI =====
st.markdown("""
    <style>
        .main .block-container {
            max-height: 100vh;
            overflow: hidden;
            padding-top: 2rem;
            padding-bottom: 1rem;
        }

        /* Tighter sidebar */
        section[data-testid="stSidebar"] [data-testid="stSidebarHeader"] {
            height: 1.5rem;
            padding: 0;
        }
        section[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding-top: 0.5rem;
        }
        section[data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
            gap: 0.75rem;
        }
        section[data-testid="stSidebar"] hr {
            margin: 0.5rem 0;
        }
        section[data-testid="stSidebar"] h2 {
            padding: 0.5rem 0 0.4rem 0;
            margin: 0;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Local RAG Research Assistant")
st.write("Upload PDF or TXT files, compare literal search with semantic retrieval, and chat with a cited, local AI assistant.")
if HOSTED_MODE:
    st.caption("Demo mode: a hosted model writes the answers, so document text is sent to an external provider. Please don't upload sensitive files. Your documents last only for this browser session.")

if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------- SIDEBAR: document management, debug tools 1-4 ----------
with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload research documents", type=["pdf", "txt"], accept_multiple_files=True
    )

    pages = extract_documents(uploaded_files) if uploaded_files else []
    chunks = chunk_pages(pages) if pages else []

    if uploaded_files:
        st.caption(f"Extracted pages: {len(pages)} \u00b7 Prepared chunks: {len(chunks)}")

    st.divider()
    st.header("Ask a question")
    question = st.text_input("Research question", value=DEFAULT_QUESTION)
    selected_sources = st.multiselect(
        "Limit retrieval to these sources; leave empty to search all", load_sources()
    )

    st.divider()
    st.header("Actions")
    run_literal = st.button("1. Test literal phrase search", disabled=not chunks, use_container_width=True)
    run_index = st.button("2. Index documents", disabled=not chunks, use_container_width=True)
    run_preview = st.button("3. Preview semantic retrieval", disabled=not question, use_container_width=True)
    run_ask = st.button("4. Ask the research assistant", disabled=not question, use_container_width=True)

# ---------- MAIN AREA: two fixed-height, independently scrollable cards ----------
col_results, col_chat = st.columns([1, 1])

with col_results:
    st.subheader("Results")
    results_container = st.container(height=420, border=True)
    with results_container:
        if uploaded_files and pages:
            st.subheader("Document previews")
            for page in pages[:3]:
                with st.expander(f"Preview: {page['source']}, page {page['page']}"):
                    st.write(page["text"][:1000])
        elif uploaded_files:
            st.write("No readable text was extracted. Scanned PDFs need OCR, which is outside this project.")

        if run_literal:
            matches = literal_phrase_search(chunks, question)
            st.subheader("Literal phrase search")
            st.write(f"Exact phrase matches: {len(matches)}")
            if not matches:
                st.write("The full question is absent even though one document discusses the idea.")

        if run_index:
            st.subheader("Indexing")
            try:
                with st.spinner("Embedding and storing chunks..."):
                    store_chunks(chunks)
                st.write(f"Stored chunks in {DB_LABEL}: {count_stored_chunks()}")
            except Exception as error:
                st.error(f"Indexing failed: {error}")

        if run_preview:
            st.subheader("Semantic retrieval results")
            if count_stored_chunks() == 0:
                st.write("Index at least one document first.")
            else:
                try:
                    evidence = retrieve(question, selected_sources)
                    render_evidence(evidence)
                except Exception as error:
                    st.error(f"Retrieval failed: {error}")

        if run_ask:
            st.subheader("Grounded answer")
            if count_stored_chunks() == 0:
                st.write("Index at least one document first.")
            else:
                try:
                    with st.spinner("Thinking..."):
                        evidence = retrieve(question, selected_sources)
                        answer_text = answer_question(question, evidence)
                    st.write(answer_text)
                    st.subheader("Retrieved evidence")
                    render_evidence(evidence)
                    st.session_state.messages.append({"role": "user", "content": question})
                    st.session_state.messages.append({"role": "assistant", "content": answer_text, "evidence": evidence})
                except Exception as error:
                    st.error(f"Could not generate an answer: {error}")

with col_chat:
    st.subheader("Chat")

    chat_container = st.container(height=420, border=True)
    with chat_container:
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if message["role"] == "assistant" and message.get("evidence"):
                    with st.expander("Show sources"):
                        render_evidence(message["evidence"])

    chat_question = st.chat_input("Ask about your documents...")

    if chat_question:
        st.session_state.messages.append({"role": "user", "content": chat_question})
        if count_stored_chunks() == 0:
            st.session_state.messages.append({"role": "assistant", "content": "Please index at least one document first (use the sidebar)."})
        else:
            try:
                with st.spinner("Thinking..."):
                    evidence = retrieve(chat_question, selected_sources)
                    answer_text = answer_question(chat_question, evidence)
                st.session_state.messages.append({"role": "assistant", "content": answer_text, "evidence": evidence})
            except Exception as error:
                st.session_state.messages.append({"role": "assistant", "content": f"Sorry, something went wrong: {error}"})
        st.rerun()

    if st.session_state.messages:
        if st.button("Clear chat history"):
            st.session_state.messages = []
            st.rerun()