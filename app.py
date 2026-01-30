import os
import uuid
import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader

from src.utils import clean_text, chunk_text
from src.embedder import Embedder
from src.endee_client import EndeeVectorDB
from src.rag import RAGPipeline

load_dotenv()

st.set_page_config(page_title="Endee RAG Assistant", page_icon="🤖", layout="wide")
st.title("🤖 Endee RAG Assistant")
st.caption("✅ Semantic Search + RAG using Endee Vector Database")

ENDEE_BASE_URL = os.getenv("ENDEE_BASE_URL", "http://localhost:8000")
ENDEE_API_KEY = os.getenv("ENDEE_API_KEY", "")

embedder = Embedder()
vectordb = EndeeVectorDB(base_url=ENDEE_BASE_URL, index_name="docs_index", api_key=ENDEE_API_KEY)
rag = RAGPipeline(vectordb, embedder)

st.sidebar.header("📂 Upload & Index Documents")

if st.sidebar.button(" Clear Database"):
    import shutil
    if os.path.exists("local_db"):
        shutil.rmtree("local_db")
    st.sidebar.success(" Local DB cleared. Now upload and re-index your document.")
    st.rerun()

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF/TXT files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

if st.sidebar.button(" Index Documents"):
    if not uploaded_files:
        st.warning(" Upload at least one document.")
    else:
        all_items = []

        for f in uploaded_files:
            raw_text = ""

            # PDF
            if f.name.endswith(".pdf"):
                reader = PdfReader(f)
                for page in reader.pages:
                    t = page.extract_text()
                    if t:
                        raw_text += t + "\n"

            # TXT
            else:
                raw_text = f.read().decode("utf-8", errors="ignore")

            raw_text = clean_text(raw_text)

            if not raw_text:
                st.warning(f"⚠️ No text extracted from {f.name}")
                continue

            chunks = chunk_text(raw_text, chunk_size=250, overlap=50)
            vectors = embedder.embed(chunks)

            for chunk, vec in zip(chunks, vectors):
                all_items.append({
                    "id": str(uuid.uuid4()),
                    "vector": vec,
                    "metadata": {"text": chunk, "source": f.name}
                })

        if all_items:
            vectordb.create_index(dim=len(all_items[0]["vector"]))
            vectordb.upsert(all_items)

            st.success(f" Indexed {len(all_items)} chunks into Endee!")
            st.rerun()
        else:
            st.error(" No chunks were created.")

st.divider()
st.subheader(" Ask Your Documents")

query = st.text_input("Type your question:")

if st.button("✨ Get Answer"):
    if not query.strip():
        st.warning(" Please enter a question.")
    else:
        with st.spinner("Searching Endee + Generating answer..."):
            contexts = rag.retrieve(query, top_k=5)
            answer = rag.generate_answer(query, contexts)

        st.markdown("##  Answer")
        st.write(answer)

        st.markdown("##  Retrieved Context")
        for i, c in enumerate(contexts, 1):
            st.markdown(f"**{i}. Source:** `{c['source']}` | Score: `{c['score']}`")
            st.write(c["text"])
            st.divider()
