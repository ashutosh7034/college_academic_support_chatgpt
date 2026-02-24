from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from rag_engine import PdfRAG


st.set_page_config(page_title="PDF RAG QA", page_icon="📄", layout="wide")
st.title("📄 PDF RAG Question Answering")
st.caption("Upload one or more PDFs, then ask questions grounded in those documents using advanced hybrid retrieval.")

if "rag" not in st.session_state:
    st.session_state.rag = PdfRAG()
if "index_ready" not in st.session_state:
    st.session_state.index_ready = False
if "indexed_chunks" not in st.session_state:
    st.session_state.indexed_chunks = 0

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True,
)

if st.button("Process PDFs", type="primary"):
    if not uploaded_files:
        st.warning("Please upload at least one PDF file.")
    else:
        with st.spinner("Reading PDFs and building hybrid index (dense + sparse + reranker)..."):
            with tempfile.TemporaryDirectory() as tmpdir:
                pdf_paths = []
                for file in uploaded_files:
                    target = Path(tmpdir) / file.name
                    target.write_bytes(file.read())
                    pdf_paths.append(target)

                indexed = st.session_state.rag.build_index(pdf_paths)
                st.session_state.index_ready = indexed > 0
                st.session_state.indexed_chunks = indexed

        if st.session_state.index_ready:
            st.success(f"Index ready with {st.session_state.indexed_chunks} chunks.")
        else:
            st.error("No extractable text found in the uploaded PDFs.")

question = st.text_input("Ask a question about your PDFs")

top_k = st.slider("Retrieved chunks", min_value=1, max_value=8, value=4)

if st.button("Get Answer"):
    if not st.session_state.index_ready:
        st.warning("Process PDF files first.")
    elif not question.strip():
        st.warning("Enter a question.")
    else:
        with st.spinner("Retrieving and generating answer..."):
            result = st.session_state.rag.answer(question=question, top_k=top_k)

        st.subheader("Answer")
        st.write(result["answer"])
        st.write(f"Confidence: {result['confidence']:.3f}")

        if result.get("best_source"):
            best_page = result.get("best_page")
            if best_page:
                st.write(f"Best source: {result['best_source']} (Page {best_page})")
            else:
                st.write(f"Best source: {result['best_source']}")

        st.subheader("Sources used")
        if result["sources"]:
            for src in result["sources"]:
                st.write(f"- {src}")
        else:
            st.write("No sources returned.")

        with st.expander("Retrieved contexts"):
            for idx, context in enumerate(result["contexts"], start=1):
                st.markdown(
                    f"**Chunk {idx}** · Source: `{context['source']}` · Page: `{context['page']}` · Score: `{context['score']:.3f}`"
                )
                st.write(context["text"])
