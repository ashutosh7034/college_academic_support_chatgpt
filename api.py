from __future__ import annotations

from pathlib import Path
import shutil
import tempfile
from typing import List

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from rag_engine import PdfRAG


app = FastAPI(title="College Academic Support Chatbot API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

rag = PdfRAG()
index_ready = False
indexed_chunks = 0

UPLOAD_ROOT = Path(tempfile.gettempdir()) / "rag_pdf_uploads"
UPLOAD_ROOT.mkdir(parents=True, exist_ok=True)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "index_ready": index_ready, "indexed_chunks": indexed_chunks}


@app.post("/index")
async def build_index(files: List[UploadFile] = File(...)) -> dict:
    global index_ready
    global indexed_chunks

    if not files:
        return {"ok": False, "message": "No PDF files received."}

    pdf_paths: list[Path] = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            continue

        safe_name = Path(file.filename).name
        destination = UPLOAD_ROOT / safe_name

        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        pdf_paths.append(destination)

    if not pdf_paths:
        index_ready = False
        indexed_chunks = 0
        return {"ok": False, "message": "No valid PDF files found."}

    indexed = rag.build_index(pdf_paths)
    index_ready = indexed > 0
    indexed_chunks = indexed

    if not index_ready:
        return {"ok": False, "message": "No extractable text found in uploaded PDFs.", "indexed_chunks": 0}

    return {
        "ok": True,
        "message": "Index built successfully.",
        "indexed_chunks": indexed_chunks,
        "files": [path.name for path in pdf_paths],
    }


def _convert_numpy_types(obj):
    """Recursively convert numpy types to native Python types for JSON serialization."""
    import numpy as np
    
    if isinstance(obj, dict):
        return {key: _convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


@app.post("/ask")
def ask_question(question: str = Form(...), top_k: int = Form(4)) -> dict:
    if not index_ready:
        return {
            "ok": False,
            "message": "Index not ready. Upload and process PDFs first.",
            "answer": None,
        }

    clean_question = question.strip()
    if not clean_question:
        return {"ok": False, "message": "Question is required.", "answer": None}

    top_k = max(1, min(int(top_k), 10))
    result = rag.answer(question=clean_question, top_k=top_k)
    
    # Convert numpy types to native Python types for JSON serialization
    result = _convert_numpy_types(result)

    return {"ok": True, "message": "Answer generated.", "result": result}
