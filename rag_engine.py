from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import List, Sequence

import faiss
import numpy as np
from pypdf import PdfReader
from sentence_transformers import CrossEncoder, SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import pipeline


@dataclass
class RetrievedChunk:
    chunk_id: int
    text: str
    score: float
    source: str
    page: int


class PdfRAG:
    def __init__(
        self,
        embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        generator_model: str = "google/flan-t5-small",
        chunk_size: int = 900,
        chunk_overlap: int = 120,
        dense_weight: float = 0.6,
        sparse_weight: float = 0.4,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.dense_weight = dense_weight
        self.sparse_weight = sparse_weight

        self.embedder = SentenceTransformer(embedding_model)
        self.reranker = CrossEncoder(reranker_model)
        self.generator = pipeline("text2text-generation", model=generator_model)
        self.extractive_qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

        self._index: faiss.Index | None = None
        self._tfidf = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), min_df=1)
        self._tfidf_matrix = None
        self._chunk_texts: List[str] = []
        self._chunk_sources: List[str] = []
        self._chunk_pages: List[int] = []

    def _read_pdf_pages(self, pdf_path: Path) -> list[tuple[int, str]]:
        reader = PdfReader(str(pdf_path))
        pages: list[tuple[int, str]] = []
        for page_num, page in enumerate(reader.pages, start=1):
            text = page.extract_text() or ""
            pages.append((page_num, text))
        return pages

    def _normalize_text(self, text: str) -> str:
        return " ".join(text.split())

    def _split_paragraphs(self, text: str) -> list[str]:
        raw_parts = re.split(r"\n\s*\n+", text)
        paragraphs = [self._normalize_text(part) for part in raw_parts]
        return [part for part in paragraphs if part]

    def _split_text(self, text: str) -> List[str]:
        if not text.strip():
            return []

        chunks: List[str] = []
        paragraphs = self._split_paragraphs(text)
        if not paragraphs:
            paragraphs = [self._normalize_text(text)]

        current_chunk = ""
        for paragraph in paragraphs:
            if not paragraph:
                continue

            candidate = f"{current_chunk} {paragraph}".strip() if current_chunk else paragraph
            if len(candidate) <= self.chunk_size:
                current_chunk = candidate
                continue

            if current_chunk:
                chunks.append(current_chunk)
            if len(paragraph) <= self.chunk_size:
                current_chunk = paragraph
                continue

            start = 0
            step = max(1, self.chunk_size - self.chunk_overlap)
            while start < len(paragraph):
                end = start + self.chunk_size
                piece = paragraph[start:end]
                if piece:
                    chunks.append(piece)
                start += step
            current_chunk = ""

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def build_index(self, pdf_paths: Sequence[Path]) -> int:
        self._chunk_texts = []
        self._chunk_sources = []
        self._chunk_pages = []

        for pdf_path in pdf_paths:
            pages = self._read_pdf_pages(pdf_path)
            for page_num, page_text in pages:
                chunks = self._split_text(page_text)
                self._chunk_texts.extend(chunks)
                self._chunk_sources.extend([pdf_path.name] * len(chunks))
                self._chunk_pages.extend([page_num] * len(chunks))

        if not self._chunk_texts:
            self._index = None
            self._tfidf_matrix = None
            return 0

        vectors = self.embedder.encode(self._chunk_texts, convert_to_numpy=True)
        vectors = vectors.astype(np.float32)
        faiss.normalize_L2(vectors)

        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        self._index = index
        self._tfidf_matrix = self._tfidf.fit_transform(self._chunk_texts)
        return len(self._chunk_texts)

    def _dense_retrieve(self, question: str, candidate_k: int) -> tuple[np.ndarray, np.ndarray]:
        question_vec = self.embedder.encode([question], convert_to_numpy=True).astype(np.float32)
        faiss.normalize_L2(question_vec)

        scores, indices = self._index.search(question_vec, candidate_k)
        return scores[0], indices[0]

    def _sparse_retrieve(self, question: str) -> np.ndarray:
        question_tfidf = self._tfidf.transform([question])
        sparse_scores = (question_tfidf @ self._tfidf_matrix.T).toarray().ravel()
        return sparse_scores

    def retrieve(self, question: str, top_k: int = 4, candidate_multiplier: int = 5) -> List[RetrievedChunk]:
        if not self._index or self._tfidf_matrix is None:
            return []

        candidate_k = min(max(top_k * candidate_multiplier, top_k), len(self._chunk_texts))

        dense_scores, dense_indices = self._dense_retrieve(question, candidate_k)
        sparse_scores_all = self._sparse_retrieve(question)

        dense_map: dict[int, float] = {}
        for score, idx in zip(dense_scores, dense_indices):
            if idx == -1:
                continue
            dense_map[idx] = float(score)

        sparse_ranking = np.argsort(-sparse_scores_all)[:candidate_k]
        candidate_ids = set(dense_map.keys()) | set(int(idx) for idx in sparse_ranking)
        if not candidate_ids:
            return []

        candidate_list = sorted(candidate_ids)

        dense_values = np.array([dense_map.get(idx, 0.0) for idx in candidate_list], dtype=np.float32)
        sparse_values = np.array([sparse_scores_all[idx] for idx in candidate_list], dtype=np.float32)

        dense_den = float(dense_values.max() - dense_values.min())
        sparse_den = float(sparse_values.max() - sparse_values.min())

        dense_norm = (
            (dense_values - dense_values.min()) / dense_den if dense_den > 1e-12 else np.zeros_like(dense_values)
        )
        sparse_norm = (
            (sparse_values - sparse_values.min()) / sparse_den if sparse_den > 1e-12 else np.zeros_like(sparse_values)
        )

        hybrid_scores = self.dense_weight * dense_norm + self.sparse_weight * sparse_norm
        hybrid_order = np.argsort(-hybrid_scores)
        top_candidates = [candidate_list[i] for i in hybrid_order[:candidate_k]]

        rerank_pairs = [(question, self._chunk_texts[idx]) for idx in top_candidates]
        rerank_scores = self.reranker.predict(rerank_pairs)
        rerank_order = np.argsort(-rerank_scores)
        reranked_ids = [top_candidates[i] for i in rerank_order[: min(top_k, len(top_candidates))]]

        score_lookup = {idx: float(score) for idx, score in zip(top_candidates, rerank_scores)}

        results: List[RetrievedChunk] = []
        for idx in reranked_ids:
            results.append(
                RetrievedChunk(
                    chunk_id=idx,
                    text=self._chunk_texts[idx],
                    score=score_lookup.get(idx, 0.0),
                    source=self._chunk_sources[idx],
                    page=self._chunk_pages[idx],
                )
            )

        return results

    def _build_prompt(self, question: str, retrieved: list[RetrievedChunk]) -> str:
        context_blocks = []
        for idx, chunk in enumerate(retrieved, start=1):
            context_blocks.append(
                f"[{idx}] Source: {chunk.source}, Page: {chunk.page}\\n{chunk.text}"
            )

        joined_context = "\n\n".join(context_blocks)
        prompt = (
            "Answer using only the provided context. "
            "If the answer is not present, say you do not know based on the documents. "
            "Include citation numbers like [1], [2].\n\n"
            f"Question: {question}\n\n"
            f"Context:\n{joined_context}\n\n"
            "Final answer:"
        )
        return prompt

    def _is_unusable_answer(self, answer: str) -> bool:
        text = (answer or "").strip()
        if not text:
            return True

        citation_only = re.fullmatch(r"(\s*[\[(]?\d+[\])]?\s*[,.]?\s*)+", text) is not None
        text_without_citations = re.sub(r"\[\d+\]", " ", text)
        alpha_tokens = re.findall(r"[A-Za-z]{2,}", text_without_citations)
        too_short = len(text) < 8
        no_meaningful_words = len(alpha_tokens) == 0
        return citation_only or too_short or no_meaningful_words

    def _extractive_fallback(self, question: str, retrieved: list[RetrievedChunk]) -> tuple[str, float]:
        best_answer = "I do not know based on the provided documents."
        best_score = 0.0
        best_idx = 1

        for idx, chunk in enumerate(retrieved, start=1):
            prediction = self.extractive_qa(question=question, context=chunk.text)
            candidate_answer = (prediction.get("answer") or "").strip()
            candidate_score = float(prediction.get("score", 0.0))

            candidate_has_words = len(re.findall(r"[A-Za-z]{2,}", candidate_answer)) > 0
            if candidate_score > best_score and candidate_answer and candidate_has_words:
                best_answer = f"{candidate_answer} [{idx}]"
                best_score = candidate_score
                best_idx = idx

        if best_score <= 0.0:
            return "I do not know based on the provided documents.", 0.0

        return best_answer if f"[{best_idx}]" in best_answer else f"{best_answer} [{best_idx}]", best_score

    def answer(self, question: str, top_k: int = 4) -> dict:
        retrieved = self.retrieve(question=question, top_k=top_k)
        if not retrieved:
            return {
                "answer": "No content is indexed yet. Upload and process at least one PDF.",
                "confidence": 0.0,
                "sources": [],
                "contexts": [],
            }

        prompt = self._build_prompt(question, retrieved)
        generation = self.generator(prompt, max_new_tokens=180, do_sample=False)
        answer_text = generation[0]["generated_text"].strip() if generation else ""

        fallback_confidence = 0.0
        if self._is_unusable_answer(answer_text):
            answer_text, fallback_confidence = self._extractive_fallback(question, retrieved)

        sources = sorted({chunk.source for chunk in retrieved})
        rerank_raw = float(max(chunk.score for chunk in retrieved))
        rerank_confidence = 1.0 / (1.0 + float(np.exp(-rerank_raw)))
        confidence = max(rerank_confidence, fallback_confidence)
        best_chunk = max(retrieved, key=lambda item: item.score)

        return {
            "answer": answer_text,
            "confidence": confidence,
            "best_source": best_chunk.source,
            "best_page": best_chunk.page,
            "sources": sources,
            "contexts": [
                {
                    "chunk_id": chunk.chunk_id,
                    "source": chunk.source,
                    "page": chunk.page,
                    "score": chunk.score,
                    "text": chunk.text,
                }
                for chunk in retrieved
            ],
        }
