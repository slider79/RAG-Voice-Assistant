"""RAG pipeline: ties the tools together and measures every stage.

One class orchestrates ingestion, dynamic add/delete, retrieval, generation,
and the Performance Monitor. Every answer carries a metrics dict (latency
broken down by stage, retrieval scores, and an optional faithfulness score),
which the UI logs and charts.
"""

from __future__ import annotations

import time

from document_loader import load_and_chunk
from embeddings import Embedder
from llm import Generator
from vector_store import VectorStore


def _ms(seconds: float) -> float:
    return round(seconds * 1000, 1)


class RAGPipeline:
    def __init__(self, embedder: Embedder, store: VectorStore, generator: Generator):
        self.embedder = embedder
        self.store = store
        self.generator = generator

    # -- data source management -------------------------------------------

    def ingest(self, filename: str, data: bytes) -> dict:
        """Add or replace one document. Returns a small report for the UI."""
        t0 = time.perf_counter()
        chunks = load_and_chunk(filename, data)
        texts = [c.text for c in chunks]
        embeddings = self.embedder.embed_documents(texts)
        added = self.store.add_document(filename, texts, embeddings)
        return {"source": filename, "chunks": added, "seconds": round(time.perf_counter() - t0, 2)}

    def delete(self, filename: str) -> None:
        self.store.delete_document(filename)

    def sources(self) -> dict[str, int]:
        return self.store.list_documents()

    def total_chunks(self) -> int:
        return self.store.chunk_count()

    # -- query ------------------------------------------------------------

    def retrieve(self, question: str, k: int = 4) -> list[dict]:
        """Embed the question and return the top-k matching chunks."""
        return self.store.query(self.embedder.embed_query(question), k=k)

    def stream_answer(self, question: str, hits: list[dict]):
        """Stream the grounded answer token by token (used by the voice backend)."""
        yield from self.generator.answer_stream(question, hits)

    def answer(self, question: str, k: int = 4, evaluate: bool = True) -> dict:
        """Retrieve, generate, and measure. Returns answer, sources, metrics."""
        t_embed = time.perf_counter()
        query_vec = self.embedder.embed_query(question)
        embed_s = time.perf_counter() - t_embed

        t_ret = time.perf_counter()
        hits = self.store.query(query_vec, k=k)
        retrieval_s = time.perf_counter() - t_ret

        t_gen = time.perf_counter()
        answer = self.generator.answer(question, hits)
        generation_s = time.perf_counter() - t_gen

        scores = [h["score"] for h in hits]
        metrics = {
            "embed_ms": _ms(embed_s),
            "retrieval_ms": _ms(retrieval_s),
            "generation_ms": _ms(generation_s),
            "total_ms": _ms(embed_s + retrieval_s + generation_s),
            "chunks_retrieved": len(hits),
            "top_score": max(scores) if scores else 0.0,
            "avg_score": round(sum(scores) / len(scores), 4) if scores else 0.0,
            "faithfulness": None,
            "supported": None,
        }

        if evaluate and hits:
            t_judge = time.perf_counter()
            verdict = self.generator.judge(answer, hits)
            metrics["faithfulness"] = verdict["faithfulness"]
            metrics["supported"] = verdict["supported"]
            metrics["judge_note"] = verdict["note"]
            metrics["judge_ms"] = _ms(time.perf_counter() - t_judge)

        return {"question": question, "answer": answer, "sources": hits, "metrics": metrics}
