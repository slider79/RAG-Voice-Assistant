"""Vector store built on Chroma (Embedding & Indexing + Retriever tools).

Chroma is used as a pure vector store: embeddings are computed elsewhere (with
Gemini) and supplied directly, so the store's only jobs are to hold vectors and
run similarity search. It was chosen because it persists to disk and supports
deleting by metadata, which is exactly what the "delete a file and drop its
vectors" requirement needs, with almost no custom code.

The collection uses cosine space, so a returned distance d maps to a similarity
score of (1 - d) in [0, 1] that is meaningful to show the user.
"""

from __future__ import annotations

import os

# On some hosts (e.g. Streamlit Cloud) the system sqlite3 is too old for Chroma.
# pysqlite3-binary ships a newer one; swap it in when available. Harmless and
# skipped on platforms (like local Windows) where pysqlite3 is not installed.
try:  # pragma: no cover - environment specific
    __import__("pysqlite3")
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")
except Exception:
    pass

DEFAULT_DIR = os.environ.get("RAG_STORE_DIR", "rag_store")
COLLECTION = "documents"


class VectorStoreError(RuntimeError):
    pass


def _chunk_id(source: str, index: int) -> str:
    return f"{source}::chunk::{index}"


class VectorStore:
    def __init__(self, persist_dir: str | None = None, client=None):
        if client is not None:
            self.client = client
        else:
            try:
                import chromadb

                self.client = chromadb.PersistentClient(path=persist_dir or DEFAULT_DIR)
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Could not open the vector store: {exc}") from None
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION, metadata={"hnsw:space": "cosine"}
        )

    # -- indexing ----------------------------------------------------------

    def add_document(self, source: str, texts: list[str], embeddings: list[list[float]]) -> int:
        """Index one document's chunks. Replaces any existing chunks for the
        same filename first, so re-uploading a file updates it cleanly."""
        self.delete_document(source)
        ids = [_chunk_id(source, i) for i in range(len(texts))]
        metadatas = [{"source": source, "index": i} for i in range(len(texts))]
        self.collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metadatas)
        return len(texts)

    def delete_document(self, source: str) -> None:
        """Remove every vector belonging to a filename from the store itself,
        so its content can never be retrieved again."""
        self.collection.delete(where={"source": source})

    # -- inspection --------------------------------------------------------

    def list_documents(self) -> dict[str, int]:
        """Map each active filename to its number of stored chunks."""
        data = self.collection.get(include=["metadatas"])
        counts: dict[str, int] = {}
        for meta in data.get("metadatas") or []:
            src = meta.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts

    def chunk_count(self) -> int:
        return self.collection.count()

    # -- retrieval ---------------------------------------------------------

    def query(self, query_embedding: list[float], k: int = 4) -> list[dict]:
        """Return the top-k chunks with a cosine similarity score in [0, 1]."""
        if self.collection.count() == 0:
            return []
        res = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=min(k, self.collection.count()),
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        for text, meta, dist in zip(docs, metas, dists):
            hits.append(
                {
                    "text": text,
                    "source": meta.get("source", "unknown"),
                    "score": round(max(0.0, 1.0 - float(dist)), 4),
                }
            )
        return hits
