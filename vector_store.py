"""Vector store on Qdrant (Embedding & Indexing + Retriever tools).

Qdrant replaced Chroma here because this backend runs on serverless hosting
(Vercel), where there is no persistent local disk: a document uploaded in one
request must be retrievable in the next, which may run on a different instance.
Qdrant is a network vector database, so state lives in one place and every
serverless invocation sees the same knowledge base.

Set QDRANT_URL and QDRANT_API_KEY (Qdrant Cloud has a free tier). With no URL
set it falls back to an in-memory instance, which is handy for local runs and
tests but does not persist.

Vectors are supplied by the caller (Gemini embeddings); Qdrant only stores and
searches them. The collection uses cosine distance, so a match score maps to a
similarity in [0, 1].
"""

from __future__ import annotations

import os
import uuid

VECTOR_DIM = int(os.environ.get("VECTOR_DIM", "768"))
COLLECTION = os.environ.get("QDRANT_COLLECTION", "documents")


class VectorStoreError(RuntimeError):
    pass


class VectorStore:
    def __init__(self, client=None):
        try:
            from qdrant_client import QdrantClient, models
        except ImportError as exc:  # pragma: no cover
            raise VectorStoreError("qdrant-client is not installed. Run: pip install qdrant-client") from exc
        self.models = models

        if client is not None:
            self.client = client
        else:
            url = os.environ.get("QDRANT_URL", "").strip()
            try:
                if url:
                    self.client = QdrantClient(url=url, api_key=os.environ.get("QDRANT_API_KEY") or None)
                else:
                    self.client = QdrantClient(location=":memory:")
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(f"Could not connect to Qdrant: {exc}") from None

        if not self.client.collection_exists(COLLECTION):
            self.client.create_collection(
                COLLECTION,
                vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.COSINE),
            )

    # -- indexing ----------------------------------------------------------

    def add_document(self, source: str, texts: list[str], embeddings: list[list[float]]) -> int:
        """Index one document's chunks, replacing any existing ones for the
        same filename first so a re-upload updates cleanly."""
        self.delete_document(source)
        points = [
            self.models.PointStruct(
                id=str(uuid.uuid4()),
                vector=embeddings[i],
                payload={"source": source, "index": i, "text": texts[i]},
            )
            for i in range(len(texts))
        ]
        self.client.upsert(COLLECTION, points=points)
        return len(texts)

    def delete_document(self, source: str) -> None:
        """Remove every vector belonging to a filename, by metadata filter."""
        self.client.delete(
            COLLECTION,
            points_selector=self.models.FilterSelector(
                filter=self.models.Filter(
                    must=[self.models.FieldCondition(key="source", match=self.models.MatchValue(value=source))]
                )
            ),
        )

    # -- inspection --------------------------------------------------------

    def list_documents(self) -> dict[str, int]:
        """Map each active filename to its number of stored chunks."""
        records, _ = self.client.scroll(COLLECTION, with_payload=["source"], limit=10000)
        counts: dict[str, int] = {}
        for rec in records:
            src = (rec.payload or {}).get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts

    def chunk_count(self) -> int:
        return self.client.count(COLLECTION).count

    # -- retrieval ---------------------------------------------------------

    def query(self, query_embedding: list[float], k: int = 4) -> list[dict]:
        """Return the top-k chunks with a cosine similarity score in [0, 1]."""
        if self.chunk_count() == 0:
            return []
        result = self.client.query_points(COLLECTION, query=query_embedding, limit=k, with_payload=True)
        hits = []
        for point in result.points:
            payload = point.payload or {}
            hits.append(
                {
                    "text": payload.get("text", ""),
                    "source": payload.get("source", "unknown"),
                    "score": round(max(0.0, float(point.score)), 4),
                }
            )
        return hits
