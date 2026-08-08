"""Embedding & Indexing Tool (embedding half): text -> vectors via Gemini.

Gemini embeddings are used instead of a local model so the app stays light
enough to deploy (no torch / sentence-transformers download) and reuses the
Gemini key the rest of the stack may already have.

gemini-embedding-001 is requested at 768 dimensions. Embedding dimensionality
matters for retrieval: higher dimensions capture more nuance but cost more to
store and compare; 768 is a solid general-purpose middle ground.
"""

from __future__ import annotations

import os

EMBED_MODEL = os.environ.get("EMBED_MODEL", "gemini-embedding-001")
# gemini-embedding-001 defaults to 3072 dimensions; we ask for 768, a strong
# general-purpose size that stores and compares faster. Chroma uses cosine
# space, which normalises internally, so the reduced vectors need no extra work.
EMBED_DIM = int(os.environ.get("EMBED_DIM", "768"))
_BATCH = 100  # larger batches mean fewer requests, which the rate limit cares about most


class EmbeddingError(RuntimeError):
    pass


def _redact(text: str) -> str:
    for var in ("GEMINI_API_KEY", "GROQ_API_KEY"):
        value = os.environ.get(var)
        if value and len(value) >= 8:
            text = text.replace(value, "[redacted]")
    return text


class Embedder:
    """Wraps the Gemini embedding endpoint. Injectable client for testing."""

    def __init__(self, api_key: str | None = None, model: str | None = None, client=None):
        self.model = model or EMBED_MODEL
        if client is not None:
            self.client = client
            return
        key = api_key or os.environ.get("GEMINI_API_KEY")
        if not key:
            raise EmbeddingError("GEMINI_API_KEY is not set; cannot embed text.")
        try:
            from google import genai

            self.client = genai.Client(api_key=key)
        except ImportError as exc:  # pragma: no cover
            raise EmbeddingError("google-genai is not installed. Run: pip install google-genai") from exc

    # Transient failures worth retrying: rate limiting and server overload. The
    # Gemini free tier returns 429 readily, and without a retry a single 429
    # aborts a whole upload part-way through.
    _RETRYABLE = (429, 500, 502, 503, 504)

    def _embed_batch(self, texts: list[str], attempts: int = 4) -> list[list[float]]:
        import time

        for attempt in range(attempts):
            try:
                result = self.client.models.embed_content(
                    model=self.model,
                    contents=texts,
                    config={"output_dimensionality": EMBED_DIM},
                )
                return [list(e.values) for e in result.embeddings]
            except Exception as exc:  # noqa: BLE001
                code = getattr(exc, "code", None)
                if code not in self._RETRYABLE or attempt == attempts - 1:
                    if code == 429:
                        raise EmbeddingError(
                            "Gemini rate limit reached (free tier). Wait a minute and try "
                            "again, or upload a smaller document."
                        ) from None
                    raise EmbeddingError(f"Embedding failed: {_redact(str(exc))}") from None
                time.sleep(2 ** attempt)  # 1s, 2s, 4s
        raise EmbeddingError("Embedding failed after retries.")

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed many chunks, batched.

        Batches are deliberately large: the free tier limits requests per minute
        far more tightly than content per request, so fewer, fuller calls are
        much less likely to be rate limited than many small ones.
        """
        vectors: list[list[float]] = []
        for start in range(0, len(texts), _BATCH):
            vectors.extend(self._embed_batch(texts[start : start + _BATCH]))
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        return self._embed_batch([text])[0]
