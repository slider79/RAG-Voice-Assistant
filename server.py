"""Delphi backend: the RAG brain Vapi talks to.

Vapi handles the voice (speech-to-text, text-to-speech, streaming, turn-taking)
and is configured to use this service as its "custom LLM". That means Vapi sends
the conversation to an OpenAI-compatible /chat/completions endpoint here; this
service runs Retrieval-Augmented Generation over your documents and streams a
grounded answer back, which Vapi speaks aloud.

Endpoints:
  POST /chat/completions      OpenAI-compatible; the endpoint Vapi calls.
  GET/POST/DELETE /documents  Manage the knowledge base the answers come from.
  GET /health                 Liveness + whether keys and documents are present.
"""

from __future__ import annotations

import json
import os
import time
import uuid

from fastapi import Depends, FastAPI, File, Header, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from document_loader import DocumentError
from embeddings import Embedder, EmbeddingError
from llm import Generator, LLMError
from rag import RAGPipeline
from vector_store import VectorStore, VectorStoreError

RETRIEVE_K = int(os.environ.get("RETRIEVE_K", "4"))
# Optional shared secret. If set, /chat/completions requires a matching bearer
# token (configure the same value as the API key on Vapi's custom LLM).
BACKEND_SECRET = os.environ.get("BACKEND_SECRET", "")

app = FastAPI(title="Delphi RAG Voice Backend")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: RAGPipeline | None = None


def get_pipeline() -> RAGPipeline:
    """Build the RAG pipeline once, lazily, so the server starts without keys."""
    global _pipeline
    if _pipeline is None:
        try:
            _pipeline = RAGPipeline(Embedder(), VectorStore(), Generator())
        except (EmbeddingError, VectorStoreError) as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from None
    return _pipeline


def require_secret(authorization: str | None = Header(default=None)) -> None:
    if not BACKEND_SECRET:
        return  # open when no secret is configured (fine for local testing)
    token = (authorization or "").removeprefix("Bearer ").strip()
    if token != BACKEND_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


# --------------------------------------------------------------------------
# Health
# --------------------------------------------------------------------------


@app.get("/health")
def health() -> dict:
    have_keys = bool(os.environ.get("GROQ_API_KEY") and os.environ.get("GEMINI_API_KEY"))
    docs = {}
    if have_keys:
        try:
            docs = get_pipeline().sources()
        except Exception:  # noqa: BLE001
            docs = {}
    return {"status": "ok", "keys_configured": have_keys, "documents": len(docs)}


# --------------------------------------------------------------------------
# Document management (the knowledge base)
# --------------------------------------------------------------------------


@app.get("/documents")
def list_documents() -> dict:
    pipe = get_pipeline()
    return {"documents": pipe.sources(), "total_chunks": pipe.total_chunks()}


@app.post("/documents")
async def add_document(file: UploadFile = File(...)) -> dict:
    pipe = get_pipeline()
    data = await file.read()
    try:
        return pipe.ingest(file.filename, data)
    except (DocumentError, EmbeddingError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from None


@app.delete("/documents/{name}")
def delete_document(name: str) -> dict:
    get_pipeline().delete(name)
    return {"deleted": name}


# --------------------------------------------------------------------------
# OpenAI-compatible chat completions (Vapi's custom-LLM endpoint)
# --------------------------------------------------------------------------


def _last_user_message(messages: list[dict]) -> str:
    for m in reversed(messages):
        if m.get("role") == "user" and m.get("content"):
            content = m["content"]
            # Vapi/OpenAI may send content as a string or a list of parts.
            if isinstance(content, list):
                return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
            return str(content)
    return ""


def _chunk(cid: str, model: str, delta: dict, finish: str | None = None) -> str:
    payload = {
        "id": cid,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
    }
    return f"data: {json.dumps(payload)}\n\n"


@app.post("/chat/completions", dependencies=[Depends(require_secret)])
async def chat_completions(body: dict) -> object:
    pipe = get_pipeline()
    messages = body.get("messages") or []
    model = body.get("model") or "delphi-rag"
    question = _last_user_message(messages)
    cid = f"chatcmpl-{uuid.uuid4().hex[:24]}"

    if not question.strip():
        raise HTTPException(status_code=400, detail="No user message to answer.")

    try:
        hits = pipe.retrieve(question, k=RETRIEVE_K)
    except EmbeddingError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None

    # Streaming path (what Vapi uses): emit OpenAI-style SSE chunks.
    if body.get("stream"):
        def event_stream():
            yield _chunk(cid, model, {"role": "assistant"})
            try:
                for token in pipe.stream_answer(question, hits):
                    yield _chunk(cid, model, {"content": token})
            except LLMError as exc:
                yield _chunk(cid, model, {"content": f" [error: {exc}]"})
            yield _chunk(cid, model, {}, finish="stop")
            yield "data: [DONE]\n\n"

        return StreamingResponse(event_stream(), media_type="text/event-stream")

    # Non-streaming path.
    try:
        answer = "".join(pipe.stream_answer(question, hits))
    except LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from None
    return {
        "id": cid,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {"index": 0, "message": {"role": "assistant", "content": answer}, "finish_reason": "stop"}
        ],
        "sources": [h["source"] for h in hits],
    }
