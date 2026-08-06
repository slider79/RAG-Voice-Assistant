"""Offline checks for the Delphi backend.

Run with:  python tests/test_backend.py

No API keys needed. The RAG pipeline is replaced with a fake, so these tests
exercise the real FastAPI contract that Vapi depends on: the OpenAI-compatible
/chat/completions endpoint (streaming and non-streaming), the document
endpoints, health, and the optional bearer-token guard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import server  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

failures: list[str] = []


def check(cond: bool, label: str) -> None:
    print(f"  {'PASS' if cond else 'FAIL'}  {label}")
    if not cond:
        failures.append(label)


class FakePipeline:
    def __init__(self):
        self.docs = {"guide.pdf": 3}

    def retrieve(self, question, k=4):
        return [{"text": "Paris is the capital of France.", "source": "guide.pdf", "score": 0.9}]

    def stream_answer(self, question, hits):
        for tok in ["Paris", " is", " the", " capital", "."]:
            yield tok

    def sources(self):
        return dict(self.docs)

    def total_chunks(self):
        return sum(self.docs.values())

    def ingest(self, name, data):
        self.docs[name] = 2
        return {"source": name, "chunks": 2, "seconds": 0.1}

    def delete(self, name):
        self.docs.pop(name, None)


def client(secret: str = "") -> TestClient:
    server._pipeline = FakePipeline()
    server.BACKEND_SECRET = secret
    return TestClient(server.app)


# ---- helper ---------------------------------------------------------------


def test_last_user_message():
    print("\n_last_user_message reads string and list content")
    check(server._last_user_message([{"role": "user", "content": "hello"}]) == "hello", "plain string content")
    parts = [{"role": "user", "content": [{"type": "text", "text": "hi there"}]}]
    check(server._last_user_message(parts) == "hi there", "OpenAI list-of-parts content")
    check(server._last_user_message([{"role": "assistant", "content": "x"}]) == "", "ignores non-user turns")


# ---- endpoints ------------------------------------------------------------


def test_health():
    print("\nGET /health reports status")
    r = client().get("/health")
    check(r.status_code == 200 and r.json()["status"] == "ok", "health returns ok")


def test_chat_completions_nonstream():
    print("\nPOST /chat/completions (non-stream) returns OpenAI shape with sources")
    r = client().post("/chat/completions", json={"messages": [{"role": "user", "content": "capital of France?"}]})
    check(r.status_code == 200, "responds 200")
    data = r.json()
    check(data["object"] == "chat.completion", "OpenAI object type")
    check(data["choices"][0]["message"]["content"] == "Paris is the capital.", "answer assembled from stream")
    check(data["choices"][0]["message"]["role"] == "assistant", "assistant role")
    check("guide.pdf" in data["sources"], "reports retrieved sources")


def test_chat_completions_stream():
    print("\nPOST /chat/completions (stream) emits OpenAI SSE chunks")
    r = client().post("/chat/completions", json={"stream": True, "messages": [{"role": "user", "content": "hi"}]})
    check(r.status_code == 200, "responds 200")
    body = r.text
    check("chat.completion.chunk" in body, "emits chunk objects")
    check('"role": "assistant"' in body or '"role":"assistant"' in body, "first chunk sets the role")
    check("data: [DONE]" in body, "terminates with [DONE]")
    # the content tokens should be present across the chunks
    contents = "".join(
        json.loads(line[6:])["choices"][0]["delta"].get("content", "")
        for line in body.splitlines()
        if line.startswith("data: ") and line[6:].strip() != "[DONE]"
    )
    check(contents == "Paris is the capital.", "streamed tokens reconstruct the answer")


def test_chat_completions_needs_user_message():
    print("\nPOST /chat/completions rejects an empty conversation")
    r = client().post("/chat/completions", json={"messages": [{"role": "system", "content": "x"}]})
    check(r.status_code == 400, "400 when there is no user message")


def test_documents_crud():
    print("\nDocument endpoints list, add, and delete")
    c = client()
    check(c.get("/documents").json()["documents"] == {"guide.pdf": 3}, "lists current documents")
    up = c.post("/documents", files={"file": ("notes.txt", b"hello world", "text/plain")})
    check(up.status_code == 200 and up.json()["chunks"] == 2, "uploads and indexes a file")
    check("notes.txt" in c.get("/documents").json()["documents"], "new file appears")
    dele = c.delete("/documents/guide.pdf")
    check(dele.status_code == 200, "delete returns 200")
    check("guide.pdf" not in c.get("/documents").json()["documents"], "deleted file is gone")


def test_vector_store_qdrant():
    print("\nVectorStore (in-memory Qdrant) indexes, retrieves, and deletes by source")
    from vector_store import VECTOR_DIM, VectorStore

    def vec(i):  # a 768-dim one-hot vector so cosine search is unambiguous
        v = [0.0] * VECTOR_DIM
        v[i] = 1.0
        return v

    vs = VectorStore()  # no QDRANT_URL -> in-memory
    vs.add_document("A.txt", ["about cats"], [vec(0)])
    vs.add_document("B.txt", ["about boats"], [vec(1)])
    check(vs.chunk_count() == 2, "two chunks indexed")
    check(set(vs.list_documents()) == {"A.txt", "B.txt"}, "lists both documents")

    hits = vs.query(vec(0), k=1)
    check(hits and hits[0]["source"] == "A.txt", "retrieves the nearest document")
    check(0.0 <= hits[0]["score"] <= 1.0, "score is a similarity in [0,1]")

    vs.delete_document("A.txt")
    check(vs.chunk_count() == 1, "delete removes the document's vectors")
    check("A.txt" not in vs.list_documents(), "deleted document leaves the list")
    check(vs.query(vec(0), k=1)[0]["source"] == "B.txt", "deleted content is no longer retrievable")


def test_index_html():
    print("\nGET / serves the built-in web UI")
    r = client().get("/")
    check(r.status_code == 200 and "Delphi" in r.text, "returns the HTML page")
    check("/chat/completions" in r.text, "page references the Vapi endpoint")


def test_bearer_secret_guard():
    print("\nBearer secret guards /chat/completions when configured")
    c = client(secret="sekret")
    body = {"messages": [{"role": "user", "content": "hi"}]}
    check(c.post("/chat/completions", json=body).status_code == 401, "rejects a missing key")
    ok = c.post("/chat/completions", json=body, headers={"Authorization": "Bearer sekret"})
    check(ok.status_code == 200, "accepts the correct key")


if __name__ == "__main__":
    for test in (
        test_last_user_message,
        test_health,
        test_chat_completions_nonstream,
        test_chat_completions_stream,
        test_chat_completions_needs_user_message,
        test_documents_crud,
        test_vector_store_qdrant,
        test_index_html,
        test_bearer_secret_guard,
    ):
        test()

    print()
    if failures:
        print(f"{len(failures)} check(s) failed:")
        for f in failures:
            print(f"  - {f}")
        sys.exit(1)
    print("All checks passed.")
