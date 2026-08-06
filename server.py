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
from fastapi.responses import HTMLResponse, StreamingResponse

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
# Built-in web UI (so the deploy is a complete app, not just an API)
# --------------------------------------------------------------------------

INDEX_HTML = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Delphi</title>
<style>
:root{--bg:#191919;--panel:#202020;--ink:#e6e6e4;--muted:#979692;--line:#333230;}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);
font-family:ui-sans-serif,-apple-system,"Segoe UI",Inter,Arial,sans-serif;line-height:1.5}
.wrap{max-width:980px;margin:0 auto;padding:36px 20px 80px}
.brand{font-size:2.3rem;font-weight:700;letter-spacing:-.02em}
.tagline{color:var(--muted);font-size:.95rem;margin:.4rem 0 0;max-width:64ch}
.rule{border-bottom:1px solid var(--line);margin:18px 0}
.label{font-size:.7rem;text-transform:uppercase;letter-spacing:.13em;color:var(--muted);font-weight:600;margin:0 0 8px}
.grid{display:grid;grid-template-columns:340px 1fr;gap:22px}
@media(max-width:760px){.grid{grid-template-columns:1fr}}
.box{border:1px solid var(--line);background:var(--panel);padding:8px 11px;margin-bottom:7px;font-size:.85rem}
.box .meta{color:var(--muted);font-size:.78rem}
button{border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:8px 14px;
cursor:pointer;font-family:inherit;font-size:.85rem}button:hover{border-color:var(--muted);background:#262625}
input[type=text]{width:100%;border:1px solid var(--line);background:var(--panel);color:var(--ink);
padding:10px 12px;font-family:inherit;font-size:.9rem}
input[type=file]{color:var(--muted);font-size:.8rem;width:100%;margin-bottom:8px}
.msg{border:1px solid #2a2a29;background:var(--panel);padding:9px 12px;margin-bottom:8px;font-size:.92rem}
.msg.u{border-left:2px solid var(--muted)}.msg .who{color:var(--muted);font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;margin-bottom:3px}
.src{color:var(--muted);font-size:.75rem;margin-top:4px}
.note{border:1px solid var(--line);background:var(--panel);padding:12px 14px;color:var(--muted);font-size:.83rem;margin-top:16px}
code{background:#161616;padding:2px 6px;border:1px solid var(--line);font-size:.82rem;word-break:break-all}
.row{display:flex;gap:8px;margin-top:8px}.del{padding:4px 9px;font-size:.75rem}
</style></head><body><div class="wrap">
<div class="brand">Delphi</div>
<div class="tagline">A voice-first knowledge assistant. Vapi speaks; this backend keeps answers grounded in your documents. Manage the knowledge base and try it in text below.</div>
<div class="rule"></div>
<div class="grid">
  <div>
    <div class="label">Add to knowledge base</div>
    <input type="file" id="file" accept=".pdf,.txt,.md,.docx" multiple>
    <button onclick="upload()">Upload</button>
    <div id="upstatus" class="meta" style="margin-top:8px"></div>
    <div class="label" style="margin-top:20px">Documents</div>
    <div id="docs"></div>
  </div>
  <div>
    <div class="label">Ask (text)</div>
    <div id="chat"></div>
    <div class="row"><input type="text" id="q" placeholder="Ask a question about your documents" onkeydown="if(event.key==='Enter')ask()">
    <button onclick="ask()">Send</button></div>
    <input type="text" id="secret" placeholder="Backend secret (only if you set one)" style="margin-top:8px;font-size:.8rem">
  </div>
</div>
<div class="note">Voice: create a Vapi assistant with model <b>Custom LLM</b> pointing at
<code id="ep"></code>, then call it from the Vapi dashboard or embed the Vapi widget.</div>
</div>
<script>
const $=id=>document.getElementById(id);
$("ep").textContent=location.origin+"/chat/completions";
function hdr(){const s=$("secret").value.trim();return s?{"Authorization":"Bearer "+s}:{};}
async function loadDocs(){const r=await fetch("/documents");const d=(await r.json()).documents||{};
 $("docs").innerHTML=Object.keys(d).length?"":'<div class="box meta">Nothing indexed yet.</div>';
 for(const[n,c]of Object.entries(d)){const el=document.createElement("div");el.className="box";
 el.innerHTML=`<div style="display:flex;justify-content:space-between;align-items:center">
 <span>${n}<br><span class="meta">${c} chunks</span></span>
 <button class="del" onclick="del('${n.replace(/'/g,"\\'")}')">Delete</button></div>`;$("docs").appendChild(el);}}
async function upload(){const f=$("file").files;if(!f.length)return;$("upstatus").textContent="Uploading…";
 for(const file of f){const fd=new FormData();fd.append("file",file);
 const r=await fetch("/documents",{method:"POST",body:fd});
 const j=await r.json();$("upstatus").textContent=r.ok?`Added ${file.name} (${j.chunks} chunks)`:("Error: "+(j.detail||r.status));}
 $("file").value="";loadDocs();}
async function del(n){await fetch("/documents/"+encodeURIComponent(n),{method:"DELETE"});loadDocs();}
function add(who,text,src){const el=document.createElement("div");el.className="msg "+(who=="You"?"u":"");
 el.innerHTML=`<div class="who">${who}</div>${text}`+(src&&src.length?`<div class="src">Sources: ${src.join(", ")}</div>`:"");
 $("chat").appendChild(el);el.scrollIntoView();}
async function ask(){const q=$("q").value.trim();if(!q)return;$("q").value="";add("You",q);
 add("Delphi","…");const last=$("chat").lastChild;
 try{const r=await fetch("/chat/completions",{method:"POST",headers:{"Content-Type":"application/json",...hdr()},
 body:JSON.stringify({messages:[{role:"user",content:q}]})});const j=await r.json();
 if(!r.ok){last.innerHTML='<div class="who">Delphi</div>Error: '+(j.detail||r.status);return;}
 last.innerHTML=`<div class="who">Delphi</div>${j.choices[0].message.content}`+
 (j.sources&&j.sources.length?`<div class="src">Sources: ${j.sources.join(", ")}</div>`:"");}
 catch(e){last.innerHTML='<div class="who">Delphi</div>Error: '+e;}}
loadDocs();
</script></body></html>"""


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    return INDEX_HTML


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
