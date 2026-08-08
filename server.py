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
from fastapi.responses import HTMLResponse, Response, StreamingResponse

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
<link rel="icon" href="/favicon.svg" type="image/svg+xml">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=JetBrains+Mono:wght@400;600&display=swap" rel="stylesheet">
<style>
:root{--bg:#121214;--panel:#1a1a1d;--ink:#ececea;--muted:#8f8e93;--line:#2e2e33;
--user:#4fd1ff;--agent:#c77dff;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);line-height:1.55;
font-family:"Space Grotesk",ui-sans-serif,-apple-system,"Segoe UI",Arial,sans-serif;
background-image:radial-gradient(circle at 50% 0%,#1b1b21 0%,#121214 55%);}
.wrap{max-width:1000px;margin:0 auto;padding:34px 20px 80px}
.brand{font-size:2.6rem;font-weight:700;letter-spacing:.14em;text-transform:uppercase}
.tagline{color:var(--muted);font-size:.92rem;margin:.5rem 0 0;max-width:64ch}
.rule{border-bottom:1px solid var(--line);margin:20px 0}
.label{font-family:"JetBrains Mono",monospace;font-size:.68rem;text-transform:uppercase;
letter-spacing:.22em;color:var(--muted);font-weight:600;margin:0 0 10px}
.grid{display:grid;grid-template-columns:330px 1fr;gap:24px}
@media(max-width:820px){.grid{grid-template-columns:1fr}}
.box{border:1px solid var(--line);background:var(--panel);padding:9px 12px;margin-bottom:7px;font-size:.85rem}
.box .meta{color:var(--muted);font-size:.76rem;font-family:"JetBrains Mono",monospace}
button{border:1px solid var(--line);background:var(--panel);color:var(--ink);padding:9px 16px;
cursor:pointer;font-family:"JetBrains Mono",monospace;font-size:.78rem;letter-spacing:.08em;
text-transform:uppercase;transition:.18s}
button:hover{border-color:var(--muted);background:#242429}
input[type=text],input[type=password]{width:100%;border:1px solid var(--line);background:var(--panel);
color:var(--ink);padding:11px 13px;font-family:inherit;font-size:.9rem}
input[type=file]{color:var(--muted);font-size:.78rem;width:100%;margin-bottom:9px;font-family:"JetBrains Mono",monospace}
.msg{border:1px solid #262629;background:var(--panel);padding:10px 13px;margin-bottom:8px;font-size:.92rem}
.msg.u{border-left:2px solid var(--user)}
.msg .who{font-family:"JetBrains Mono",monospace;color:var(--muted);font-size:.66rem;
text-transform:uppercase;letter-spacing:.16em;margin-bottom:4px}
.src{color:var(--muted);font-size:.74rem;margin-top:5px;font-family:"JetBrains Mono",monospace}
.note{border:1px solid var(--line);background:var(--panel);padding:13px 15px;color:var(--muted);
font-size:.82rem;margin-top:18px}
code{background:#0e0e10;padding:2px 7px;border:1px solid var(--line);font-family:"JetBrains Mono",monospace;
font-size:.78rem;word-break:break-all}
.row{display:flex;gap:8px;margin-top:9px}.del{padding:5px 10px;font-size:.66rem}
.tabs{display:flex;gap:0;border-bottom:1px solid var(--line);margin-bottom:22px}
.tab{font-family:"JetBrains Mono",monospace;font-size:.74rem;letter-spacing:.16em;text-transform:uppercase;
padding:11px 20px;cursor:pointer;color:var(--muted);border-bottom:2px solid transparent;margin-bottom:-1px}
.tab.on{color:var(--ink);border-bottom-color:var(--ink)}
.pane{display:none}.pane.on{display:block}

/* ---- the orb ---- */
.stage{display:flex;flex-direction:column;align-items:center;justify-content:center;padding:34px 0 10px}
.orb{position:relative;width:210px;height:210px;border-radius:50%;cursor:pointer;
display:grid;place-items:center;transition:transform .3s}
.orb:active{transform:scale(.97)}
.orb .core{position:absolute;inset:26%;border-radius:50%;
background:radial-gradient(circle at 35% 30%,#3a3a42,#1c1c20 70%);
box-shadow:inset 0 0 30px rgba(0,0,0,.6);transition:background .5s,box-shadow .5s}
.orb .ring{position:absolute;inset:0;border-radius:50%;border:1px solid var(--line);
opacity:.55;transition:border-color .5s}
.orb .glow{position:absolute;inset:8%;border-radius:50%;opacity:0;transition:opacity .5s;
background:radial-gradient(circle,rgba(255,255,255,.16) 0%,transparent 62%)}
.orb .pulse{position:absolute;inset:0;border-radius:50%;border:1px solid transparent;opacity:0}

/* idle: slow breathing */
.orb.idle .core{animation:breathe 4.4s ease-in-out infinite}
@keyframes breathe{0%,100%{transform:scale(1)}50%{transform:scale(1.045)}}

/* live states */
.orb.live .glow{opacity:1}
.orb.live.user .core{background:radial-gradient(circle at 35% 30%,#8ee6ff,#0e6f92 72%);
box-shadow:0 0 46px 8px rgba(79,209,255,.42),inset 0 0 26px rgba(255,255,255,.22)}
.orb.live.user .ring{border-color:rgba(79,209,255,.55)}
.orb.live.agent .core{background:radial-gradient(circle at 35% 30%,#e2b6ff,#6c2bb3 72%);
box-shadow:0 0 52px 10px rgba(199,125,255,.46),inset 0 0 26px rgba(255,255,255,.22)}
.orb.live.agent .ring{border-color:rgba(199,125,255,.6)}
.orb.live.user .core{animation:pulseFast 1.15s ease-in-out infinite}
.orb.live.agent .core{animation:pulseFast .82s ease-in-out infinite}
@keyframes pulseFast{0%,100%{transform:scale(1)}50%{transform:scale(1.11)}}
/* expanding rings while someone speaks */
.orb.user .pulse,.orb.agent .pulse{animation:ripple 1.7s ease-out infinite}
.orb.user .pulse{border-color:rgba(79,209,255,.5)}
.orb.agent .pulse{border-color:rgba(199,125,255,.55)}
@keyframes ripple{0%{transform:scale(.72);opacity:.85}100%{transform:scale(1.35);opacity:0}}

.orbstatus{font-family:"JetBrains Mono",monospace;font-size:.74rem;letter-spacing:.2em;
text-transform:uppercase;color:var(--muted);margin-top:24px;min-height:1.2em;text-align:center}
.speaker{font-family:"JetBrains Mono",monospace;font-size:.7rem;letter-spacing:.2em;
text-transform:uppercase;margin-top:7px;min-height:1.1em;text-align:center;color:var(--muted)}
.speaker.user{color:var(--user)}.speaker.agent{color:var(--agent)}
.vfields{display:grid;grid-template-columns:1fr 1fr;gap:10px;max-width:620px;margin:26px auto 0}
@media(max-width:620px){.vfields{grid-template-columns:1fr}}
</style></head><body><div class="wrap">
<div class="brand">Delphi</div>
<div class="tagline">A voice-first knowledge assistant. Vapi listens and speaks; this backend keeps every answer grounded in your own documents.</div>
<div class="rule"></div>

<div class="tabs">
  <div class="tab on" data-p="voice" onclick="tab(this)">Voice</div>
  <div class="tab" data-p="text" onclick="tab(this)">Text</div>
  <div class="tab" data-p="docs" onclick="tab(this)">Documents</div>
</div>

<div class="pane on" id="p-voice">
  <div class="stage">
    <div class="orb idle" id="orb" onclick="toggleCall()">
      <div class="ring"></div><div class="pulse"></div>
      <div class="glow"></div><div class="core"></div>
    </div>
    <div class="orbstatus" id="orbstatus">Tap the orb to start</div>
    <div class="speaker" id="speaker"></div>
  </div>
  <div class="vfields" id="vfields">
    <input type="password" id="vkey" placeholder="Vapi public key">
    <input type="text" id="vassist" placeholder="Vapi assistant ID">
  </div>
  <div class="note" id="vnote">Create a Vapi assistant with model <b>Custom LLM</b> pointing at
  <code id="ep"></code>, then paste its public key and assistant ID above and tap the orb.
  The orb glows <b style="color:var(--user)">blue</b> while you speak and
  <b style="color:var(--agent)">violet</b> while Delphi answers.</div>
</div>

<div class="pane" id="p-text">
  <div class="label">Ask in text</div>
  <div id="chat"></div>
  <div class="row"><input type="text" id="q" placeholder="Ask a question about your documents" onkeydown="if(event.key==='Enter')ask()">
  <button onclick="ask()">Send</button></div>
  <input type="text" id="secret" placeholder="Backend secret (only if you set one)" style="margin-top:9px;font-size:.8rem">
</div>

<div class="pane" id="p-docs">
  <div class="grid">
    <div>
      <div class="label">Add to knowledge base</div>
      <input type="file" id="file" accept=".pdf,.txt,.md,.docx" multiple>
      <button onclick="upload()">Upload</button>
      <div id="upstatus" class="meta" style="margin-top:9px;font-family:'JetBrains Mono',monospace;font-size:.76rem;color:var(--muted)"></div>
    </div>
    <div>
      <div class="label">Documents</div>
      <div id="docs"></div>
    </div>
  </div>
</div>
</div>
<script type="module">
import Vapi from "https://esm.sh/@vapi-ai/web@2.3.9";

const $=id=>document.getElementById(id);
$("ep").textContent=location.origin+"/chat/completions";

/* Server-injected Vapi credentials. The public key is meant to be used in the
   browser, so shipping it here is by design; the private key never appears.
   When both are configured the input fields are hidden and the orb is one tap. */
const PRESET_KEY="__VAPI_PUBLIC_KEY__", PRESET_ID="__VAPI_ASSISTANT_ID__";
const PRESET_READY = PRESET_KEY && PRESET_ID;
if(PRESET_READY){
  $("vfields").style.display="none";
  $("vnote").innerHTML='Voice is wired to <code>'+location.origin+'/chat/completions</code>. '+
    'Tap the orb to talk. It glows <b style="color:var(--user)">blue</b> while you speak and '+
    '<b style="color:var(--agent)">violet</b> while Delphi answers.';
}

window.tab=el=>{document.querySelectorAll(".tab").forEach(t=>t.classList.remove("on"));
 document.querySelectorAll(".pane").forEach(p=>p.classList.remove("on"));
 el.classList.add("on");$("p-"+el.dataset.p).classList.add("on");};

function hdr(){const s=$("secret").value.trim();return s?{"Authorization":"Bearer "+s}:{};}

window.loadDocs=async function(){const r=await fetch("/documents");const d=(await r.json()).documents||{};
 $("docs").innerHTML=Object.keys(d).length?"":'<div class="box"><span class="meta">Nothing indexed yet.</span></div>';
 for(const[n,c]of Object.entries(d)){const el=document.createElement("div");el.className="box";
 el.innerHTML=`<div style="display:flex;justify-content:space-between;align-items:center;gap:10px">
 <span>${n}<br><span class="meta">${c} chunks</span></span></div>`;
 const b=document.createElement("button");b.className="del";b.textContent="Delete";
 b.onclick=()=>del(n);el.firstElementChild.appendChild(b);$("docs").appendChild(el);}};

window.upload=async function(){const f=$("file").files;if(!f.length)return;$("upstatus").textContent="Uploading...";
 for(const file of f){const fd=new FormData();fd.append("file",file);
 const r=await fetch("/documents",{method:"POST",body:fd});const j=await r.json();
 $("upstatus").textContent=r.ok?`Added ${file.name} (${j.chunks} chunks)`:("Error: "+(j.detail||r.status));}
 $("file").value="";loadDocs();};

window.del=async function(n){await fetch("/documents/"+encodeURIComponent(n),{method:"DELETE"});loadDocs();};

function add(who,text,src){const el=document.createElement("div");el.className="msg "+(who=="You"?"u":"");
 el.innerHTML=`<div class="who">${who}</div>${text}`+(src&&src.length?`<div class="src">Sources: ${src.join(", ")}</div>`:"");
 $("chat").appendChild(el);el.scrollIntoView({block:"nearest"});return el;}

window.ask=async function(){const q=$("q").value.trim();if(!q)return;$("q").value="";add("You",q);
 const last=add("Delphi","...");
 try{const r=await fetch("/chat/completions",{method:"POST",headers:{"Content-Type":"application/json",...hdr()},
 body:JSON.stringify({messages:[{role:"user",content:q}]})});const j=await r.json();
 if(!r.ok){last.innerHTML='<div class="who">Delphi</div>Error: '+(j.detail||r.status);return;}
 last.innerHTML=`<div class="who">Delphi</div>${j.choices[0].message.content}`+
 (j.sources&&j.sources.length?`<div class="src">Sources: ${j.sources.join(", ")}</div>`:"");}
 catch(e){last.innerHTML='<div class="who">Delphi</div>Error: '+e;}};

/* ---------------- voice orb ---------------- */
const orb=$("orb"),ostat=$("orbstatus"),spk=$("speaker");
let vapi=null,live=false;

function setState(s){
  orb.classList.remove("idle","live","user","agent");
  if(s==="idle"){orb.classList.add("idle");spk.textContent="";spk.className="speaker";return;}
  orb.classList.add("live");
  if(s==="user"){orb.classList.add("user");spk.textContent="You are speaking";spk.className="speaker user";}
  else if(s==="agent"){orb.classList.add("agent");spk.textContent="Delphi is speaking";spk.className="speaker agent";}
  else{spk.textContent="Listening";spk.className="speaker";}
}

window.toggleCall=function(){
  const key=PRESET_KEY||$("vkey").value.trim(), id=PRESET_ID||$("vassist").value.trim();
  if(live){vapi&&vapi.stop();return;}
  if(!key||!id){ostat.textContent="Enter your Vapi key and assistant ID";return;}
  try{
    if(!vapi){
      vapi=new Vapi(key);
      vapi.on("call-start",()=>{live=true;ostat.textContent="Connected \u2014 tap to end";setState("live");});
      vapi.on("call-end",()=>{live=false;ostat.textContent="Call ended \u2014 tap to start";setState("idle");});
      vapi.on("speech-start",()=>setState("agent"));   // assistant began speaking
      vapi.on("speech-end",()=>setState("live"));      // assistant stopped
      vapi.on("volume-level",v=>{                       // user's mic level
        if(!live)return;
        if(orb.classList.contains("agent"))return;      // assistant has priority
        setState(v>0.045?"user":"live");
      });
      vapi.on("error",e=>{ostat.textContent="Error: "+((e&&e.message)||e);setState("idle");live=false;});
    }
    ostat.textContent="Connecting...";
    vapi.start(id);
  }catch(e){ostat.textContent="Error: "+e;}
};

loadDocs();
</script></body></html>"""


def _js_safe(value: str) -> str:
    """Escape a value for embedding inside a JavaScript string literal."""
    return (
        value.replace("\\", "\\\\").replace('"', '\\"').replace("<", "\\u003c").replace("\n", "")
    )


@app.get("/", response_class=HTMLResponse)
def index() -> str:
    """Serve the UI, injecting the Vapi credentials when they are configured.

    VAPI_PUBLIC_KEY is a browser-side key by design (it is what the Vapi web SDK
    expects), so embedding it in the page is intended usage, not a leak. The
    private/server key is never referenced here. With both values set, the key
    fields disappear and the orb starts a call in one tap.
    """
    return INDEX_HTML.replace(
        "__VAPI_PUBLIC_KEY__", _js_safe(os.environ.get("VAPI_PUBLIC_KEY", ""))
    ).replace("__VAPI_ASSISTANT_ID__", _js_safe(os.environ.get("VAPI_ASSISTANT_ID", "")))


# A glowing orb favicon, drawn as SVG so it needs no binary asset.
FAVICON_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 64 64">
  <defs>
    <radialGradient id="g" cx="38%" cy="32%">
      <stop offset="0%" stop-color="#e2b6ff"/>
      <stop offset="55%" stop-color="#8b3fe0"/>
      <stop offset="100%" stop-color="#4a1580"/>
    </radialGradient>
    <filter id="b" x="-60%" y="-60%" width="220%" height="220%">
      <feGaussianBlur stdDeviation="4"/>
    </filter>
  </defs>
  <rect width="64" height="64" rx="12" fill="#121214"/>
  <circle cx="32" cy="32" r="17" fill="#c77dff" opacity=".45" filter="url(#b)"/>
  <circle cx="32" cy="32" r="14" fill="url(#g)"/>
  <circle cx="32" cy="32" r="21" fill="none" stroke="#c77dff" stroke-opacity=".5"/>
</svg>"""


@app.get("/favicon.svg")
def favicon() -> Response:
    return Response(content=FAVICON_SVG, media_type="image/svg+xml")


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
