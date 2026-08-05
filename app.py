"""Delphi - companion UI (thin client of the backend).

This app holds no keys and runs no RAG itself. It talks to the deployed backend
(server.py on Render) over HTTP: it lists / uploads / deletes documents, sends
chat questions, and launches the Vapi voice widget. Because the backend is the
single source of truth, a document added here is the same one the voice agent
answers from.

Point it at your backend by setting BACKEND_URL (in secrets or the sidebar).
"""

from __future__ import annotations

import os

import httpx
import streamlit as st
import streamlit.components.v1 as components

st.set_page_config(page_title="Delphi", page_icon="◼", layout="wide")

st.markdown(
    """
    <style>
      :root { --bg:#191919; --panel:#202020; --ink:#e6e6e4; --muted:#979692; --line:#333230; --line-soft:#2a2a29; }
      .stApp { background: var(--bg); color: var(--ink); }
      #MainMenu, footer { visibility: hidden; }
      [data-testid="stHeader"] { background: transparent; }
      .block-container { padding-top: 2.6rem; max-width: 1080px; }
      html, body, [class*="css"], input, textarea, button {
        font-family: ui-sans-serif, -apple-system, "Segoe UI", Inter, Helvetica, Arial, sans-serif;
      }
      .stButton button, [data-testid="stChatInput"], [data-baseweb="input"], input, textarea,
      [data-testid="stExpander"], [data-testid="stFileUploaderDropzone"],
      [data-baseweb="tab-list"], [data-baseweb="tab"], .stAlert { border-radius: 0 !important; }
      .brand { font-size: 2.3rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1; }
      .tagline { color: var(--muted); font-size: 0.95rem; margin-top: 0.5rem; max-width: 64ch; }
      .rule { border-bottom: 1px solid var(--line); margin: 1.1rem 0 0.4rem; }
      .label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.13em; color: var(--muted);
               font-weight: 600; margin: 0.2rem 0 0.5rem; }
      [data-testid="stChatMessage"] { border: 1px solid var(--line-soft); background: var(--panel);
                                       padding: 0.6rem 0.9rem; margin-bottom: 0.5rem; }
      .doc-row { border: 1px solid var(--line); background: var(--panel); padding: 8px 11px; margin-bottom: 6px;
                 font-size: 0.85rem; line-height: 1.4; }
      .doc-row .name { color: var(--ink); font-weight: 500; word-break: break-all; }
      .doc-row .meta { color: var(--muted); font-size: 0.78rem; }
      .notebox { border: 1px solid var(--line); background: var(--panel); padding: 14px 16px; color: var(--muted);
                 font-size: 0.9rem; line-height: 1.6; }
      .metaline { border: 1px solid var(--line-soft); background: var(--panel); padding: 7px 10px; margin-top: 6px;
                  color: var(--muted); font-size: 0.78rem; }
      .stButton button { border: 1px solid var(--line); background: var(--panel); color: var(--ink); font-weight: 500; }
      .stButton button:hover { border-color: var(--muted); color: #fff; background: #262625; }
      [data-testid="stSidebar"] { background: #1c1c1b; border-right: 1px solid var(--line); }
      [data-baseweb="tab-list"] { border-bottom: 1px solid var(--line); gap: 0; }
    </style>
    """,
    unsafe_allow_html=True,
)


# --------------------------------------------------------------------------
# Backend configuration
# --------------------------------------------------------------------------


def resolve_backend() -> tuple[str, str]:
    url = st.session_state.get("backend_url")
    if not url:
        try:
            url = st.secrets.get("BACKEND_URL")
        except Exception:
            url = None
        url = url or os.environ.get("BACKEND_URL", "")
    secret = st.session_state.get("backend_secret") or os.environ.get("BACKEND_SECRET", "")
    try:
        secret = secret or st.secrets.get("BACKEND_SECRET", "")
    except Exception:
        pass
    return (url or "").rstrip("/"), secret


def api(method: str, path: str, backend: str, secret: str, **kwargs) -> httpx.Response:
    headers = kwargs.pop("headers", {})
    if secret:
        headers["Authorization"] = f"Bearer {secret}"
    # Generous timeout: Render free tier can cold-start slowly.
    return httpx.request(method, f"{backend}{path}", headers=headers, timeout=90, **kwargs)


# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------

st.markdown('<div class="brand">Delphi</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">A voice-first knowledge assistant. Vapi listens and speaks; a retrieval-augmented '
    'backend keeps every answer grounded in your documents. This page manages that backend and lets you try it '
    'in text before you talk to it.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

backend, secret = resolve_backend()

if not backend:
    with st.sidebar:
        st.markdown('<div class="label">Backend</div>', unsafe_allow_html=True)
        st.text_input("Backend URL", key="backend_url", placeholder="https://delphi-backend.onrender.com")
        st.text_input("Backend secret (optional)", key="backend_secret", type="password")
    st.markdown(
        '<div class="notebox">Enter the URL of your deployed backend in the sidebar to begin '
        '(the Render service from <code>render.yaml</code>). Everything runs there; this page is just a client.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

# Confirm the backend is reachable.
try:
    health = api("GET", "/health", backend, secret).json()
except Exception as exc:  # noqa: BLE001
    st.error(f"Could not reach the backend at {backend}. It may be starting up (Render free tier cold start can take up to a minute). Details: {exc}")
    st.stop()

if not health.get("keys_configured"):
    st.warning("The backend is reachable but its GROQ_API_KEY / GEMINI_API_KEY are not set. Add them in the Render dashboard.")


# --------------------------------------------------------------------------
# Sidebar: document manager (over HTTP)
# --------------------------------------------------------------------------


def load_docs() -> dict:
    try:
        return api("GET", "/documents", backend, secret).json().get("documents", {})
    except Exception:  # noqa: BLE001
        return {}


st.session_state.setdefault("uploader_round", 0)
st.session_state.setdefault("history", [])

with st.sidebar:
    st.markdown('<div class="brand" style="font-size:1.4rem;">Delphi</div>', unsafe_allow_html=True)
    st.caption(f"Backend: {backend}")
    st.markdown('<div class="label" style="margin-top:0.9rem;">Add to knowledge base</div>', unsafe_allow_html=True)

    round_key = st.session_state["uploader_round"]
    uploads = st.file_uploader(
        "PDF, TXT, MD or DOCX", type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True, key=f"uploader_{round_key}", label_visibility="collapsed",
    )
    if uploads:
        existing = load_docs()
        for f in uploads:
            if f.name in existing:
                continue
            try:
                with st.spinner(f"Uploading {f.name}…"):
                    r = api("POST", "/documents", backend, secret, files={"file": (f.name, f.getvalue())})
                if r.status_code == 200:
                    st.success(f"Added {f.name} ({r.json().get('chunks', '?')} chunks)")
                else:
                    st.error(f"{f.name}: {r.json().get('detail', r.text)}")
            except Exception as exc:  # noqa: BLE001
                st.error(f"{f.name}: {exc}")
        st.session_state["uploader_round"] += 1
        st.rerun()

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    docs = load_docs()
    st.markdown(f'<div class="label">Documents · {len(docs)} files</div>', unsafe_allow_html=True)
    if not docs:
        st.markdown('<div class="doc-row meta">Nothing indexed yet.</div>', unsafe_allow_html=True)
    for name, count in sorted(docs.items()):
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f'<div class="doc-row"><div class="name">{name}</div><div class="meta">{count} chunks</div></div>', unsafe_allow_html=True)
        with col_b:
            if st.button("Delete", key=f"del_{name}", use_container_width=True):
                api("DELETE", f"/documents/{name}", backend, secret)
                st.rerun()

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="doc-row meta">Built by Shuja Jamal · Voice Vapi · Backend on Render · LLM Groq · Embeddings Gemini · Vector store Chroma</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Tabs
# --------------------------------------------------------------------------

chat_tab, voice_tab = st.tabs(["Chat", "Voice"])

with chat_tab:
    if not load_docs():
        st.markdown('<div class="notebox">Upload a document in the sidebar, then ask a question. Answers come only from what you add.</div>', unsafe_allow_html=True)

    for rec in st.session_state["history"]:
        with st.chat_message("user"):
            st.markdown(rec["q"])
        with st.chat_message("assistant"):
            st.markdown(rec["a"])
            if rec.get("sources"):
                st.markdown(f'<div class="metaline">Sources: {", ".join(rec["sources"])}</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask a question about your documents")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Asking the backend…"):
                    r = api("POST", "/chat/completions", backend, secret,
                            json={"messages": [{"role": "user", "content": question}]})
                data = r.json()
                answer = data["choices"][0]["message"]["content"]
                sources = data.get("sources", [])
            except Exception as exc:  # noqa: BLE001
                st.error(f"Backend error: {exc}")
                st.stop()
            st.markdown(answer)
            if sources:
                st.markdown(f'<div class="metaline">Sources: {", ".join(sources)}</div>', unsafe_allow_html=True)
        st.session_state["history"].append({"q": question, "a": answer, "sources": sources})
        st.rerun()

with voice_tab:
    st.markdown('<div class="label">Talk to Delphi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="notebox">The voice loop runs through <b>Vapi</b>. In the Vapi dashboard, create an assistant '
        'whose model is <b>Custom LLM</b> pointing at <code>' + backend + '/chat/completions</code>, then paste your '
        'Vapi <b>public key</b> and <b>assistant ID</b> below. Delphi will answer aloud from this same knowledge base.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    c1, c2 = st.columns(2)
    public_key = c1.text_input("Vapi public key", type="password")
    assistant_id = c2.text_input("Vapi assistant ID")

    if public_key and assistant_id:
        widget = f"""
        <div style="display:flex;justify-content:center;padding:24px 0;">
          <button id="call" style="background:#202020;color:#e6e6e4;border:1px solid #333230;
            padding:14px 26px;font-size:15px;cursor:pointer;font-family:sans-serif;letter-spacing:.02em;">
            Start voice call
          </button>
          <span id="status" style="color:#979692;margin-left:16px;align-self:center;font-family:sans-serif;font-size:13px;"></span>
        </div>
        <script type="module">
          import Vapi from "https://cdn.jsdelivr.net/npm/@vapi-ai/web@latest/dist/index.js";
          const vapi = new Vapi("{public_key}");
          const btn = document.getElementById("call");
          const status = document.getElementById("status");
          let live = false;
          btn.onclick = () => {{ if (live) {{ vapi.stop(); }} else {{ vapi.start("{assistant_id}"); }} }};
          vapi.on("call-start", () => {{ live = true; btn.textContent = "End call"; status.textContent = "Connected, speak now."; }});
          vapi.on("call-end", () => {{ live = false; btn.textContent = "Start voice call"; status.textContent = "Call ended."; }});
          vapi.on("error", (e) => {{ status.textContent = "Error: " + (e?.message || e); }});
        </script>
        """
        components.html(widget, height=140)
        st.caption("If the microphone does not activate inside this embed, open the widget on a standalone page (see README), since some browsers restrict mic access in iframes.")
    else:
        st.markdown('<div class="doc-row meta">Enter your Vapi public key and assistant ID to launch the voice widget.</div>', unsafe_allow_html=True)
