"""Delphi - Streamlit companion UI for the voice RAG assistant.

This is the optional text-and-management surface: upload the documents the
assistant answers from, try the RAG in text form, and launch the Vapi voice
widget. The voice loop itself runs through Vapi talking to server.py; this app
shares the same vector store, so a document added here is available to voice.
"""

from __future__ import annotations

import os

import streamlit as st
import streamlit.components.v1 as components

from document_loader import DocumentError
from embeddings import Embedder, EmbeddingError
from llm import Generator
from rag import RAGPipeline
from vector_store import VectorStore, VectorStoreError

st.set_page_config(page_title="Delphi", page_icon="◼", layout="wide")

# --------------------------------------------------------------------------
# Styling: dark, boxy, Notion-like.
# --------------------------------------------------------------------------

st.markdown(
    """
    <style>
      :root {
        --bg:#191919; --panel:#202020; --ink:#e6e6e4; --muted:#979692;
        --line:#333230; --line-soft:#2a2a29;
      }
      .stApp { background: var(--bg); color: var(--ink); }
      #MainMenu, footer { visibility: hidden; }
      [data-testid="stHeader"] { background: transparent; }
      .block-container { padding-top: 2.6rem; max-width: 1080px; }
      html, body, [class*="css"], input, textarea, button {
        font-family: ui-sans-serif, -apple-system, "Segoe UI", Inter, Helvetica, Arial, sans-serif;
      }
      .stButton button, [data-testid="stChatInput"], [data-baseweb="input"], input, textarea,
      [data-testid="stExpander"], [data-testid="stFileUploaderDropzone"],
      [data-baseweb="tab-list"], [data-baseweb="tab"], .stAlert {
        border-radius: 0 !important;
      }
      .brand { font-size: 2.3rem; font-weight: 700; letter-spacing: -0.02em; line-height: 1; }
      .tagline { color: var(--muted); font-size: 0.95rem; margin-top: 0.5rem; max-width: 64ch; }
      .rule { border-bottom: 1px solid var(--line); margin: 1.1rem 0 0.4rem; }
      .label { font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.13em;
               color: var(--muted); font-weight: 600; margin: 0.2rem 0 0.5rem; }
      [data-testid="stExpander"] { border: 1px solid var(--line); background: var(--panel); }
      [data-testid="stChatMessage"] { border: 1px solid var(--line-soft); background: var(--panel);
                                       padding: 0.6rem 0.9rem; margin-bottom: 0.5rem; }
      .doc-row { border: 1px solid var(--line); background: var(--panel); padding: 8px 11px;
                 margin-bottom: 6px; font-size: 0.85rem; line-height: 1.4; }
      .doc-row .name { color: var(--ink); font-weight: 500; word-break: break-all; }
      .doc-row .meta { color: var(--muted); font-size: 0.78rem; }
      .notebox { border: 1px solid var(--line); background: var(--panel); padding: 14px 16px;
                 color: var(--muted); font-size: 0.9rem; line-height: 1.6; }
      .metaline { border: 1px solid var(--line-soft); background: var(--panel); padding: 7px 10px;
                  margin-top: 6px; color: var(--muted); font-size: 0.78rem; }
      .stButton button { border: 1px solid var(--line); background: var(--panel); color: var(--ink); font-weight: 500; }
      .stButton button:hover { border-color: var(--muted); color: #fff; background: #262625; }
      [data-testid="stSidebar"] { background: #1c1c1b; border-right: 1px solid var(--line); }
      [data-baseweb="tab-list"] { border-bottom: 1px solid var(--line); gap: 0; }
      ol, ul { color: var(--ink); }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Keys
# --------------------------------------------------------------------------

KEY_SPECS = {
    "GROQ_API_KEY": {"state": "groq_key", "widget": "groq_in", "label": "Groq API key (LLM)", "help": "console.groq.com/keys"},
    "GEMINI_API_KEY": {"state": "gemini_key", "widget": "gemini_in", "label": "Gemini API key (embeddings)", "help": "aistudio.google.com/apikey"},
}


def resolve_key(env_name: str) -> str | None:
    spec = KEY_SPECS[env_name]
    if st.session_state.get(spec["state"]):
        return st.session_state[spec["state"]]
    try:
        secret = st.secrets.get(env_name)
        if secret:
            return str(secret)
    except Exception:
        pass
    return os.environ.get(env_name)


def key_input(env_name: str) -> None:
    spec = KEY_SPECS[env_name]

    def _remember() -> None:
        typed = (st.session_state.get(spec["widget"]) or "").strip()
        if typed:
            st.session_state[spec["state"]] = typed

    st.text_input(spec["label"], type="password", key=spec["widget"], on_change=_remember)
    st.caption(spec["help"])
    _remember()


@st.cache_resource(show_spinner=False)
def build_pipeline(groq_key: str, gemini_key: str) -> RAGPipeline:
    return RAGPipeline(Embedder(api_key=gemini_key), VectorStore(), Generator(api_key=groq_key))


# --------------------------------------------------------------------------
# Header + gate
# --------------------------------------------------------------------------

st.markdown('<div class="brand">Delphi</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="tagline">A voice-first knowledge assistant. Vapi listens and speaks; a '
    'retrieval-augmented backend keeps every answer grounded in your own documents. Manage the '
    'knowledge base here, try it in text, then talk to it.</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="rule"></div>', unsafe_allow_html=True)

groq_key = resolve_key("GROQ_API_KEY")
gemini_key = resolve_key("GEMINI_API_KEY")

if not groq_key or not gemini_key:
    with st.sidebar:
        st.markdown('<div class="label">API keys</div>', unsafe_allow_html=True)
        for env_name in KEY_SPECS:
            if not resolve_key(env_name):
                key_input(env_name)
    st.markdown(
        '<div class="notebox">Add your Groq and Gemini API keys in the sidebar to begin.</div>',
        unsafe_allow_html=True,
    )
    st.stop()

try:
    pipeline = build_pipeline(groq_key, gemini_key)
except (EmbeddingError, VectorStoreError) as exc:
    st.error(str(exc))
    st.stop()

st.session_state.setdefault("history", [])
st.session_state.setdefault("uploader_round", 0)

# --------------------------------------------------------------------------
# Sidebar: document manager
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown('<div class="brand" style="font-size:1.4rem;">Delphi</div>', unsafe_allow_html=True)
    st.markdown('<div class="label" style="margin-top:0.9rem;">Add to knowledge base</div>', unsafe_allow_html=True)

    round_key = st.session_state["uploader_round"]
    uploads = st.file_uploader(
        "PDF, TXT, MD or DOCX", type=["pdf", "txt", "md", "docx"],
        accept_multiple_files=True, key=f"uploader_{round_key}", label_visibility="collapsed",
    )
    if uploads:
        existing = pipeline.sources()
        for f in uploads:
            if f.name in existing:
                continue
            try:
                with st.spinner(f"Indexing {f.name}…"):
                    report = pipeline.ingest(f.name, f.getvalue())
                st.success(f"Added {f.name} ({report['chunks']} chunks)")
            except (DocumentError, EmbeddingError) as exc:
                st.error(f"{f.name}: {exc}")
        st.session_state["uploader_round"] += 1
        st.rerun()

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    sources = pipeline.sources()
    st.markdown(f'<div class="label">Documents · {len(sources)} files · {pipeline.total_chunks()} chunks</div>', unsafe_allow_html=True)
    if not sources:
        st.markdown('<div class="doc-row meta">Nothing indexed yet.</div>', unsafe_allow_html=True)
    for name, count in sorted(sources.items()):
        col_a, col_b = st.columns([3, 1])
        with col_a:
            st.markdown(f'<div class="doc-row"><div class="name">{name}</div><div class="meta">{count} chunks</div></div>', unsafe_allow_html=True)
        with col_b:
            if st.button("Delete", key=f"del_{name}", use_container_width=True):
                pipeline.delete(name)
                st.rerun()

    st.markdown('<div class="rule"></div>', unsafe_allow_html=True)
    st.markdown('<div class="doc-row meta">Built by Shuja Jamal · Voice Vapi · LLM Groq · Embeddings Gemini · Vector store Chroma</div>', unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Tabs: Text chat + Voice
# --------------------------------------------------------------------------

chat_tab, voice_tab = st.tabs(["Chat", "Voice"])

with chat_tab:
    if not pipeline.sources():
        st.markdown('<div class="notebox">Upload a document in the sidebar, then ask a question. Answers come only from what you add.</div>', unsafe_allow_html=True)

    for rec in st.session_state["history"]:
        with st.chat_message("user"):
            st.markdown(rec["question"])
        with st.chat_message("assistant"):
            st.markdown(rec["answer"])
            if rec.get("sources"):
                srcs = ", ".join(sorted({h["source"] for h in rec["sources"]}))
                st.markdown(f'<div class="metaline">Sources: {srcs}</div>', unsafe_allow_html=True)

    question = st.chat_input("Ask a question about your documents")
    if question:
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Retrieving and answering…"):
                    result = pipeline.answer(question, evaluate=False)
            except Exception as exc:  # noqa: BLE001
                st.error(str(exc))
                st.stop()
            st.markdown(result["answer"])
            if result["sources"]:
                srcs = ", ".join(sorted({h["source"] for h in result["sources"]}))
                st.markdown(f'<div class="metaline">Sources: {srcs}</div>', unsafe_allow_html=True)
        st.session_state["history"].append(result)
        st.rerun()

with voice_tab:
    st.markdown('<div class="label">Talk to Delphi</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="notebox">The voice loop runs through <b>Vapi</b>. Create an assistant in the Vapi '
        'dashboard, set its model to <b>Custom LLM</b> pointing at this project\'s <code>/chat/completions</code> '
        'endpoint (see the README), then paste your Vapi <b>public key</b> and <b>assistant ID</b> below to launch '
        'the voice widget. The assistant will answer out loud from the documents in this knowledge base.</div>',
        unsafe_allow_html=True,
    )
    st.write("")
    col1, col2 = st.columns(2)
    public_key = col1.text_input("Vapi public key", type="password")
    assistant_id = col2.text_input("Vapi assistant ID")

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
          btn.onclick = () => {{
            if (live) {{ vapi.stop(); }} else {{ vapi.start("{assistant_id}"); }}
          }};
          vapi.on("call-start", () => {{ live = true; btn.textContent = "End call"; status.textContent = "Connected, speak now."; }});
          vapi.on("call-end", () => {{ live = false; btn.textContent = "Start voice call"; status.textContent = "Call ended."; }});
          vapi.on("error", (e) => {{ status.textContent = "Error: " + (e?.message || e); }});
        </script>
        """
        components.html(widget, height=140)
        st.caption("If the microphone does not activate inside this embed, open the Vapi widget on a standalone page (see README), since some browsers restrict mic access in iframes.")
    else:
        st.markdown('<div class="doc-row meta">Enter your Vapi public key and assistant ID above to launch the voice widget.</div>', unsafe_allow_html=True)
