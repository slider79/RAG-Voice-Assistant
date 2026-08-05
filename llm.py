"""LLM Response Generator: answer from retrieved context, and judge faithfulness.

Two Groq calls live here:
  answer()   - generate a grounded answer from the retrieved chunks.
  judge()    - score how faithful that answer is to the chunks (0..1), for the
               performance monitor. Optional, since it costs a second call.
"""

from __future__ import annotations

import json
import os
import re

LLM_MODEL = os.environ.get("LLM_MODEL", "llama-3.3-70b-versatile")

ANSWER_SYSTEM = (
    "You are a careful assistant that answers strictly from the provided context. "
    "Rules:\n"
    "- Use only the context below. Do not use outside knowledge.\n"
    "- If the context does not contain the answer, say so plainly rather than guessing.\n"
    "- Cite the source filenames you used in square brackets, e.g. [report.pdf].\n"
    "- Be concise and directly answer the question."
)

JUDGE_SYSTEM = (
    "You are a strict evaluator. Given a CONTEXT and an ANSWER, decide how much of "
    "the answer is supported by the context. Reply with only a JSON object: "
    '{"faithfulness": <0.0-1.0>, "supported": true|false, "note": "<short reason>"}. '
    "faithfulness is the fraction of the answer's claims backed by the context. "
    "supported is false if the answer introduces facts not in the context."
)


class LLMError(RuntimeError):
    pass


def _redact(text: str) -> str:
    for var in ("GROQ_API_KEY", "GEMINI_API_KEY"):
        value = os.environ.get(var)
        if value and len(value) >= 8:
            text = text.replace(value, "[redacted]")
    return text


def _format_context(chunks: list[dict]) -> str:
    blocks = []
    for c in chunks:
        blocks.append(f"[{c['source']}]\n{c['text']}")
    return "\n\n---\n\n".join(blocks)


class Generator:
    def __init__(self, api_key: str | None = None, model: str | None = None, client=None):
        self.model = model or LLM_MODEL
        if client is not None:
            self.client = client
            return
        try:
            from groq import Groq

            self.client = Groq(api_key=api_key or os.environ.get("GROQ_API_KEY"))
        except ImportError as exc:  # pragma: no cover
            raise LLMError("groq is not installed. Run: pip install groq") from exc

    def answer(self, question: str, chunks: list[dict]) -> str:
        if not chunks:
            return "I don't have any documents to answer from yet. Upload something first."
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=self._messages(question, chunks), temperature=0.2
            )
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Answer generation failed: {_redact(str(exc))}") from None
        return (response.choices[0].message.content or "").strip()

    def _messages(self, question: str, chunks: list[dict]) -> list[dict]:
        return [
            {"role": "system", "content": ANSWER_SYSTEM},
            {"role": "user", "content": f"CONTEXT:\n{_format_context(chunks)}\n\nQUESTION: {question}"},
        ]

    def answer_stream(self, question: str, chunks: list[dict]):
        """Yield the answer token by token, for the Vapi-facing streaming endpoint.

        Streaming matters here: Vapi begins speaking as soon as the first tokens
        arrive, so the caller hears a reply sooner instead of waiting for the
        whole answer."""
        if not chunks:
            yield "I don't have any documents to answer from yet. Please add some first."
            return
        try:
            stream = self.client.chat.completions.create(
                model=self.model, messages=self._messages(question, chunks), temperature=0.2, stream=True
            )
            for part in stream:
                token = part.choices[0].delta.content or ""
                if token:
                    yield token
        except Exception as exc:  # noqa: BLE001
            raise LLMError(f"Answer generation failed: {_redact(str(exc))}") from None

    def judge(self, answer: str, chunks: list[dict]) -> dict:
        """Score how faithful an answer is to its context. Never raises: on any
        problem it returns a neutral, clearly-marked result so the chat is not
        blocked by the evaluation step."""
        context = _format_context(chunks)
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": f"CONTEXT:\n{context}\n\nANSWER:\n{answer}"},
                ],
                temperature=0.0,
            )
            raw = (response.choices[0].message.content or "").strip()
            match = re.search(r"\{.*\}", raw, re.S)
            data = json.loads(match.group(0) if match else raw)
            return {
                "faithfulness": float(data.get("faithfulness", 0.0)),
                "supported": bool(data.get("supported", False)),
                "note": str(data.get("note", "")),
            }
        except Exception:  # noqa: BLE001
            return {"faithfulness": None, "supported": None, "note": "evaluation unavailable"}
