"""
Conversational Agent (Phase 1)
- Parses free-text requirement into structured intent/details.
- Uses OpenAI if OPENAI_API_KEY is present, otherwise falls back to
  a deterministic rule-based parser.
"""
import os
import re
from typing import Dict, Any

OPENAI_KEY = os.getenv("OPENAI_API_KEY")


class ConversationalAgent:
    def __init__(self):
        self.history = []

    def parse_requirement(self, text: str) -> Dict[str, Any]:
        """Return structured dict: { intent: [...], details: {...} }"""
        self.history.append({"user": text})

        # Try using OpenAI if configured (keeps call minimal and robust)
        if OPENAI_KEY:
            try:
                from openai import OpenAI
                client = OpenAI(api_key=OPENAI_KEY)
                prompt = self._build_prompt(text)
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.0,
                )
                raw = resp.choices[0].message.content
                import json
                try:
                    parsed = json.loads(raw)
                    return parsed
                except Exception:
                    # fall back to deterministic parsing below
                    pass
            except Exception:
                # ignore any OpenAI runtime errors and fallback
                pass

        # Deterministic heuristics (safe fallback)
        intents = []
        text_l = (text or "").lower()
        if any(k in text_l for k in ["ui", "frontend", "login", "page", "click", "type"]):
            intents.append("ui_testing")
        if any(k in text_l for k in ["api", "endpoint", "json", "rest", "http"]):
            intents.append("api_testing")
        if any(k in text_l for k in ["perf", "performance", "load", "latency", "benchmark"]):
            intents.append("performance_testing")

        # URL extraction (first http(s) token)
        url = None
        m = re.search(r"(https?://\S+)", text)
        if m:
            url = m.group(1).rstrip(".,;")  # strip trailing punctuation

        return {"intent": intents or ["ui_testing"], "details": {"url": url, "raw_text": text}}

    def _build_prompt(self, text: str) -> str:
        return f"Parse the following QA requirement into a JSON with keys: intent (list), details (object). Requirement: {text}"
