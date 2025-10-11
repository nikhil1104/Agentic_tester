# modules/conversational_agent.py
"""
Conversational Agent (Phase 1 — Hardened & Schema-Oriented)
-----------------------------------------------------------
Parses free-text requirements into a structured plan seed.

Key traits:
- Backward-compatible output shape:
    { "intent": [...], "details": {"url": <str>|None, "raw_text": <str>, ...} }
- Optional LLM assist (OpenAI) with strict JSON-only prompting and timeouts.
- Deterministic heuristics for robust fallback & unit-testability.
- Extracts useful signals: endpoints, performance targets, browsers/devices, auth hints.
- Adds confidence scoring and provenance.

Env vars:
- AGENT_USE_LLM=true|false           (default false)
- OPENAI_API_KEY=...                 (optional; required if AGENT_USE_LLM=true)
- AGENT_MODEL=gpt-4o-mini            (default)
- AGENT_TIMEOUT_S=12                 (default)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# -------------------------
# Configuration (env-driven)
# -------------------------
AGENT_USE_LLM = os.getenv("AGENT_USE_LLM", "false").strip().lower() == "true"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")  # optional, only used when AGENT_USE_LLM=true
AGENT_MODEL = os.getenv("AGENT_MODEL", "gpt-4o-mini")
AGENT_TIMEOUT_S = float(os.getenv("AGENT_TIMEOUT_S", "12"))

# -------------------------
# Small bounded memory
# -------------------------
MAX_MEM_TURNS = 20


@dataclass
class ParseResult:
    """
    Internal normalized representation (we return dict at the end for compatibility).
    """
    intent: List[str]
    url: Optional[str]
    details: Dict[str, Any]
    confidence: float
    provenance: str  # "llm" | "heuristic" | "mixed"


class ConversationalAgent:
    """
    Conversational agent that turns unstructured text into a structured QA ask.
    - Uses LLM when allowed, with strict JSON-output prompt.
    - Always falls back to deterministic heuristics.
    """

    def __init__(self) -> None:
        self.history: List[Dict[str, str]] = []

    # -------------------------
    # Public API
    # -------------------------
    def parse_requirement(self, text: str) -> Dict[str, Any]:
        """
        Return a dict with at least:
          {
            "intent": [...],                 # e.g., ["ui_testing", "api_testing"]
            "details": {
                "url": <str|None>,
                "raw_text": <original>,
                ...optional enriched fields...
            }
          }
        """
        # Memory (bounded)
        self._remember({"role": "user", "content": text})

        llm_parsed: Optional[Dict[str, Any]] = None
        if AGENT_USE_LLM and OPENAI_KEY:
            llm_parsed = self._try_llm_parse(text)

        # Heuristic baseline (always)
        heur = self._heuristic_parse(text)

        # If LLM present, merge conservatively
        if llm_parsed:
            merged = self._merge(llm_parsed, heur)
            result = ParseResult(
                intent=merged.get("intent", []) or ["ui_testing"],
                url=(merged.get("details") or {}).get("url"),
                details=merged.get("details", {}),
                confidence=merged.get("confidence", 0.75),
                provenance="mixed",
            )
        else:
            result = ParseResult(
                intent=heur.get("intent", []) or ["ui_testing"],
                url=(heur.get("details") or {}).get("url"),
                details=heur.get("details", {}),
                confidence=0.6,
                provenance="heuristic",
            )

        # Backward-compatible shape + enrich
        out: Dict[str, Any] = {
            "intent": result.intent,
            "details": {
                "url": result.url,
                "raw_text": text,
                **{k: v for k, v in result.details.items() if k != "url"},
                "_agent": {
                    "confidence": result.confidence,
                    "provenance": result.provenance,
                },
            },
        }
        return out

    def reset_history(self) -> None:
        self.history.clear()

    def get_history(self) -> List[Dict[str, str]]:
        return list(self.history)

    # -------------------------
    # LLM path (optional)
    # -------------------------
    def _try_llm_parse(self, text: str) -> Optional[Dict[str, Any]]:
        """
        Ask the LLM to produce strict JSON following our schema.
        Any error/timeouts return None (we never fail the pipeline).
        """
        try:
            # Lazy import to avoid hard dependency in environments without openai package
            from openai import OpenAI
            client = OpenAI(api_key=OPENAI_KEY)

            prompt = self._build_llm_prompt(text)
            resp = client.chat.completions.create(
                model=AGENT_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                timeout=AGENT_TIMEOUT_S,  # type: ignore (some client versions accept 'timeout')
            )
            raw = resp.choices[0].message.content
            parsed = self._safe_json_loads(raw)
            if not isinstance(parsed, dict):
                return None

            # Minimal validation
            if "intent" not in parsed or "details" not in parsed:
                return None
            if not isinstance(parsed["intent"], list):
                return None
            if not isinstance(parsed["details"], dict):
                return None

            self._remember({"role": "assistant", "content": "[LLM parse OK]"})
            return parsed
        except Exception as e:
            logger.debug("LLM parse failed (non-fatal): %s", e, exc_info=True)
            self._remember({"role": "assistant", "content": "[LLM parse failed → fallback]"})
            return None

    @staticmethod
    def _build_llm_prompt(text: str) -> str:
        """
        Constrain the model to emit JSON only. No prose.
        """
        return (
            "You are a strict JSON generator for QA intents.\n"
            "Return ONLY a JSON object matching this schema:\n"
            "{\n"
            '  "intent": ["ui_testing" | "api_testing" | "performance_testing", ...],\n'
            '  "details": {\n'
            '     "url": string|null,\n'
            '     "ui_targets": string[]?,\n'
            '     "api_targets": { "endpoints": string[]?, "methods": string[]? }?,\n'
            '     "perf_targets": { "threshold_ms": number? }?,\n'
            '     "constraints": { "browsers": string[]?, "devices": string[]?, "headless": boolean? }?,\n'
            '     "auth_required": boolean?,\n'
            '     "env": "dev"|"staging"|"prod"|"unknown"?,\n'
            '     "notes": string? \n'
            "  }\n"
            "}\n"
            "Rules:\n"
            "- If unsure, omit the field.\n"
            "- Detect performance targets like 'under 2s' as threshold_ms=2000.\n"
            "- Extract first http(s) URL if present.\n"
            "- No extra text; output valid JSON only.\n"
            f'Input: """{text}"""'
        )

    @staticmethod
    def _safe_json_loads(raw: str) -> Any:
        try:
            return json.loads(raw or "{}")
        except Exception:
            # try to salvage a JSON object within the response
            m = re.search(r"\{.*\}", raw or "", re.S)
            if m:
                try:
                    return json.loads(m.group(0))
                except Exception:
                    return {}
            return {}

    # -------------------------
    # Heuristic fallback
    # -------------------------
    def _heuristic_parse(self, text: str) -> Dict[str, Any]:
        t = (text or "").strip()
        tl = t.lower()

        # intents
        intents: List[str] = []
        if re.search(r"\b(ui|frontend|page|click|type|playwright|selector)\b", tl):
            intents.append("ui_testing")
        if re.search(r"\b(api|endpoint|json|rest|http|swagger|openapi)\b", tl):
            intents.append("api_testing")
        if re.search(r"\b(perf|performance|load|latency|stress|soak|benchmark)\b", tl):
            intents.append("performance_testing")
        if not intents:
            intents = ["ui_testing"]

        # url
        url = self._extract_url(t)

        # api targets
        api_endpoints, api_methods = self._extract_api_targets(t)

        # perf targets
        perf_ms = self._extract_perf_target_ms(t)

        # constraints (browsers/devices/headless)
        browsers = self._extract_browsers(tl)
        devices = self._extract_devices(tl)
        headless = self._extract_headless(tl)

        # environment hint
        env = self._extract_env(tl)

        # auth hints
        auth_required = bool(re.search(r"\b(login|auth|authenticated|sso|oauth|okta|keycloak)\b", tl))

        # UI targets (rough nouns around “page”, “button”, “form”, “checkout”, etc.)
        ui_targets = self._extract_ui_targets(t)

        details: Dict[str, Any] = {
            "url": url,
            "ui_targets": ui_targets or None,
            "api_targets": {"endpoints": api_endpoints or None, "methods": api_methods or None},
            "perf_targets": {"threshold_ms": perf_ms} if perf_ms is not None else None,
            "constraints": {
                "browsers": browsers or None,
                "devices": devices or None,
                "headless": headless,
            },
            "auth_required": auth_required or None,
            "env": env or None,
            "notes": None,
        }
        # remove Nones to keep output tidy
        details = {k: v for k, v in details.items() if v is not None}
        return {"intent": intents, "details": details}

    # -------------------------
    # Heuristic extractors
    # -------------------------
    @staticmethod
    def _extract_url(text: str) -> Optional[str]:
        m = re.search(r"(https?://[^\s'\"<>)]+)", text)
        if not m:
            return None
        url = m.group(1).rstrip(".,;)")
        # basic sanity
        try:
            pu = urlparse(url)
            if pu.scheme in ("http", "https") and pu.netloc:
                return url
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_api_targets(text: str) -> Tuple[List[str], List[str]]:
        endpoints = re.findall(r"\b(/[A-Za-z0-9._~:/?#\[\]@!$&'()*+,;=%-]+)", text)
        endpoints = [e for e in endpoints if e.startswith("/")]
        methods = re.findall(r"\b(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\b", text, flags=re.I)
        methods = list({m.upper() for m in methods})
        return (list(dict.fromkeys(endpoints))[:20], methods[:10])

    @staticmethod
    def _extract_perf_target_ms(text: str) -> Optional[int]:
        """
        Recognize 'under 2s', '< 3s', 'p95 500 ms', 'TTFB < 200ms' patterns.
        Return threshold in ms (best-effort).
        """
        t = text.lower()

        # < N s/ms
        m = re.search(r"(?:<|under|below)\s*(\d+(?:\.\d+)?)\s*(s|ms)\b", t)
        if m:
            val = float(m.group(1))
            unit = m.group(2)
            return int(val * 1000) if unit == "s" else int(val)

        # p95/p90 500ms
        m = re.search(r"\bp9[05]\s*[:=]?\s*(\d+)\s*ms\b", t)
        if m:
            return int(m.group(1))

        return None

    @staticmethod
    def _extract_browsers(tl: str) -> List[str]:
        out: List[str] = []
        for b in ("chromium", "chrome", "firefox", "webkit", "safari", "edge"):
            if b in tl:
                out.append(b if b != "chrome" else "chromium")
        return list(dict.fromkeys(out))

    @staticmethod
    def _extract_devices(tl: str) -> List[str]:
        out: List[str] = []
        if "mobile" in tl or "iphone" in tl or "android" in tl:
            out.append("mobile")
        if "tablet" in tl or "ipad" in tl:
            out.append("tablet")
        if "desktop" in tl:
            out.append("desktop")
        return list(dict.fromkeys(out))

    @staticmethod
    def _extract_headless(tl: str) -> Optional[bool]:
        if "headless true" in tl or "run headless" in tl:
            return True
        if "headless false" in tl or "run headed" in tl or "non-headless" in tl:
            return False
        return None

    @staticmethod
    def _extract_env(tl: str) -> Optional[str]:
        if "staging" in tl or "stage" in tl:
            return "staging"
        if "prod" in tl or "production" in tl:
            return "prod"
        if "dev" in tl or "development" in tl:
            return "dev"
        return None

    @staticmethod
    def _extract_ui_targets(text: str) -> List[str]:
        # naive noun-ish capture around common UI terms
        candidates: List[str] = []
        patterns = [
            r"(login|signin|authentication)\s+(page|form)",
            r"(signup|register)\s+(page|form)",
            r"(checkout|cart|basket)",
            r"(profile|settings|preferences)",
            r"(dashboard|home|landing)",
            r"(search)\s+(bar|box|page)",
            r"(language|locale)\s+(selector|dropdown)",
        ]
        for p in patterns:
            for m in re.finditer(p, text, flags=re.I):
                candidates.append(m.group(0).lower())
        return list(dict.fromkeys(candidates))[:20]

    # -------------------------
    # Merge LLM and heuristic results
    # -------------------------
    def _merge(self, llm: Dict[str, Any], heur: Dict[str, Any]) -> Dict[str, Any]:
        """
        Conservative merge:
        - intents: union
        - details.url: prefer llm if valid, else heur
        - details subfields: llm wins if not empty; fallback to heur
        - confidence: boost if both agree
        """
        out: Dict[str, Any] = {"intent": [], "details": {}, "confidence": 0.75}
        # intents
        i_llm = [s for s in llm.get("intent", []) if isinstance(s, str)]
        i_heur = [s for s in heur.get("intent", []) if isinstance(s, str)]
        intents = list(dict.fromkeys(i_llm + i_heur)) or ["ui_testing"]
        out["intent"] = intents

        # details
        d_llm = llm.get("details", {}) or {}
        d_heur = heur.get("details", {}) or {}

        # URL
        url = d_llm.get("url") or d_heur.get("url")
        out["details"]["url"] = url

        # copy structured blocks
        for key in (
            "ui_targets",
            "api_targets",
            "perf_targets",
            "constraints",
            "auth_required",
            "env",
            "notes",
        ):
            val = d_llm.get(key, None)
            if val in (None, [], {}):
                val = d_heur.get(key, None)
            if val not in (None, [], {}):
                out["details"][key] = val

        # confidence heuristic
        agree = set(i_llm) == set(i_heur) and (d_llm.get("url") == d_heur.get("url") or url is not None)
        out["confidence"] = 0.85 if agree else 0.75
        return out

    # -------------------------
    # Memory helpers
    # -------------------------
    def _remember(self, msg: Dict[str, str]) -> None:
        self.history.append(msg)
        if len(self.history) > MAX_MEM_TURNS:
            # simple FIFO trim
            self.history = self.history[-MAX_MEM_TURNS:]
