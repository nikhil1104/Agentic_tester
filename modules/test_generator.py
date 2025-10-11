"""
Test Generator (Phase 4.9 — Components, Risk, OpenAPI-aware)
------------------------------------------------------------
Purpose:
- Generate structured, AI-driven test plans from:
  - Parsed requirements (UI/API/Performance)
  - DOM insights (SemanticActionEngine)
  - Optional OpenAPI (dict or JSON file path via req/scan_res)

Key Improvements:
- Component tags inferred from DOM
- Heuristic risk scoring → auto priority (P0/P1/P2)
- OpenAPI expansion (safe, capped)
- Deterministic limits + structured logging
"""

from __future__ import annotations

import os
import json
import uuid
import logging
from typing import Dict, Any, List, Optional, Tuple

from modules.semantic_action_engine import SemanticActionEngine

logger = logging.getLogger(__name__)

# -----------------------------
# Tunables (env-overridable)
# -----------------------------
PLAN_SCHEMA_VERSION = "4.9.0"
MAX_STEPS_PER_SUITE = int(os.environ.get("TG_MAX_STEPS_PER_SUITE", "50"))
MAX_SUITES_PER_KIND = int(os.environ.get("TG_MAX_SUITES_PER_KIND", "25"))
MAX_API_ENDPOINTS = int(os.environ.get("TG_MAX_API_ENDPOINTS", "50"))

# Intent normalization
_UI_INTENTS = {"ui", "ui_testing", "ui-tests", "ui-automation"}
_API_INTENTS = {"api", "api_testing", "api-tests"}
_PERF_INTENTS = {"performance", "performance_testing", "perf", "load", "perf-tests"}

# Priority ranking helper
_PRIORITY_RANK = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}


class TestGenerator:
    """Generates intelligent, multi-suite test plans using scanned HTML, DOM analysis, and (optional) OpenAPI."""

    def __init__(self, html_cache_dir: str = "data/scraped_docs"):
        self.engine = SemanticActionEngine(html_cache_dir=html_cache_dir)
        self.html_cache_dir = html_cache_dir

    # ============================================================
    # Public API — Generate Test Plan
    # ============================================================
    def generate_plan(self, req: Dict[str, Any], scan_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a structured test plan.

        Args:
            req: Parsed requirement dict, e.g. {"intent": [...], "details": {"url": "..."},
                                                "api": {"openapi": <dict or json-file-path>}}
            scan_res: Result of WebScraper.quick_scan(), may include {"html": "..."} and/or {"openapi": {...}}

        Returns:
            Plan dict: {"project": <url>, "plan_meta": {...}, "suites": {...}}
        """
        project_url = self._safe_str((req.get("details", {}) or {}).get("url"))
        intents = {self._safe_str(i).lower() for i in (req.get("intent") or [])}

        logger.info("🧠 Generating plan (schema=%s) for project=%s", PLAN_SCHEMA_VERSION, project_url or "<unknown>")

        # Resolve HTML (scan_res preferred, else cached)
        html = self._resolve_html(scan_res)

        # Extract testable elements (safe fallback)
        parsed = self._safe_extract(html)

        # Derive component tags from parsed DOM snapshot
        component_tags = self._derive_component_tags(parsed)

        suites: Dict[str, List[Dict[str, Any]]] = {}

        # ----------------- UI Suites -----------------
        if intents & _UI_INTENTS:
            ui_suites: List[Dict[str, Any]] = []
            ui_suites.extend(self._suite_inputs(parsed))
            ui_suites.extend(self._suite_selects(parsed))
            ui_suites.extend(self._suite_languages(parsed))
            ui_suites.extend(self._suite_buttons_and_clickables(parsed))
            ui_suites.extend(self._suite_headings(parsed))
            ui_suites.extend(self._suite_images(parsed))
            ui_suites.extend(self._suite_links(parsed, project_url))
            ui_suites.append(self._suite_smoke(project_url))

            # Compute risk score per suite and auto-assign priority
            for s in ui_suites:
                s["risk_score"] = self._risk_score_suite(s)
                if s.get("priority") is None:
                    s["priority"] = self._priority_from_risk(s["risk_score"])
                # augment tags with components
                s.setdefault("tags", []).extend(component_tags)

            # Sort by priority + risk desc + name
            ui_suites = self._prioritize_and_cap_suites(ui_suites, MAX_SUITES_PER_KIND)
            suites["ui"] = ui_suites

        # ----------------- API Suites -----------------
        api_suites: List[Dict[str, Any]] = []
        if intents & _API_INTENTS:
            # a) Health baseline
            api_suites.extend(self._suite_api_baseline())
            # b) OpenAPI expansion if present
            openapi = self._resolve_openapi(req, scan_res)
            if openapi:
                api_suites.extend(self._suite_api_from_openapi(openapi))
            suites["api"] = self._cap_list(api_suites, MAX_SUITES_PER_KIND)

        # ----------------- Performance Suites -----------------
        if intents & _PERF_INTENTS:
            suites["performance"] = self._cap_list(self._suite_perf(req, scan_res), MAX_SUITES_PER_KIND)

        # Risk & components summary for plan meta
        risk_summary = self._risk_summary(suites)

        # Plan metadata
        plan_meta = {
            "schema_version": PLAN_SCHEMA_VERSION,
            "plan_id": str(uuid.uuid4())[:8],
            "source": "TestGenerator",
            "limits": {
                "max_steps_per_suite": MAX_STEPS_PER_SUITE,
                "max_suites_per_kind": MAX_SUITES_PER_KIND,
                "max_api_endpoints": MAX_API_ENDPOINTS,
            },
            "inputs": {
                "project_url_present": bool(project_url),
                "html_present": bool(html),
                "intents": list(intents),
                "components": sorted(component_tags),
            },
            "risk_summary": risk_summary,
        }

        plan = {"project": project_url, "plan_meta": plan_meta, "suites": suites}
        logger.info(
            "✅ Plan generated: ui=%d, api=%d, perf=%d | top_risk=%s",
            len(suites.get("ui", [])),
            len(suites.get("api", [])),
            len(suites.get("performance", [])),
            risk_summary.get("top_suite", "<none>"),
        )
        return plan

    # ============================================================
    # Internal: HTML & OpenAPI resolution
    # ============================================================
    def _resolve_html(self, scan_res: Dict[str, Any]) -> Optional[str]:
        """Prefer HTML from scan_res; fallback to latest cached snapshot."""
        try:
            if isinstance(scan_res, dict):
                for key in ("html", "rendered_html", "snapshot_html"):
                    if scan_res.get(key):
                        return self._safe_str(scan_res[key])
            return self.engine._load_latest_html()
        except Exception:
            logger.debug("No HTML available from scan or cache", exc_info=True)
            return None

    def _resolve_openapi(self, req: Dict[str, Any], scan_res: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Accepts:
          - dict already loaded (req['api']['openapi'] or scan_res['openapi'])
          - path to JSON file
        YAML is intentionally not parsed to avoid extra deps; can be added later.
        """
        # 1) Direct dict
        for src in (req.get("api", {}) or {}, scan_res or {}):
            if isinstance(src.get("openapi"), dict):
                return src["openapi"]

        # 2) JSON path
        for src in (req.get("api", {}) or {}, scan_res or {}):
            val = src.get("openapi")
            if isinstance(val, str):
                path = val.strip()
                if path and os.path.exists(path):
                    try:
                        with open(path, "r", encoding="utf-8") as fh:
                            return json.load(fh)
                    except Exception:
                        logger.exception("Failed to load OpenAPI JSON from %s", path)
        return None

    # ============================================================
    # Internal: Extraction & Components
    # ============================================================
    def _safe_extract(self, html: Optional[str]) -> Dict[str, Any]:
        if not html:
            logger.warning("No HTML content; generating minimal plan.")
            return {
                "inputs": [], "selects": [], "buttons": [],
                "links": [], "images": [], "headings": [],
                "languages": [], "custom_clickables": []
            }
        try:
            return self.engine.extract_testable_elements(html)
        except Exception:
            logger.exception("extract_testable_elements failed; using empty structure")
            return {
                "inputs": [], "selects": [], "buttons": [],
                "links": [], "images": [], "headings": [],
                "languages": [], "custom_clickables": []
            }

    def _derive_component_tags(self, parsed: Dict[str, Any]) -> List[str]:
        """Infer broad components to tag suites for filtering and routing."""
        tags = set()
        inputs = parsed.get("inputs") or []
        selects = parsed.get("selects") or []
        buttons = parsed.get("buttons") or []
        links = parsed.get("links") or []
        images = parsed.get("images") or []
        headings = parsed.get("headings") or []
        languages = parsed.get("languages") or []

        if inputs:
            tags.add("forms")
            if any("password" in (i.get("name", "") or "").lower() for i in inputs):
                tags.add("auth")
        if selects:
            tags.add("dropdowns")
        if buttons:
            tags.add("interactions")
        if links:
            tags.add("nav")
            if any(("login" in (l.get("text", "") or "").lower()) for l in links):
                tags.add("auth")
            if any(("cart" in (l.get("text", "") or "").lower() or "checkout" in (l.get("text", "") or "").lower()) for l in links):
                tags.add("checkout")
        if images:
            tags.add("media")
        if headings:
            tags.add("content")
        if languages:
            tags.add("i18n")

        return sorted(tags)

    # ============================================================
    # Internal: UI suite builders
    # ============================================================
    def _suite_inputs(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        suites: List[Dict[str, Any]] = []
        inputs = parsed.get("inputs") or []
        if not inputs:
            return suites

        steps: List[str] = []
        for inp in inputs:
            field = self._first_not_empty(
                inp.get("name"), inp.get("id"), inp.get("aria-label"), inp.get("placeholder"), "input"
            )
            steps.append(f"type sample data into '{field.strip()}' field")

        suites.append(self._mk_suite("form fields – input typing & validation", steps))
        return suites

    def _suite_selects(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        suites: List[Dict[str, Any]] = []
        selects = parsed.get("selects") or []
        for sel in selects:
            label = self._first_not_empty(sel.get("name"), sel.get("id"), "dropdown")
            opts = [o for o in (sel.get("options") or []) if isinstance(o, str) and o.strip()]
            steps: List[str] = []

            if opts:
                display = ", ".join(f"'{o}'" for o in opts[:20])
                steps.append(f"verify dropdown '{label}' includes [{display}]")
                for o in opts[:MAX_STEPS_PER_SUITE - len(steps)]:
                    steps.append(f"verify '{o}' option visible in dropdown '{label}'")
            else:
                steps.append(f"verify dropdown '{label}' dynamically loads options at runtime")

            suites.append(self._mk_suite(f"dropdown – {label} verification", steps))
        return self._cap_list(suites, MAX_SUITES_PER_KIND)

    def _suite_languages(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        suites: List[Dict[str, Any]] = []
        langs = parsed.get("languages") or []
        for lang in langs:
            label = self._first_not_empty(lang.get("name"), lang.get("text"), "language")
            opts = [o for o in (lang.get("options") or []) if isinstance(o, str) and o.strip()]
            steps: List[str] = []
            if opts:
                display = ", ".join(f"'{o}'" for o in opts[:20])
                steps.append(f"verify language dropdown '{label}' includes [{display}]")
                for o in opts[:MAX_STEPS_PER_SUITE - len(steps)]:
                    steps.append(f"verify '{o}' language option visible")
            else:
                steps.append(f"verify '{label}' language selector is visible and clickable")

            suites.append(self._mk_suite(f"language selector – {label}", steps))
        return self._cap_list(suites, MAX_SUITES_PER_KIND)

    def _suite_buttons_and_clickables(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        suites: List[Dict[str, Any]] = []
        steps: List[str] = []
        for btn in parsed.get("buttons") or []:
            text = self._safe_str(btn.get("text")) or "button"
            steps.append(f"click '{text}' button")
        for c in parsed.get("custom_clickables") or []:
            text = self._safe_str(c.get("text")) or "element"
            steps.append(f"click '{text}' element")

        if steps:
            suites.append(self._mk_suite("buttons & clickables – interactions", steps))
        return suites

    def _suite_headings(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        suites: List[Dict[str, Any]] = []
        heads = [h for h in (parsed.get("headings") or []) if self._safe_str(h)]
        if heads:
            display = heads[:25]
            suites.append(self._mk_suite("headings – visibility checks", [f"verify headings visible: {display}"]))
        return suites

    def _suite_images(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        suites: List[Dict[str, Any]] = []
        imgs = parsed.get("images") or []
        if not imgs:
            return suites

        steps: List[str] = []
        for img in imgs[:MAX_STEPS_PER_SUITE]:
            alt = self._safe_str((img.get("alt") or "").strip())
            if alt:
                steps.append(f"verify image '{alt}' has alt text")
            else:
                steps.append("verify image without alt is handled gracefully")

        suites.append(self._mk_suite("images – accessibility & alt validation", steps))
        return suites

    def _suite_links(self, parsed: Dict[str, Any], project_url: str) -> List[Dict[str, Any]]:
        suites: List[Dict[str, Any]] = []
        links = parsed.get("links") or []
        if not links:
            return suites

        steps: List[str] = []
        for link in links[:MAX_STEPS_PER_SUITE]:
            text = self._safe_str(link.get("text")) or "link"
            href = self._safe_str(link.get("href"))
            if href:
                steps.append(f"verify link '{text}' navigates to '{href}'")
            else:
                steps.append(f"click '{text}' link")

        suites.append(self._mk_suite("links – href verification", steps))
        return suites

    def _suite_smoke(self, project_url: str) -> Dict[str, Any]:
        url = project_url or "about:blank"
        steps = [f"goto {url}", "verify page title is visible"]
        return self._mk_suite("smoke – page title visible", steps, priority="P0", tags=["smoke", "critical"])

    # ============================================================
    # Internal: API suites (baseline + OpenAPI)
    # ============================================================
    def _suite_api_baseline(self) -> List[Dict[str, Any]]:
        return [{
            "name": "API Health Check",
            "steps": ["GET /health", "expect status 200"],
            "priority": "P1",
            "tags": ["api", "health"],
            "risk_score": 30
        }]

    def _suite_api_from_openapi(self, spec: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Create small, safe baseline tests for each path/method.
        We do not validate schema bodies here (keeps dependency-light).
        """
        out: List[Dict[str, Any]] = []
        paths = (spec.get("paths") or {}) if isinstance(spec, dict) else {}
        if not isinstance(paths, dict):
            return out

        # Flatten to (method, path) tuples
        items: List[Tuple[str, str, Dict[str, Any]]] = []
        for path, methods in paths.items():
            if not isinstance(methods, dict):
                continue
            for method, op in methods.items():
                if method.upper() not in {"GET", "POST", "PUT", "PATCH", "DELETE"}:
                    continue
                items.append((method.upper(), path, op if isinstance(op, dict) else {}))

        # Cap endpoints
        items = items[:MAX_API_ENDPOINTS]

        for method, path, op in items:
            name = op.get("summary") or f"{method} {path}"
            steps = [f"{method} {path}", "expect status 2xx"]
            tags = ["api", "openapi"]
            # simple risk heuristic: write-y methods are riskier
            risk = 40 if method == "GET" else 65
            out.append({
                "name": name,
                "steps": self._dedup_and_cap(steps),
                "priority": self._priority_from_risk(risk),
                "tags": tags,
                "risk_score": risk
            })
        return out

    # ============================================================
    # Internal: Performance suite builder
    # ============================================================
    def _suite_perf(self, req: Dict[str, Any], scan_res: Dict[str, Any]) -> List[Dict[str, Any]]:
        suites = [{
            "name": "Performance Baseline",
            "steps": ["measure page load time", "assert < 3s"],
            "priority": "P2",
            "tags": ["perf", "baseline"],
            "risk_score": 35
        }]
        return suites

    # ============================================================
    # Risk scoring / Prioritization
    # ============================================================
    def _risk_score_suite(self, suite: Dict[str, Any]) -> int:
        """
        Heuristic: + weights for auth/checkout/admin, #steps, write ops,
        and presence of external links.
        """
        score = 20
        name = (suite.get("name") or "").lower()
        steps = suite.get("steps") or []
        joined = " ".join(steps).lower()

        # Domain-specific keywords
        if any(k in joined or k in name for k in ("password", "login", "signin", "auth")):
            score += 35
        if any(k in joined or k in name for k in ("checkout", "payment", "billing", "card")):
            score += 30
        if any(k in joined or k in name for k in ("admin", "settings", "profile")):
            score += 15

        # Write operations (API)
        if any(k in joined for k in ("post ", "put ", "patch ", "delete ")):
            score += 15

        # Steps volume (larger flows carry more risk)
        score += min(len(steps), 30) // 3  # +1 per ~3 steps, capped

        # External link checks
        if "http://" in joined or "https://" in joined:
            score += 5

        return max(0, min(score, 100))

    def _priority_from_risk(self, risk: int) -> str:
        if risk >= 80:
            return "P0"
        if risk >= 50:
            return "P1"
        return "P2"

    def _prioritize_and_cap_suites(self, suites: List[Dict[str, Any]], cap: int) -> List[Dict[str, Any]]:
        def key(s: Dict[str, Any]):
            pr = s.get("priority") or "P2"
            return (_PRIORITY_RANK.get(pr, 2), -int(s.get("risk_score", 0)), s.get("name", ""))

        suites_sorted = sorted(suites, key=key)
        return self._cap_list(suites_sorted, cap)

    def _risk_summary(self, suites: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
        all_suites = [*suites.get("ui", []), *suites.get("api", []), *suites.get("performance", [])]
        if not all_suites:
            return {"top_suite": None, "top_risk": None, "avg_risk": None}
        top = max(all_suites, key=lambda s: int(s.get("risk_score", 0))) if all_suites else None
        avg = round(sum(int(s.get("risk_score", 0)) for s in all_suites) / len(all_suites), 2)
        return {"top_suite": top.get("name") if top else None, "top_risk": int(top.get("risk_score", 0)) if top else None, "avg_risk": avg}

    # ============================================================
    # Helpers
    # ============================================================
    @staticmethod
    def _safe_str(v: Any) -> str:
        try:
            return str(v or "").strip()
        except Exception:
            return ""

    @staticmethod
    def _first_not_empty(*vals: Optional[str]) -> str:
        for v in vals:
            if v and str(v).strip():
                return str(v).strip()
        return ""

    @staticmethod
    def _cap_list(items: List[Any], limit: int) -> List[Any]:
        if limit <= 0:
            return items
        return items[:limit]

    @staticmethod
    def _dedup_and_cap(steps: List[str]) -> List[str]:
        seen, out = set(), []
        for s in steps:
            s2 = (s or "").strip()
            if s2 and s2 not in seen:
                seen.add(s2)
                out.append(s2)
            if len(out) >= MAX_STEPS_PER_SUITE:
                break
        return out

    @staticmethod
    def _mk_suite(name: str, steps: List[str], priority: Optional[str] = None, tags: Optional[List[str]] = None) -> Dict[str, Any]:
        return {
            "name": name,
            "steps": TestGenerator._dedup_and_cap(steps),
            "priority": priority,           # may be None → will be set by risk policy
            "tags": tags or [],
        }
