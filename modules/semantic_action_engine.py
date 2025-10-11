# modules/semantic_action_engine.py
"""
Semantic Action Engine (async-safe, no playwright.sync_api)
-----------------------------------------------------------
Purpose:
  Convert high-level, human-readable test steps into Playwright-ready
  JavaScript/TypeScript actions with resilient, self-healing locators.

Key Design Goals:
  1) Resilience   – Prefer Playwright's recommended locators (role/label/testId),
                    fall back to text/CSS, handle minor UI changes.
  2) Clarity      – Generated JS is readable and expect-aware.
  3) Extensibility– Hooks to add future strategies (visual, embeddings, LLM).
  4) Safety       – No top-level Playwright imports; any runtime enrichment
                    uses playwright.async_api lazily in async methods.

Public API (backward compatible):
  - extract_testable_elements(html) -> Dict[str, Any]      # sync (static parse only)
  - resolve_locator(html, name) -> str                     # sync
  - generate_js_action(step, html) -> str                  # sync JS/TS snippet

Optional async enrichment (NEW, opt-in):
  - detect_dynamic_dropdowns_async(html) -> Dict[str, List[str]]
  - enrich_selects_with_runtime_options_async(html, selects) -> List[Dict]
"""

from __future__ import annotations

import os
import re
import logging
from typing import Dict, List, Any, Optional, Tuple
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

# ------------------------------
# Utility helpers
# ------------------------------

_QUOTE_RX = re.compile(r"""['"]([^'"]+)['"]""")
_RX_TYPE_INTO = re.compile(
    r"""(?:type|enter)\s+['"]([^'"]+)['"]\s+(?:into|in)\s+['"]([^'"]+)['"]""", re.I
)
_RX_SELECT_FROM = re.compile(
    r"""select\s+['"]([^'"]+)['"]\s+(?:from|in)\s+['"]([^'"]+)['"]""", re.I
)


def _js_escape(s: str) -> str:
    """Escape text for safe inclusion in JS double-quoted strings."""
    return (s or "").replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n")


def _norm_space(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


class SemanticActionEngine:
    """
    Translates DOM + natural language steps into executable JS actions.
    """

    def __init__(self, html_cache_dir: str = "data/scraped_docs"):
        self.html_cache_dir = html_cache_dir

    # ----------------------------------------------------------------------
    # (optional) load latest cached HTML file
    # ----------------------------------------------------------------------
    def _load_latest_html(self) -> str:
        """
        Loads the latest HTML/MD snapshot from the cache directory.
        Kept for convenience. Not used automatically.
        """
        if not os.path.exists(self.html_cache_dir):
            raise FileNotFoundError(f"{self.html_cache_dir} not found.")
        files = sorted(
            f for f in os.listdir(self.html_cache_dir) if f.lower().endswith((".md", ".html"))
        )
        if not files:
            raise FileNotFoundError("No HTML snapshots found in scraped_docs/")
        latest = os.path.join(self.html_cache_dir, files[-1])
        with open(latest, "r", encoding="utf-8") as fh:
            return fh.read()

    # ----------------------------------------------------------------------
    # ELEMENT EXTRACTION (static, sync)
    # ----------------------------------------------------------------------
    def extract_testable_elements(self, html: str) -> Dict[str, Any]:
        """
        Parse HTML, extract testable elements and metadata to guide action generation.
        Static only (no JS execution). For dynamic dropdowns, see
        detect_dynamic_dropdowns_async/enrich_selects_with_runtime_options_async.

        Returns:
            {
              "inputs": [{"name":..., "type":...}],
              "selects": [{"name":..., "options":[...]}],
              "buttons": [{"text":...}],
              "links": [{"text":..., "href":...}],
              "images": [{"alt":..., "src":...}],
              "headings": [..],
              "languages": [...mixed from selects/buttons...],
              "custom_clickables": [{"text":...}],
            }
        """
        soup = BeautifulSoup(html or "", "html.parser")

        # --- INPUT FIELDS ---
        inputs: List[Dict[str, str]] = []
        for inp in soup.find_all(["input", "textarea"]):
            name = (
                inp.get("data-testid")
                or inp.get("name")
                or inp.get("id")
                or inp.get("aria-label")
                or inp.get("placeholder")
            )
            itype = (inp.get("type") or "text").lower()
            if not name:
                continue
            # Skip security tokens/hidden
            if any(k in (name or "").lower() for k in ("csrf", "token")):
                continue
            if inp.get("type", "").lower() == "hidden":
                continue
            inputs.append({"name": _norm_space(name), "type": itype})

        # --- DROPDOWNS ---
        selects: List[Dict[str, Any]] = []
        for sel in soup.find_all("select"):
            name = (
                sel.get("data-testid")
                or sel.get("name")
                or sel.get("id")
                or sel.get("aria-label")
                or "select"
            )
            opts = [
                o.get_text(strip=True) for o in sel.find_all("option") if o.get_text(strip=True)
            ]
            selects.append({"name": _norm_space(name), "options": opts})

        # --- BUTTONS ---
        buttons: List[Dict[str, str]] = []
        for btn in soup.find_all("button"):
            txt = _norm_space(btn.get_text(strip=True))
            if txt:
                buttons.append({"text": txt})
        for inp in soup.find_all("input", {"type": re.compile("button|submit", re.I)}):
            txt = _norm_space(inp.get("value") or inp.get("aria-label") or "")
            if txt:
                buttons.append({"text": txt})

        # --- LINKS ---
        links: List[Dict[str, str]] = []
        for a in soup.find_all("a", href=True):
            text = _norm_space(a.get_text(strip=True))
            href = a.get("href", "")
            if text:
                links.append({"text": text, "href": href})

        # --- IMAGES ---
        images = [
            {"alt": img.get("alt"), "src": img.get("src")} for img in soup.find_all("img")
        ]

        # --- HEADINGS ---
        headings: List[str] = []
        for tag in ("h1", "h2", "h3", "h4"):
            for h in soup.find_all(tag):
                txt = _norm_space(h.get_text(strip=True))
                if txt:
                    headings.append(txt)

        # --- CUSTOM CLICKABLES ---
        custom_clickables: List[Dict[str, str]] = []
        for tag in soup.find_all(["div", "span"], attrs={"role": re.compile("button|link", re.I)}):
            txt = _norm_space(tag.get_text(strip=True))
            if txt:
                custom_clickables.append({"text": txt})
        for tag in soup.find_all(["div", "span"], attrs={"onclick": True}):
            txt = _norm_space(tag.get_text(strip=True))
            if txt:
                custom_clickables.append({"text": txt})

        # --- LANGUAGE ELEMENTS ---
        languages: List[Dict[str, Any]] = []
        for sel in selects:
            if any(k in sel["name"].lower() for k in ["lang", "locale", "language"]):
                languages.append(sel)
        for b in buttons:
            bt = b["text"].lower()
            if "language" in bt or re.search(r"\b(lang)\b", bt):
                languages.append(b)

        # --- DEDUPLICATION ---
        def _dedup(items: List[Dict[str, Any]], key: str = "text") -> List[Dict[str, Any]]:
            seen, out = set(), []
            for it in items:
                v = it.get(key)
                if v and v not in seen:
                    seen.add(v)
                    out.append(it)
            return out

        buttons = _dedup(buttons, "text")
        links = _dedup(links, "text")
        headings = list(dict.fromkeys(headings))

        return {
            "inputs": inputs,
            "selects": selects,
            "buttons": buttons,
            "links": links,
            "images": images,
            "headings": headings,
            "languages": languages,
            "custom_clickables": custom_clickables,
        }

    # ----------------------------------------------------------------------
    # OPTIONAL DYNAMIC ENRICHMENT (async-only; never imported at module import)
    # ----------------------------------------------------------------------
    async def detect_dynamic_dropdowns_async(self, html: str) -> Dict[str, List[str]]:
        """
        Render provided HTML with Playwright (async_api) to capture JS-populated
        <select> options. This method is OPT-IN and async to avoid greenlet issues.

        Returns: { select_id_or_name_or_testid: [option_texts...] }
        """
        from playwright.async_api import async_playwright  # lazy import

        results: Dict[str, List[str]] = {}
        if not html:
            return results

        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=True)
        context = await browser.new_context(ignore_https_errors=True)
        page = await context.new_page()
        try:
            await page.set_content(html)
            try:
                # best-effort: give the page a chance to populate dropdowns
                await page.wait_for_selector("select option", timeout=1000)
            except Exception:
                pass

            selects = page.locator("select")
            count = await selects.count()
            for i in range(count):
                sel = selects.nth(i)
                key = (
                    await sel.get_attribute("id")
                    or await sel.get_attribute("name")
                    or await sel.get_attribute("data-testid")
                    or f"select_{i}"
                )
                opts: List[str] = []
                all_opts = sel.locator("option")
                ocount = await all_opts.count()
                for j in range(ocount):
                    try:
                        txt = (await all_opts.nth(j).inner_text()).strip()
                        if txt:
                            opts.append(txt)
                    except Exception:
                        continue
                if opts:
                    results[key] = list(dict.fromkeys(opts))
        finally:
            try:
                await context.close()
            except Exception:
                pass
            try:
                await browser.close()
            except Exception:
                pass
            try:
                await pw.stop()
            except Exception:
                pass

        return results

    async def enrich_selects_with_runtime_options_async(
        self, html: str, selects: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Helper: merge dynamic options into the static 'selects' list.
        """
        dynamic = await self.detect_dynamic_dropdowns_async(html)
        if not dynamic:
            return selects

        merged: List[Dict[str, Any]] = []
        for sel in selects:
            name = sel.get("name")
            opts = list(sel.get("options", []))
            # match by name/testid/id key if available
            if name and name in dynamic:
                opts = list(dict.fromkeys(opts + dynamic[name]))
            merged.append({**sel, "options": opts})
        return merged

    # ----------------------------------------------------------------------
    # SEMANTIC MAPPING
    # ----------------------------------------------------------------------
    def action_type(self, step: str) -> str:
        """Classify natural language step → action type."""
        s = (step or "").lower()
        if s.startswith("click") or " click " in s:
            return "click"
        if _RX_TYPE_INTO.search(s) or s.startswith("type") or " enter " in s:
            return "type"
        if _RX_SELECT_FROM.search(s) or s.startswith("select") or " choose " in s:
            return "select"
        if s.startswith("verify") or " assert " in s:
            return "verify"
        if s.startswith("goto") or " navigate " in s:
            return "goto"
        return "unknown"

    # ----------------------------------------------------------------------
    # LOCATOR RESOLUTION
    # ----------------------------------------------------------------------
    def _candidate_locators(self, html: str, name: str) -> List[Tuple[str, float, str]]:
        """
        Build a ranked list of Playwright locator JS expressions with confidence:
        Returns list of (js_expr_without_page_prefix, confidence, rationale)
        e.g., ('getByLabel("Email")', 0.95, 'label')
        """
        soup = BeautifulSoup(html or "", "html.parser")
        n = _norm_space(name)

        candidates: List[Tuple[str, float, str]] = []

        # data-testid (strong when present)
        el = soup.find(attrs={"data-testid": re.compile(re.escape(n), re.I)})
        if el and el.get("data-testid"):
            candidates.append((f'getByTestId("{_js_escape(el.get("data-testid"))}")', 0.99, "testId"))

        # label[for] + aria-label
        lab = soup.find("label", string=re.compile(re.escape(n), re.I))
        if lab and lab.get("for"):
            candidates.append((f'getByLabel("{_js_escape(n)}")', 0.96, "label-for"))
        if soup.find(attrs={"aria-label": re.compile(re.escape(n), re.I)}):
            candidates.append((f'getByLabel("{_js_escape(n)}")', 0.94, "aria-label"))

        # placeholder
        if soup.find(attrs={"placeholder": re.compile(re.escape(n), re.I)}):
            candidates.append((f'getByPlaceholder("{_js_escape(n)}")', 0.92, "placeholder"))

        # role-based buttons/links with accessible name
        btn = soup.find(lambda t: t.name in ("button", "a") and n.lower() in t.get_text(strip=True).lower())
        if btn:
            role = "button" if btn.name == "button" else "link"
            candidates.append((f'getByRole("{role}", {{ name: "{_js_escape(n)}" }})', 0.90, "role+name"))

        # id / name attribute (falls back to CSS)
        attr_hit = soup.find(attrs={"id": re.compile(re.escape(n), re.I)})
        if attr_hit and attr_hit.get("id"):
            candidates.append((f'locator("#{_js_escape(attr_hit.get("id"))}")', 0.85, "id"))
        attr_hit = soup.find(attrs={"name": re.compile(re.escape(n), re.I)})
        if attr_hit and attr_hit.get("name"):
            candidates.append((f'locator("[name=\\"{_js_escape(attr_hit.get("name"))}\\"]")', 0.83, "name-attr"))

        # text fallback (least reliable)
        candidates.append((f'getByText("{_js_escape(n)}")', 0.70, "text"))

        # unique, keep highest confidence per js_expr
        uniq: Dict[str, Tuple[str, float, str]] = {}
        for js, conf, why in candidates:
            if js not in uniq or conf > uniq[js][1]:
                uniq[js] = (js, conf, why)
        return sorted(uniq.values(), key=lambda x: x[1], reverse=True)

    def resolve_locator(self, html: str, name: str) -> str:
        """
        Legacy: return a CSS/text style string for page.locator("...").
        Delegates to best candidate and converts to legacy form where possible.
        """
        if not name:
            return "text=UNKNOWN"

        cands = self._candidate_locators(html, name)
        if not cands:
            return f'text={_norm_space(name)}'

        best = cands[0][0]  # e.g., getByLabel("Email")
        if best.startswith('getByLabel("'):
            return f'[aria-label="{_norm_space(name)}"]'
        if best.startswith('getByPlaceholder("'):
            return f'[placeholder="{_norm_space(name)}"]'
        if best.startswith('getByRole("button"'):
            return f'text={_norm_space(name)}'
        if best.startswith('locator("'):
            # already a CSS locator string
            inner = best[len('locator("'):-2]
            return inner
        if best.startswith('getByText("'):
            return f'text={_norm_space(name)}'
        return f'text={_norm_space(name)}'

    # ----------------------------------------------------------------------
    # JS ACTION GENERATOR
    # ----------------------------------------------------------------------
    def _emit_click(self, page_expr: str, html: str, target: str) -> str:
        """Generate a robust click snippet with fallbacks using ranked candidates."""
        cands = self._candidate_locators(html, target)
        if not cands:
            return f'await {page_expr}.getByText("{_js_escape(target)}").click();'

        primary = cands[0][0]
        fallback = cands[1][0] if len(cands) > 1 else None

        code = [
            f'await {page_expr}.{primary}.waitFor({{ state: "visible", timeout: 10000 }});',
            f'await {page_expr}.{primary}.click();',
        ]
        if fallback:
            code = [
                "try {",
                f'  {code[0]}',
                f'  {code[1]}',
                "} catch (e) {",
                f'  console.warn("Primary click failed, using fallback:", e);',
                f'  await {page_expr}.{fallback}.click();',
                "}",
            ]
        return "\n".join(code)

    def _emit_type(self, page_expr: str, html: str, field: str, value: str) -> str:
        cands = self._candidate_locators(html, field)
        if not cands:
            return f'await {page_expr}.getByText("{_js_escape(field)}").fill("{_js_escape(value)}");'
        primary = cands[0][0]
        fallback = cands[1][0] if len(cands) > 1 else None

        code = [
            f'await {page_expr}.{primary}.waitFor({{ state: "visible", timeout: 10000 }});',
            f'await {page_expr}.{primary}.fill("{_js_escape(value)}");',
        ]
        if fallback:
            code = [
                "try {",
                f'  {code[0]}',
                f'  {code[1]}',
                "} catch (e) {",
                f'  console.warn("Primary fill failed, using fallback:", e);',
                f'  await {page_expr}.{fallback}.fill("{_js_escape(value)}");',
                "}",
            ]
        return "\n".join(code)

    def _emit_select(self, page_expr: str, html: str, field: str, option: Optional[str]) -> str:
        cands = self._candidate_locators(html, field)
        if not cands:
            if option:
                return (
                    f'const dd = {page_expr}.getByText("{_js_escape(field)}");\n'
                    f'await dd.click();\n'
                    f'await {page_expr}.getByText("{_js_escape(option)}").click();'
                )
            return f'await {page_expr}.getByText("{_js_escape(field)}").press("Enter");'

        primary = cands[0][0]
        if option:
            return (
                f'await {page_expr}.{primary}.click();\n'
                f'await {page_expr}.getByText("{_js_escape(option)}").click();'
            )
        return f'await {page_expr}.{primary}.selectOption({{ index: 0 }});'

    def generate_js_action(self, step: str, html: str) -> str:
        """
        Convert a natural-language step into an executable Playwright action.
        Emission targets the '@playwright/test' runtime (page, expect available).
        """
        act = self.action_type(step)
        s = _norm_space(step)
        page_expr = "page"

        # ---------- NAVIGATE ----------
        if act == "goto":
            m = re.search(r"(?:goto|navigate)\s+(\S+)", s, re.I)
            url = m.group(1) if m else "baseURL"
            return f'await {page_expr}.goto("{_js_escape(url)}");'

        # ---------- CLICK ----------
        if act == "click":
            qm = _QUOTE_RX.search(step)
            target = qm.group(1) if qm else _norm_space(s.split("click", 1)[-1])
            if not target:
                return f'console.log("No click target provided for: {_js_escape(step)}");'
            return self._emit_click(page_expr, html, target)

        # ---------- TYPE ----------
        if act == "type":
            m = _RX_TYPE_INTO.search(step)
            if m:
                value, field = m.groups()
            else:
                allq = _QUOTE_RX.findall(step)
                value = allq[0] if allq else ""
                field = allq[1] if len(allq) > 1 else _norm_space(step.split("type", 1)[-1])
            if not field:
                return f'console.log("No field specified for type: {_js_escape(step)}");'
            if not value:
                value = self._get_sample_value(field)
            return self._emit_type(page_expr, html, field, value)

        # ---------- SELECT ----------
        if act == "select":
            m = _RX_SELECT_FROM.search(step)
            option, field = (m.groups() if m else (None, None))
            if not field:
                q = _QUOTE_RX.search(step)
                field = q.group(1) if q else _norm_space(step.split("select", 1)[-1])
            return self._emit_select(page_expr, html, field, option)

        # ---------- VERIFY ----------
        if act == "verify":
            ls = s.lower()
            if "page title" in ls:
                return 'await expect(page).toHaveTitle(/\\S+/);'
            if "language options" in ls:
                return (
                    'const opts = await page.locator("select,[role=\\"listbox\\"],[aria-label*=Language]").allInnerTexts();\n'
                    'console.log("Language options:", opts);\n'
                    'expect(opts.length).toBeGreaterThan(0);'
                )
            if "headings" in ls:
                return (
                    'const heads = await page.locator("h1,h2,h3").allInnerTexts();\n'
                    'console.log("Detected headings:", heads);\n'
                    'expect(heads.length).toBeGreaterThan(0);'
                )
            if "image" in ls or "alt" in ls:
                # Playwright has no toHaveCountGreaterThan — check count numerically
                return (
                    'const imgCount = await page.locator("img").count();\n'
                    'expect(imgCount).toBeGreaterThan(0);'
                )
            q = _QUOTE_RX.search(step)
            if q:
                text = q.group(1)
                return f'await expect(page.getByText("{_js_escape(text)}")).toBeVisible();'
            return f'console.log("Verify step executed (generic): {_js_escape(step)}");'

        # ---------- DEFAULT ----------
        return f'console.log("TODO: Unhandled step → {_js_escape(step)}");'

    # ----------------------------------------------------------------------
    # SAMPLE DATA GENERATION
    # ----------------------------------------------------------------------
    def _get_sample_value(self, field_name: str) -> str:
        """Provide intelligent mock data for form fields based on name."""
        n = (field_name or "").lower()
        if "email" in n:
            return "test.user@example.com"
        if "password" in n:
            return "Test@1234"
        if "phone" in n or "mobile" in n:
            return "9876543210"
        if "url" in n or "site" in n:
            return "https://example.com"
        if "name" in n:
            return "John Doe"
        if "date" in n:
            return "2025-10-05"
        return "sample text"
