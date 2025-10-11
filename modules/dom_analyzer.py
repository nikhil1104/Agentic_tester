# modules/dom_analyzer.py
"""
DOM Analyzer – Phase 6 (Hardened, RAG-friendly)
----------------------------------------------
Parses saved HTML/MD snapshots and returns a structured, deduplicated summary
optimized for:
  • test generation (locator candidates per element)
  • self-healing heuristics (labels/text/aria/name/id)
  • RAG/document indexing (counts, title, links)
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import urljoin

from bs4 import BeautifulSoup

# Optional markdown → HTML (kept optional to avoid hard dep)
try:
    import markdown as _md  # type: ignore
    _MD_AVAILABLE = True
except Exception:
    _MD_AVAILABLE = False

# Prefer 'lxml' for robustness if installed, else stdlib parser
try:
    import lxml  # noqa: F401
    _DEFAULT_PARSER = "lxml"
except Exception:
    _DEFAULT_PARSER = "html.parser"

logger = logging.getLogger(__name__)


# ----------------------------- Data Models -----------------------------
@dataclass
class ElementMeta:
    """Generic element record with locator candidates."""
    kind: str                     # button|input|link|select
    text: Optional[str] = None
    href: Optional[str] = None
    id: Optional[str] = None
    name: Optional[str] = None
    aria_label: Optional[str] = None
    placeholder: Optional[str] = None
    type: Optional[str] = None       # input type, button type, etc.
    label_text: Optional[str] = None # resolved <label for=...> text if present
    role: Optional[str] = None
    extra: Optional[Dict[str, Any]] = None  # extensibility


# ----------------------------- Analyzer -----------------------------
class DOMAnalyzer:
    def __init__(
        self,
        *,
        parser: str = _DEFAULT_PARSER,
        max_items_per_category: int = 200,
        include_headings: bool = True,
    ):
        """
        parser: bs4 parser to use ('lxml' if available, else 'html.parser')
        max_items_per_category: cap results for predictability/perf
        include_headings: optionally include H1..H4 strings
        """
        self.parser = parser
        self.max_items = max_items_per_category
        self.include_headings = include_headings

    # ------------------------ Public API ------------------------
    def analyze_page(self, html_content: str, url: str = "") -> Dict[str, Any]:
        """Return structured summary for a single HTML/MD content string."""
        if not html_content:
            return self._empty_summary(url)

        # If this looks like raw markdown, try to convert (optional)
        if self._looks_like_markdown(html_content):
            html_content = self._md_to_html(html_content)

        soup = BeautifulSoup(html_content, self.parser)
        self._strip_noise(soup)

        title = (soup.title.string.strip() if soup.title and soup.title.string else "") or "Untitled"

        buttons = self._collect_buttons(soup)
        inputs = self._collect_inputs(soup)
        selects = self._collect_selects(soup)
        links = self._collect_links(soup, base_url=url)
        headings = self._collect_headings(soup) if self.include_headings else []

        # Deduplicate + cap
        buttons = self._dedup_cap(buttons, key=("text", "id", "name", "aria_label"))
        inputs = self._dedup_cap(inputs, key=("id", "name", "aria_label", "placeholder"))
        selects = self._dedup_cap(selects, key=("id", "name", "aria_label"))
        links = self._dedup_cap(links, key=("text", "href"))
        headings = self._dedup_list(headings)

        summary = {
            "url": url,
            "title": title,
            "counts": {
                "buttons": len(buttons),
                "inputs": len(inputs),
                "selects": len(selects),
                "links": len(links),
                "headings": len(headings),
            },
            "buttons": [asdict(b) for b in buttons],
            "inputs": [asdict(i) for i in inputs],
            "selects": [asdict(s) for s in selects],
            "links": [asdict(l) for l in links],
            "headings": headings,
        }
        return summary

    def analyze_all(self, scraped_docs_dir: str = "data/scraped_docs") -> List[Dict[str, Any]]:
        """Analyze every .html/.md file in a directory, sorted by filename."""
        out: List[Dict[str, Any]] = []
        root = Path(scraped_docs_dir)
        if not root.exists():
            logger.info("scraped_docs_dir %s not found", scraped_docs_dir)
            return out

        files = sorted([p for p in root.iterdir() if p.suffix.lower() in (".html", ".md")])
        for p in files:
            try:
                html = p.read_text(encoding="utf-8")
            except Exception:
                logger.exception("Failed to read %s", p)
                continue
            out.append(self.analyze_page(html, url=str(p.name)))
        return out

    # ------------------------ Helpers: parsing & cleanup ------------------------
    @staticmethod
    def _looks_like_markdown(text: str) -> bool:
        """Heuristic: if raw HTML tags are rare and MD syntax is present."""
        if "<html" in text.lower() or "<body" in text.lower():
            return False
        # simple md markers
        return bool(re.search(r"(^#\s+)|(\[.+?\]\(.+?\))|(^-\s+)|(^\*\s+)", text, re.M))

    @staticmethod
    def _md_to_html(md: str) -> str:
        """Best-effort MD → HTML (optional)."""
        if _MD_AVAILABLE:
            try:
                return _md.markdown(md)
            except Exception:
                pass
        # ultra-minimal fallback: convert basic links and headings
        html = md
        html = re.sub(r"^# (.+)$", r"<h1>\1</h1>", html, flags=re.M)
        html = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', html)
        return f"<html><body>{html}</body></html>"

    @staticmethod
    def _strip_noise(soup: BeautifulSoup) -> None:
        """Remove scripts/styles to keep only meaningful DOM."""
        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()

    @staticmethod
    def _text(tag) -> str:
        return (tag.get_text(" ", strip=True) or "").strip()

    @staticmethod
    def _label_for(soup: BeautifulSoup, elem_id: Optional[str]) -> Optional[str]:
        if not elem_id:
            return None
        lbl = soup.find("label", attrs={"for": elem_id})
        return (lbl.get_text(" ", strip=True).strip() if lbl else None)

    # ------------------------ Collectors ------------------------
    def _collect_buttons(self, soup: BeautifulSoup) -> List[ElementMeta]:
        out: List[ElementMeta] = []

        # <button>...</button>
        for btn in soup.find_all("button"):
            out.append(ElementMeta(
                kind="button",
                text=self._text(btn) or btn.get("value") or None,
                id=btn.get("id"),
                name=btn.get("name"),
                aria_label=btn.get("aria-label"),
                type=btn.get("type"),
                role=btn.get("role"),
            ))

        # <input type=button|submit|reset>
        for inp in soup.find_all("input", {"type": re.compile(r"^(button|submit|reset)$", re.I)}):
            out.append(ElementMeta(
                kind="button",
                text=inp.get("value") or inp.get("aria-label"),
                id=inp.get("id"),
                name=inp.get("name"),
                aria_label=inp.get("aria-label"),
                type=inp.get("type"),
                role=inp.get("role"),
            ))

        # role="button" on div/span/etc
        for tag in soup.find_all(attrs={"role": re.compile(r"button", re.I)}):
            out.append(ElementMeta(
                kind="button",
                text=self._text(tag) or tag.get("aria-label"),
                id=tag.get("id"),
                name=tag.get("name"),
                aria_label=tag.get("aria-label"),
                role=tag.get("role"),
            ))

        return out

    def _collect_inputs(self, soup: BeautifulSoup) -> List[ElementMeta]:
        out: List[ElementMeta] = []

        # <input> (all types except hidden)
        for inp in soup.find_all("input"):
            itype = (inp.get("type") or "text").lower()
            if itype == "hidden":
                continue
            em = ElementMeta(
                kind="input",
                text=None,
                id=inp.get("id"),
                name=inp.get("name"),
                aria_label=inp.get("aria-label"),
                placeholder=inp.get("placeholder"),
                type=itype,
                role=inp.get("role"),
            )
            em.label_text = self._label_for(soup, em.id)
            out.append(em)

        # <textarea>
        for ta in soup.find_all("textarea"):
            em = ElementMeta(
                kind="input",
                id=ta.get("id"),
                name=ta.get("name"),
                aria_label=ta.get("aria-label"),
                placeholder=ta.get("placeholder"),
                type="textarea",
                role=ta.get("role"),
            )
            em.label_text = self._label_for(soup, em.id)
            out.append(em)

        return out

    def _collect_selects(self, soup: BeautifulSoup) -> List[ElementMeta]:
        out: List[ElementMeta] = []
        for sel in soup.find_all("select"):
            options = [self._text(o) for o in sel.find_all("option") if self._text(o)]
            out.append(ElementMeta(
                kind="select",
                text=None,
                id=sel.get("id"),
                name=sel.get("name"),
                aria_label=sel.get("aria-label"),
                type="select",
                role=sel.get("role"),
                extra={"options": options[:100]} if options else None,
            ))
        return out

    def _collect_links(self, soup: BeautifulSoup, base_url: str = "") -> List[ElementMeta]:
        out: List[ElementMeta] = []
        for a in soup.find_all("a", href=True):
            href = a.get("href", "").strip()
            if not href or href.lower().startswith(("javascript:", "mailto:")):
                continue
            abs_href = urljoin(base_url, href) if base_url else href
            out.append(ElementMeta(
                kind="link",
                text=self._text(a) or a.get("title"),
                href=abs_href,
                id=a.get("id"),
                name=a.get("name"),
                aria_label=a.get("aria-label"),
                role=a.get("role"),
            ))
        return out

    def _collect_headings(self, soup: BeautifulSoup) -> List[str]:
        heads: List[str] = []
        for tag in ("h1", "h2", "h3", "h4"):
            for h in soup.find_all(tag):
                t = self._text(h)
                if t:
                    heads.append(t)
        return heads

    # ------------------------ Dedup & Limits ------------------------
    def _dedup_cap(self, items: List[ElementMeta], key: Tuple[str, ...]) -> List[ElementMeta]:
        seen: set = set()
        out: List[ElementMeta] = []
        for it in items:
            k = tuple(getattr(it, fld, None) for fld in key)
            if k in seen:
                continue
            seen.add(k)
            out.append(it)
            if len(out) >= self.max_items:
                break
        return out

    @staticmethod
    def _dedup_list(items: List[str]) -> List[str]:
        seen = set()
        out = []
        for x in items:
            if x not in seen:
                seen.add(x)
                out.append(x)
        return out

    # ------------------------ Misc ------------------------
    @staticmethod
    def _empty_summary(url: str) -> Dict[str, Any]:
        return {
            "url": url,
            "title": "",
            "counts": {"buttons": 0, "inputs": 0, "selects": 0, "links": 0, "headings": 0},
            "buttons": [],
            "inputs": [],
            "selects": [],
            "links": [],
            "headings": [],
        }

    # Optional: export helper (convenience for debugging / offline)
    @staticmethod
    def save_summary(summary: Dict[str, Any], path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(summary, indent=2), encoding="utf-8")
