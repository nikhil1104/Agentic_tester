"""
DOM Analyzer – Phase 2.5
------------------------
Inspects saved HTML pages (from web_scraper) and extracts actionable UI elements.
Produces a structured summary the TestGenerator can consume.
"""

from bs4 import BeautifulSoup
import json, re, os

class DOMAnalyzer:
    def __init__(self):
        pass

    def analyze_page(self, html_content: str, url: str = "") -> dict:
        """Return high-level element summary."""
        soup = BeautifulSoup(html_content, "html.parser")

        buttons = [
            self._text_or_attr(btn, ["value", "aria-label", "alt", "title"])
            for btn in soup.find_all(["button", "input"], {"type": re.compile("button|submit")})
        ]
        inputs = [
            self._text_or_attr(inp, ["name", "placeholder", "id", "aria-label"])
            for inp in soup.find_all("input", {"type": re.compile("text|email|password|search")})
        ]
        links = [
            self._text_or_attr(a, ["href", "title"])
            for a in soup.find_all("a") if a.get("href")
        ]
        selects = [
            self._text_or_attr(sel, ["name", "id"])
            for sel in soup.find_all("select")
        ]

        return {
            "url": url,
            "buttons": [b for b in buttons if b],
            "inputs": [i for i in inputs if i],
            "links": [l for l in links if l],
            "selects": [s for s in selects if s],
        }

    def analyze_all(self, scraped_docs_dir="data/scraped_docs") -> list:
        """Iterate all saved HTML files."""
        summaries = []
        for file in os.listdir(scraped_docs_dir):
            if not file.endswith(".html") and not file.endswith(".md"):
                continue
            with open(os.path.join(scraped_docs_dir, file), "r", encoding="utf-8") as f:
                html = f.read()
            summaries.append(self.analyze_page(html, url=file))
        return summaries

    def _text_or_attr(self, tag, attrs):
        if tag is None:
            return None
        txt = tag.get_text(strip=True)
        if txt:
            return txt
        for a in attrs:
            val = tag.get(a)
            if val:
                return val
        return None
