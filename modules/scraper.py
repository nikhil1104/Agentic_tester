# modules/scraper.py
"""
Compatibility shim so legacy imports keep working:
`from modules.scraper import Scraper` now points to the async-only implementation.
"""

from __future__ import annotations

from .async_scraper import AsyncScraper as Scraper, CrawlConfig

__all__ = ["Scraper", "CrawlConfig"]

