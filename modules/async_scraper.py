# modules/async_scraper.py
"""
AsyncScraper — Playwright async-only crawler (API + Security Aware)

- Pure async Playwright (no playwright.sync_api) → no greenlet/thread issues
- HTML snapshots to snapshot_dir
- API capture via context.on('request'/'response'/'requestfailed')
- Response-time + body capture (bounded & timed)
- Security headers summary for 'document' responses
- Results include: pages, api_calls, api_calls_count, security_headers_summary
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Set, Tuple, Dict, Optional, Any
from urllib.parse import urljoin, urldefrag, urlparse

from playwright.async_api import async_playwright, TimeoutError as PWTimeout

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class CrawlConfig:
    # Core crawl settings
    max_pages: int = 15
    max_depth: int = 1
    concurrency: int = 3
    headless: bool = True
    nav_timeout_ms: int = 30_000
    snapshot_dir: str | Path = "data/scraped_docs"  # accepts str or Path
    same_origin_only: bool = True

    # Compatibility with WebScraper wrapper
    start_url: Optional[str] = None
    timeout_ms: int = 30_000
    post_load_wait_ms: int = 1_500
    same_origin: bool = True
    capture_api: bool = True
    api_body_max_bytes: int = 128_000
    rate_limit_delay_s: float = 1.0
    save_text: bool = True
    respect_robots: bool = False
    exclude_patterns: list = field(default_factory=list)
    use_batch_extraction: bool = True
    html5lib_fallback: bool = True
    custom_headers: dict = field(default_factory=dict)
    storage_state_path: Optional[str] = None
    save_metadata: bool = True
    html_dir: str | Path = "data/scraped_docs"
    save_apis_file: bool = True
    timezone_id: Optional[str] = None
    locale: str = "en-US"
    viewport: Dict[str, int] = field(default_factory=lambda: {"width": 1920, "height": 1080})
    user_agent: Optional[str] = None
    blocked_resource_types: set = field(default_factory=lambda: {"image", "font", "media"})
    extra_stealth: bool = True
    api_url_filters: list = field(default_factory=list)
    max_retries: int = 2
    export_har: bool = True
    generate_test_metadata: bool = True
    allowed_api_hosts: list = field(default_factory=list)

    # Browser flags helpful for picky sites (optional)
    disable_http2: bool = False

    def __post_init__(self):
        # Path coercion
        if isinstance(self.snapshot_dir, str):
            self.snapshot_dir = Path(self.snapshot_dir)
        if isinstance(self.html_dir, str):
            self.html_dir = Path(self.html_dir)

        # Ensure dirs
        try:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Failed to create snapshot_dir: %s", e)
        try:
            self.html_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Failed to create html_dir: %s", e)

        # Keep mirrored options in sync
        if self.timeout_ms != self.nav_timeout_ms:
            self.nav_timeout_ms = self.timeout_ms
        self.same_origin_only = self.same_origin

        # Realistic UA by default (helps with some CDNs)
        if not self.user_agent:
            self.user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )


# ==================== Main AsyncScraper ====================

class AsyncScraper:
    """
    Async Playwright crawler. All browser calls remain on a single asyncio loop.
    """

    def __init__(
        self,
        cfg: CrawlConfig | None = None,
        progress_cb: Optional[callable] = None,
        stop_event: Optional[asyncio.Event] = None,
    ):
        self.cfg = cfg or CrawlConfig()
        self._seen: Set[str] = set()
        self._count: int = 0
        self._count_lock = asyncio.Lock()
        self._progress_cb = progress_cb
        self._stop_event = stop_event

        # API tracking (ring buffer)
        self._api_order: deque[str] = deque()
        self._api_tracking: Dict[str, Dict[str, Any]] = {}
        self._max_api_entries = 600

        # Security headers snapshot
        self._sec_hdr_summary = {"pages_checked": 0, "missing_counts": {}}

        logger.info(
            "✅ AsyncScraper initialized (max_pages=%d, max_depth=%d, concurrency=%d)",
            self.cfg.max_pages,
            self.cfg.max_depth,
            self.cfg.concurrency,
        )

    # ---------------- Sync wrapper ----------------

    def deep_scan(
        self,
        start_url: str,
        *,
        max_pages: int | None = None,
        max_depth: int | None = None,
    ) -> Dict[str, Any]:
        if max_pages is not None:
            self.cfg.max_pages = max_pages
        if max_depth is not None:
            self.cfg.max_depth = max_depth
        try:
            return asyncio.run(self.deep_scan_async(start_url))
        except Exception as e:
            logger.error("deep_scan failed: %s", e, exc_info=True)
            return {
                "visited": 0,
                "saved": 0,
                "enqueued": 0,
                "errors": 1,
                "pages": [],
                "api_calls": [],
                "api_calls_count": 0,
                "security_headers_summary": {},
                "error": str(e),
                "start_url": start_url,
            }

    # ---------------- Async entry ----------------

    async def deep_scan_async(self, start_url: str) -> Dict[str, Any]:
        self._seen.clear()
        self._count = 0
        self._api_tracking.clear()
        self._api_order.clear()
        self._sec_hdr_summary = {"pages_checked": 0, "missing_counts": {}}

        start_url = start_url.strip()
        base = urlparse(start_url)
        origin = f"{base.scheme}://{base.netloc}"

        # Ensure dir exists (safe duplicate of __post_init__)
        self.cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)

        sem = asyncio.Semaphore(max(1, self.cfg.concurrency))
        queue: asyncio.Queue[Tuple[str, int]] = asyncio.Queue()
        await queue.put((start_url, 0))

        results: Dict[str, Any] = {
            "visited": 0,
            "saved": 0,
            "enqueued": 0,
            "errors": 0,
            "pages": [],
            "start_url": start_url,
        }

        browser_args = [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ]
        if self.cfg.disable_http2:
            browser_args.append("--disable-http2")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.cfg.headless, args=browser_args)
            context = await browser.new_context(
                ignore_https_errors=True,
                viewport=self.cfg.viewport,
                locale=self.cfg.locale,
                timezone_id=self.cfg.timezone_id,
                user_agent=self.cfg.user_agent,
                extra_http_headers=self.cfg.custom_headers or None,
            )

            # light stealth
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                window.chrome = { runtime: {} };
                Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
            """)

            # -------- API capture handlers --------
            def _host_allowed(url: str) -> bool:
                if not self.cfg.allowed_api_hosts:
                    return True
                try:
                    host = urlparse(url).netloc
                except Exception:
                    return True
                return any(host == h or host.endswith("." + h) for h in self.cfg.allowed_api_hosts)

            def _filter_url(url: str) -> bool:
                if self.cfg.api_url_filters and not any(p in url for p in self.cfg.api_url_filters):
                    return False
                return True

            def _ring_add(key: str, record: Dict[str, Any]):
                # evict if full
                if len(self._api_order) >= self._max_api_entries:
                    old = self._api_order.popleft()
                    self._api_tracking.pop(old, None)
                self._api_tracking[key] = record
                self._api_order.append(key)

            def _on_request(req):
                try:
                    if not self.cfg.capture_api:
                        return
                    rt = req.resource_type or ""
                    accept = (req.headers or {}).get("accept", "")
                    url_raw = req.url

                    if not _filter_url(url_raw) or not _host_allowed(url_raw):
                        return
                    # keep XHR/fetch or explicit JSON accepts
                    if rt not in ("xhr", "fetch") and "application/json" not in str(accept).lower():
                        return

                    key = f"{id(req)}::{req.method}::{url_raw}"
                    rec = {
                        "request": {
                            "method": req.method,
                            "url": url_raw,
                            "headers": dict(req.headers) if req.headers else {},
                            "post_data": getattr(req, "post_data", None),
                            "timestamp": time.time(),
                        }
                    }
                    _ring_add(key, rec)
                except Exception:
                    pass

            async def _handle_response(resp):
                try:
                    if not self.cfg.capture_api:
                        return
                    req = resp.request
                    url_raw = resp.url
                    if not _filter_url(url_raw) or not _host_allowed(url_raw):
                        return

                    # Security headers for 'document'
                    if req.resource_type == "document":
                        self._security_headers_snapshot(resp.headers or {})

                    key = f"{id(req)}::{req.method}::{req.url}"
                    rec = self._api_tracking.get(key)
                    if rec is None:
                        rec = {
                            "request": {
                                "method": req.method,
                                "url": req.url,
                                "timestamp": time.time(),
                            }
                        }
                        _ring_add(key, rec)

                    headers_lower = {k.lower(): v for k, v in (resp.headers or {}).items()}
                    ctype = headers_lower.get("content-type", "") or ""
                    rts = time.time()

                    payload: Dict[str, Any] = {
                        "status": resp.status,
                        "headers": dict(resp.headers or {}),
                        "content_type": ctype,
                        "timestamp": rts,
                    }

                    # body (bounded + timed)
                    if "application/json" in ctype.lower():
                        try:
                            body = await asyncio.wait_for(resp.body(), timeout=5)
                            if body and len(body) <= self.cfg.api_body_max_bytes:
                                try:
                                    import json as _json
                                    payload["json"] = _json.loads(body.decode("utf-8", "ignore"))
                                except Exception:
                                    payload["text"] = body.decode("utf-8", "ignore")[:1000]
                                payload["body_size"] = len(body)
                            else:
                                payload["truncated"] = True
                                payload["actual_size"] = len(body) if body else 0
                        except Exception:
                            payload["body_timeout_or_error"] = True

                    rec["response"] = payload
                    req_ts = rec.get("request", {}).get("timestamp", rts)
                    rec["response_time_ms"] = int((rts - req_ts) * 1000)
                except Exception:
                    pass

            def _on_response(resp):
                # schedule async handler
                asyncio.create_task(_handle_response(resp))

            def _on_request_failed(req):
                try:
                    if not self.cfg.capture_api:
                        return
                    url_raw = req.url
                    if not _filter_url(url_raw) or not _host_allowed(url_raw):
                        return
                    key = f"{id(req)}::{req.method}::{req.url}"
                    started = time.time()
                    rec = {
                        "request": {
                            "method": req.method,
                            "url": req.url,
                            "headers": dict(req.headers) if req.headers else {},
                            "post_data": getattr(req, "post_data", None),
                            "timestamp": started,
                        },
                        "response": {
                            "status": None,
                            "error": getattr(getattr(req, "failure", None), "errorText", None) or "request failed",
                            "timestamp": time.time(),
                        },
                    }
                    rec["response_time_ms"] = int((rec["response"]["timestamp"] - started) * 1000)
                    _ring_add(key, rec)
                except Exception:
                    pass

            context.on("request", _on_request)
            context.on("response", _on_response)
            context.on("requestfailed", _on_request_failed)

            try:
                tasks: list[asyncio.Task] = []
                while not queue.empty() and self._count < self.cfg.max_pages:
                    if self._stop_event and self._stop_event.is_set():
                        logger.info("🛑 Stop signal received, halting crawl")
                        results["stopped"] = True
                        break

                    url, depth = await queue.get()
                    if url in self._seen or depth > self.cfg.max_depth:
                        queue.task_done()
                        continue

                    self._seen.add(url)
                    tasks.append(
                        asyncio.create_task(
                            self._visit_and_extract(
                                url, depth, context, sem, origin, queue, results
                            )
                        )
                    )

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                results["visited"] = self._count

                # Flatten API calls
                api_calls_list: list[Dict[str, Any]] = []
                for key in list(self._api_order):
                    data = self._api_tracking.get(key) or {}
                    resp = data.get("response", {}) or {}
                    api_calls_list.append(
                        {
                            "method": data.get("request", {}).get("method"),
                            "url": data.get("request", {}).get("url"),
                            "request": data.get("request"),
                            "response": resp,
                            "response_time_ms": data.get("response_time_ms", 0),
                            "status": resp.get("status"),
                            "content_type": resp.get("content_type"),
                            "phase": "response" if resp else "request",
                        }
                    )

                results["api_calls"] = api_calls_list
                results["api_calls_count"] = len(api_calls_list)
                results["security_headers_summary"] = self._sec_hdr_summary

                # Optionally persist API telemetry
                if self.cfg.save_apis_file and api_calls_list:
                    import json as _json
                    (self.cfg.html_dir / "apis.json").write_text(
                        _json.dumps({"api_calls": api_calls_list}, indent=2), encoding="utf-8"
                    )

                # Minimal test metadata (avg response time per endpoint)
                if self.cfg.generate_test_metadata and api_calls_list:
                    meta: Dict[str, Any] = {"endpoints": {}, "total_endpoints": 0}
                    for rec in api_calls_list:
                        u = (rec.get("url") or "").split("?")[0]
                        rt = rec.get("response_time_ms", 0) or 0
                        ep = meta["endpoints"].setdefault(u, {"count": 0, "sum_rt": 0, "methods": set(), "statuses": set()})
                        ep["count"] += 1
                        ep["sum_rt"] += rt
                        if rec.get("method"):
                            ep["methods"].add(rec["method"])
                        if rec.get("status") is not None:
                            ep["statuses"].add(rec["status"])
                    for u, ep in meta["endpoints"].items():
                        ep["avg_response_time_ms"] = round(ep["sum_rt"] / max(1, ep["count"]), 2)
                        ep["methods"] = sorted(list(ep["methods"]))
                        ep["statuses"] = sorted(list(ep["statuses"]))
                        del ep["sum_rt"]
                    meta["total_endpoints"] = len(meta["endpoints"])
                    import json as _json
                    (self.cfg.html_dir / "test_metadata.json").write_text(
                        _json.dumps(meta, indent=2), encoding="utf-8"
                    )

                return results
            finally:
                await context.close()
                await browser.close()

    # ---------------- Worker ----------------

    async def _visit_and_extract(
        self,
        url: str,
        depth: int,
        context,
        sem: asyncio.Semaphore,
        origin: str,
        queue: asyncio.Queue,
        results: Dict[str, Any],
    ):
        async with sem:
            async with self._count_lock:
                self._count += 1
                idx = self._count
                total = self.cfg.max_pages

            logger.info("🔍 Scraping [%d/%d] depth=%d: %s", idx, total, depth, url)

            if self._progress_cb:
                try:
                    self._progress_cb(
                        {"type": "page_start", "url": url, "depth": depth, "index": idx}
                    )
                except Exception as e:
                    logger.debug("progress_cb failed: %s", e)

            page = await context.new_page()
            page.set_default_timeout(self.cfg.nav_timeout_ms)

            try:
                try:
                    await page.goto(url, wait_until="networkidle", timeout=self.cfg.nav_timeout_ms)
                except PWTimeout:
                    logger.debug("networkidle timeout → domcontentloaded fallback: %s", url)
                    await page.goto(url, wait_until="domcontentloaded", timeout=self.cfg.nav_timeout_ms)
            except Exception as e:
                logger.error("Navigation failed: %s (%s)", url, e)
                results["errors"] += 1
                await page.close()
                return

            try:
                await page.wait_for_load_state(
                    "networkidle", timeout=min(8_000, self.cfg.nav_timeout_ms)
                )
            except Exception:
                pass

            await page.wait_for_timeout(self.cfg.post_load_wait_ms)

            # Snapshot
            try:
                html = await page.content()
                title = await page.title()
                out_path = self._save_snapshot(url, html)
                results["saved"] += 1
                results["pages"].append(
                    {"url": url, "path": str(out_path), "title": title, "depth": depth}
                )
                if self._progress_cb:
                    try:
                        self._progress_cb(
                            {
                                "type": "page_done",
                                "url": url,
                                "title": title,
                                "depth": depth,
                                "length": len(html),
                            }
                        )
                    except Exception as e:
                        logger.debug("progress_cb failed: %s", e)
            except Exception as e:
                logger.warning("Snapshot failed for %s: %s", url, e)
                results["errors"] += 1

            # Links
            try:
                links = await self._extract_links(page, url)
                enq = 0
                for link in links:
                    if self._should_enqueue(link, origin) and link not in self._seen:
                        if self._count + enq >= self.cfg.max_pages:
                            break
                        await queue.put((link, depth + 1))
                        enq += 1
                results["enqueued"] += enq
                if enq:
                    logger.debug("   🔗 Added %d new links to queue (depth %d)", enq, depth + 1)
            except Exception as e:
                logger.debug("Link extraction failed for %s: %s", url, e)

            await page.close()

    # ---------------- Helpers ----------------

    async def _extract_links(self, page, base_url: str) -> Iterable[str]:
        try:
            hrefs = await page.eval_on_selector_all(
                "a[href]", "els => els.map(e => e.getAttribute('href'))"
            )
        except Exception as e:
            logger.debug("Link extraction failed: %s", e)
            return set()

        links: Set[str] = set()
        for href in hrefs:
            if not href:
                continue
            href = href.strip()
            if re.match(r"^(javascript:|mailto:|tel:|#)", href, re.I):
                continue
            absolute = urljoin(base_url, href)
            absolute, _ = urldefrag(absolute)
            if absolute.startswith(("http://", "https://")):
                links.add(absolute)
        return links

    def _should_enqueue(self, url: str, origin: str) -> bool:
        if not self.cfg.same_origin_only:
            return True
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}" == origin

    def _safe_name(self, url: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", url)[:240]

    def _save_snapshot(self, url: str, html: str) -> Path:
        out = self.cfg.snapshot_dir / f"{self._safe_name(url)}.html"
        out.write_text(html, encoding="utf-8")
        logger.debug("💾 Saved snapshot: %s", out)
        return out

    # -------- Security headers quick snapshot --------
    _REQ_HEADERS_REQUIRED = {
        "content-security-policy",
        "x-content-type-options",
        "x-frame-options",
        "referrer-policy",
        "strict-transport-security",
        "permissions-policy",
    }

    def _security_headers_snapshot(self, headers: Dict[str, str]) -> None:
        low = {k.lower(): v for k, v in (headers or {}).items()}
        missing = [h for h in self._REQ_HEADERS_REQUIRED if not low.get(h)]
        self._sec_hdr_summary["pages_checked"] += 1
        for h in missing:
            self._sec_hdr_summary["missing_counts"][h] = self._sec_hdr_summary["missing_counts"].get(h, 0) + 1

    # ---------------- Compatibility ----------------

    async def run(self) -> Dict[str, Any]:
        if not self.cfg.start_url:
            raise ValueError("start_url not set in config")
        return await self.deep_scan_async(self.cfg.start_url)


# ---------------- CLI demo ----------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m modules.async_scraper <url> [max_pages] [max_depth]")
        sys.exit(1)

    url = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    cfg = CrawlConfig(max_pages=max_pages, max_depth=max_depth, concurrency=3, headless=True)
    scraper = AsyncScraper(cfg)
    res = scraper.deep_scan(url)
    print(res)

# modules/async_scraper.py
"""
AsyncScraper — Playwright async-only crawler (API + Security Aware)

- Pure async Playwright (no playwright.sync_api) → no greenlet/thread issues
- HTML snapshots to snapshot_dir
- API capture via context.on('request'/'response'/'requestfailed')
- Response-time + body capture (bounded & timed)
- Security headers summary for 'document' responses
- Results include: pages, api_calls, api_calls_count, security_headers_summary
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Set, Tuple, Dict, Optional, Any
from urllib.parse import urljoin, urldefrag, urlparse

from playwright.async_api import async_playwright, TimeoutError as PWTimeout

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class CrawlConfig:
    # Core crawl settings
    max_pages: int = 15
    max_depth: int = 1
    concurrency: int = 3
    headless: bool = True
    nav_timeout_ms: int = 30_000
    snapshot_dir: str | Path = "data/scraped_docs"  # accepts str or Path
    same_origin_only: bool = True

    # Compatibility with WebScraper wrapper
    start_url: Optional[str] = None
    timeout_ms: int = 30_000
    post_load_wait_ms: int = 1_500
    same_origin: bool = True
    capture_api: bool = True
    api_body_max_bytes: int = 128_000
    rate_limit_delay_s: float = 1.0
    save_text: bool = True
    respect_robots: bool = False
    exclude_patterns: list = field(default_factory=list)
    use_batch_extraction: bool = True
    html5lib_fallback: bool = True
    custom_headers: dict = field(default_factory=dict)
    storage_state_path: Optional[str] = None
    save_metadata: bool = True
    html_dir: str | Path = "data/scraped_docs"
    save_apis_file: bool = True
    timezone_id: Optional[str] = None
    locale: str = "en-US"
    viewport: Dict[str, int] = field(default_factory=lambda: {"width": 1920, "height": 1080})
    user_agent: Optional[str] = None
    blocked_resource_types: set = field(default_factory=lambda: {"image", "font", "media"})
    extra_stealth: bool = True
    api_url_filters: list = field(default_factory=list)
    max_retries: int = 2
    export_har: bool = True
    generate_test_metadata: bool = True
    allowed_api_hosts: list = field(default_factory=list)

    # Browser flags helpful for picky sites (optional)
    disable_http2: bool = False

    def __post_init__(self):
        # Path coercion
        if isinstance(self.snapshot_dir, str):
            self.snapshot_dir = Path(self.snapshot_dir)
        if isinstance(self.html_dir, str):
            self.html_dir = Path(self.html_dir)

        # Ensure dirs
        try:
            self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Failed to create snapshot_dir: %s", e)
        try:
            self.html_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.warning("Failed to create html_dir: %s", e)

        # Keep mirrored options in sync
        if self.timeout_ms != self.nav_timeout_ms:
            self.nav_timeout_ms = self.timeout_ms
        self.same_origin_only = self.same_origin

        # Realistic UA by default (helps with some CDNs)
        if not self.user_agent:
            self.user_agent = (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            )


# ==================== Main AsyncScraper ====================

class AsyncScraper:
    """
    Async Playwright crawler. All browser calls remain on a single asyncio loop.
    """

    def __init__(
        self,
        cfg: CrawlConfig | None = None,
        progress_cb: Optional[callable] = None,
        stop_event: Optional[asyncio.Event] = None,
    ):
        self.cfg = cfg or CrawlConfig()
        self._seen: Set[str] = set()
        self._count: int = 0
        self._count_lock = asyncio.Lock()
        self._progress_cb = progress_cb
        self._stop_event = stop_event

        # API tracking (ring buffer)
        self._api_order: deque[str] = deque()
        self._api_tracking: Dict[str, Dict[str, Any]] = {}
        self._max_api_entries = 600

        # Security headers snapshot
        self._sec_hdr_summary = {"pages_checked": 0, "missing_counts": {}}

        logger.info(
            "✅ AsyncScraper initialized (max_pages=%d, max_depth=%d, concurrency=%d)",
            self.cfg.max_pages,
            self.cfg.max_depth,
            self.cfg.concurrency,
        )

    # ---------------- Sync wrapper ----------------

    def deep_scan(
        self,
        start_url: str,
        *,
        max_pages: int | None = None,
        max_depth: int | None = None,
    ) -> Dict[str, Any]:
        if max_pages is not None:
            self.cfg.max_pages = max_pages
        if max_depth is not None:
            self.cfg.max_depth = max_depth
        try:
            return asyncio.run(self.deep_scan_async(start_url))
        except Exception as e:
            logger.error("deep_scan failed: %s", e, exc_info=True)
            return {
                "visited": 0,
                "saved": 0,
                "enqueued": 0,
                "errors": 1,
                "pages": [],
                "api_calls": [],
                "api_calls_count": 0,
                "security_headers_summary": {},
                "error": str(e),
                "start_url": start_url,
            }

    # ---------------- Async entry ----------------

    async def deep_scan_async(self, start_url: str) -> Dict[str, Any]:
        self._seen.clear()
        self._count = 0
        self._api_tracking.clear()
        self._api_order.clear()
        self._sec_hdr_summary = {"pages_checked": 0, "missing_counts": {}}

        start_url = start_url.strip()
        base = urlparse(start_url)
        origin = f"{base.scheme}://{base.netloc}"

        # Ensure dir exists (safe duplicate of __post_init__)
        self.cfg.snapshot_dir.mkdir(parents=True, exist_ok=True)

        sem = asyncio.Semaphore(max(1, self.cfg.concurrency))
        queue: asyncio.Queue[Tuple[str, int]] = asyncio.Queue()
        await queue.put((start_url, 0))

        results: Dict[str, Any] = {
            "visited": 0,
            "saved": 0,
            "enqueued": 0,
            "errors": 0,
            "pages": [],
            "start_url": start_url,
        }

        browser_args = [
            "--no-sandbox",
            "--disable-setuid-sandbox",
            "--disable-dev-shm-usage",
            "--disable-blink-features=AutomationControlled",
        ]
        if self.cfg.disable_http2:
            browser_args.append("--disable-http2")

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=self.cfg.headless, args=browser_args)
            context = await browser.new_context(
                ignore_https_errors=True,
                viewport=self.cfg.viewport,
                locale=self.cfg.locale,
                timezone_id=self.cfg.timezone_id,
                user_agent=self.cfg.user_agent,
                extra_http_headers=self.cfg.custom_headers or None,
            )

            # light stealth
            await context.add_init_script("""
                Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
                Object.defineProperty(navigator, 'languages', { get: () => ['en-US', 'en'] });
                window.chrome = { runtime: {} };
                Object.defineProperty(navigator, 'plugins', { get: () => [1,2,3,4,5] });
            """)

            # -------- API capture handlers --------
            def _host_allowed(url: str) -> bool:
                if not self.cfg.allowed_api_hosts:
                    return True
                try:
                    host = urlparse(url).netloc
                except Exception:
                    return True
                return any(host == h or host.endswith("." + h) for h in self.cfg.allowed_api_hosts)

            def _filter_url(url: str) -> bool:
                if self.cfg.api_url_filters and not any(p in url for p in self.cfg.api_url_filters):
                    return False
                return True

            def _ring_add(key: str, record: Dict[str, Any]):
                # evict if full
                if len(self._api_order) >= self._max_api_entries:
                    old = self._api_order.popleft()
                    self._api_tracking.pop(old, None)
                self._api_tracking[key] = record
                self._api_order.append(key)

            def _on_request(req):
                try:
                    if not self.cfg.capture_api:
                        return
                    rt = req.resource_type or ""
                    accept = (req.headers or {}).get("accept", "")
                    url_raw = req.url

                    if not _filter_url(url_raw) or not _host_allowed(url_raw):
                        return
                    # keep XHR/fetch or explicit JSON accepts
                    if rt not in ("xhr", "fetch") and "application/json" not in str(accept).lower():
                        return

                    key = f"{id(req)}::{req.method}::{url_raw}"
                    rec = {
                        "request": {
                            "method": req.method,
                            "url": url_raw,
                            "headers": dict(req.headers) if req.headers else {},
                            "post_data": getattr(req, "post_data", None),
                            "timestamp": time.time(),
                        }
                    }
                    _ring_add(key, rec)
                except Exception:
                    pass

            async def _handle_response(resp):
                try:
                    if not self.cfg.capture_api:
                        return
                    req = resp.request
                    url_raw = resp.url
                    if not _filter_url(url_raw) or not _host_allowed(url_raw):
                        return

                    # Security headers for 'document'
                    if req.resource_type == "document":
                        self._security_headers_snapshot(resp.headers or {})

                    key = f"{id(req)}::{req.method}::{req.url}"
                    rec = self._api_tracking.get(key)
                    if rec is None:
                        rec = {
                            "request": {
                                "method": req.method,
                                "url": req.url,
                                "timestamp": time.time(),
                            }
                        }
                        _ring_add(key, rec)

                    headers_lower = {k.lower(): v for k, v in (resp.headers or {}).items()}
                    ctype = headers_lower.get("content-type", "") or ""
                    rts = time.time()

                    payload: Dict[str, Any] = {
                        "status": resp.status,
                        "headers": dict(resp.headers or {}),
                        "content_type": ctype,
                        "timestamp": rts,
                    }

                    # body (bounded + timed)
                    if "application/json" in ctype.lower():
                        try:
                            body = await asyncio.wait_for(resp.body(), timeout=5)
                            if body and len(body) <= self.cfg.api_body_max_bytes:
                                try:
                                    import json as _json
                                    payload["json"] = _json.loads(body.decode("utf-8", "ignore"))
                                except Exception:
                                    payload["text"] = body.decode("utf-8", "ignore")[:1000]
                                payload["body_size"] = len(body)
                            else:
                                payload["truncated"] = True
                                payload["actual_size"] = len(body) if body else 0
                        except Exception:
                            payload["body_timeout_or_error"] = True

                    rec["response"] = payload
                    req_ts = rec.get("request", {}).get("timestamp", rts)
                    rec["response_time_ms"] = int((rts - req_ts) * 1000)
                except Exception:
                    pass

            def _on_response(resp):
                # schedule async handler
                asyncio.create_task(_handle_response(resp))

            def _on_request_failed(req):
                try:
                    if not self.cfg.capture_api:
                        return
                    url_raw = req.url
                    if not _filter_url(url_raw) or not _host_allowed(url_raw):
                        return
                    key = f"{id(req)}::{req.method}::{req.url}"
                    started = time.time()
                    rec = {
                        "request": {
                            "method": req.method,
                            "url": req.url,
                            "headers": dict(req.headers) if req.headers else {},
                            "post_data": getattr(req, "post_data", None),
                            "timestamp": started,
                        },
                        "response": {
                            "status": None,
                            "error": getattr(getattr(req, "failure", None), "errorText", None) or "request failed",
                            "timestamp": time.time(),
                        },
                    }
                    rec["response_time_ms"] = int((rec["response"]["timestamp"] - started) * 1000)
                    _ring_add(key, rec)
                except Exception:
                    pass

            context.on("request", _on_request)
            context.on("response", _on_response)
            context.on("requestfailed", _on_request_failed)

            try:
                tasks: list[asyncio.Task] = []
                while not queue.empty() and self._count < self.cfg.max_pages:
                    if self._stop_event and self._stop_event.is_set():
                        logger.info("🛑 Stop signal received, halting crawl")
                        results["stopped"] = True
                        break

                    url, depth = await queue.get()
                    if url in self._seen or depth > self.cfg.max_depth:
                        queue.task_done()
                        continue

                    self._seen.add(url)
                    tasks.append(
                        asyncio.create_task(
                            self._visit_and_extract(
                                url, depth, context, sem, origin, queue, results
                            )
                        )
                    )

                if tasks:
                    await asyncio.gather(*tasks, return_exceptions=True)

                results["visited"] = self._count

                # Flatten API calls
                api_calls_list: list[Dict[str, Any]] = []
                for key in list(self._api_order):
                    data = self._api_tracking.get(key) or {}
                    resp = data.get("response", {}) or {}
                    api_calls_list.append(
                        {
                            "method": data.get("request", {}).get("method"),
                            "url": data.get("request", {}).get("url"),
                            "request": data.get("request"),
                            "response": resp,
                            "response_time_ms": data.get("response_time_ms", 0),
                            "status": resp.get("status"),
                            "content_type": resp.get("content_type"),
                            "phase": "response" if resp else "request",
                        }
                    )

                results["api_calls"] = api_calls_list
                results["api_calls_count"] = len(api_calls_list)
                results["security_headers_summary"] = self._sec_hdr_summary

                # Optionally persist API telemetry
                if self.cfg.save_apis_file and api_calls_list:
                    import json as _json
                    (self.cfg.html_dir / "apis.json").write_text(
                        _json.dumps({"api_calls": api_calls_list}, indent=2), encoding="utf-8"
                    )

                # Minimal test metadata (avg response time per endpoint)
                if self.cfg.generate_test_metadata and api_calls_list:
                    meta: Dict[str, Any] = {"endpoints": {}, "total_endpoints": 0}
                    for rec in api_calls_list:
                        u = (rec.get("url") or "").split("?")[0]
                        rt = rec.get("response_time_ms", 0) or 0
                        ep = meta["endpoints"].setdefault(u, {"count": 0, "sum_rt": 0, "methods": set(), "statuses": set()})
                        ep["count"] += 1
                        ep["sum_rt"] += rt
                        if rec.get("method"):
                            ep["methods"].add(rec["method"])
                        if rec.get("status") is not None:
                            ep["statuses"].add(rec["status"])
                    for u, ep in meta["endpoints"].items():
                        ep["avg_response_time_ms"] = round(ep["sum_rt"] / max(1, ep["count"]), 2)
                        ep["methods"] = sorted(list(ep["methods"]))
                        ep["statuses"] = sorted(list(ep["statuses"]))
                        del ep["sum_rt"]
                    meta["total_endpoints"] = len(meta["endpoints"])
                    import json as _json
                    (self.cfg.html_dir / "test_metadata.json").write_text(
                        _json.dumps(meta, indent=2), encoding="utf-8"
                    )

                return results
            finally:
                await context.close()
                await browser.close()

    # ---------------- Worker ----------------

    async def _visit_and_extract(
        self,
        url: str,
        depth: int,
        context,
        sem: asyncio.Semaphore,
        origin: str,
        queue: asyncio.Queue,
        results: Dict[str, Any],
    ):
        async with sem:
            async with self._count_lock:
                self._count += 1
                idx = self._count
                total = self.cfg.max_pages

            logger.info("🔍 Scraping [%d/%d] depth=%d: %s", idx, total, depth, url)

            if self._progress_cb:
                try:
                    self._progress_cb(
                        {"type": "page_start", "url": url, "depth": depth, "index": idx}
                    )
                except Exception as e:
                    logger.debug("progress_cb failed: %s", e)

            page = await context.new_page()
            page.set_default_timeout(self.cfg.nav_timeout_ms)

            try:
                try:
                    await page.goto(url, wait_until="networkidle", timeout=self.cfg.nav_timeout_ms)
                except PWTimeout:
                    logger.debug("networkidle timeout → domcontentloaded fallback: %s", url)
                    await page.goto(url, wait_until="domcontentloaded", timeout=self.cfg.nav_timeout_ms)
            except Exception as e:
                logger.error("Navigation failed: %s (%s)", url, e)
                results["errors"] += 1
                await page.close()
                return

            try:
                await page.wait_for_load_state(
                    "networkidle", timeout=min(8_000, self.cfg.nav_timeout_ms)
                )
            except Exception:
                pass

            await page.wait_for_timeout(self.cfg.post_load_wait_ms)

            # Snapshot
            try:
                html = await page.content()
                title = await page.title()
                out_path = self._save_snapshot(url, html)
                results["saved"] += 1
                results["pages"].append(
                    {"url": url, "path": str(out_path), "title": title, "depth": depth}
                )
                if self._progress_cb:
                    try:
                        self._progress_cb(
                            {
                                "type": "page_done",
                                "url": url,
                                "title": title,
                                "depth": depth,
                                "length": len(html),
                            }
                        )
                    except Exception as e:
                        logger.debug("progress_cb failed: %s", e)
            except Exception as e:
                logger.warning("Snapshot failed for %s: %s", url, e)
                results["errors"] += 1

            # Links
            try:
                links = await self._extract_links(page, url)
                enq = 0
                for link in links:
                    if self._should_enqueue(link, origin) and link not in self._seen:
                        if self._count + enq >= self.cfg.max_pages:
                            break
                        await queue.put((link, depth + 1))
                        enq += 1
                results["enqueued"] += enq
                if enq:
                    logger.debug("   🔗 Added %d new links to queue (depth %d)", enq, depth + 1)
            except Exception as e:
                logger.debug("Link extraction failed for %s: %s", url, e)

            await page.close()

    # ---------------- Helpers ----------------

    async def _extract_links(self, page, base_url: str) -> Iterable[str]:
        try:
            hrefs = await page.eval_on_selector_all(
                "a[href]", "els => els.map(e => e.getAttribute('href'))"
            )
        except Exception as e:
            logger.debug("Link extraction failed: %s", e)
            return set()

        links: Set[str] = set()
        for href in hrefs:
            if not href:
                continue
            href = href.strip()
            if re.match(r"^(javascript:|mailto:|tel:|#)", href, re.I):
                continue
            absolute = urljoin(base_url, href)
            absolute, _ = urldefrag(absolute)
            if absolute.startswith(("http://", "https://")):
                links.add(absolute)
        return links

    def _should_enqueue(self, url: str, origin: str) -> bool:
        if not self.cfg.same_origin_only:
            return True
        p = urlparse(url)
        return f"{p.scheme}://{p.netloc}" == origin

    def _safe_name(self, url: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", url)[:240]

    def _save_snapshot(self, url: str, html: str) -> Path:
        out = self.cfg.snapshot_dir / f"{self._safe_name(url)}.html"
        out.write_text(html, encoding="utf-8")
        logger.debug("💾 Saved snapshot: %s", out)
        return out

    # -------- Security headers quick snapshot --------
    _REQ_HEADERS_REQUIRED = {
        "content-security-policy",
        "x-content-type-options",
        "x-frame-options",
        "referrer-policy",
        "strict-transport-security",
        "permissions-policy",
    }

    def _security_headers_snapshot(self, headers: Dict[str, str]) -> None:
        low = {k.lower(): v for k, v in (headers or {}).items()}
        missing = [h for h in self._REQ_HEADERS_REQUIRED if not low.get(h)]
        self._sec_hdr_summary["pages_checked"] += 1
        for h in missing:
            self._sec_hdr_summary["missing_counts"][h] = self._sec_hdr_summary["missing_counts"].get(h, 0) + 1

    # ---------------- Compatibility ----------------

    async def run(self) -> Dict[str, Any]:
        if not self.cfg.start_url:
            raise ValueError("start_url not set in config")
        return await self.deep_scan_async(self.cfg.start_url)


# ---------------- CLI demo ----------------

if __name__ == "__main__":
    import sys

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
    if len(sys.argv) < 2:
        print("Usage: python -m modules.async_scraper <url> [max_pages] [max_depth]")
        sys.exit(1)

    url = sys.argv[1]
    max_pages = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    max_depth = int(sys.argv[3]) if len(sys.argv) > 3 else 1

    cfg = CrawlConfig(max_pages=max_pages, max_depth=max_depth, concurrency=3, headless=True)
    scraper = AsyncScraper(cfg)
    res = scraper.deep_scan(url)
    print(res)
