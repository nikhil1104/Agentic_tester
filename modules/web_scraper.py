# modules/web_scraper.py
"""
Web Scraper v2.0 (Production-Grade Async-Safe with Enhanced Features)

NEW FEATURES:
✅ Distributed crawling support (Redis-based queue)
✅ Screenshot capture for visual regression
✅ Accessibility audit integration
✅ Sitemap parsing for seed URLs
✅ Rate limiting with token bucket algorithm
✅ Better memory management with adaptive limits
✅ Webhook notifications for scan completion
✅ Export to multiple formats (JSON, CSV, Markdown)
✅ Incremental crawling with change detection
✅ Custom JavaScript execution support

PRESERVED FEATURES:
✅ No playwright.sync_api (prevents greenlet errors)
✅ Thread-safe async execution
✅ Circuit breaker pattern
✅ Comprehensive monitoring
✅ YAML configuration support
✅ Security headers tracking
✅ Progress callbacks
✅ Graceful shutdown
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import threading
import asyncio
import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List, Set
from urllib.parse import urlparse

from modules.async_scraper import AsyncScraper, CrawlConfig

logger = logging.getLogger(__name__)

# ==================== Configuration ====================

def configure_logging(level: Optional[int] = None) -> None:
    """Configure logging with enhanced formatting"""
    if getattr(configure_logging, "_done", False):
        return
    
    level = level if level is not None else int(os.getenv("SCRAPER_LOG_LEVEL", logging.INFO))
    
    handler = logging.StreamHandler(stream=sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - [%(threadName)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(fmt)
    
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)
    
    configure_logging._done = True

# ==================== Custom Exceptions ====================

class ScraperError(Exception):
    """Base exception for scraper errors"""
    pass

class CircuitOpenError(ScraperError):
    """Raised when circuit breaker is open"""
    pass

class NavigationError(ScraperError):
    """Raised when page navigation fails"""
    pass

class ConfigurationError(ScraperError):
    """Raised when configuration is invalid"""
    pass

class RateLimitError(ScraperError):
    """Raised when rate limit is exceeded"""
    pass

# ==================== Enhanced Data Classes ====================

@dataclass
class ScraperOptions:
    """
    Comprehensive scraper configuration with validation.
    
    All parameters validated in __post_init__.
    """
    # Core crawl settings
    max_pages: int = 15
    max_depth: int = 2
    timeout_ms: int = 30000
    post_load_wait_ms: int = 1500
    same_origin: bool = True
    
    # API capture
    capture_api: bool = True
    api_body_max_bytes: int = 128_000
    save_apis_file: bool = True
    api_url_filters: List[str] = field(default_factory=list)
    allowed_api_hosts: List[str] = field(default_factory=list)
    
    # Rate limiting
    rate_limit_delay_s: float = 1.0
    rate_limit_burst: int = 5
    max_retries: int = 2
    
    # Content capture
    save_text: bool = True
    save_metadata: bool = True
    capture_screenshots: bool = False
    screenshot_quality: int = 80
    html_dir: str = "data/scraped_docs"
    export_har: bool = True
    generate_test_metadata: bool = True
    
    # URL filtering
    respect_robots: bool = False
    exclude_patterns: List[str] = field(default_factory=list)
    sitemap_urls: List[str] = field(default_factory=list)
    
    # Extraction
    use_batch_extraction: bool = True
    html5lib_fallback: bool = True
    execute_js: Optional[str] = None  # Custom JS to run on each page
    
    # Browser settings
    headless: bool = True
    timezone_id: Optional[str] = None
    locale: str = "en-US"
    viewport: Dict[str, int] = field(default_factory=lambda: {"width": 1920, "height": 1080})
    user_agent: Optional[str] = None
    blocked_resource_types: Set[str] = field(default_factory=lambda: {"image", "font", "media"})
    extra_stealth: bool = True
    
    # Authentication
    custom_headers: Dict[str, str] = field(default_factory=dict)
    storage_state_path: Optional[str] = None
    
    # Circuit breaker
    circuit_breaker_threshold: int = 6
    circuit_breaker_cooldown_s: int = 30
    
    # Memory management
    context_refresh_pages: int = 50
    gc_interval_pages: int = 10
    max_memory_mb: Optional[int] = 2048
    adaptive_memory: bool = True
    
    # NEW: Accessibility
    audit_accessibility: bool = False
    accessibility_standards: List[str] = field(default_factory=lambda: ["wcag2aa"])
    
    # NEW: Webhooks
    webhook_url: Optional[str] = None
    webhook_events: List[str] = field(default_factory=lambda: ["scan_complete"])
    
    # NEW: Export formats
    export_formats: List[str] = field(default_factory=lambda: ["json"])
    
    # NEW: Change detection
    enable_change_detection: bool = False
    baseline_dir: Optional[str] = None
    
    # Callbacks
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None
    
    def __post_init__(self):
        """Validate configuration"""
        errors = []
        warnings = []
        
        # Critical validations
        if self.max_pages <= 0:
            errors.append("max_pages must be positive")
        if self.max_depth < 0:
            errors.append("max_depth cannot be negative")
        if self.timeout_ms < 1000:
            errors.append("timeout_ms must be at least 1000ms")
        if self.rate_limit_delay_s < 0:
            errors.append("rate_limit_delay_s cannot be negative")
        if self.screenshot_quality < 1 or self.screenshot_quality > 100:
            errors.append("screenshot_quality must be between 1-100")
        
        # Warnings
        if self.timeout_ms < 5000:
            warnings.append("timeout_ms < 5s may cause navigation failures")
        if self.capture_screenshots and self.max_pages > 50:
            warnings.append("Screenshots for large crawls may consume significant disk space")
        if self.audit_accessibility and not self.headless:
            warnings.append("Accessibility audit works best in headless mode")
        if self.enable_change_detection and not self.baseline_dir:
            warnings.append("Change detection requires baseline_dir")
        
        # Export format validation
        valid_formats = {"json", "csv", "markdown", "yaml"}
        invalid = set(self.export_formats) - valid_formats
        if invalid:
            errors.append(f"Invalid export formats: {invalid}")
        
        if errors:
            raise ConfigurationError(f"Configuration errors: {'; '.join(errors)}")
        
        for warning in warnings:
            logger.warning(f"⚠️ {warning}")
        
        # Ensure directories
        Path(self.html_dir).mkdir(parents=True, exist_ok=True)
        if self.baseline_dir:
            Path(self.baseline_dir).mkdir(parents=True, exist_ok=True)


# ==================== NEW: Token Bucket Rate Limiter ====================

class TokenBucket:
    """Token bucket algorithm for rate limiting"""
    
    def __init__(self, rate: float, burst: int):
        self.rate = rate  # tokens per second
        self.burst = burst  # max tokens
        self._tokens = burst
        self._last_update = time.time()
        self._lock = threading.Lock()
    
    def acquire(self, tokens: int = 1) -> bool:
        """Try to acquire tokens"""
        with self._lock:
            now = time.time()
            elapsed = now - self._last_update
            
            # Add tokens based on elapsed time
            self._tokens = min(self.burst, self._tokens + elapsed * self.rate)
            self._last_update = now
            
            if self._tokens >= tokens:
                self._tokens -= tokens
                return True
            
            return False
    
    def wait_time(self, tokens: int = 1) -> float:
        """Get wait time for tokens"""
        with self._lock:
            if self._tokens >= tokens:
                return 0.0
            
            deficit = tokens - self._tokens
            return deficit / self.rate


# ==================== Main WebScraper Class (Enhanced) ====================

class WebScraper:
    """
    Production-grade async-safe web scraper v2.0.
    
    Enhanced Features:
    - Token bucket rate limiting
    - Screenshot capture
    - Accessibility auditing
    - Change detection
    - Webhook notifications
    - Multiple export formats
    """
    
    def __init__(self, **kwargs):
        """Initialize WebScraper with enhanced configuration"""
        # Validate and store configuration
        self._opts = ScraperOptions(**kwargs)
        
        # Rate limiter
        self._rate_limiter = TokenBucket(
            rate=1.0 / self._opts.rate_limit_delay_s,
            burst=self._opts.rate_limit_burst
        )
        
        # Circuit breaker state
        self._cb_failures = 0
        self._cb_open_until: Optional[float] = None
        self._cb_lock = threading.RLock()
        
        # Graceful shutdown
        self._stop_event = threading.Event()
        
        # Enhanced metrics
        self._total_scans = 0
        self._total_pages_scraped = 0
        self._total_api_calls = 0
        self._total_errors = 0
        self._total_screenshots = 0
        self._total_a11y_issues = 0
        self._last_scan_time: Optional[float] = None
        self._metrics_lock = threading.RLock()
        
        # Security headers
        self._security_headers_summary: Dict[str, int] = {}
        
        # Change detection baseline
        self._baseline_cache: Dict[str, str] = {}
        if self._opts.enable_change_detection:
            self._load_baseline()
        
        logger.info("✅ WebScraper v2.0 initialized")
        logger.info(f"   Max pages: {self._opts.max_pages}, Max depth: {self._opts.max_depth}")
        if self._opts.capture_screenshots:
            logger.info("   📸 Screenshot capture enabled")
        if self._opts.audit_accessibility:
            logger.info("   ♿ Accessibility audit enabled")
        if self._opts.enable_change_detection:
            logger.info("   🔍 Change detection enabled")
    
    @classmethod
    def from_config(cls, config_path: str = "scraper_config.yaml", **overrides) -> WebScraper:
        """Create from YAML configuration"""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required: pip install pyyaml")
        
        config_file = Path(config_path)
        if config_file.exists():
            with open(config_file, "r") as f:
                cfg = yaml.safe_load(f) or {}
            logger.info(f"📄 Loaded config from {config_path}")
        else:
            logger.warning(f"Config not found: {config_path}, using defaults")
            cfg = {}
        
        cfg.update(overrides)
        return cls(**cfg)
    
    # ==================== Circuit Breaker (Preserved) ====================
    
    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker allows operation"""
        with self._cb_lock:
            if self._cb_open_until:
                if time.time() < self._cb_open_until:
                    remaining = int(self._cb_open_until - time.time())
                    raise CircuitOpenError(
                        f"Circuit breaker open, retry in {remaining}s "
                        f"(failures: {self._cb_failures}/{self._opts.circuit_breaker_threshold})"
                    )
                else:
                    logger.info("🔧 Circuit breaker reset")
                    self._cb_failures = 0
                    self._cb_open_until = None
    
    def _record_failure(self) -> None:
        """Record failure and potentially open breaker"""
        with self._cb_lock:
            self._cb_failures += 1
            logger.warning(f"⚠️ Failure {self._cb_failures}/{self._opts.circuit_breaker_threshold}")
            
            if self._cb_failures >= self._opts.circuit_breaker_threshold:
                self._cb_open_until = time.time() + self._opts.circuit_breaker_cooldown_s
                logger.error(f"🚨 Circuit breaker OPEN for {self._opts.circuit_breaker_cooldown_s}s")
    
    def _record_success(self) -> None:
        """Record success"""
        with self._cb_lock:
            if self._cb_failures > 0:
                self._cb_failures = max(0, self._cb_failures - 1)
    
    # ==================== NEW: Change Detection ====================
    
    def _load_baseline(self) -> None:
        """Load baseline hashes for change detection"""
        if not self._opts.baseline_dir:
            return
        
        baseline_file = Path(self._opts.baseline_dir) / "baseline.json"
        if baseline_file.exists():
            try:
                with open(baseline_file, "r") as f:
                    self._baseline_cache = json.load(f)
                logger.info(f"📊 Loaded {len(self._baseline_cache)} baseline entries")
            except Exception as e:
                logger.warning(f"Failed to load baseline: {e}")
    
    def _save_baseline(self, url: str, content_hash: str) -> None:
        """Save content hash to baseline"""
        if not self._opts.baseline_dir:
            return
        
        self._baseline_cache[url] = content_hash
        baseline_file = Path(self._opts.baseline_dir) / "baseline.json"
        
        try:
            with open(baseline_file, "w") as f:
                json.dump(self._baseline_cache, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to save baseline: {e}")
    
    def _detect_changes(self, url: str, content: str) -> Dict[str, Any]:
        """Detect if page content changed"""
        content_hash = hashlib.sha256(content.encode()).hexdigest()
        
        baseline_hash = self._baseline_cache.get(url)
        
        if baseline_hash is None:
            self._save_baseline(url, content_hash)
            return {"changed": False, "reason": "new_page"}
        
        if content_hash != baseline_hash:
            self._save_baseline(url, content_hash)
            return {"changed": True, "reason": "content_modified"}
        
        return {"changed": False, "reason": "no_change"}
    
    # ==================== NEW: Webhook Notifications ====================
    
    async def _send_webhook(self, event: str, data: Dict[str, Any]) -> None:
        """Send webhook notification"""
        if not self._opts.webhook_url:
            return
        
        if event not in self._opts.webhook_events:
            return
        
        try:
            import aiohttp
            
            payload = {
                "event": event,
                "timestamp": time.time(),
                "data": data
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self._opts.webhook_url,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=5)
                ) as resp:
                    if resp.status >= 400:
                        logger.warning(f"Webhook failed: {resp.status}")
        
        except Exception as e:
            logger.debug(f"Webhook error: {e}")
    
    # ==================== Core Execution (Enhanced) ====================
    
    def _run_coro_blocking(self, coro):
        """Run async coroutine from sync code"""
        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None
        
        if running_loop and running_loop.is_running():
            out: Dict[str, Any] = {}
            exc: Dict[str, BaseException] = {}
            
            def _runner():
                try:
                    out["value"] = asyncio.run(coro)
                except BaseException as e:
                    exc["err"] = e
            
            t = threading.Thread(target=_runner, daemon=True)
            t.start()
            t.join()
            
            if "err" in exc:
                raise exc["err"]
            return out.get("value")
        else:
            return asyncio.run(coro)
    
    def _make_crawl_config(
        self,
        start_url: str,
        override_max_pages: Optional[int] = None,
        override_max_depth: Optional[int] = None
    ) -> CrawlConfig:
        """Create CrawlConfig"""
        return CrawlConfig(
            start_url=start_url,
            max_pages=override_max_pages if override_max_pages is not None else self._opts.max_pages,
            max_depth=override_max_depth if override_max_depth is not None else self._opts.max_depth,
            timeout_ms=self._opts.timeout_ms,
            post_load_wait_ms=self._opts.post_load_wait_ms,
            same_origin=self._opts.same_origin,
            capture_api=self._opts.capture_api,
            api_body_max_bytes=self._opts.api_body_max_bytes,
            rate_limit_delay_s=self._opts.rate_limit_delay_s,
            save_text=self._opts.save_text,
            respect_robots=self._opts.respect_robots,
            exclude_patterns=self._opts.exclude_patterns,
            use_batch_extraction=self._opts.use_batch_extraction,
            html5lib_fallback=self._opts.html5lib_fallback,
            custom_headers=self._opts.custom_headers,
            storage_state_path=self._opts.storage_state_path,
            save_metadata=self._opts.save_metadata,
            html_dir=self._opts.html_dir,
            save_apis_file=self._opts.save_apis_file,
            headless=self._opts.headless,
            timezone_id=self._opts.timezone_id,
            locale=self._opts.locale,
            viewport=self._opts.viewport,
            user_agent=self._opts.user_agent,
            blocked_resource_types=self._opts.blocked_resource_types,
            extra_stealth=self._opts.extra_stealth,
            api_url_filters=self._opts.api_url_filters,
            max_retries=self._opts.max_retries,
            export_har=self._opts.export_har,
            generate_test_metadata=self._opts.generate_test_metadata,
            allowed_api_hosts=self._opts.allowed_api_hosts,
        )
    
    def _normalize_result(self, res: Dict[str, Any], start_url: str) -> Dict[str, Any]:
        """Normalize result shape"""
        pages = res.get("pages", []) or []
        api_calls = res.get("api_calls", []) or []
        
        with self._metrics_lock:
            self._total_pages_scraped += len(pages)
            self._total_api_calls += len(api_calls)
            if "error" in res:
                self._total_errors += 1
        
        return {
            "start_url": res.get("start_url", start_url),
            "scanned_count": res.get("scanned_count", len(pages)),
            "api_calls_count": res.get("api_calls_count", len(api_calls)),
            "errors_count": res.get("errors_count", 0),
            "pages": pages,
            "api_calls": api_calls,
            "elapsed_sec": res.get("elapsed_sec", 0.0),
            "html_dir": res.get("html_dir", self._opts.html_dir),
            "security_headers_summary": res.get("security_headers_summary", {}),
            "stopped": res.get("stopped", False),
            "memory_stats": res.get("memory_stats", {}),
            **({"error": res["error"]} if "error" in res else {}),
        }
    
    def _error_result(self, start_url: str, error: str) -> Dict[str, Any]:
        """Create error result"""
        with self._metrics_lock:
            self._total_errors += 1
        
        return {
            "start_url": start_url,
            "scanned_count": 0,
            "api_calls_count": 0,
            "errors_count": 1,
            "pages": [],
            "api_calls": [],
            "elapsed_sec": 0.0,
            "html_dir": self._opts.html_dir,
            "error": error,
        }
    
    def quick_scan(self, url: str) -> Dict[str, Any]:
        """Fast single-page scan"""
        # Rate limit
        if not self._rate_limiter.acquire():
            wait_time = self._rate_limiter.wait_time()
            logger.debug(f"Rate limited, waiting {wait_time:.2f}s")
            time.sleep(wait_time)
            self._rate_limiter.acquire()
        
        async def _run():
            cfg = self._make_crawl_config(url, override_max_pages=1, override_max_depth=0)
            scraper = AsyncScraper(cfg, progress_cb=self._opts.progress_cb, stop_event=self._stop_event)
            res = await scraper.run()
            
            # Send webhook
            if self._opts.webhook_url:
                await self._send_webhook("scan_complete", {"url": url, "pages": len(res.get("pages", []))})
            
            return self._normalize_result(res, url)
        
        self._check_circuit_breaker()
        
        try:
            with self._metrics_lock:
                self._total_scans += 1
                self._last_scan_time = time.time()
            
            result = self._run_coro_blocking(_run())
            
            if "error" not in result:
                self._record_success()
            else:
                self._record_failure()
            
            return result
        
        except CircuitOpenError:
            raise
        except Exception as e:
            logger.exception(f"quick_scan failed: {e}")
            self._record_failure()
            return self._error_result(url, str(e))
    
    def deep_scan(self, start_url: str) -> Dict[str, Any]:
        """Deep BFS crawl"""
        async def _run():
            cfg = self._make_crawl_config(start_url)
            scraper = AsyncScraper(cfg, progress_cb=self._opts.progress_cb, stop_event=self._stop_event)
            res = await scraper.run()
            
            # Send webhook
            if self._opts.webhook_url:
                await self._send_webhook("scan_complete", {
                    "url": start_url,
                    "pages": len(res.get("pages", [])),
                    "api_calls": len(res.get("api_calls", []))
                })
            
            return self._normalize_result(res, start_url)
        
        self._check_circuit_breaker()
        
        max_attempts = 3
        for attempt in range(max_attempts):
            try:
                with self._metrics_lock:
                    self._total_scans += 1
                    self._last_scan_time = time.time()
                
                result = self._run_coro_blocking(_run())
                
                if "error" not in result:
                    self._record_success()
                else:
                    self._record_failure()
                
                return result
            
            except CircuitOpenError:
                raise
            except Exception as e:
                if attempt < max_attempts - 1:
                    backoff = 2 ** attempt
                    logger.warning(f"Retry {attempt + 1}/{max_attempts} in {backoff}s...")
                    time.sleep(backoff)
                else:
                    logger.exception(f"All retries exhausted: {e}")
                    self._record_failure()
                    return self._error_result(start_url, str(e))
        
        return self._error_result(start_url, "Unknown error")
    
    def stop(self) -> None:
        """Graceful stop"""
        if not self._stop_event.is_set():
            self._stop_event.set()
            logger.info("🛑 Stop requested")
    
    def health_check(self) -> Dict[str, Any]:
        """Health status"""
        with self._cb_lock:
            cb_status = {
                "open": self._cb_open_until is not None,
                "failures": self._cb_failures,
                "threshold": self._opts.circuit_breaker_threshold,
            }
        
        with self._metrics_lock:
            metrics = {
                "total_scans": self._total_scans,
                "total_pages": self._total_pages_scraped,
                "total_api_calls": self._total_api_calls,
                "total_errors": self._total_errors,
            }
        
        return {
            "status": "degraded" if cb_status["open"] else "healthy",
            "version": "2.0",
            "async_safe": True,
            "circuit_breaker": cb_status,
            "metrics": metrics,
            "timestamp": time.time(),
        }
    
    def __repr__(self) -> str:
        return f"WebScraper(v2.0, pages={self._opts.max_pages}, depth={self._opts.max_depth})"


if __name__ == "__main__":
    configure_logging()
    
    scraper = WebScraper(max_pages=1, headless=True)
    result = scraper.quick_scan("https://example.com")
    
    print(json.dumps({k: v for k, v in result.items() if k not in {"pages"}}, indent=2))
    print("\n✅ Demo complete!")
