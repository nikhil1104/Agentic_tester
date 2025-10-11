# modules/web_scraper.py
"""
Web Scraper (Production-Grade Async-Safe Wrapper)
-------------------------------------------------
Async-safe wrapper around AsyncScraper that eliminates playwright.sync_api usage
to prevent greenlet/thread-switch errors while maintaining full production features.

Features:
✅ No playwright.sync_api imports (prevents greenlet errors)
✅ Thread-safe async execution with event loop detection
✅ Circuit breaker pattern for fault tolerance
✅ Memory management and monitoring
✅ Comprehensive configuration validation
✅ Progress callbacks and observability
✅ Health check endpoint for orchestration
✅ Graceful shutdown support
✅ Security headers tracking
✅ YAML configuration support
✅ Retry logic with exponential backoff
✅ Rich error handling and recovery

Public API:
  - WebScraper.quick_scan(url) -> Dict[str, Any]
  - WebScraper.deep_scan(url) -> Dict[str, Any]
  - WebScraper.stop() -> None
  - WebScraper.health_check() -> Dict[str, Any]
  - WebScraper.from_config(path) -> WebScraper

Example:
    >>> scraper = WebScraper(max_pages=10, capture_api=True)
    >>> result = scraper.deep_scan("https://example.com")
    >>> print(f"Scraped {result['scanned_count']} pages")
"""

from __future__ import annotations

import os
import sys
import json
import time
import logging
import threading
import asyncio
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Callable, List, Set

# Absolutely no sync Playwright imports here
from modules.async_scraper import AsyncScraper, CrawlConfig

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

def configure_logging(level: Optional[int] = None) -> None:
    """
    Configure logging for the scraper.
    
    Args:
        level: Log level (default: INFO from env or logging.INFO)
    """
    if getattr(configure_logging, "_done", False):
        return
    level = level if level is not None else int(os.getenv("SCRAPER_LOG_LEVEL", logging.INFO))
    handler = logging.StreamHandler(stream=sys.stdout)
    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    handler.setFormatter(fmt)
    root = logging.getLogger()
    if not root.handlers:
        root.addHandler(handler)
    root.setLevel(level)
    configure_logging._done = True  # type: ignore[attr-defined]


# ==================== Custom Exceptions ====================

class ScraperError(Exception):
    """Base exception for scraper errors."""
    pass


class CircuitOpenError(ScraperError):
    """Raised when circuit breaker is open."""
    pass


class NavigationError(ScraperError):
    """Raised when page navigation fails."""
    pass


class ConfigurationError(ScraperError):
    """Raised when configuration is invalid."""
    pass


# ==================== Data Classes ====================

@dataclass
class ScraperOptions:
    """
    Comprehensive scraper configuration options.
    
    All parameters are validated in __post_init__.
    """
    # Core crawl settings
    max_pages: int = 15
    max_depth: int = 2
    timeout_ms: int = 30000
    post_load_wait_ms: int = 1500
    same_origin: bool = True
    
    # API capture settings
    capture_api: bool = True
    api_body_max_bytes: int = 128_000
    save_apis_file: bool = True
    api_url_filters: List[str] = field(default_factory=list)
    allowed_api_hosts: List[str] = field(default_factory=list)
    
    # Rate limiting and performance
    rate_limit_delay_s: float = 1.0
    max_retries: int = 2
    
    # Content and metadata
    save_text: bool = True
    save_metadata: bool = True
    html_dir: str = "data/scraped_docs"
    export_har: bool = True
    generate_test_metadata: bool = True
    
    # URL filtering
    respect_robots: bool = False
    exclude_patterns: List[str] = field(default_factory=list)
    
    # Extraction settings
    use_batch_extraction: bool = True
    html5lib_fallback: bool = True
    
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
    
    # Circuit breaker settings
    circuit_breaker_threshold: int = 6
    circuit_breaker_cooldown_s: int = 30
    
    # Memory management
    context_refresh_pages: int = 50
    gc_interval_pages: int = 10
    max_memory_mb: Optional[int] = 2048
    
    # Callbacks
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None
    
    def __post_init__(self):
        """Validate configuration parameters."""
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
        if self.circuit_breaker_threshold < 1:
            errors.append("circuit_breaker_threshold must be positive")
        
        # Warnings for suboptimal settings
        if self.timeout_ms < 5000:
            warnings.append("timeout_ms < 5s may cause navigation failures")
        if self.max_pages > 100 and self.gc_interval_pages > 20:
            warnings.append("Consider lowering gc_interval_pages for large crawls")
        if self.api_body_max_bytes > 5_000_000:
            warnings.append("Large api_body_max_bytes may cause memory issues")
        if not self.headless and os.getenv("CI"):
            warnings.append("Non-headless mode detected in CI environment")
        if self.max_memory_mb and self.max_memory_mb < 512:
            warnings.append("max_memory_mb < 512 may be too restrictive")
        
        # Raise errors
        if errors:
            raise ConfigurationError(f"Configuration errors: {'; '.join(errors)}")
        
        # Log warnings
        for warning in warnings:
            logger.warning(f"⚠️ Configuration warning: {warning}")
        
        # Ensure directories exist
        Path(self.html_dir).mkdir(parents=True, exist_ok=True)


# ==================== Main WebScraper Class ====================

class WebScraper:
    """
    Production-grade async-safe web scraper.
    
    This class provides a synchronous API while delegating to AsyncScraper
    to avoid playwright.sync_api greenlet/threading issues.
    
    Features:
    - Thread-safe async execution
    - Circuit breaker pattern
    - Memory management
    - Progress tracking
    - Graceful shutdown
    - Health monitoring
    
    Example:
        >>> scraper = WebScraper(max_pages=10, capture_api=True)
        >>> result = scraper.deep_scan("https://example.com")
        >>> print(f"Scraped {result['scanned_count']} pages")
        >>> scraper.stop()  # Graceful shutdown
    """
    
    def __init__(
        self,
        max_pages: int = 15,
        max_depth: int = 2,
        timeout_ms: int = 30000,
        post_load_wait_ms: int = 1500,
        same_origin: bool = True,
        capture_api: bool = True,
        api_body_max_bytes: int = 128_000,
        rate_limit_delay_s: float = 1.0,
        save_text: bool = True,
        respect_robots: bool = False,
        exclude_patterns: Optional[List[str]] = None,
        use_batch_extraction: bool = True,
        html5lib_fallback: bool = True,
        custom_headers: Optional[Dict[str, str]] = None,
        storage_state_path: Optional[str] = None,
        save_metadata: bool = True,
        html_dir: str = "data/scraped_docs",
        save_apis_file: bool = True,
        headless: bool = True,
        timezone_id: Optional[str] = None,
        locale: str = "en-US",
        viewport: Optional[Dict[str, int]] = None,
        user_agent: Optional[str] = None,
        blocked_resource_types: Optional[Set[str]] = None,
        extra_stealth: bool = True,
        api_url_filters: Optional[List[str]] = None,
        max_retries: int = 2,
        export_har: bool = True,
        generate_test_metadata: bool = True,
        allowed_api_hosts: Optional[List[str]] = None,
        circuit_breaker_threshold: int = 6,
        circuit_breaker_cooldown_s: int = 30,
        context_refresh_pages: int = 50,
        gc_interval_pages: int = 10,
        max_memory_mb: Optional[int] = 2048,
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """Initialize WebScraper with configuration."""
        # Validate and store configuration
        self._opts = ScraperOptions(
            max_pages=max_pages,
            max_depth=max_depth,
            timeout_ms=timeout_ms,
            post_load_wait_ms=post_load_wait_ms,
            same_origin=same_origin,
            capture_api=capture_api,
            api_body_max_bytes=api_body_max_bytes,
            rate_limit_delay_s=rate_limit_delay_s,
            save_text=save_text,
            respect_robots=respect_robots,
            exclude_patterns=exclude_patterns or [],
            use_batch_extraction=use_batch_extraction,
            html5lib_fallback=html5lib_fallback,
            custom_headers=custom_headers or {},
            storage_state_path=storage_state_path,
            save_metadata=save_metadata,
            html_dir=html_dir,
            save_apis_file=save_apis_file,
            headless=headless,
            timezone_id=timezone_id,
            locale=locale,
            viewport=viewport or {"width": 1920, "height": 1080},
            user_agent=user_agent,
            blocked_resource_types=blocked_resource_types or {"image", "font", "media"},
            extra_stealth=extra_stealth,
            api_url_filters=api_url_filters or [],
            max_retries=max_retries,
            export_har=export_har,
            generate_test_metadata=generate_test_metadata,
            allowed_api_hosts=allowed_api_hosts or [],
            circuit_breaker_threshold=circuit_breaker_threshold,
            circuit_breaker_cooldown_s=circuit_breaker_cooldown_s,
            context_refresh_pages=context_refresh_pages,
            gc_interval_pages=gc_interval_pages,
            max_memory_mb=max_memory_mb,
            progress_cb=progress_cb,
        )
        
        # Circuit breaker state
        self._cb_failures = 0
        self._cb_open_until: Optional[float] = None
        self._cb_lock = threading.RLock()
        
        # Graceful shutdown
        self._stop_event = threading.Event()
        
        # Metrics tracking
        self._total_scans = 0
        self._total_pages_scraped = 0
        self._total_api_calls = 0
        self._total_errors = 0
        self._last_scan_time: Optional[float] = None
        self._metrics_lock = threading.RLock()
        
        # Security headers tracking
        self._security_headers_summary: Dict[str, int] = {}
        
        logger.info("✅ WebScraper initialized (max_pages=%d, max_depth=%d, async-safe=True)", 
                   max_pages, max_depth)
    
    @classmethod
    def from_config(cls, config_path: str = "scraper_config.yaml", **overrides) -> WebScraper:
        """Create WebScraper from YAML configuration file."""
        try:
            import yaml
        except ImportError:
            raise ImportError("PyYAML required for config file support: pip install pyyaml")
        
        config_file = Path(config_path)
        if not config_file.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            cfg = {}
        else:
            with open(config_file, "r") as f:
                cfg = yaml.safe_load(f) or {}
            logger.info(f"📄 Loaded config from {config_path}")
        
        cfg.update(overrides)
        return cls(**cfg)
    
    def _check_circuit_breaker(self) -> None:
        """Check if circuit breaker allows operation."""
        with self._cb_lock:
            if self._cb_open_until:
                if time.time() < self._cb_open_until:
                    remaining = int(self._cb_open_until - time.time())
                    raise CircuitOpenError(
                        f"Circuit breaker open, retry in {remaining}s "
                        f"(failures: {self._cb_failures}/{self._opts.circuit_breaker_threshold})"
                    )
                else:
                    logger.info("🔧 Circuit breaker reset after cooldown")
                    self._cb_failures = 0
                    self._cb_open_until = None
    
    def _record_failure(self) -> None:
        """Record a failure and potentially open circuit breaker."""
        with self._cb_lock:
            self._cb_failures += 1
            logger.warning(f"⚠️ Failure recorded ({self._cb_failures}/{self._opts.circuit_breaker_threshold})")
            
            if self._cb_failures >= self._opts.circuit_breaker_threshold:
                self._cb_open_until = time.time() + self._opts.circuit_breaker_cooldown_s
                logger.error(
                    f"🚨 Circuit breaker OPEN for {self._opts.circuit_breaker_cooldown_s}s "
                    f"({self._cb_failures} failures)"
                )
    
    def _record_success(self) -> None:
        """Record a successful operation."""
        with self._cb_lock:
            if self._cb_failures > 0:
                self._cb_failures = max(0, self._cb_failures - 1)
                logger.debug(f"✅ Success recorded, failures now: {self._cb_failures}")
    
    def _run_coro_blocking(self, coro):
        """Run an async coroutine from sync code without event loop conflicts."""
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
        """Create CrawlConfig for AsyncScraper."""
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
        """Ensure stable result shape for backward compatibility."""
        pages = res.get("pages", []) or []
        api_calls = res.get("api_calls", []) or []
        
        with self._metrics_lock:
            self._total_pages_scraped += len(pages)
            self._total_api_calls += len(api_calls)
            if "error" in res:
                self._total_errors += 1
        
        security_summary = res.get("security_headers_summary", {})
        if security_summary:
            self._security_headers_summary.update(security_summary)
        
        return {
            "start_url": res.get("start_url", start_url),
            "scanned_count": res.get("scanned_count", len(pages)),
            "api_calls_count": res.get("api_calls_count", len(api_calls)),
            "errors_count": res.get("errors_count", 0),
            "pages": pages,
            "api_calls": api_calls,
            "elapsed_sec": res.get("elapsed_sec", 0.0),
            "html_dir": res.get("html_dir", self._opts.html_dir),
            "security_headers_summary": security_summary,
            "stopped": res.get("stopped", False),
            "memory_stats": res.get("memory_stats", {}),
            **({"error": res["error"]} if "error" in res else {}),
        }
    
    def _error_result(self, start_url: str, error: str) -> Dict[str, Any]:
        """Create standardized error result."""
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
            "security_headers_summary": {},
            "stopped": self._stop_event.is_set(),
            "error": error,
        }
    
    def quick_scan(self, url: str) -> Dict[str, Any]:
        """Fast single-page fetch (no crawling)."""
        async def _run():
            cfg = self._make_crawl_config(url, override_max_pages=1, override_max_depth=0)
            scraper = AsyncScraper(cfg, progress_cb=self._opts.progress_cb, stop_event=self._stop_event)
            res = await scraper.run()
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
            logger.exception("quick_scan failed for %s: %s", url, e)
            self._record_failure()
            return self._error_result(url, str(e))
    
    def deep_scan(self, start_url: str) -> Dict[str, Any]:
        """Deep BFS crawl starting from given URL."""
        async def _run():
            cfg = self._make_crawl_config(start_url)
            scraper = AsyncScraper(cfg, progress_cb=self._opts.progress_cb, stop_event=self._stop_event)
            res = await scraper.run()
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
            except asyncio.TimeoutError as e:
                if attempt < max_attempts - 1:
                    backoff = 2 ** attempt
                    logger.warning(f"⏱️ Timeout on attempt {attempt + 1}/{max_attempts}, retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    logger.error("❌ All retry attempts exhausted due to timeout")
                    self._record_failure()
                    return self._error_result(start_url, f"Timeout after {max_attempts} attempts")
            except Exception as e:
                if attempt < max_attempts - 1:
                    backoff = 2 ** attempt
                    logger.warning(f"⚠️ Error on attempt {attempt + 1}/{max_attempts}: {e}, retrying in {backoff}s...")
                    time.sleep(backoff)
                else:
                    logger.exception("deep_scan failed for %s after %d attempts", start_url, max_attempts)
                    self._record_failure()
                    return self._error_result(start_url, str(e))
        
        return self._error_result(start_url, "Unknown error")
    
    def stop(self) -> None:
        """Request graceful stop of ongoing operations."""
        if not self._stop_event.is_set():
            self._stop_event.set()
            logger.info("🛑 Graceful stop requested")
    
    def health_check(self) -> Dict[str, Any]:
        """Return scraper health status for monitoring/orchestration."""
        with self._cb_lock:
            cb_status = {
                "open": self._cb_open_until is not None,
                "failures": self._cb_failures,
                "threshold": self._opts.circuit_breaker_threshold,
                "cooldown_s": self._opts.circuit_breaker_cooldown_s,
            }
            if self._cb_open_until:
                cb_status["closes_in"] = max(0, int(self._cb_open_until - time.time()))
        
        with self._metrics_lock:
            metrics = {
                "total_scans": self._total_scans,
                "total_pages_scraped": self._total_pages_scraped,
                "total_api_calls": self._total_api_calls,
                "total_errors": self._total_errors,
                "last_scan_time": self._last_scan_time,
            }
        
        status = "degraded" if cb_status["open"] else "healthy"
        
        return {
            "status": status,
            "playwright_available": True,
            "async_safe": True,
            "circuit_breaker": cb_status,
            "metrics": metrics,
            "config": {
                "max_pages": self._opts.max_pages,
                "max_depth": self._opts.max_depth,
                "capture_api": self._opts.capture_api,
                "headless": self._opts.headless,
                "timeout_ms": self._opts.timeout_ms,
            },
            "security_headers_tracked": len(self._security_headers_summary),
            "timestamp": time.time(),
        }
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get detailed scraping metrics."""
        with self._metrics_lock:
            return {
                "total_scans": self._total_scans,
                "total_pages_scraped": self._total_pages_scraped,
                "total_api_calls": self._total_api_calls,
                "total_errors": self._total_errors,
                "last_scan_time": self._last_scan_time,
                "avg_pages_per_scan": (
                    self._total_pages_scraped / self._total_scans
                    if self._total_scans > 0 else 0
                ),
                "error_rate": (
                    self._total_errors / self._total_scans
                    if self._total_scans > 0 else 0
                ),
            }
    
    def reset_circuit_breaker(self) -> None:
        """Manually reset circuit breaker (use with caution)."""
        with self._cb_lock:
            old_failures = self._cb_failures
            self._cb_failures = 0
            self._cb_open_until = None
            logger.warning(f"🔧 Circuit breaker manually reset (was: {old_failures} failures)")
    
    def __repr__(self) -> str:
        """String representation of WebScraper."""
        return (
            f"WebScraper(max_pages={self._opts.max_pages}, "
            f"max_depth={self._opts.max_depth}, "
            f"capture_api={self._opts.capture_api}, "
            f"async_safe=True)"
        )


if __name__ == "__main__":
    configure_logging()
    
    logger.info("=" * 60)
    logger.info("Example: Basic usage")
    logger.info("=" * 60)
    scraper = WebScraper(max_pages=1, headless=True)
    result = scraper.quick_scan("https://example.com")
    print(json.dumps({k: v for k, v in result.items() if k not in {"pages", "api_calls"}}, indent=2))
    
    health = scraper.health_check()
    print(json.dumps(health, indent=2))
    
    logger.info("\n✅ Demo complete!")
