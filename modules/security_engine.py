# modules/security_engine.py
"""
Security Engine (Phase 6.1) - Modular Architecture
Main orchestration engine with plugin-based security checks.
"""

from __future__ import annotations
import asyncio
import logging
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List
import httpx

from modules.security_types import SecurityCheckResult, CheckStatus
from modules.security_checks import CHECK_REGISTRY, get_enabled_checks

logger = logging.getLogger(__name__)


class SecurityEngine:
    """
    Production-grade modular security validation engine.
    """
    
    def __init__(
        self,
        timeout_s: float = 15.0,
        enabled_checks: Optional[List[str]] = None,
        enable_async: bool = True,
        max_concurrent_checks: int = 3,
    ):
        self.timeout_s = timeout_s
        self.enable_async = enable_async
        self.max_concurrent_checks = max_concurrent_checks
        self.enabled_checks = enabled_checks or list(CHECK_REGISTRY.keys())
        self.check_classes = get_enabled_checks(self.enabled_checks)
        logger.info(
            "SecurityEngine initialized: enabled_checks=%s, async=%s",
            self.enabled_checks,
            enable_async,
        )
    
    def plan_from_base_url(self, base_url: str) -> Optional[Dict[str, Any]]:
        if not base_url:
            return None
        return {
            "name": "OWASP Security Validation Suite",
            "description": "Modular security checks with CWE/OWASP mapping",
            "steps": [f"run {check} security check on {base_url}" for check in self.enabled_checks],
            "priority": "P1",
            "tags": ["security", "owasp"] + self.enabled_checks,
            "metadata": {"check_types": self.enabled_checks, "standard": "OWASP"},
        }

    # ---- New: runner-friendly API used by modules/runner.py ----
    def run_suites(
        self,
        suites: List[Dict[str, Any]],
        base_url: Optional[str] = None,
        export_dir: str = "reports/security",
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute security scan(s) for the given suites.
        We currently scan the base_url once (typical case). If not provided,
        we try to extract URLs from suite steps like: "run <check> ... on <url>".
        """
        urls: List[str] = []
        if base_url:
            urls.append(base_url)
        else:
            for suite in suites or []:
                for step in suite.get("steps", []):
                    m = re.search(r"(https?://\S+)", step)
                    if m:
                        urls.append(m.group(1))
        # dedupe while preserving order
        seen = set()
        urls = [u for u in urls if not (u in seen or seen.add(u))]

        exports: List[Dict[str, str]] = []
        results: List[Dict[str, Any]] = []
        failed = False

        out_dir = Path(export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rid = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")

        for url in (urls or []):
            res = asyncio.run(self.scan_async(url)) if self.enable_async else self.scan_sync(url)
            results.append(res.to_dict())
            if res.overall_status in (CheckStatus.FAIL, CheckStatus.ERROR):
                failed = True

            # export artifacts
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")[:60]
            json_path = out_dir / f"{rid}_{slug}.json"
            md_path = out_dir / f"{rid}_{slug}.md"
            try:
                json_path.write_text(res.to_json(), encoding="utf-8")
                md_path.write_text(res.to_markdown(), encoding="utf-8")
                exports.append({"json": str(json_path), "md": str(md_path), "url": url})
            except Exception as e:
                logger.warning("Failed to export security report for %s: %s", url, e)

        summary = {
            "targets": urls,
            "result_count": len(results),
            "failed": int(failed),
            "success": not failed,
            "exports": exports,
        }
        return {**summary, "results": results}

    async def scan_async(self, url: str) -> SecurityCheckResult:
        start_time = time.time()
        result = SecurityCheckResult(
            url=url,
            timestamp=datetime.utcnow().isoformat() + "Z",
            overall_status=CheckStatus.PASS,
        )
        logger.info("🔒 Starting async security scan: %s", url)
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.timeout_s,
                verify=True,
            ) as client:
                checks = [check_class(timeout_s=self.timeout_s) for check_class in self.check_classes]
                sem = asyncio.Semaphore(self.max_concurrent_checks)
                async def bounded(check):
                    async with sem:
                        await check.execute_async(url, client, result)
                await asyncio.gather(*[bounded(c) for c in checks])
        except Exception as e:
            logger.error("Security scan failed for %s: %s", url, e)
        result.duration_s = round(time.time() - start_time, 2)
        result.calculate_summary()
        logger.info("✅ Scan completed: %s (risk_score=%s, findings=%d)",
                    url, result.summary.get("risk_score", 0), len(result.findings))
        return result
    
    def scan_sync(self, url: str) -> SecurityCheckResult:
        start_time = time.time()
        result = SecurityCheckResult(
            url=url,
            timestamp=datetime.utcnow().isoformat() + "Z",
            overall_status=CheckStatus.PASS,
        )
        logger.info("🔒 Starting sync security scan: %s", url)
        try:
            with httpx.Client(
                follow_redirects=True,
                timeout=self.timeout_s,
                verify=True,
            ) as client:
                checks = [check_class(timeout_s=self.timeout_s) for check_class in self.check_classes]
                for check in checks:
                    check.execute_sync(url, client, result)
        except Exception as e:
            logger.error("Security scan failed for %s: %s", url, e)
        result.duration_s = round(time.time() - start_time, 2)
        result.calculate_summary()
        return result
