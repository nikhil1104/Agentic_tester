# modules/performance_test_engine.py
"""
PerformanceTestEngine (async-safe, no Playwright sync API)

Two modes:
  - Default: Safe stub that returns "skipped" unless explicitly enabled.
  - Optional: If Lighthouse (Node) is available, run via subprocess to avoid
              mixing Python event loops and Playwright.

This avoids playwright.sync_api entirely.
"""

from __future__ import annotations
import json
import shutil
import subprocess
from typing import Dict, Any, Optional


class PerformanceTestEngine:
    def __init__(self, use_lighthouse: bool = False):
        self.use_lighthouse = use_lighthouse and bool(shutil.which("npx"))

    def run(self, url: str) -> Dict[str, Any]:
        if not url:
            return {"skipped": True, "reason": "no_url"}

        if not self.use_lighthouse:
            # Safe default: do nothing unless explicitly enabled
            return {
                "skipped": True,
                "reason": "lighthouse_disabled",
                "hint": "Init with use_lighthouse=True and ensure Node+npx installed.",
            }

        # Run Lighthouse in a subprocess (no Playwright in-process)
        try:
            cmd = [
                "npx",
                "lighthouse",
                url,
                "--quiet",
                "--chrome-flags=--headless=new",
                "--output=json",
                "--output-path=stdout",
            ]
            proc = subprocess.run(cmd, check=False, text=True, capture_output=True)
            if proc.returncode != 0:
                return {
                    "skipped": True,
                    "reason": "lighthouse_failed",
                    "stderr": proc.stderr[-1000:],
                }
            report = json.loads(proc.stdout or "{}")
            score = (
                report.get("categories", {})
                .get("performance", {})
                .get("score", None)
            )
            return {
                "skipped": False,
                "tool": "lighthouse",
                "performance_score": score,
                "raw": report.get("categories", {}).get("performance", {}),
            }
        except Exception as e:
            return {"skipped": True, "reason": "lighthouse_exception", "error": str(e)}
