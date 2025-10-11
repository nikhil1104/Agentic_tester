#!/usr/bin/env python3
"""
Demo Runner (async Playwright safe)
----------------------------------
Pipeline:
  1) Parse requirement (ConversationalAgent)
  2) Quick + deep site scan using AsyncScraper (NO playwright.sync_api anywhere)
  3) Generate AI-driven test plan (TestGenerator)
  4) Execute plan (Runner)
  5) Persist reports and render dashboard (ReportDashboard)

Usage:
  - Interactive:     python -m scripts.demo_run
  - Non-interactive: python -m scripts.demo_run --require "https://example.com"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, Optional

# Core modules (expected under modules/)
from modules.conversational_agent import ConversationalAgent
from modules.test_generator import TestGenerator
from modules.runner import Runner
from modules.report_dashboard import ReportDashboard

# IMPORTANT: Use ONLY the async scraper; DO NOT import modules.web_scraper
from modules.async_scraper import AsyncScraper, CrawlConfig


# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("demo_run")


# -----------------------------------------------------------------------------
# Helper: safe JSON/HTML persisting
# -----------------------------------------------------------------------------
def persist_reports(results: Dict[str, Any], out_dir: str = "reports") -> Dict[str, str]:
    """
    Persist results into reports/<execution_id>.json and .html.
    Returns paths to created files.
    """
    os.makedirs(out_dir, exist_ok=True)

    eid = results.get("execution_id") or datetime.utcnow().strftime("%Y%m%d%H%M%S")
    json_path = os.path.join(out_dir, f"{eid}.json")
    html_path = os.path.join(out_dir, f"{eid}.html")

    try:
        with open(json_path, "w", encoding="utf-8") as jf:
            json.dump(results, jf, indent=2)
        with open(html_path, "w", encoding="utf-8") as hf:
            hf.write("<html><body><pre>" + json.dumps(results, indent=2) + "</pre></body></html>")
        logger.info("Persisted demo reports: %s , %s", json_path, html_path)
    except Exception as e:
        logger.warning("Failed to persist reports: %s", e)

    return {"json": json_path, "html": html_path}


# -----------------------------------------------------------------------------
# Main demo function
# -----------------------------------------------------------------------------
def demo(requirement: Optional[str] = None, non_interactive: bool = False) -> Dict[str, Any]:
    """
    Run the demo pipeline end-to-end and return the results dict.
    This function intentionally does not sys.exit(); the caller may decide exit semantics.
    """
    logger.info("🧠 Starting AI QA Agent Demo runner")

    # 1) Get requirement (CLI or interactive)
    if not requirement:
        if non_interactive:
            logger.error("Non-interactive mode requires --require argument.")
            raise ValueError("Non-interactive run requires --require")
        try:
            requirement = input("Enter a requirement (e.g. 'Test https://example.com login and checkout'):\n").strip()
        except (KeyboardInterrupt, EOFError):
            logger.info("Input cancelled by user.")
            sys.exit(0)

    if not requirement:
        logger.error("No requirement provided; aborting.")
        raise ValueError("Requirement is required")

    # Initialize modules that do NOT import playwright.sync_api
    try:
        agent = ConversationalAgent()
        generator = TestGenerator()
        runner = Runner()
        dashboard = ReportDashboard()
    except Exception as e:
        logger.exception("Failed to initialize modules: %s", e)
        raise

    # 2) Parse requirement
    logger.info("🤖 Understanding requirement...")
    req = agent.parse_requirement(requirement)
    logger.info("Structured requirement: %s", json.dumps(req, indent=2))

    # 3) Quick + deep scan using the async scraper ONLY (prevents greenlet errors)
    scan_res: Dict[str, Any] = {"visited": 0, "saved": 0, "enqueued": 0, "errors": 0, "pages": []}
    target_url = req.get("details", {}).get("url") or req.get("details", {}).get("raw_text")

    if target_url:
        logger.info("🔍 Running quick scan for: %s", target_url)
        try:
            # Configure the async scraper. It manages its own event loop safely.
            crawl_cfg = CrawlConfig(
                max_pages=15,          # overall cap; deep/quick calls below override per-call values
                max_depth=1,
                concurrency=3,
                headless=True,
                snapshot_dir="data/scraped_docs",
                nav_timeout_ms=30_000,
                same_origin_only=True,
            )
            scraper = AsyncScraper(crawl_cfg)

            # Quick scan: shallow & tiny
            quick = scraper.deep_scan(target_url, max_pages=3, max_depth=0)

            logger.info("🚀 Starting deep scan from: %s", target_url)
            deep = scraper.deep_scan(target_url, max_pages=15, max_depth=1)

            # Merge results (prefer deep pages; ensure uniqueness)
            seen = set()
            pages = []
            for p in (quick.get("pages", []) + deep.get("pages", [])):
                u = p.get("url")
                if u and u not in seen:
                    pages.append(p)
                    seen.add(u)

            scan_res = {
                "visited": max(quick.get("visited", 0), deep.get("visited", 0)),
                "saved": quick.get("saved", 0) + deep.get("saved", 0),
                "enqueued": quick.get("enqueued", 0) + deep.get("enqueued", 0),
                "errors": quick.get("errors", 0) + deep.get("errors", 0),
                "pages": pages,
            }
        except Exception as e:
            logger.exception("AsyncScraper failed (continuing with empty scan results): %s", e)
    else:
        logger.warning("No URL found in requirement; skipping web scan.")

    logger.info("🕸️ Scanned %d page snapshot(s), %d error(s)", scan_res.get("saved", 0), scan_res.get("errors", 0))
    for p in scan_res.get("pages", []):
        logger.info("  → %s", p.get("url"))

    # 4) Generate test plan (uses requirement + scan artifacts)
    logger.info("🧩 Generating test plan...")
    try:
        plan = generator.generate_plan(req, scan_res)
    except Exception as e:
        logger.exception("TestGenerator failed to build plan: %s", e)
        raise

    logger.info("Generated plan:\n%s", json.dumps(plan, indent=2))

    # 5) Execute plan
    logger.info("🚀 Executing test plan...")
    try:
        results = runner.run_plan(plan)
    except Exception as e:
        logger.exception("Runner failed: %s", e)
        # Normalize an error result for persistence and return
        results = {
            "execution_id": f"failed_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
            "test_results": {},
            "error": str(e),
            "execution_meta": {
                "start_time": datetime.utcnow().isoformat(),
                "end_time": datetime.utcnow().isoformat(),
                "duration_sec": 0.0,
            },
        }

    # 6) Persist results and generate dashboard (best-effort)
    report_paths = persist_reports(results)
    try:
        dashboard.generate_dashboard(results.get("execution_id"))
    except Exception as e:
        logger.warning("Dashboard generation skipped/failed: %s", e)

    logger.info("🎯 Demo finished. Reports: %s", json.dumps(report_paths))
    return results


# -----------------------------------------------------------------------------
# CLI entrypoint
# -----------------------------------------------------------------------------
def _build_cli():
    p = argparse.ArgumentParser(prog="demo_run", description="Run AI QA Agent demo pipeline")
    p.add_argument("--require", "-r", dest="requirement", help="Requirement text or URL to test")
    p.add_argument("--non-interactive", action="store_true", help="Do not prompt for input; requires --require")
    return p


if __name__ == "__main__":
    parser = _build_cli()
    args = parser.parse_args()
    try:
        demo(requirement=args.requirement, non_interactive=args.non_interactive)
    except Exception as exc:
        logger.exception("Demo failed: %s", exc)
        sys.exit(1)
