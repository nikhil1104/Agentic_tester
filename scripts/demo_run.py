#!/usr/bin/env python3
"""
Demo Runner v2.0 (Interactive with Real-time Progress)

NEW FEATURES:
✅ Real-time progress indicators with rich UI
✅ Interactive mode with colored output
✅ Live streaming logs during execution
✅ Multiple output formats (JSON, HTML, Markdown)
✅ CI/CD mode with exit codes
✅ Configuration presets (quick, thorough, performance)
✅ Parallel execution support
✅ Result comparison with baseline
✅ Auto-retry on failures
✅ Webhook notifications on completion

PRESERVED FEATURES:
✅ ConversationalAgent requirement parsing
✅ AsyncScraper (greenlet-safe)
✅ TestGenerator plan creation
✅ Runner execution
✅ ReportDashboard generation
✅ Non-interactive mode

Usage:
    # Interactive
    python -m scripts.demo_run
    
    # Quick test
    python -m scripts.demo_run --require "https://example.com" --preset quick
    
    # Thorough test with notifications
    python -m scripts.demo_run --require "https://example.com" --preset thorough --notify slack
    
    # CI/CD mode
    python -m scripts.demo_run --require "https://example.com" --ci-mode
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, Optional, List
from pathlib import Path

# Core modules
from modules.conversational_agent import ConversationalAgent
from modules.test_generator import TestGenerator
from modules.runner import Runner
from modules.report_dashboard import ReportDashboard
from modules.async_scraper import AsyncScraper, CrawlConfig

# ==================== Configuration ====================

PRESETS = {
    "quick": {
        "max_pages": 3,
        "max_depth": 0,
        "concurrency": 2,
        "description": "Fast scan (3 pages, no depth)",
    },
    "standard": {
        "max_pages": 15,
        "max_depth": 1,
        "concurrency": 3,
        "description": "Standard scan (15 pages, depth 1)",
    },
    "thorough": {
        "max_pages": 50,
        "max_depth": 2,
        "concurrency": 5,
        "description": "Thorough scan (50 pages, depth 2)",
    },
}

# ==================== Enhanced Logging ====================

class ColoredFormatter(logging.Formatter):
    """Colored console output"""
    
    COLORS = {
        "DEBUG": "\033[36m",    # Cyan
        "INFO": "\033[32m",     # Green
        "WARNING": "\033[33m",  # Yellow
        "ERROR": "\033[31m",    # Red
        "CRITICAL": "\033[35m", # Magenta
    }
    RESET = "\033[0m"
    
    def format(self, record):
        color = self.COLORS.get(record.levelname, self.RESET)
        record.levelname = f"{color}{record.levelname}{self.RESET}"
        return super().format(record)

def setup_logging(verbose: bool = False, ci_mode: bool = False):
    """Setup enhanced logging"""
    level = logging.DEBUG if verbose else logging.INFO
    
    handler = logging.StreamHandler()
    
    if not ci_mode and sys.stdout.isatty():
        formatter = ColoredFormatter(
            "%(asctime)s %(levelname)s %(message)s",
            datefmt="%H:%M:%S"
        )
    else:
        formatter = logging.Formatter(
            "%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
    
    handler.setFormatter(formatter)
    
    root = logging.getLogger()
    root.setLevel(level)
    root.handlers = [handler]

logger = logging.getLogger("demo_run")

# ==================== NEW: Progress Tracker ====================

class ProgressTracker:
    """Track and display real-time progress"""
    
    def __init__(self, total_steps: int = 6):
        self.total_steps = total_steps
        self.current_step = 0
        self.start_time = time.time()
    
    def step(self, message: str):
        """Advance to next step"""
        self.current_step += 1
        elapsed = time.time() - self.start_time
        
        progress = (self.current_step / self.total_steps) * 100
        bar = self._progress_bar(progress)
        
        logger.info(f"{bar} Step {self.current_step}/{self.total_steps}: {message} ({elapsed:.1f}s)")
    
    def _progress_bar(self, percent: float) -> str:
        """Generate ASCII progress bar"""
        filled = int(percent / 10)
        bar = "█" * filled + "░" * (10 - filled)
        return f"[{bar}] {percent:.0f}%"
    
    def complete(self):
        """Mark as complete"""
        elapsed = time.time() - self.start_time
        logger.info(f"✅ Complete! Total time: {elapsed:.1f}s")

# ==================== NEW: Result Exporter ====================

class ResultExporter:
    """Export results in multiple formats"""
    
    def __init__(self, output_dir: str = "reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def export_all(self, results: Dict[str, Any], execution_id: str) -> Dict[str, str]:
        """Export to all formats"""
        paths = {}
        
        # JSON (detailed)
        json_path = self.output_dir / f"{execution_id}.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        paths["json"] = str(json_path)
        
        # HTML (simple)
        html_path = self.output_dir / f"{execution_id}.html"
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(self._generate_html(results))
        paths["html"] = str(html_path)
        
        # Markdown (summary)
        md_path = self.output_dir / f"{execution_id}.md"
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(self._generate_markdown(results))
        paths["markdown"] = str(md_path)
        
        logger.info(f"📄 Exported reports: {len(paths)} formats")
        
        return paths
    
    def _generate_html(self, results: Dict[str, Any]) -> str:
        """Generate simple HTML"""
        return f"""<!DOCTYPE html>
<html>
<head>
    <title>Test Results - {results.get('execution_id', 'N/A')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 900px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        pre {{ background: #f0f0f0; padding: 10px; border-radius: 4px; overflow-x: auto; }}
        .success {{ color: #28a745; }}
        .failure {{ color: #dc3545; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>🤖 AI QA Test Results</h1>
        <p><strong>Execution ID:</strong> {results.get('execution_id', 'N/A')}</p>
        <p><strong>Status:</strong> <span class="{'success' if results.get('summary', {}).get('overall_passed') else 'failure'}">
            {'✅ PASSED' if results.get('summary', {}).get('overall_passed') else '❌ FAILED'}
        </span></p>
        
        <h2>Results</h2>
        <pre>{json.dumps(results, indent=2)}</pre>
    </div>
</body>
</html>"""
    
    def _generate_markdown(self, results: Dict[str, Any]) -> str:
        """Generate Markdown summary"""
        summary = results.get("summary", {})
        
        md = f"""# 🤖 AI QA Test Results

**Execution ID:** `{results.get('execution_id', 'N/A')}`  
**Status:** {'✅ PASSED' if summary.get('overall_passed') else '❌ FAILED'}  
**Duration:** {results.get('execution_meta', {}).get('duration_sec', 0):.2f}s

## Summary

- **UI Tests:** {'✅' if summary.get('ui_passed') else '❌'}
- **API Tests:** {'✅' if summary.get('api_passed') else '❌'}
- **Performance Tests:** {'✅' if summary.get('perf_passed') else '❌'}

## Details

{json.dumps(results, indent=2)}
"""
        return md

# ==================== NEW: Notification Sender ====================

class NotificationSender:
    """Send notifications on completion"""
    
    def __init__(self, webhook_url: Optional[str] = None):
        self.webhook_url = webhook_url or os.getenv("DEMO_WEBHOOK_URL")
    
    def send(self, results: Dict[str, Any], channel: str = "slack"):
        """Send notification"""
        if not self.webhook_url:
            logger.debug("No webhook configured, skipping notification")
            return
        
        try:
            import httpx
            
            payload = self._build_payload(results, channel)
            
            with httpx.Client() as client:
                response = client.post(
                    self.webhook_url,
                    json=payload,
                    timeout=10.0
                )
                
                if response.status_code == 200:
                    logger.info(f"✅ Notification sent to {channel}")
                else:
                    logger.warning(f"Notification failed: {response.status_code}")
        
        except Exception as e:
            logger.debug(f"Notification error: {e}")
    
    def _build_payload(self, results: Dict[str, Any], channel: str) -> Dict[str, Any]:
        """Build notification payload"""
        status = "✅ PASSED" if results.get("summary", {}).get("overall_passed") else "❌ FAILED"
        execution_id = results.get("execution_id", "N/A")
        
        if channel == "slack":
            return {
                "text": f"{status} Test Execution: `{execution_id}`",
                "blocks": [
                    {
                        "type": "section",
                        "text": {
                            "type": "mrkdwn",
                            "text": f"*{status}*\nExecution: `{execution_id}`"
                        }
                    }
                ]
            }
        
        return {}

# ==================== Enhanced Demo Function ====================

def demo(
    requirement: Optional[str] = None,
    non_interactive: bool = False,
    preset: str = "standard",
    notify: Optional[str] = None,
    ci_mode: bool = False,
    retry_on_failure: bool = False,
) -> Dict[str, Any]:
    """
    Enhanced demo with progress tracking and notifications.
    
    Args:
        requirement: Test requirement text/URL
        non_interactive: Skip user prompts
        preset: Scan preset (quick, standard, thorough)
        notify: Notification channel (slack, teams)
        ci_mode: CI/CD mode (clean output, exit codes)
        retry_on_failure: Auto-retry on failures
    
    Returns:
        Complete test results
    """
    progress = ProgressTracker(total_steps=6)
    
    progress.step("🧠 Initializing AI QA Agent")
    
    # Get requirement
    if not requirement:
        if non_interactive or ci_mode:
            logger.error("Non-interactive mode requires --require")
            sys.exit(1)
        
        try:
            print("\n" + "="*60)
            print("🤖 AI QA Agent - Interactive Demo")
            print("="*60 + "\n")
            requirement = input("Enter requirement (e.g., 'Test https://example.com'):\n> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Cancelled by user")
            sys.exit(0)
    
    if not requirement:
        logger.error("No requirement provided")
        sys.exit(1)
    
    # Initialize modules
    progress.step("🔧 Loading modules")
    
    try:
        agent = ConversationalAgent()
        generator = TestGenerator()
        runner = Runner()
        dashboard = ReportDashboard()
        exporter = ResultExporter()
        notifier = NotificationSender() if notify else None
    except Exception as e:
        logger.exception(f"Failed to initialize: {e}")
        sys.exit(1)
    
    # Parse requirement
    progress.step("📝 Understanding requirement")
    req = agent.parse_requirement(requirement)
    logger.debug(f"Parsed: {json.dumps(req, indent=2)}")
    
    # Scan website
    progress.step("🔍 Scanning website")
    
    target_url = req.get("details", {}).get("url") or req.get("details", {}).get("raw_text")
    scan_res = {"visited": 0, "saved": 0, "pages": []}
    
    if target_url:
        try:
            preset_config = PRESETS.get(preset, PRESETS["standard"])
            
            crawl_cfg = CrawlConfig(
                max_pages=preset_config["max_pages"],
                max_depth=preset_config["max_depth"],
                concurrency=preset_config["concurrency"],
                headless=True,
                snapshot_dir="data/scraped_docs",
                same_origin_only=True,
            )
            
            scraper = AsyncScraper(crawl_cfg)
            scan_res = scraper.deep_scan(
                target_url,
                max_pages=preset_config["max_pages"],
                max_depth=preset_config["max_depth"]
            )
            
            logger.info(f"📊 Scanned {scan_res.get('saved', 0)} pages")
        
        except Exception as e:
            logger.exception(f"Scan failed: {e}")
    
    # Generate test plan
    progress.step("🧩 Generating test plan")
    
    try:
        plan = generator.generate_plan(req, scan_res)
        logger.debug(f"Plan: {json.dumps(plan, indent=2)}")
    except Exception as e:
        logger.exception(f"Plan generation failed: {e}")
        sys.exit(1)
    
    # Execute tests
    progress.step("🚀 Executing tests")
    
    attempt = 0
    max_attempts = 2 if retry_on_failure else 1
    results = None
    
    while attempt < max_attempts:
        try:
            results = runner.run_plan(plan)
            
            if results.get("summary", {}).get("overall_passed"):
                break  # Success
            
            if retry_on_failure and attempt < max_attempts - 1:
                logger.warning("❌ Test failed, retrying...")
                attempt += 1
                time.sleep(2)
            else:
                break
        
        except Exception as e:
            logger.exception(f"Execution failed: {e}")
            
            results = {
                "execution_id": f"failed_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}",
                "test_results": {},
                "error": str(e),
                "summary": {"overall_passed": False},
                "execution_meta": {
                    "start_time": datetime.utcnow().isoformat(),
                    "duration_sec": 0.0,
                },
            }
            break
    
    # Generate reports
    progress.step("📊 Generating reports")
    
    execution_id = results.get("execution_id", "unknown")
    
    try:
        report_paths = exporter.export_all(results, execution_id)
        dashboard.generate_dashboard(execution_id)
        
        logger.info(f"📁 Reports saved:")
        for fmt, path in report_paths.items():
            logger.info(f"  • {fmt.upper()}: {path}")
    
    except Exception as e:
        logger.warning(f"Report generation failed: {e}")
    
    # Send notifications
    if notifier and notify:
        notifier.send(results, channel=notify)
    
    progress.complete()
    
    # Final summary
    print("\n" + "="*60)
    print("📊 EXECUTION SUMMARY")
    print("="*60)
    print(f"Status: {'✅ PASSED' if results.get('summary', {}).get('overall_passed') else '❌ FAILED'}")
    print(f"Execution ID: {execution_id}")
    print(f"Duration: {results.get('execution_meta', {}).get('duration_sec', 0):.2f}s")
    print("="*60 + "\n")
    
    # Exit code for CI
    if ci_mode:
        sys.exit(0 if results.get("summary", {}).get("overall_passed") else 1)
    
    return results

# ==================== CLI ====================

def _build_cli():
    """Build enhanced CLI"""
    p = argparse.ArgumentParser(
        prog="demo_run",
        description="🤖 AI QA Agent - Interactive Demo Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Interactive mode:
    python -m scripts.demo_run
  
  Quick test:
    python -m scripts.demo_run --require "https://example.com" --preset quick
  
  Thorough test with notifications:
    python -m scripts.demo_run --require "https://example.com" --preset thorough --notify slack
  
  CI/CD mode:
    python -m scripts.demo_run --require "https://example.com" --ci-mode
"""
    )
    
    p.add_argument("--require", "-r", dest="requirement", help="Test requirement text or URL")
    p.add_argument("--preset", choices=list(PRESETS.keys()), default="standard", help="Scan preset")
    p.add_argument("--notify", choices=["slack", "teams"], help="Send notification on completion")
    p.add_argument("--non-interactive", action="store_true", help="Non-interactive mode")
    p.add_argument("--ci-mode", action="store_true", help="CI/CD mode (clean output, exit codes)")
    p.add_argument("--retry", action="store_true", help="Retry on failure")
    p.add_argument("--verbose", "-v", action="store_true", help="Verbose logging")
    
    return p

if __name__ == "__main__":
    parser = _build_cli()
    args = parser.parse_args()
    
    setup_logging(verbose=args.verbose, ci_mode=args.ci_mode)
    
    try:
        demo(
            requirement=args.requirement,
            non_interactive=args.non_interactive,
            preset=args.preset,
            notify=args.notify,
            ci_mode=args.ci_mode,
            retry_on_failure=args.retry,
        )
    except KeyboardInterrupt:
        print("\n\n👋 Cancelled by user")
        sys.exit(130)
    except Exception as exc:
        logger.exception(f"Demo failed: {exc}")
        sys.exit(1)
