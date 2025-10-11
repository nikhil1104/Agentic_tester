# modules/runner.py
"""
Runner (Phase 6.1.2) — Production-Grade Test Execution Engine

Enhancements vs. 6.1:
- UI generation (which may use Playwright Sync API) runs in a worker thread to avoid
  greenlet/thread-switch errors when the runner uses asyncio.
- Robust fallback JSON parsing for `npx playwright test --reporter=json` (parse stdout).
- Event-loop safe sync execution: avoid asyncio.run() inside a running loop.
- Minor hardening and comments.

Feature set:
✅ Stage registry pattern for extensibility
✅ Event system for observability
✅ Retry logic with exponential backoff
✅ Async stage execution with concurrency control
✅ Structured metrics and progress tracking
✅ Plan validation (JSON Schema)
✅ JUnit XML export for CI/CD
✅ Stage dependencies and ordering
✅ Dry-run mode for validation
✅ Circuit breaker pattern
✅ Comprehensive error handling
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import asyncio
import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, List, Set, Callable, Tuple
from enum import Enum

logger = logging.getLogger(__name__)

# Optional imports with graceful fallback
try:
    from modules.reporter import Reporter
    REPORTER_AVAILABLE = True
except Exception:
    Reporter = None
    REPORTER_AVAILABLE = False

try:
    from modules.ui_framework_generator import UIFrameworkGenerator
    UI_GENERATOR_AVAILABLE = True
except Exception:
    UIFrameworkGenerator = None
    UI_GENERATOR_AVAILABLE = False

try:
    from modules.security_engine import SecurityEngine
    SECURITY_ENGINE_AVAILABLE = True
except Exception:
    SecurityEngine = None
    SECURITY_ENGINE_AVAILABLE = False

try:
    from jsonschema import validate, ValidationError as JSONSchemaValidationError
    JSON_SCHEMA_AVAILABLE = True
except ImportError:
    JSON_SCHEMA_AVAILABLE = False

try:
    import junit_xml
    JUNIT_AVAILABLE = True
except ImportError:
    JUNIT_AVAILABLE = False
    logger.info("junit-xml not installed. JUnit export disabled. Install: pip install junit-xml")


# ==================== Types and Enums ====================

class StageStatus(Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"


class RunnerEvent(Enum):
    RUN_START = "run_start"
    RUN_COMPLETE = "run_complete"
    STAGE_START = "stage_start"
    STAGE_COMPLETE = "stage_complete"
    STAGE_RETRY = "stage_retry"
    VALIDATION_FAILED = "validation_failed"
    PROGRESS_UPDATE = "progress_update"


@dataclass
class StageMetrics:
    stage_name: str
    status: StageStatus
    start_time: float
    end_time: Optional[float] = None
    duration_s: float = 0.0
    retry_count: int = 0
    error: Optional[str] = None
    details: Dict[str, Any] = field(default_factory=dict)

    def complete(self, status: StageStatus, error: Optional[str] = None):
        self.end_time = time.time()
        self.duration_s = round(self.end_time - self.start_time, 2)
        self.status = status
        self.error = error


@dataclass
class RunMetrics:
    execution_id: str
    start_time: float
    end_time: Optional[float] = None
    duration_s: float = 0.0
    stages: List[StageMetrics] = field(default_factory=list)
    overall_status: StageStatus = StageStatus.SUCCESS

    def add_stage(self, stage: StageMetrics):
        self.stages.append(stage)

    def complete(self, success: bool):
        self.end_time = time.time()
        self.duration_s = round(self.end_time - self.start_time, 2)
        self.overall_status = StageStatus.SUCCESS if success else StageStatus.FAILURE

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "stages": [{**asdict(s), "status": s.status.value} for s in self.stages],
            "overall_status": self.overall_status.value,
        }


# ==================== Configuration ====================

@dataclass
class RunnerConfig:
    reports_dir: str = "reports"
    workspace_dir: str = "workspace"

    # Playwright settings
    browsers: str = "chromium"
    playwright_timeout_s: int = 3600
    playwright_workers: int = 4

    # Execution settings
    retry_max_attempts: int = 3
    retry_delay_s: float = 2.0
    stage_timeout_s: int = 7200
    enable_async: bool = True
    max_concurrent_stages: int = 2

    # Features
    enable_dry_run: bool = False
    enable_junit_export: bool = True
    enable_progress_tracking: bool = True
    validate_plan: bool = True

    # Circuit breaker
    circuit_breaker_threshold: int = 5
    circuit_breaker_timeout_s: int = 300

    @classmethod
    def from_env(cls) -> "RunnerConfig":
        return cls(
            browsers=os.getenv("BROWSERS", "chromium"),
            enable_dry_run=os.getenv("DRY_RUN", "false").lower() == "true",
            enable_async=os.getenv("RUNNER_ASYNC", "true").lower() == "true",
        )


# ==================== Exceptions ====================

class RunnerError(Exception):
    pass


class StageExecutionError(RunnerError):
    def __init__(self, stage_name: str, original_error: Exception):
        self.stage_name = stage_name
        self.original_error = original_error
        super().__init__(f"Stage '{stage_name}' execution failed: {original_error}")


class PlanValidationError(RunnerError):
    pass


# ==================== Event System ====================

@dataclass
class RunnerEventData:
    event_type: RunnerEvent
    execution_id: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


EventCallback = Callable[[RunnerEventData], None]


class EventEmitter:
    def __init__(self):
        self._callbacks: Dict[RunnerEvent, List[EventCallback]] = {
            event_type: [] for event_type in RunnerEvent
        }

    def on(self, event_type: RunnerEvent, callback: EventCallback):
        self._callbacks[event_type].append(callback)

    def emit(self, event: RunnerEventData):
        for callback in self._callbacks.get(event.event_type, []):
            try:
                callback(event)
            except Exception as e:
                logger.error("Event callback failed: %s", e)


# ==================== Circuit Breaker ====================

class CircuitBreaker:
    def __init__(self, threshold: int = 5, timeout_s: int = 300):
        self.threshold = threshold
        self.timeout_s = timeout_s
        self.failure_count = 0
        self.last_failure_time: Optional[float] = None
        self.is_open = False

    def record_success(self):
        self.failure_count = 0
        self.is_open = False

    def record_failure(self):
        self.failure_count += 1
        self.last_failure_time = time.time()
        if self.failure_count >= self.threshold:
            self.is_open = True
            logger.warning("⚠️ Circuit breaker opened after %d failures", self.failure_count)

    def can_execute(self) -> bool:
        if not self.is_open:
            return True
        if self.last_failure_time:
            elapsed = time.time() - self.last_failure_time
            if elapsed >= self.timeout_s:
                logger.info("Circuit breaker attempting recovery...")
                self.is_open = False
                self.failure_count = 0
                return True
        return False


# ==================== Helpers ====================

def _run_coro_blocking(coro):
    """
    Run an async coroutine from any context:
    - If no loop is running, use asyncio.run()
    - If a loop is running (e.g., notebook / other framework), execute in a new loop on a temp thread.
    """
    try:
        asyncio.get_running_loop()
        loop_running = True
    except RuntimeError:
        loop_running = False

    if not loop_running:
        return asyncio.run(coro)

    result_holder = {}
    error_holder = {}

    def _worker():
        new_loop = asyncio.new_event_loop()
        try:
            asyncio.set_event_loop(new_loop)
            result_holder["result"] = new_loop.run_until_complete(coro)
        except Exception as e:
            error_holder["error"] = e
        finally:
            new_loop.close()

    t = threading.Thread(target=_worker, daemon=True)
    t.start()
    t.join()

    if "error" in error_holder:
        raise error_holder["error"]
    return result_holder.get("result")


# ==================== Abstract Stage ====================

class AbstractStage(ABC):
    """
    Abstract base class for execution stages.
    Similar to AbstractSecurityCheck pattern.
    """

    def __init__(self, config: RunnerConfig):
        self.config = config

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def dependencies(self) -> List[str]:
        return []

    @abstractmethod
    async def execute_async(
        self,
        plan: Dict[str, Any],
        execution_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        pass

    def execute_sync(
        self,
        plan: Dict[str, Any],
        execution_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        # Make sync calls safe even if a loop is already running
        return _run_coro_blocking(self.execute_async(plan, execution_id, context))


# ==================== Concrete Stages ====================

class UIStage(AbstractStage):
    """Playwright UI test execution stage."""

    @property
    def name(self) -> str:
        return "ui"

    def is_applicable(self, plan: Dict[str, Any]) -> bool:
        return bool(plan.get("suites", {}).get("ui"))

    async def execute_async(
        self,
        plan: Dict[str, Any],
        execution_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Execute UI tests with Playwright."""
        if not UI_GENERATOR_AVAILABLE:
            raise RunnerError("UIFrameworkGenerator not available")

        # Generate workspace once and cache (avoid double UI generation)
        plan.setdefault("artifacts", {})
        ui_workspace = plan["artifacts"].get("ui_workspace")

        if not ui_workspace:
            logger.info("🏗️ Generating Playwright workspace (first time)")
            gen = UIFrameworkGenerator(
                plan=plan,
                html_cache_dir="data/scraped_docs",
                output_type="ts",
                execution_id=execution_id,
                split_cases=False,
            )
            # IMPORTANT:
            # If the generator internally uses Playwright *Sync* API, running it directly here
            # (on the asyncio event loop thread) can cause greenlet/thread-switch errors.
            # Run in a thread instead.
            loop = asyncio.get_running_loop()
            ui_workspace = await loop.run_in_executor(None, gen.generate)
            plan["artifacts"]["ui_workspace"] = ui_workspace
            context["ui_workspace"] = ui_workspace
        else:
            logger.info("♻️ Reusing workspace: %s", ui_workspace)

        # Execute Playwright tests
        result = await self._run_playwright(ui_workspace)
        result["workspace"] = ui_workspace
        return result

    async def _run_playwright(self, workspace: str) -> Dict[str, Any]:
        """Run Playwright tests in workspace (subprocess-based, async-friendly)."""
        ws = Path(workspace).resolve()
        if not ws.exists():
            raise FileNotFoundError(f"Workspace not found: {ws}")

        # Isolated environment (no global pollution)
        npm_cache = ws / ".npm-cache"
        pw_browsers = ws / ".pw-browsers"
        npm_cache.mkdir(parents=True, exist_ok=True)
        pw_browsers.mkdir(parents=True, exist_ok=True)

        env = os.environ.copy()
        env.update({
            "BROWSERS": self.config.browsers,
            "NPM_CONFIG_CACHE": str(npm_cache),
            "PLAYWRIGHT_BROWSERS_PATH": str(pw_browsers),
            "HOME": str(ws),
        })

        # Ensure chromium present (best-effort)
        def _chromium_present(p: Path) -> bool:
            try:
                return any("chromium" in d.name or "chrome" in d.name for d in p.iterdir())
            except Exception:
                return False

        if not _chromium_present(pw_browsers):
            logger.info("⬇️ Installing Playwright browsers (chromium)…")
            # Do not fail the whole run if this step exits non-zero (network hiccups, etc.)
            await asyncio.get_running_loop().run_in_executor(
                None, lambda: subprocess.run(["npx", "playwright", "install", "chromium"], cwd=ws, env=env, check=False)
            )

        # Primary run: respect reporters from repo config
        cmd = ["npx", "playwright", "test"]
        logger.info("▶️ Running Playwright: %s", " ".join(cmd))

        loop = asyncio.get_running_loop()
        proc = await loop.run_in_executor(
            None,
            lambda: subprocess.run(
                cmd,
                cwd=ws,
                env=env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=self.config.playwright_timeout_s,
                check=False,
                text=True,
            )
        )

        if proc.stdout:
            for line in proc.stdout.splitlines():
                if line.strip():
                    logger.info(line)
        if proc.stderr:
            for line in proc.stderr.splitlines():
                if line.strip():
                    logger.warning(line)

        # Parse JSON report produced by config (if any)
        report_path = ws / "reports" / "playwright" / "report.json"
        stats = self._parse_playwright_report_file(report_path)

        # Fallback: if no report totals, rerun with explicit JSON reporter and parse stdout
        need_fallback = (stats.get("total", 0) == 0) and (not report_path.exists())
        if need_fallback:
            logger.warning("No Playwright JSON found; retrying with --reporter=json")
            proc2 = await loop.run_in_executor(
                None,
                lambda: subprocess.run(
                    ["npx", "playwright", "test", "--reporter=json"],
                    cwd=ws,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=self.config.playwright_timeout_s,
                    check=False,
                    text=True,
                )
            )
            if proc2.stdout:
                # Parse JSON from stdout (official behavior of reporter=json)
                stats = self._parse_playwright_json_stdout(proc2.stdout)
                for line in proc2.stdout.splitlines():
                    if line.strip():
                        logger.info(line)
            if proc2.stderr:
                for line in proc2.stderr.splitlines():
                    if line.strip():
                        logger.warning(line)

            return {
                **stats,
                "success": stats.get("failed", 0) == 0,
                "report_path": str(report_path),
                "return_code": proc2.returncode,
            }

        return {
            **stats,
            "success": stats.get("failed", 0) == 0,
            "report_path": str(report_path),
            "return_code": proc.returncode,
        }

    def _parse_playwright_report_file(self, report_path: Path) -> Dict[str, int]:
        """Parse Playwright JSON report written to disk by config."""
        if not report_path.exists():
            logger.warning("Playwright report not found: %s", report_path)
            return {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "flaky": 0}
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
            return self._count_statuses(data)
        except Exception as e:
            logger.error("Failed to parse Playwright report file: %s", e)
            return {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "flaky": 0}

    def _parse_playwright_json_stdout(self, stdout: str) -> Dict[str, int]:
        """Parse JSON emitted by `--reporter=json` (stdout)."""
        try:
            data = json.loads(stdout)
            return self._count_statuses(data)
        except Exception as e:
            logger.error("Failed to parse Playwright JSON from stdout: %s", e)
            return {"total": 0, "passed": 0, "failed": 0, "skipped": 0, "flaky": 0}

    def _count_statuses(self, report: Dict[str, Any]) -> Dict[str, int]:
        """Recursively count test statuses across the report object."""
        counts = {"passed": 0, "failed": 0, "skipped": 0, "flaky": 0}

        def visit(node):
            if isinstance(node, dict):
                status = (node.get("status") or "").lower()
                if status in ("passed", "ok"):
                    counts["passed"] += 1
                elif status == "failed":
                    counts["failed"] += 1
                elif status in ("skipped", "skip"):
                    counts["skipped"] += 1
                elif status == "flaky":
                    counts["flaky"] += 1
                for v in node.values():
                    visit(v)
            elif isinstance(node, list):
                for item in node:
                    visit(item)

        visit(report)
        counts["total"] = sum(counts.values())
        logger.info("✅ Playwright results: passed=%d failed=%d skipped=%d flaky=%d",
                    counts["passed"], counts["failed"], counts["skipped"], counts["flaky"])
        return counts


class SecurityStage(AbstractStage):
    """Security validation stage."""

    @property
    def name(self) -> str:
        return "security"

    def is_applicable(self, plan: Dict[str, Any]) -> bool:
        return bool(plan.get("suites", {}).get("security"))

    async def execute_async(
        self,
        plan: Dict[str, Any],
        execution_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        if not SECURITY_ENGINE_AVAILABLE:
            logger.warning("SecurityEngine not available, skipping security stage")
            return {"skipped": True, "reason": "SecurityEngine not available"}

        try:
            suites = (plan.get("suites") or {}).get("security") or []
            enabled = None
            if suites:
                first = suites[0]
                if isinstance(first, dict) and "enabled" in first:
                    enabled = first["enabled"]
                elif isinstance(first, list):
                    enabled = first

            engine = SecurityEngine(enabled_checks=enabled)

            base_url = (
                (plan.get("details") or {}).get("url")
                or plan.get("project")
                or context.get("base_url")
            )
            if not base_url:
                logger.warning("No base URL found for security checks")
                return {"skipped": True, "reason": "No base URL"}

            result = await engine.scan_async(base_url)
            report = result.to_dict()

            failed = sum(1 for f in report["findings"] if f.get("status") in ("FAIL", "ERROR"))
            warnings = sum(1 for f in report["findings"] if f.get("status") == "WARNING")

            return {
                "success": failed == 0,
                "results": [report],
                "total_checks": len(report["findings"]),
                "failed": failed,
                "warnings": warnings,
                "risk_score": (report.get("summary") or {}).get("risk_score", 0),
                "duration_s": report.get("duration_s", 0.0),
            }

        except Exception as e:
            logger.error("Security stage failed: %s", e)
            raise StageExecutionError(self.name, e)


class APIStage(AbstractStage):
    """API test execution stage."""

    @property
    def name(self) -> str:
        return "api"

    def is_applicable(self, plan: Dict[str, Any]) -> bool:
        return bool(plan.get("suites", {}).get("api"))

    async def execute_async(
        self,
        plan: Dict[str, Any],
        execution_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        try:
            from modules.api_test_engine import APITestEngine as _Engine
        except ImportError:
            try:
                from modules.api_test_engine import ApiTestEngine as _Engine  # legacy
            except ImportError:
                logger.warning("API test engine not available, skipping API stage")
                return {"skipped": True, "reason": "ApiTestEngine not available"}

        try:
            engine = _Engine()
            suites = (plan.get("suites") or {}).get("api") or []
            base_url = (
                (plan.get("details") or {}).get("url")
                or plan.get("project")
                or context.get("base_url")
                or ""
            )

            if getattr(engine, "cfg", None) is not None and base_url:
                engine.cfg["base_url"] = base_url

            if hasattr(engine, "run_suites_async"):
                summary = await engine.run_suites_async(suites)
            else:
                # CPU-bound or blocking? Run in thread to avoid blocking event loop.
                summary = await asyncio.get_running_loop().run_in_executor(None, lambda: engine.run_suites(suites))

            if isinstance(summary, list):
                cases = summary
                passed_cases = 0
                failed_cases = 0
                for c in cases:
                    steps = c.get("steps", [])
                    case_failed = any((s.get("status") or "").upper() == "FAIL" for s in steps)
                    if case_failed:
                        failed_cases += 1
                    else:
                        passed_cases += 1
                return {
                    "success": failed_cases == 0,
                    "cases": cases,
                    "passed": passed_cases,
                    "failed": failed_cases,
                }
            else:
                return {
                    "success": summary.get("failed", 0) == 0,
                    **summary,
                }

        except Exception as e:
            logger.error("API stage failed: %s", e)
            raise StageExecutionError(self.name, e)


# ==================== Stage Registry ====================

STAGE_REGISTRY: Dict[str, type[AbstractStage]] = {
    "ui": UIStage,
    "security": SecurityStage,
    "api": APIStage,
}


def get_enabled_stages(
    plan: Dict[str, Any],
    config: RunnerConfig,
) -> List[AbstractStage]:
    stages = []
    for stage_name, stage_class in STAGE_REGISTRY.items():
        stage = stage_class(config)
        if stage.is_applicable(plan):
            stages.append(stage)
            logger.info("✓ Stage '%s' is applicable", stage_name)
        else:
            logger.debug("✗ Stage '%s' is not applicable", stage_name)
    return stages


# ==================== Plan Validation ====================

PLAN_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["suites"],
    "properties": {
        "suites": {"type": "object", "minProperties": 1},
        "ui_config": {"type": "object"},
        "artifacts": {"type": "object"},
    },
}


def validate_plan(plan: Dict[str, Any]) -> None:
    if not JSON_SCHEMA_AVAILABLE:
        logger.warning("JSON Schema validation skipped (jsonschema not installed)")
        return
    try:
        validate(instance=plan, schema=PLAN_SCHEMA)
        logger.debug("✓ Plan validation passed")
    except JSONSchemaValidationError as e:
        raise PlanValidationError(f"Plan validation failed: {e.message}")


# ==================== JUnit Export ====================

def export_junit_xml(results: Dict[str, Any], output_path: Path) -> None:
    if not JUNIT_AVAILABLE:
        logger.warning("JUnit export skipped (junit-xml not installed)")
        return
    try:
        test_suites = []

        for stage_name, stage_result in results.get("stage_results", {}).items():
            if stage_result.get("skipped"):
                continue

            test_cases = []
            if stage_name == "ui":
                passed = stage_result.get("passed", 0)
                failed = stage_result.get("failed", 0)
                skipped = stage_result.get("skipped", 0)
                for i in range(passed):
                    test_cases.append(junit_xml.TestCase(f"ui_test_{i+1}", stage_name))
                for i in range(failed):
                    tc = junit_xml.TestCase(f"ui_test_failed_{i+1}", stage_name)
                    tc.add_failure_info("Test failed")
                    test_cases.append(tc)
                for i in range(skipped):
                    tc = junit_xml.TestCase(f"ui_test_skipped_{i+1}", stage_name)
                    tc.add_skipped_info("Skipped by Playwright")
                    test_cases.append(tc)

            elif stage_name == "security":
                for res in stage_result.get("results", []):
                    url = res.get("url", "target")
                    for f in res.get("findings", []):
                        name = f.get("check_name", "security_check")
                        status = (f.get("status") or "").upper()
                        msg = f.get("message", "")
                        tc = junit_xml.TestCase(f"{name}::{url}", stage_name)
                        if status in ("FAIL", "ERROR"):
                            tc.add_failure_info(msg or "Security failure")
                        elif status == "WARNING":
                            tc.add_skipped_info(msg or "Security warning")
                        test_cases.append(tc)

            elif stage_name == "api":
                passed = stage_result.get("passed", 0)
                failed = stage_result.get("failed", 0)
                for i in range(passed):
                    test_cases.append(junit_xml.TestCase(f"api_test_{i+1}", stage_name))
                for i in range(failed):
                    tc = junit_xml.TestCase(f"api_test_failed_{i+1}", stage_name)
                    tc.add_failure_info("API test failed")
                    test_cases.append(tc)

            test_suites.append(junit_xml.TestSuite(stage_name, test_cases))

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            junit_xml.TestSuite.to_file(f, test_suites, prettyprint=True)

        logger.info("✓ JUnit XML exported to %s", output_path)
    except Exception as e:
        logger.error("Failed to export JUnit XML: %s", e)


# ==================== Main Runner ====================

class Runner:
    """Production-grade test execution engine."""

    def __init__(self, config: Optional[RunnerConfig] = None):
        self.config = config or RunnerConfig()
        self._reports_root = Path(self.config.reports_dir)
        self._reports_root.mkdir(parents=True, exist_ok=True)

        self.events = EventEmitter()
        self._circuit_breakers: Dict[str, CircuitBreaker] = {}

        self._reporter = None
        if REPORTER_AVAILABLE:
            try:
                self._reporter = Reporter(reports_dir=str(self._reports_root))
            except Exception as e:
                logger.warning("Failed to initialize Reporter: %s", e)

        logger.info("Runner initialized (async=%s)", self.config.enable_async)

    def add_event_listener(self, event_type: RunnerEvent, callback: EventCallback):
        self.events.on(event_type, callback)

    def run_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        execution_id = (
            plan.get("plan_meta", {}).get("plan_id")
            or datetime.utcnow().strftime("%Y%m%d%H%M%S")
        )

        metrics = RunMetrics(execution_id=execution_id, start_time=time.time())

        logger.info("=" * 60)
        logger.info("[%s] Starting test execution", execution_id)
        logger.info("=" * 60)

        self.events.emit(RunnerEventData(
            event_type=RunnerEvent.RUN_START,
            execution_id=execution_id,
            timestamp=time.time(),
            data={"plan": plan},
        ))

        try:
            if self.config.validate_plan:
                validate_plan(plan)

            if self.config.enable_dry_run:
                logger.warning("[%s] DRY-RUN MODE: Skipping execution", execution_id)
                return self._create_dry_run_result(execution_id, plan)

            if self.config.enable_async:
                results = _run_coro_blocking(self._execute_stages_async(plan, execution_id, metrics))
            else:
                results = self._execute_stages_sync(plan, execution_id, metrics)

            overall_success = all(
                stage.status == StageStatus.SUCCESS
                for stage in metrics.stages
                if stage.status != StageStatus.SKIPPED
            )

            metrics.complete(overall_success)

            final_results = {
                "execution_id": execution_id,
                "overall_success": overall_success,
                "start_time": datetime.fromtimestamp(metrics.start_time).isoformat() + "Z",
                "end_time": datetime.fromtimestamp(metrics.end_time).isoformat() + "Z",
                "duration_s": metrics.duration_s,
                "stage_results": results,
                "metrics": metrics.to_dict(),
            }

            self._persist_reports(final_results, execution_id)

            if self.config.enable_junit_export:
                junit_path = self._reports_root / f"{execution_id}.xml"
                export_junit_xml(final_results, junit_path)
                final_results["junit_report"] = str(junit_path)

            self.events.emit(RunnerEventData(
                event_type=RunnerEvent.RUN_COMPLETE,
                execution_id=execution_id,
                timestamp=time.time(),
                data={"success": overall_success},
            ))

            if self._reporter and hasattr(self._reporter, "on_run_complete"):
                try:
                    self._reporter.on_run_complete(final_results)
                except Exception as e:
                    logger.warning("Reporter hook failed: %s", e)

            logger.info("=" * 60)
            logger.info(
                "[%s] Execution completed: success=%s, duration=%.2fs",
                execution_id,
                overall_success,
                metrics.duration_s,
            )
            logger.info("=" * 60)

            return final_results

        except Exception as e:
            logger.error("[%s] Execution failed: %s", execution_id, e, exc_info=True)
            metrics.complete(False)
            return {
                "execution_id": execution_id,
                "overall_success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "metrics": metrics.to_dict(),
            }

    # ==================== Stage Execution ====================

    async def _execute_stages_async(
        self,
        plan: Dict[str, Any],
        execution_id: str,
        metrics: RunMetrics,
    ) -> Dict[str, Any]:
        stages = get_enabled_stages(plan, self.config)
        context: Dict[str, Any] = {}
        results: Dict[str, Any] = {}

        sorted_stages = self._topological_sort(stages)

        for batch in sorted_stages:
            coros: List[Tuple[str, asyncio.Future]] = [
                (stage.name, self._execute_stage_async(stage, plan, execution_id, context, metrics))
                for stage in batch
            ]
            i = 0
            while i < len(coros):
                # Enforce concurrency window
                chunk = coros[i:i + max(1, self.config.max_concurrent_stages)]
                names = [n for n, _ in chunk]
                tasks = [c for _, c in chunk]
                try:
                    chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
                except Exception as e:
                    chunk_results = [e]
                for name, res in zip(names, chunk_results):
                    if isinstance(res, Exception):
                        logger.error("Stage '%s' failed: %s", name, res)
                        results[name] = {"error": str(res), "status": "ERROR"}
                    else:
                        results[name] = res
                i += len(chunk)

        return results

    async def _execute_stage_async(
        self,
        stage: AbstractStage,
        plan: Dict[str, Any],
        execution_id: str,
        context: Dict[str, Any],
        metrics: RunMetrics,
    ) -> Dict[str, Any]:
        stage_metrics = StageMetrics(
            stage_name=stage.name,
            status=StageStatus.SUCCESS,
            start_time=time.time(),
        )

        cb = self._get_circuit_breaker(stage.name)
        if not cb.can_execute():
            logger.warning("⚠️ Circuit breaker open for stage '%s'", stage.name)
            stage_metrics.complete(StageStatus.SKIPPED, "Circuit breaker open")
            metrics.add_stage(stage_metrics)
            return {"skipped": True, "reason": "circuit_breaker_open"}

        self.events.emit(RunnerEventData(
            event_type=RunnerEvent.STAGE_START,
            execution_id=execution_id,
            timestamp=time.time(),
            data={"stage": stage.name},
        ))

        last_error = None

        for attempt in range(1, self.config.retry_max_attempts + 1):
            try:
                logger.info("[%s] Stage '%s': attempt %d/%d",
                            execution_id, stage.name, attempt, self.config.retry_max_attempts)

                result = await asyncio.wait_for(
                    stage.execute_async(plan, execution_id, context),
                    timeout=self.config.stage_timeout_s,
                )

                stage_metrics.complete(StageStatus.SUCCESS)
                stage_metrics.retry_count = attempt - 1
                cb.record_success()
                metrics.add_stage(stage_metrics)

                self.events.emit(RunnerEventData(
                    event_type=RunnerEvent.STAGE_COMPLETE,
                    execution_id=execution_id,
                    timestamp=time.time(),
                    data={"stage": stage.name, "success": True},
                ))
                return result

            except asyncio.TimeoutError:
                last_error = TimeoutError(f"Stage timed out after {self.config.stage_timeout_s}s")
                logger.warning("[%s] Stage '%s' timed out (attempt %d/%d)",
                               execution_id, stage.name, attempt, self.config.retry_max_attempts)
            except Exception as e:
                last_error = e
                logger.warning("[%s] Stage '%s' failed (attempt %d/%d): %s",
                               execution_id, stage.name, attempt, self.config.retry_max_attempts, e)

            if attempt < self.config.retry_max_attempts:
                delay = self.config.retry_delay_s * (2 ** (attempt - 1))
                logger.info("Retrying in %.1fs...", delay)
                self.events.emit(RunnerEventData(
                    event_type=RunnerEvent.STAGE_RETRY,
                    execution_id=execution_id,
                    timestamp=time.time(),
                    data={"stage": stage.name, "attempt": attempt, "delay": delay},
                ))
                await asyncio.sleep(delay)

        # All retries exhausted
        stage_metrics.complete(StageStatus.FAILURE, str(last_error))
        stage_metrics.retry_count = self.config.retry_max_attempts
        cb.record_failure()
        metrics.add_stage(stage_metrics)

        self.events.emit(RunnerEventData(
            event_type=RunnerEvent.STAGE_COMPLETE,
            execution_id=execution_id,
            timestamp=time.time(),
            data={"stage": stage.name, "success": False, "error": str(last_error)},
        ))
        return {"error": str(last_error), "status": "FAILURE"}

    def _execute_stages_sync(
        self,
        plan: Dict[str, Any],
        execution_id: str,
        metrics: RunMetrics,
    ) -> Dict[str, Any]:
        stages = get_enabled_stages(plan, self.config)
        context: Dict[str, Any] = {}
        results: Dict[str, Any] = {}

        for stage in stages:
            try:
                result = stage.execute_sync(plan, execution_id, context)
                results[stage.name] = result
            except Exception as e:
                logger.error("Stage '%s' failed: %s", stage.name, e)
                results[stage.name] = {"error": str(e), "status": "ERROR"}
        return results

    # ==================== Utilities ====================

    def _topological_sort(self, stages: List[AbstractStage]) -> List[List[AbstractStage]]:
        """
        Topologically sort stages by dependencies.
        Returns list of batches that can run concurrently.
        """
        graph: Dict[str, Set[str]] = {}
        in_degree: Dict[str, int] = {}
        stage_map: Dict[str, AbstractStage] = {}

        for stage in stages:
            stage_map[stage.name] = stage
            graph[stage.name] = set(stage.dependencies)
            in_degree[stage.name] = len(stage.dependencies)

        batches: List[List[AbstractStage]] = []
        remaining = set(stage_map.keys())

        while remaining:
            batch = [stage_map[name] for name in remaining if in_degree[name] == 0]
            if not batch:
                logger.warning("Circular dependency detected in stages")
                batch = [stage_map[name] for name in remaining]
                batches.append(batch)
                break

            batches.append(batch)
            for stage in batch:
                remaining.remove(stage.name)
                for name in list(remaining):
                    if stage.name in graph[name]:
                        in_degree[name] -= 1

        return batches

    def _get_circuit_breaker(self, stage_name: str) -> CircuitBreaker:
        cb = self._circuit_breakers.get(stage_name)
        if cb is None:
            cb = CircuitBreaker(
                threshold=self.config.circuit_breaker_threshold,
                timeout_s=self.config.circuit_breaker_timeout_s,
            )
            self._circuit_breakers[stage_name] = cb
        return cb

    def _create_dry_run_result(self, execution_id: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        stages = get_enabled_stages(plan, self.config)
        return {
            "execution_id": execution_id,
            "dry_run": True,
            "overall_success": True,
            "message": "Dry-run completed successfully",
            "applicable_stages": [stage.name for stage in stages],
            "plan_valid": True,
        }

    def _persist_reports(self, results: Dict[str, Any], execution_id: str) -> None:
        json_path = self._reports_root / f"{execution_id}.json"
        json_path.parent.mkdir(parents=True, exist_ok=True)

        tmp_path = json_path.with_suffix(".tmp")
        tmp_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        os.replace(tmp_path, json_path)

        html_path = self._reports_root / f"{execution_id}.html"
        html_content = self._generate_html_report(results)
        tmp2_path = html_path.with_suffix(".tmp")
        tmp2_path.write_text(html_content, encoding="utf-8")
        os.replace(tmp2_path, html_path)

        logger.info("✓ Reports saved: %s, %s", json_path, html_path)
        results["final_reports"] = {"json": str(json_path), "html": str(html_path)}

    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        return f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Test Report - {results['execution_id']}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                .success {{ color: green; }}
                .failure {{ color: red; }}
                .header {{ background: #f0f0f0; padding: 10px; }}
                pre {{ background: #f5f5f5; padding: 10px; overflow-x: auto; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Test Execution Report</h1>
                <p>Execution ID: {results['execution_id']}</p>
                <p class="{{'success' if results['overall_success'] else 'failure'}}">
                    Status: {'SUCCESS' if results['overall_success'] else 'FAILURE'}
                </p>
                <p>Duration: {results['duration_s']}s</p>
            </div>
            <h2>Results</h2>
            <pre>{json.dumps(results, indent=2)}</pre>
        </body>
        </html>
        """


# ==================== CLI Support ====================

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    if len(sys.argv) < 2:
        print("Usage: python -m modules.runner <plan.json>")
        sys.exit(1)

    plan_path = Path(sys.argv[1])
    if not plan_path.exists():
        print(f"Plan file not found: {plan_path}")
        sys.exit(1)

    plan = json.loads(plan_path.read_text())

    config = RunnerConfig.from_env()
    runner = Runner(config=config)

    result = runner.run_plan(plan)

    print("\n" + "=" * 60)
    print("EXECUTION RESULT")
    print("=" * 60)
    print(f"Success: {result['overall_success']}")
    print(f"Duration: {result['duration_s']}s")
    print("=" * 60)

    sys.exit(0 if result['overall_success'] else 1)
