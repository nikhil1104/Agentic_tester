# modules/spec_orchestrator.py
"""
Spec Orchestrator (Phase 6.1+) - Production-Ready Implementation
Enhanced with:
✅ State recovery (resume from checkpoint)
✅ UUID-based run IDs for traceability
✅ Telemetry export (Prometheus + JSONL)
✅ Dry-run mode for safe testing
✅ Event hooks for extensibility
✅ Graceful security engine fallback
✅ JSON Schema plan validation (with optional external schema)
"""

from __future__ import annotations
import json
import logging
import time
import uuid
from typing import Dict, Any, Optional, List, Protocol, Callable
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from enum import Enum

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Core modules
# ------------------------------------------------------------------------------
from modules.conversational_agent import ConversationalAgent
from modules.web_scraper import WebScraper
from modules.test_generator import TestGenerator
from modules.runner import Runner

# Optional: Security engine (graceful fallback if missing)
try:
    from modules.security_engine import SecurityEngine
    SECURITY_ENGINE_AVAILABLE = True
except Exception as e:
    SECURITY_ENGINE_AVAILABLE = False
    logger.warning("SecurityEngine not available: %s. Security checks disabled.", e)

# Optional: JSON Schema validation
try:
    from jsonschema import validate as jsonschema_validate, ValidationError as JSONSchemaValidationError
    JSON_SCHEMA_AVAILABLE = True
except Exception:
    JSON_SCHEMA_AVAILABLE = False
    logger.info("jsonschema not installed. Plan validation disabled. Install: pip install jsonschema")

# Optional: Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, start_http_server, generate_latest
    PROMETHEUS_AVAILABLE = True
except Exception:
    PROMETHEUS_AVAILABLE = False
    logger.info("prometheus_client not installed. Metrics export disabled. Install: pip install prometheus-client")


# ==================== Configuration ====================

@dataclass
class OrchestratorConfig:
    """Centralized configuration for the orchestrator."""
    # Scraper settings
    scraper_max_pages: int = 20
    scraper_max_depth: int = 2
    scraper_timeout_ms: int = 40000
    scraper_post_load_wait_ms: int = 1500   # maps to WebScraper.post_load_wait
    scraper_same_origin: bool = True
    scraper_capture_api: bool = True
    scraper_rate_limit_s: float = 1.5       # maps to WebScraper.rate_limit_delay
    scraper_respect_robots: bool = False
    scraper_exclude: Optional[List[str]] = None

    # Generator settings
    html_cache_dir: str = "data/scraped_docs"

    # UI config defaults
    ui_browsers: str = "chromium"
    ui_headless: bool = True
    ui_test_timeout_ms: int = 45000
    ui_expect_timeout_ms: int = 8000
    ui_workers: int = 4
    ui_retries: int = 1
    ui_fully_parallel: bool = False
    ui_forbid_only: bool = True
    ui_enable_junit: bool = True

    # Orchestrator settings
    enable_api_discovery: bool = True
    enable_security_checks: bool = True
    api_discovery_limit: int = 50
    retry_max_attempts: int = 3
    retry_delay_s: float = 2.0
    enable_state_persistence: bool = True
    state_file: str = "data/orchestrator_state.json"

    # ✨ NEW: Dry-run mode
    dry_run: bool = False

    # ✨ NEW: Plan validation
    enable_plan_validation: bool = True
    plan_schema_file: Optional[str] = "schemas/test_plan.schema.json"

    # ✨ NEW: Telemetry
    enable_prometheus_metrics: bool = False
    prometheus_port: int = 9090
    enable_jsonl_export: bool = True
    metrics_export_dir: str = "data/metrics"

    @classmethod
    def from_env(cls) -> "OrchestratorConfig":
        """Load configuration from environment variables."""
        import os
        return cls(
            scraper_max_pages=int(os.getenv("SCRAPER_MAX_PAGES", "20")),
            scraper_max_depth=int(os.getenv("SCRAPER_MAX_DEPTH", "2")),
            scraper_timeout_ms=int(os.getenv("SCRAPER_TIMEOUT_MS", "40000")),
            scraper_post_load_wait_ms=int(os.getenv("SCRAPER_POST_LOAD_WAIT_MS", "1500")),
            scraper_same_origin=os.getenv("SCRAPER_SAME_ORIGIN", "true").lower() == "true",
            scraper_capture_api=os.getenv("SCRAPER_CAPTURE_API", "true").lower() == "true",
            scraper_rate_limit_s=float(os.getenv("SCRAPER_RATE_LIMIT_S", "1.5")),
            dry_run=os.getenv("DRY_RUN", "false").lower() == "true",
            enable_prometheus_metrics=os.getenv("ENABLE_PROMETHEUS", "false").lower() == "true",
        )

    @classmethod
    def from_yaml(cls, path: str) -> "OrchestratorConfig":
        """Load configuration from YAML file."""
        import yaml
        with open(path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f) or {}
        return cls(**config_dict)


# ==================== Prometheus Metrics ====================

class PrometheusMetrics:
    """Prometheus metrics collector for orchestrator."""

    def __init__(self, enabled: bool = False, port: int = 9090):
        self.enabled = enabled and PROMETHEUS_AVAILABLE

        if self.enabled:
            # Orchestration counters
            self.orchestrations_total = Counter(
                'orchestrator_runs_total',
                'Total number of orchestration runs',
                ['status']  # success/failure
            )

            self.stages_total = Counter(
                'orchestrator_stages_total',
                'Total number of stage executions',
                ['stage', 'status']
            )

            # Duration histograms
            self.orchestration_duration = Histogram(
                'orchestrator_run_duration_seconds',
                'Duration of complete orchestration runs',
                buckets=[10, 30, 60, 120, 300, 600, 1800]
            )

            self.stage_duration = Histogram(
                'orchestrator_stage_duration_seconds',
                'Duration of individual stages',
                ['stage'],
                buckets=[1, 5, 10, 30, 60, 120]
            )

            # Gauges
            self.active_orchestrations = Gauge(
                'orchestrator_active_runs',
                'Number of currently active orchestration runs'
            )

            # Start HTTP server
            try:
                start_http_server(port)
                logger.info("✅ Prometheus metrics server started on port %d", port)
            except Exception as e:
                logger.warning("Failed to start Prometheus server: %s", e)
                self.enabled = False

    def record_orchestration_start(self):
        if self.enabled:
            self.active_orchestrations.inc()

    def record_orchestration_complete(self, success: bool, duration: float):
        if self.enabled:
            status = 'success' if success else 'failure'
            self.orchestrations_total.labels(status=status).inc()
            self.orchestration_duration.observe(duration)
            self.active_orchestrations.dec()

    def record_stage_complete(self, stage: str, success: bool, duration: float):
        if self.enabled:
            status = 'success' if success else 'failure'
            self.stages_total.labels(stage=stage, status=status).inc()
            self.stage_duration.labels(stage=stage).observe(duration)

    def export_latest(self) -> bytes:
        if self.enabled:
            return generate_latest()
        return b""


# ==================== Metrics & Observability ====================

@dataclass
class StageMetrics:
    stage_name: str
    start_time: float
    end_time: Optional[float] = None
    duration_s: Optional[float] = None
    success: bool = False
    error: Optional[str] = None
    retry_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def complete(self, success: bool = True, error: Optional[str] = None):
        self.end_time = time.time()
        self.duration_s = round(self.end_time - self.start_time, 2)
        self.success = success
        self.error = error


@dataclass
class OrchestrationMetrics:
    run_id: str
    start_time: float
    end_time: Optional[float] = None
    total_duration_s: Optional[float] = None
    stages: List[StageMetrics] = field(default_factory=list)
    overall_success: bool = False
    requirement_text: Optional[str] = None
    target_url: Optional[str] = None

    def add_stage(self, stage: StageMetrics):
        self.stages.append(stage)
        logger.info(
            "[%s] Stage '%s' completed: success=%s, duration=%.2fs, retries=%d",
            self.run_id, stage.stage_name, stage.success, stage.duration_s or 0, stage.retry_count,
        )

    def complete(self, success: bool = True):
        self.end_time = time.time()
        self.total_duration_s = round(self.end_time - self.start_time, 2)
        self.overall_success = success
        logger.info(
            "[%s] Orchestration completed: success=%s, duration=%.2fs, stages=%d",
            self.run_id, success, self.total_duration_s, len(self.stages),
        )

    def to_dict(self) -> Dict[str, Any]:
        return {**asdict(self), "stages": [asdict(s) for s in self.stages]}

    def export_jsonl(self, filepath: str):
        try:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(json.dumps(self.to_dict()) + "\n")
            logger.info("[%s] Exported metrics to %s", self.run_id, filepath)
        except Exception as e:
            logger.error("[%s] Failed to export metrics: %s", self.run_id, e)


# ==================== Event System ====================

class EventType(Enum):
    ORCHESTRATION_START = "orchestration_start"
    ORCHESTRATION_COMPLETE = "orchestration_complete"
    STAGE_START = "stage_start"
    STAGE_COMPLETE = "stage_complete"
    ERROR_OCCURRED = "error_occurred"
    VALIDATION_FAILED = "validation_failed"


@dataclass
class Event:
    event_type: EventType
    run_id: str
    timestamp: float
    data: Dict[str, Any] = field(default_factory=dict)


EventHandler = Callable[[Event], None]


class EventManager:
    def __init__(self):
        self._handlers: Dict[EventType, List[EventHandler]] = {event_type: [] for event_type in EventType}

    def subscribe(self, event_type: EventType, handler: EventHandler):
        self._handlers[event_type].append(handler)

    def unsubscribe(self, event_type: EventType, handler: EventHandler):
        if handler in self._handlers[event_type]:
            self._handlers[event_type].remove(handler)

    def fire(self, event: Event):
        handlers = self._handlers.get(event.event_type, [])
        for handler in handlers:
            try:
                handler(event)
            except Exception as e:
                logger.error("[%s] Event handler failed for %s: %s", event.run_id, event.event_type.value, e)


# ==================== Exceptions ====================

class OrchestrationError(Exception):
    pass


class StageFailureError(OrchestrationError):
    def __init__(self, stage_name: str, original_error: Exception):
        self.stage_name = stage_name
        self.original_error = original_error
        super().__init__(f"Stage '{stage_name}' failed: {original_error}")


class ValidationError(OrchestrationError):
    pass


class ResumeError(OrchestrationError):
    pass


# ==================== Dependency Injection Protocols ====================

class AgentProtocol(Protocol):
    def parse_requirement(self, text: str) -> Dict[str, Any]: ...


class ScraperProtocol(Protocol):
    def deep_scan(self, url: str) -> Dict[str, Any]: ...


class GeneratorProtocol(Protocol):
    def generate_plan(self, req: Dict[str, Any], scan: Dict[str, Any]) -> Dict[str, Any]: ...


class SecurityProtocol(Protocol):
    def plan_from_base_url(self, url: str) -> Optional[Dict[str, Any]]: ...


class RunnerProtocol(Protocol):
    def run_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]: ...


# ==================== State Management ====================

@dataclass
class OrchestrationState:
    run_id: str
    stage: str
    timestamp: str
    requirement: Optional[Dict[str, Any]] = None
    scan_result: Optional[Dict[str, Any]] = None
    plan: Optional[Dict[str, Any]] = None
    execution_result: Optional[Dict[str, Any]] = None

    def save(self, path: str):
        state_dict = asdict(self)
        tmp_path = path + ".tmp"
        Path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(state_dict, f, indent=2)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(Path(tmp_path).read_text(encoding="utf-8"), encoding="utf-8")
        Path(tmp_path).unlink(missing_ok=True)
        logger.debug("[%s] Saved state to %s", self.run_id, path)

    @classmethod
    def load(cls, path: str) -> Optional["OrchestrationState"]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                state_dict = json.load(f)
            logger.info("Loaded state from %s (run_id=%s)", path, state_dict.get("run_id"))
            return cls(**state_dict)
        except FileNotFoundError:
            logger.debug("No state file found at %s", path)
            return None
        except Exception as e:
            logger.warning("Failed to load state from %s: %s", path, e)
            return None


# ==================== Plan Validation ====================

# Embedded JSON Schema (used if external file missing)
PLAN_SCHEMA = {
    "$schema": "http://json-schema.org/draft-07/schema#",
    "type": "object",
    "required": ["suites"],
    "properties": {
        "project": {"type": "string"},
        "suites": {
            "type": "object",
            "minProperties": 1,
            "additionalProperties": {
                "type": "array",
                "items": {
                    "type": "object",
                    "required": ["name", "steps"],
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "steps": {"type": "array", "items": {"type": "string"}},
                        "priority": {"type": "string"},
                        "tags": {"type": "array", "items": {"type": "string"}}
                    }
                }
            }
        },
        "ui_config": {
            "type": "object",
            "properties": {
                "browsers": {"type": "string"},
                "headless": {"type": "boolean"},
                "test_timeout_ms": {"type": "integer"},
                "expect_timeout_ms": {"type": "integer"},
                "workers": {"type": "integer"},
                "retries": {"type": "integer"}
            }
        }
    }
}


def validate_plan_schema(plan: Dict[str, Any], schema: Optional[Dict[str, Any]] = None, schema_file: Optional[str] = None) -> None:
    """
    Validate plan against JSON schema.
    Uses external schema_file if provided and exists, otherwise the embedded PLAN_SCHEMA.
    """
    if not JSON_SCHEMA_AVAILABLE:
        logger.warning("JSON Schema validation skipped (jsonschema not installed)")
        return

    # Prefer external schema if present
    if schema_file and Path(schema_file).exists():
        try:
            schema = json.loads(Path(schema_file).read_text(encoding="utf-8"))
            logger.info("Using external plan schema: %s", schema_file)
        except Exception as e:
            logger.warning("Failed to read external schema (%s): %s; falling back to embedded", schema_file, e)
            schema = schema or PLAN_SCHEMA
    else:
        schema = schema or PLAN_SCHEMA

    try:
        jsonschema_validate(instance=plan, schema=schema)
        logger.debug("Plan schema validation passed")
    except JSONSchemaValidationError as e:
        raise ValidationError(f"Plan schema validation failed: {e.message}")


# ==================== Main Orchestrator ====================

class SpecOrchestrator:
    """Production-ready test specification orchestrator."""

    def __init__(
        self,
        config: Optional[OrchestratorConfig] = None,
        agent: Optional[AgentProtocol] = None,
        scraper: Optional[ScraperProtocol] = None,
        generator: Optional[GeneratorProtocol] = None,
        security: Optional[SecurityProtocol] = None,
        runner: Optional[RunnerProtocol] = None,
        event_manager: Optional[EventManager] = None,
    ):
        self.config = config or OrchestratorConfig()

        # Dependency injection with defaults
        self.agent = agent or ConversationalAgent()
        self.scraper = scraper or self._create_default_scraper()
        self.generator = generator or TestGenerator(html_cache_dir=self.config.html_cache_dir)

        # ✨ Graceful security engine fallback
        if SECURITY_ENGINE_AVAILABLE:
            self.security = security or SecurityEngine()
        else:
            self.security = None
            logger.warning("Security checks disabled (SecurityEngine not available)")

        self.runner = runner or Runner()

        # ✨ Event system & Prometheus
        self.events = event_manager or EventManager()
        self.prometheus = PrometheusMetrics(
            enabled=self.config.enable_prometheus_metrics,
            port=self.config.prometheus_port
        )

        logger.info("SpecOrchestrator initialized (run_id will be generated per run)")

    def _create_default_scraper(self) -> WebScraper:
        """
        Factory method for creating default scraper with config.
        NOTE: Aligns with your WebScraper signature (post_load_wait, rate_limit_delay).
        """
        return WebScraper(
            max_pages=self.config.scraper_max_pages,
            max_depth=self.config.scraper_max_depth,
            timeout_ms=self.config.scraper_timeout_ms,
            post_load_wait=self.config.scraper_post_load_wait_ms,
            same_origin=self.config.scraper_same_origin,
            capture_api=self.config.scraper_capture_api,
            rate_limit_delay=self.config.scraper_rate_limit_s,
            respect_robots=self.config.scraper_respect_robots,
            exclude_patterns=self.config.scraper_exclude or [],
            # Make the crawl authenticated when possible
            storage_state_path="auth/session_state.json",
        )

    # ==================== Event Hooks ====================

    def add_hook(self, event_type: EventType, handler: EventHandler):
        self.events.subscribe(event_type, handler)
        logger.info("Added hook for %s", event_type.value)

    def remove_hook(self, event_type: EventType, handler: EventHandler):
        self.events.unsubscribe(event_type, handler)
        logger.info("Removed hook for %s", event_type.value)

    # ==================== Validation ====================

    def _validate_requirement(self, req: Dict[str, Any]) -> None:
        if not req:
            raise ValidationError("Requirement parsing returned empty result")
        details = req.get("details", {}) or {}
        url = details.get("url")
        if not url:
            raise ValidationError("No URL found in requirement details")
        if not url.startswith(("http://", "https://")):
            raise ValidationError(f"Invalid URL format: {url}")
        logger.debug("Requirement validation passed: url=%s", url)

    def _validate_scan(self, scan: Dict[str, Any]) -> None:
        if not scan:
            raise ValidationError("Scan returned empty result")
        if scan.get("errors_count", 0) > 0 and scan.get("scanned_count", 0) == 0:
            raise ValidationError("Scan failed to retrieve any pages")
        logger.debug(
            "Scan validation passed: pages=%d, apis=%d",
            scan.get("scanned_count", 0),
            scan.get("api_calls_count", 0),
        )

    def _validate_plan(self, plan: Dict[str, Any]) -> None:
        if not plan:
            raise ValidationError("Plan generation returned empty result")
        if not plan.get("suites"):
            raise ValidationError("Plan has no test suites")
        if self.config.enable_plan_validation:
            validate_plan_schema(plan, schema=None, schema_file=self.config.plan_schema_file)
        logger.debug("Plan validation passed: suites=%d", len(plan.get("suites", {})))

    # ==================== Retry Logic ====================

    def _retry_with_backoff(self, func, stage_name: str, *args, **kwargs) -> Any:
        last_error = None
        for attempt in range(1, self.config.retry_max_attempts + 1):
            try:
                logger.info("Stage '%s': attempt %d/%d", stage_name, attempt, self.config.retry_max_attempts)
                result = func(*args, **kwargs)
                if attempt > 1:
                    logger.info("Stage '%s': succeeded on attempt %d", stage_name, attempt)
                return result
            except Exception as e:
                last_error = e
                logger.warning(
                    "Stage '%s': attempt %d/%d failed: %s",
                    stage_name, attempt, self.config.retry_max_attempts, e,
                )
                if attempt < self.config.retry_max_attempts:
                    delay = self.config.retry_delay_s * (2 ** (attempt - 1))
                    logger.info("Stage '%s': retrying in %.1fs...", stage_name, delay)
                    time.sleep(delay)
        raise StageFailureError(stage_name, last_error)

    # ==================== Stage Methods ====================

    def _parse_requirement_stage(self, requirement_text: str, metrics: OrchestrationMetrics, state: OrchestrationState) -> Dict[str, Any]:
        stage = StageMetrics(stage_name="parse_requirement", start_time=time.time())
        self.events.fire(Event(EventType.STAGE_START, metrics.run_id, time.time(), {"stage": "parse_requirement"}))
        try:
            req = self._retry_with_backoff(self.agent.parse_requirement, "parse_requirement", requirement_text)
            self._validate_requirement(req)
            state.requirement = req
            state.stage = "parsed"
            if self.config.enable_state_persistence:
                state.save(self.config.state_file)
            stage.complete(success=True)
            stage.metadata = {"has_url": bool(req.get("details", {}).get("url"))}
            return req
        except Exception as e:
            stage.complete(success=False, error=str(e))
            self.events.fire(Event(EventType.ERROR_OCCURRED, metrics.run_id, time.time(), {"stage": "parse_requirement", "error": str(e)}))
            raise
        finally:
            metrics.add_stage(stage)
            self.prometheus.record_stage_complete(stage.stage_name, stage.success, stage.duration_s or 0)
            self.events.fire(Event(EventType.STAGE_COMPLETE, metrics.run_id, time.time(), {"stage": "parse_requirement", "success": stage.success}))

    def _scan_website_stage(self, target_url: str, metrics: OrchestrationMetrics, state: OrchestrationState) -> Dict[str, Any]:
        stage = StageMetrics(stage_name="scan_website", start_time=time.time())
        self.events.fire(Event(EventType.STAGE_START, metrics.run_id, time.time(), {"stage": "scan_website", "url": target_url}))
        try:
            scan = self._retry_with_backoff(self.scraper.deep_scan, "scan_website", target_url)
            self._validate_scan(scan)
            state.scan_result = scan
            state.stage = "scanned"
            if self.config.enable_state_persistence:
                state.save(self.config.state_file)
            stage.complete(success=True)
            stage.metadata = {
                "pages_scraped": scan.get("scanned_count", 0),
                "api_calls_found": scan.get("api_calls_count", 0),
                "errors": scan.get("errors_count", 0),
            }
            return scan
        except Exception as e:
            stage.complete(success=False, error=str(e))
            self.events.fire(Event(EventType.ERROR_OCCURRED, metrics.run_id, time.time(), {"stage": "scan_website", "error": str(e)}))
            raise
        finally:
            metrics.add_stage(stage)
            self.prometheus.record_stage_complete(stage.stage_name, stage.success, stage.duration_s or 0)
            self.events.fire(Event(EventType.STAGE_COMPLETE, metrics.run_id, time.time(), {"stage": "scan_website", "success": stage.success}))

    def _generate_plan_stage(self, req: Dict[str, Any], scan: Dict[str, Any], metrics: OrchestrationMetrics, state: OrchestrationState) -> Dict[str, Any]:
        stage = StageMetrics(stage_name="generate_plan", start_time=time.time())
        self.events.fire(Event(EventType.STAGE_START, metrics.run_id, time.time(), {"stage": "generate_plan"}))
        try:
            plan = self._retry_with_backoff(self.generator.generate_plan, "generate_plan", req, scan)
            plan = self._augment_plan(plan, scan, req)
            self._validate_plan(plan)
            state.plan = plan
            state.stage = "planned"
            if self.config.enable_state_persistence:
                state.save(self.config.state_file)
            stage.complete(success=True)
            stage.metadata = {
                "suites_count": len(plan.get("suites", {})),
                "has_api_suite": "api" in plan.get("suites", {}),
                "has_security_suite": "security" in plan.get("suites", {}),
            }
            return plan
        except ValidationError as e:
            stage.complete(success=False, error=str(e))
            self.events.fire(Event(EventType.VALIDATION_FAILED, metrics.run_id, time.time(), {"stage": "generate_plan", "error": str(e)}))
            raise
        except Exception as e:
            stage.complete(success=False, error=str(e))
            self.events.fire(Event(EventType.ERROR_OCCURRED, metrics.run_id, time.time(), {"stage": "generate_plan", "error": str(e)}))
            raise
        finally:
            metrics.add_stage(stage)
            self.prometheus.record_stage_complete(stage.stage_name, stage.success, stage.duration_s or 0)
            self.events.fire(Event(EventType.STAGE_COMPLETE, metrics.run_id, time.time(), {"stage": "generate_plan", "success": stage.success}))

    def _execute_tests_stage(self, plan: Dict[str, Any], metrics: OrchestrationMetrics, state: OrchestrationState) -> Dict[str, Any]:
        stage = StageMetrics(stage_name="execute_tests", start_time=time.time())
        self.events.fire(Event(EventType.STAGE_START, metrics.run_id, time.time(), {"stage": "execute_tests"}))

        if self.config.dry_run:
            logger.warning("[%s] DRY-RUN MODE: Skipping test execution", metrics.run_id)
            stage.complete(success=True)
            stage.metadata = {"dry_run": True}
            metrics.add_stage(stage)
            return {"dry_run": True, "message": "Execution skipped (dry-run mode)", "plan_valid": True}

        try:
            results = self._retry_with_backoff(self.runner.run_plan, "execute_tests", plan)
            state.execution_result = results
            state.stage = "executed"
            if self.config.enable_state_persistence:
                state.save(self.config.state_file)

            # Defensive extraction for metrics
            passed = results.get("total", {}).get("PASS") if isinstance(results.get("total"), dict) else results.get("passed", 0)
            failed = results.get("total", {}).get("FAIL") if isinstance(results.get("total"), dict) else results.get("failed", 0)
            total_tests = (passed or 0) + (failed or 0)

            stage.complete(success=True)
            stage.metadata = {"total_tests": total_tests, "passed": passed or 0, "failed": failed or 0}
            return results
        except Exception as e:
            stage.complete(success=False, error=str(e))
            self.events.fire(Event(EventType.ERROR_OCCURRED, metrics.run_id, time.time(), {"stage": "execute_tests", "error": str(e)}))
            raise
        finally:
            metrics.add_stage(stage)
            self.prometheus.record_stage_complete(stage.stage_name, stage.success, stage.duration_s or 0)
            self.events.fire(Event(EventType.STAGE_COMPLETE, metrics.run_id, time.time(), {"stage": "execute_tests", "success": stage.success}))

    # ==================== Plan Augmentation ====================

    def _augment_plan(self, plan: Dict[str, Any], scan: Dict[str, Any], requirement: Dict[str, Any]) -> Dict[str, Any]:
        """Augment plan with API discovery, security checks, and UI config."""
        plan.setdefault("plan_meta", {})
        plan.setdefault("suites", {})
        plan.setdefault("artifacts", {})

        base_url = (requirement.get("details", {}) or {}).get("url") or plan.get("project") or ""

        # 1) API Discovery Suite (robust to scraper output shape)
        if self.config.enable_api_discovery:
            api_calls: List[Dict[str, Any]] = scan.get("api_calls", []) or []
            api_suite = self._build_api_suite(api_calls, base_url)
            if api_suite:
                plan["suites"].setdefault("api", [])
                plan["suites"]["api"].append(api_suite)
                logger.info("Added API discovery suite with %d step(s)", len(api_suite.get("steps", [])))

        # 2) Security Suite (graceful fallback)
        if self.config.enable_security_checks and self.security:
            try:
                sec_suite = self.security.plan_from_base_url(base_url)
                if sec_suite:
                    plan["suites"].setdefault("security", [])
                    plan["suites"]["security"].append(sec_suite)
                    logger.info("Added security suite")
            except Exception as e:
                logger.warning("Failed to generate security suite (non-fatal): %s", e)

        # 3) UI Configuration
        plan["ui_config"] = {
            "browsers": self.config.ui_browsers,
            "headless": self.config.ui_headless,
            "test_timeout_ms": self.config.ui_test_timeout_ms,
            "expect_timeout_ms": self.config.ui_expect_timeout_ms,
            "workers": self.config.ui_workers,
            "retries": self.config.ui_retries,
            "fully_parallel": self.config.ui_fully_parallel,
            "forbid_only": self.config.ui_forbid_only,
            "enable_junit": self.config.ui_enable_junit,
        }

        # 4) Metadata
        plan["artifacts"].setdefault("ui_workspace", None)
        plan["plan_meta"]["orchestrated_at"] = datetime.utcnow().isoformat() + "Z"
        plan["plan_meta"]["scan_stats"] = {
            "pages": scan.get("scanned_count", 0),
            "api_calls": scan.get("api_calls_count", 0),
            "errors": scan.get("errors_count", 0),
        }

        return plan

    def _build_api_suite(self, api_calls: List[Dict[str, Any]], base_url: str) -> Optional[Dict[str, Any]]:
        if not api_calls:
            return None

        seen = set()
        steps = []

        for call in api_calls[: self.config.api_discovery_limit]:
            method = (call.get("method") or "GET").upper()
            url = call.get("url")
            if not url:
                continue

            key = f"{method}:{url}"
            if key in seen:
                continue
            seen.add(key)

            status = call.get("status")
            ctype = (call.get("content_type") or "").lower()

            if status is not None:
                if 200 <= int(status) < 300:
                    step = f"{method} {url} expect {status}"
                elif 300 <= int(status) < 400:
                    step = f"{method} {url} expect 3xx"
                else:
                    step = f"{method} {url} expect {status}"
            else:
                # no status recorded → generic health
                if "json" in ctype:
                    step = f"{method} {url} expect 2xx application/json"
                else:
                    step = f"{method} {url} expect 2xx"

            steps.append(step)

        if not steps:
            return None

        return {
            "name": "Discovered API health checks",
            "description": f"Auto-discovered {len(steps)} API endpoints",
            "steps": steps,
            "priority": "P1",
            "tags": ["discovery", "health", "api"],
        }


    # ==================== Main Entry Point ====================

    def run(self, requirement_text: str, run_id: Optional[str] = None) -> Dict[str, Any]:
        """Execute complete orchestration workflow."""
        run_id = run_id or f"orch_{uuid.uuid4().hex[:8]}"
        metrics = OrchestrationMetrics(run_id=run_id, start_time=time.time(), requirement_text=requirement_text)
        state = OrchestrationState(run_id=run_id, stage="init", timestamp=datetime.utcnow().isoformat() + "Z")

        logger.info("=" * 60)
        logger.info("[%s] Starting orchestration", run_id)
        if self.config.dry_run:
            logger.warning("[%s] DRY-RUN MODE ENABLED", run_id)
        logger.info("=" * 60)

        self.events.fire(Event(EventType.ORCHESTRATION_START, run_id, time.time(), {"requirement_text": requirement_text}))
        self.prometheus.record_orchestration_start()

        try:
            req = self._parse_requirement_stage(requirement_text, metrics, state)
            target_url = (req.get("details", {}) or {}).get("url")
            metrics.target_url = target_url

            scan = self._scan_website_stage(target_url, metrics, state)
            plan = self._generate_plan_stage(req, scan, metrics, state)
            results = self._execute_tests_stage(plan, metrics, state)

            metrics.complete(success=True)
            logger.info("=" * 60)
            logger.info("[%s] Orchestration completed successfully", run_id)
            logger.info("=" * 60)

            self.events.fire(Event(EventType.ORCHESTRATION_COMPLETE, run_id, time.time(), {"success": True}))
            self.prometheus.record_orchestration_complete(True, metrics.total_duration_s or 0)

            if self.config.enable_jsonl_export:
                metrics_file = Path(self.config.metrics_export_dir) / f"metrics_{run_id}.jsonl"
                metrics.export_jsonl(str(metrics_file))

            return {"run_id": run_id, "success": True, "results": results, "metrics": metrics.to_dict(), "plan": plan}

        except Exception as e:
            metrics.complete(success=False)
            logger.error("=" * 60)
            logger.error("[%s] Orchestration failed: %s", run_id, e, exc_info=True)
            logger.error("=" * 60)

            self.events.fire(Event(EventType.ORCHESTRATION_COMPLETE, run_id, time.time(), {"success": False, "error": str(e)}))
            self.prometheus.record_orchestration_complete(False, metrics.total_duration_s or 0)

            if self.config.enable_jsonl_export:
                metrics_file = Path(self.config.metrics_export_dir) / f"metrics_{run_id}.jsonl"
                metrics.export_jsonl(str(metrics_file))

            return {
                "run_id": run_id,
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__,
                "metrics": metrics.to_dict(),
                "partial_state": asdict(state),
            }

    # ==================== State Recovery ====================

    def resume(self, state_file: Optional[str] = None) -> Dict[str, Any]:
        """Resume orchestration from saved state."""
        state_file = state_file or self.config.state_file
        state = OrchestrationState.load(state_file)
        if not state:
            raise ResumeError(f"No state file found at {state_file}")

        logger.info("[%s] Resuming orchestration from stage: %s", state.run_id, state.stage)
        metrics = OrchestrationMetrics(run_id=state.run_id, start_time=time.time())

        try:
            if state.stage in ("init", "parsed"):
                if not state.requirement:
                    raise ResumeError("Cannot resume: no requirement in state")
                req = state.requirement
                target_url = (req.get("details", {}) or {}).get("url")
                scan = self._scan_website_stage(target_url, metrics, state)
                plan = self._generate_plan_stage(req, scan, metrics, state)
                results = self._execute_tests_stage(plan, metrics, state)
            elif state.stage == "scanned":
                if not state.requirement or not state.scan_result:
                    raise ResumeError("Cannot resume: missing requirement or scan result")
                plan = self._generate_plan_stage(state.requirement, state.scan_result, metrics, state)
                results = self._execute_tests_stage(plan, metrics, state)
            elif state.stage == "planned":
                if not state.plan:
                    raise ResumeError("Cannot resume: no plan in state")
                results = self._execute_tests_stage(state.plan, metrics, state)
            elif state.stage == "executed":
                logger.info("[%s] Orchestration already completed", state.run_id)
                return {"run_id": state.run_id, "success": True, "resumed": True, "results": state.execution_result, "message": "Orchestration was already complete"}
            else:
                raise ResumeError(f"Unknown stage: {state.stage}")

            metrics.complete(success=True)
            logger.info("[%s] Resume completed successfully", state.run_id)
            return {"run_id": state.run_id, "success": True, "resumed": True, "results": results, "metrics": metrics.to_dict()}
        except Exception as e:
            metrics.complete(success=False)
            logger.error("[%s] Resume failed: %s", state.run_id, e, exc_info=True)
            return {"run_id": state.run_id, "success": False, "resumed": True, "error": str(e), "error_type": type(e).__name__, "metrics": metrics.to_dict()}


# ==================== Example Usage ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    def log_stage_completion(event: Event):
        print(f"📊 [Hook] Stage completed: {event.data.get('stage')}, success={event.data.get('success')}")

    def alert_on_error(event: Event):
        print(f"🚨 [Hook] Error in {event.data.get('stage')}: {event.data.get('error')}")

    config = OrchestratorConfig(scraper_max_pages=20, dry_run=False, enable_prometheus_metrics=False, enable_jsonl_export=True)
    orchestrator = SpecOrchestrator(config=config)
    orchestrator.add_hook(EventType.STAGE_COMPLETE, log_stage_completion)
    orchestrator.add_hook(EventType.ERROR_OCCURRED, alert_on_error)

    requirement = "Test the login functionality of https://example.com"
    result = orchestrator.run(requirement)
    print("\n" + "=" * 60)
    print("ORCHESTRATION RESULT")
    print("=" * 60)
    print(f"Run ID: {result['run_id']}")
    print(f"Success: {result['success']}")
    print(f"Duration: {result.get('metrics', {}).get('total_duration_s')}s")
    print("=" * 60 + "\n")
