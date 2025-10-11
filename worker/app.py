# worker/app.py
import os
import uuid
import json
import logging
import threading
import traceback
import datetime
import subprocess
import signal
import sys
from enum import Enum
from typing import Dict, Any, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, Future

from fastapi import FastAPI, HTTPException, Header, Request, status
from fastapi.responses import JSONResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware
from pydantic import BaseModel, Field, field_validator

from worker.config import Settings

# Repo modules (ensure PYTHONPATH points to repo root)
from modules.ui_framework_generator import UIFrameworkGenerator
from modules.auth_manager import AuthManager
from modules.api_test_engine import APITestEngine
from modules.performance_engine import PerformanceEngine

# ------------------- Settings & logging -------------------
settings = Settings()
logging.basicConfig(level=settings.log_level)
logger = logging.getLogger("worker")

# Optional Prometheus metrics
try:
    from prometheus_client import (
        Counter, Gauge, Histogram, start_http_server,
        generate_latest, CONTENT_TYPE_LATEST
    )
    METRICS_ENABLED = bool(settings.enable_metrics)
except Exception:
    METRICS_ENABLED = False

API_KEY = settings.api_key                      # optional auth
DEFAULT_TIMEOUT = settings.default_timeout      # seconds
MAX_THREADS = settings.max_threads
BASE_REPORTS_DIR = settings.reports_dir

app = FastAPI(title="QA Worker Agent", version="1.1.0")

# ---------------- In-memory job store (demo) ----------------
_jobs_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}
_executor: Optional[ThreadPoolExecutor] = None  # created on startup

# Simple idempotency cache: Idempotency-Key -> job_id
_idem_lock = threading.Lock()
_idem: Dict[str, str] = {}

# ---------------- Middleware ----------------
class RequestIdMiddleware(BaseHTTPMiddleware):
    """Inject a request-id for easier tracing in logs."""
    async def dispatch(self, request: Request, call_next):
        rid = request.headers.get("X-Request-ID") or str(uuid.uuid4())
        request.state.request_id = rid
        logger.info("req start rid=%s %s %s", rid, request.method, request.url.path)
        resp = await call_next(request)
        resp.headers["X-Request-ID"] = rid
        logger.info("req end   rid=%s status=%s", rid, resp.status_code)
        return resp

class BodyLimitMiddleware(BaseHTTPMiddleware):
    """Reject overly large payloads to protect the worker."""
    def __init__(self, app, max_bytes: int):
        super().__init__(app)
        self.max_bytes = max_bytes
    async def dispatch(self, request: Request, call_next):
        body = await request.body()
        if len(body) > self.max_bytes:
            return JSONResponse({"detail": "payload too large"}, status_code=413)
        # Re-inject body so FastAPI can parse it
        request._body = body  # type: ignore[attr-defined]
        return await call_next(request)

app.add_middleware(RequestIdMiddleware)
app.add_middleware(BodyLimitMiddleware, max_bytes=settings.max_body_bytes)

# ---------------- Metrics ----------------
if METRICS_ENABLED:
    start_http_server(settings.metrics_port)
    JOBS_TOTAL = Counter("worker_jobs_total", "Jobs created", ["stage"])
    JOBS_RUNNING = Gauge("worker_jobs_running", "Jobs running")
    JOB_DURATION = Histogram("worker_job_duration_seconds", "Job duration", ["stage"])
else:
    JOBS_TOTAL = JOBS_RUNNING = JOB_DURATION = None  # type: ignore

# Optionally expose /metrics from the app as well (handy behind a single ingress)
@app.get("/metrics")
def metrics_endpoint():
    if not METRICS_ENABLED:
        raise HTTPException(status_code=404, detail="metrics disabled")
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

# ---------------- Models & Validation ----------------
class Stage(str, Enum):
    UI = "UI"
    API = "API"
    PERF = "PERF"
    PERFORMANCE = "PERFORMANCE"  # alias for PERF

class RunRequest(BaseModel):
    stage: Stage = Field(..., description="Stage to run: UI | API | PERF")
    plan: Dict[str, Any] = Field(..., description="Plan dict consumed by engines")
    execution_id: Optional[str] = Field(None, description="Optional external execution id")
    reports_dir: Optional[str] = Field(None, description="Optional subdir for artifacts")
    timeout_sec: Optional[int] = Field(None, ge=1, le=86400)

    @field_validator("plan")
    @classmethod
    def _validate_plan(cls, v: Dict[str, Any]) -> Dict[str, Any]:
        if not isinstance(v, dict):
            raise ValueError("plan must be an object")
        suites = v.get("suites")
        if suites is None or not isinstance(suites, dict):
            raise ValueError("plan.suites must be a dict")
        for k in ("ui", "api", "performance"):
            if k in suites and not isinstance(suites[k], list):
                raise ValueError(f"plan.suites.{k} must be a list")
        return v

class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    created_at: str
    started_at: Optional[str] = None
    finished_at: Optional[str] = None

class JobResultResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None

# ---------------- Helpers ----------------
def _utc_now() -> str:
    return datetime.datetime.utcnow().isoformat() + "Z"

def _require_api_key(key: Optional[str]) -> None:
    if API_KEY and key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid X-API-KEY")

def _resolve_reports_dir(user_dir: Optional[str]) -> str:
    """
    Ensure artifacts land under RUNNER_REPORTS_DIR and avoid path traversal.
    """
    base = os.path.abspath(BASE_REPORTS_DIR)
    os.makedirs(base, exist_ok=True)
    if not user_dir:
        return base
    sub = os.path.abspath(os.path.join(base, user_dir))
    if not sub.startswith(base):
        logger.warning("reports_dir %s escaped base %s; using base", sub, base)
        return base
    os.makedirs(sub, exist_ok=True)
    return sub

def _mk_job(stage: Stage, payload: RunRequest) -> str:
    job_id = str(uuid.uuid4())
    with _jobs_lock:
        _jobs[job_id] = {
            "stage": stage.value,
            "status": "queued",
            "created_at": _utc_now(),
            "payload": payload.model_dump(),  # Pydantic v2
            "result": None,
            "error": None,
            "started_at": None,
            "finished_at": None,
            "future": None,
            "stop_event": threading.Event(),  # cancel propagation
        }
    if METRICS_ENABLED:
        JOBS_TOTAL.labels(stage=stage.value).inc()  # type: ignore[union-attr]
    return job_id

def _update_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(kwargs)

def _auth_like_failure(stdout: str, stderr: str) -> bool:
    text = ((stdout or "") + "\n" + (stderr or "")).lower()
    patterns = [
        "401", "unauthorized", "authentication", "not authenticated",
        "login required", "session expired", "invalid credentials",
        "403", "access denied", "please login", "sign in"
    ]
    return any(p in text for p in patterns)

# ---------------- Subprocess runner (UI Playwright) ----------------
def _run_playwright_with_cancel(
    workspace: str,
    project: str,
    timeout: int,
    stop_event: threading.Event,
) -> Tuple[int, str, str]:
    """
    Run Playwright via Popen with:
      - hard timeout
      - cooperative cancel (kill on stop_event)
      - stdout/stderr capture
      - process group isolation (POSIX) for clean termination
    """
    cmd = ["npx", "playwright", "test", "--reporter", "list", f"--project={project}"]
    proc: Optional[subprocess.Popen] = None
    creationflags = 0
    preexec_fn = None

    # Create new process group/session for safer termination
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0x00000200)
    else:  # POSIX
        preexec_fn = os.setsid  # create a new session

    def _terminate_group():
        if proc and proc.poll() is None:
            try:
                if os.name == "nt":
                    # Best-effort: terminate process (group signals are limited on Windows)
                    proc.terminate()
                else:
                    # Kill the whole group
                    os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            except Exception:
                logger.debug("first terminate failed", exc_info=True)
            # Hard kill fallback
            try:
                if os.name == "nt":
                    proc.kill()
                else:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except Exception:
                logger.debug("kill fallback failed", exc_info=True)

    def _watch_cancel():
        stop_event.wait()
        _terminate_group()

    try:
        proc = subprocess.Popen(
            cmd,
            cwd=workspace,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=preexec_fn,
            creationflags=creationflags,
        )
        threading.Thread(target=_watch_cancel, daemon=True).start()
        stdout, stderr = proc.communicate(timeout=timeout)
        return proc.returncode or 0, stdout or "", stderr or ""
    except subprocess.TimeoutExpired:
        _terminate_group()
        try:
            stdout, stderr = proc.communicate(timeout=5) if proc else ("", "timeout")
        except Exception:
            stdout, stderr = "", "timeout"
        return 124, stdout, stderr
    except Exception as e:
        return 1, "", f"playwright error: {e}"

# ---------------- Core worker logic ----------------
def _run_stage_logic(job_id: str) -> None:
    """
    Pull job from store, perform the stage and persist result.
    WARNING: worker must run in the same environment as repo modules.
    """
    start_ts = datetime.datetime.utcnow()
    with _jobs_lock:
        job = _jobs[job_id]
        payload = RunRequest(**job["payload"])
        job["status"] = "running"
        job["started_at"] = _utc_now()
        stop_event: threading.Event = job["stop_event"]

    stage = payload.stage.value.upper()
    if METRICS_ENABLED:
        JOBS_RUNNING.inc()  # type: ignore[union-attr]

    try:
        reports_dir = _resolve_reports_dir(payload.reports_dir)
        execution_id = payload.execution_id or f"job-{job_id[:8]}"
        timeout = payload.timeout_sec or DEFAULT_TIMEOUT

        result: Any = None

        if stage == "UI":
            generator = UIFrameworkGenerator(payload.plan)
            workspace = generator.generate()
            if not workspace:
                result = {"error": "framework_generation_failed", "summary": {"passed": False}}
            else:
                raw_out = os.path.join(reports_dir, f"{execution_id}_ui_raw_output.txt")

                # First attempt
                rc1, out1, err1 = _run_playwright_with_cancel(
                    workspace=workspace, project="chromium", timeout=timeout, stop_event=stop_event,
                )
                # Persist logs (best-effort)
                try:
                    with open(raw_out, "w", encoding="utf-8") as fh:
                        fh.write("=== RUN STDOUT ===\n" + (out1 or "") +
                                 "\n\n=== RUN STDERR ===\n" + (err1 or ""))
                except Exception:
                    logger.exception("failed to write raw log")

                attempted_relogin = False
                final_rc = rc1

                # Optional auth retry
                if _auth_like_failure(out1, err1) and not stop_event.is_set():
                    attempted_relogin = True
                    try:
                        AuthManager().login_and_save_session(force=True)
                    except Exception:
                        logger.exception("auth manager login failed")
                    rc2, out2, err2 = _run_playwright_with_cancel(
                        workspace=workspace, project="chromium", timeout=timeout, stop_event=stop_event,
                    )
                    final_rc = rc2
                    try:
                        with open(raw_out, "a", encoding="utf-8") as fh:
                            fh.write("\n\n=== SECOND RUN STDOUT ===\n" + (out2 or "") +
                                     "\n\n=== SECOND RUN STDERR ===\n" + (err2 or ""))
                    except Exception:
                        logger.exception("failed to append raw log")

                # Prefer Playwright JSON if present
                pw_path = os.path.join(workspace, "reports", "playwright", "report.json")
                if os.path.exists(pw_path):
                    try:
                        with open(pw_path, "r", encoding="utf-8") as fh:
                            pj = json.load(fh)
                        result = {"playwright_report": pj}
                    except Exception:
                        result = {"error": "failed_to_load_playwright_report"}
                else:
                    result = {"cases": [], "note": "no_playwright_json_present_fallback_to_step_logs"}

                result["execution_meta"] = {"return_code": final_rc, "attempted_relogin": attempted_relogin}
                result["summary"] = {"passed": (final_rc == 0)}
                result["artifacts"] = {
                    "raw_log": raw_out,
                    "playwright_report": pw_path if os.path.exists(pw_path) else None
                }

        elif stage == "API":
            suites = payload.plan.get("suites", {}).get("api", []) or []
            engine = APITestEngine(payload.plan.get("project", ""))
            result = [engine.run_suite(s) for s in suites]

        elif stage in ("PERF", "PERFORMANCE"):
            suites = payload.plan.get("suites", {}).get("performance", []) or []
            engine = PerformanceEngine(payload.plan.get("project", ""), workspace=reports_dir)
            # TODO: teach PerformanceEngine to accept a stop_event for graceful cancel
            result = engine.run_all({"suites": suites}) if hasattr(engine, "run_all") else [engine.run_suite(s) for s in suites]

        else:
            raise RuntimeError(f"Unknown stage: {stage!r}")

        _update_job(job_id, status="done", result=result, finished_at=_utc_now())

    except Exception as exc:
        tb = traceback.format_exc()
        logger.exception("Job %s failed", job_id)
        _update_job(
            job_id,
            status="failed",
            error=str(exc),
            result={"error": str(exc), "trace": tb},
            finished_at=_utc_now()
        )
    finally:
        if METRICS_ENABLED:
            dur = (datetime.datetime.utcnow() - start_ts).total_seconds()
            JOB_DURATION.labels(stage=stage).observe(dur)  # type: ignore[union-attr]
            JOBS_RUNNING.dec()  # type: ignore[union-attr]

# ---------------- Health ----------------
@app.get("/health/live")
def live():
    return {"status": "ok", "time": _utc_now()}

@app.get("/health/ready")
def ready():
    ok = True
    # writeability check
    try:
        os.makedirs(BASE_REPORTS_DIR, exist_ok=True)
        probe = os.path.join(BASE_REPORTS_DIR, ".probe")
        open(probe, "w").close()
        os.remove(probe)
    except Exception:
        ok = False
    # presence of Node Playwright (optional but useful if you run UI here)
    try:
        subprocess.run(["npx", "playwright", "--version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=False, timeout=5)
    except Exception:
        ok = False
    return {"status": "ok" if ok else "not_ready"}

# ---------------- API endpoints (versioned) ----------------
@app.on_event("startup")
def _startup():
    global _executor
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=MAX_THREADS)
    os.makedirs(BASE_REPORTS_DIR, exist_ok=True)
    logger.info("Worker started (threads=%s, reports_dir=%s)", MAX_THREADS, BASE_REPORTS_DIR)

@app.on_event("shutdown")
def _shutdown():
    global _executor
    if _executor:
        _executor.shutdown(cancel_futures=True)
        _executor = None
    logger.info("Worker stopped")

@app.post("/v1/run", status_code=status.HTTP_202_ACCEPTED)
def run_endpoint(
    req: RunRequest,
    x_api_key: Optional[str] = Header(None),
    x_idempotency_key: Optional[str] = Header(None),
):
    _require_api_key(x_api_key)
    if _executor is None:
        raise HTTPException(status_code=503, detail="Executor not ready")

    # Idempotency (return the original job for the same key)
    if x_idempotency_key:
        with _idem_lock:
            if x_idempotency_key in _idem:
                jid = _idem[x_idempotency_key]
                with _jobs_lock:
                    status_ = _jobs.get(jid, {}).get("status", "unknown")
                return JSONResponse(
                    status_code=status.HTTP_202_ACCEPTED,
                    content={"job_id": jid, "status": status_},
                    headers={"Location": f"/v1/status/{jid}"}
                )

    job_id = _mk_job(req.stage, req)
    future: Future = _executor.submit(_run_stage_logic, job_id)
    with _jobs_lock:
        _jobs[job_id]["future"] = future

    if x_idempotency_key:
        with _idem_lock:
            _idem[x_idempotency_key] = job_id

    return JSONResponse(
        status_code=status.HTTP_202_ACCEPTED,
        content={"job_id": job_id, "status": "queued"},
        headers={"Location": f"/v1/status/{job_id}"}
    )

@app.get("/v1/status/{job_id}", response_model=JobStatusResponse)
def status_endpoint(job_id: str, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        return JobStatusResponse(
            job_id=job_id,
            status=job["status"],
            created_at=job["created_at"],
            started_at=job.get("started_at"),
            finished_at=job.get("finished_at"),
        )

@app.get("/v1/result/{job_id}", response_model=JobResultResponse)
def result_endpoint(job_id: str, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        if job["status"] in ("done", "failed", "cancelled"):
            return JobResultResponse(
                job_id=job_id,
                status=job["status"],
                result=job["result"],
                error=job.get("error"),
            )
        return JobResultResponse(job_id=job_id, status=job["status"], result=None, error=None)

@app.post("/v1/cancel/{job_id}")
def cancel_endpoint(job_id: str, x_api_key: Optional[str] = Header(None)):
    _require_api_key(x_api_key)
    with _jobs_lock:
        job = _jobs.get(job_id)
        if not job:
            raise HTTPException(status_code=404, detail="job not found")
        job["status"] = "cancelled"
        job["finished_at"] = _utc_now()
        stop: threading.Event = job.get("stop_event")  # type: ignore
        if stop:
            stop.set()  # triggers subprocess termination in UI stage
        future: Optional[Future] = job.get("future")
        if future and not future.done():
            future.cancel()  # cancels Python task; subprocess handled via stop_event
    return {"job_id": job_id, "status": "cancelled"}
