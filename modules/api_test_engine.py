# modules/api_test_engine.py
"""
API Test Engine (Phase 6.2 — async httpx, circuit breaker, progress, summary)
-----------------------------------------------------------------------------
Executes API suites from the plan using httpx.AsyncClient.

Key features
- Async httpx client with connection pooling
- Per-request timeout + retry/backoff (network & 5xx)
- Base URL + default headers via config/env (config/api_config.json)
- Step templating with {{var}} (headers/body/url/params)
- Extractors: dotted JSON paths, headers, $status → saved to context
- Assertions: status, json_path_equals, json_contains, header_equals, header_contains, response_time_ms_lte
- Structured per-step logs: reports/step_logs_api/<step_id>.json
- Returns a JUnit-friendly summary dict: total/passed/failed/skipped/cases/duration_s
- Circuit breaker: consecutive failures open the circuit for a cooldown (configurable)
- Progress callbacks + graceful stop

Expected plan shape:
suites.api = [
  {
    "name": "Health",
    "steps": [
      {"request": {"method":"GET","path":"/health"}, "expect": {"status":200}},
      # or strings via fallback: "GET /health", "expect status 200"
    ]
  }
]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union, Callable

import httpx

logger = logging.getLogger(__name__)

# -------- Paths / constants --------
REPORTS_DIR = Path(os.environ.get("REPORTS_DIR", "reports"))
STEP_LOG_DIR = REPORTS_DIR / "step_logs_api"
STEP_LOG_DIR.mkdir(parents=True, exist_ok=True)

# -------- Config via env / file --------
DEFAULT_API_CONFIG_PATH = os.environ.get("API_CONFIG_PATH", "config/api_config.json")

_SENSITIVE_KEYS = {
    "authorization", "x-api-key", "api_key", "apikey", "token",
    "access_token", "cookie", "x-auth-token", "x-access-token"
}

def _load_api_config() -> Dict[str, Any]:
    cfg: Dict[str, Any] = {}
    p = Path(DEFAULT_API_CONFIG_PATH)
    if p.exists():
        try:
            with p.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict):
                cfg.update(data)
        except Exception as e:
            logger.warning("API config read error (%s): %s", p, e, exc_info=True)

    # env overrides
    if os.environ.get("API_BASE_URL"):
        cfg["base_url"] = os.environ["API_BASE_URL"]
    if os.environ.get("API_DEFAULT_HEADERS"):
        try:
            cfg["default_headers"] = json.loads(os.environ["API_DEFAULT_HEADERS"])
        except Exception:
            logger.warning("API_DEFAULT_HEADERS not valid JSON; ignoring")
    if os.environ.get("API_TIMEOUT_SEC"):
        try:
            cfg["timeout_sec"] = float(os.environ["API_TIMEOUT_SEC"])
        except Exception:
            logger.warning("API_TIMEOUT_SEC invalid; ignoring")
    if os.environ.get("API_RETRIES"):
        try:
            cfg["retries"] = int(os.environ["API_RETRIES"])
        except Exception:
            logger.warning("API_RETRIES invalid; ignoring")
    if os.environ.get("API_BACKOFF_BASE_SEC"):
        try:
            cfg["backoff_base_sec"] = float(os.environ["API_BACKOFF_BASE_SEC"])
        except Exception:
            logger.warning("API_BACKOFF_BASE_SEC invalid; ignoring")
    if os.environ.get("API_MAX_CONCURRENCY"):
        try:
            cfg["max_concurrency"] = int(os.environ["API_MAX_CONCURRENCY"])
        except Exception:
            logger.warning("API_MAX_CONCURRENCY invalid; ignoring")
    if os.environ.get("API_VERIFY_SSL"):
        cfg["verify_ssl"] = os.environ["API_VERIFY_SSL"].strip().lower() in {"1", "true", "yes"}

    # Circuit breaker
    if os.environ.get("API_CB_THRESHOLD"):
        try:
            cfg["cb_threshold"] = int(os.environ["API_CB_THRESHOLD"])
        except Exception:
            logger.warning("API_CB_THRESHOLD invalid; ignoring")
    if os.environ.get("API_CB_COOLDOWN_SEC"):
        try:
            cfg["cb_cooldown_sec"] = float(os.environ["API_CB_COOLDOWN_SEC"])
        except Exception:
            logger.warning("API_CB_COOLDOWN_SEC invalid; ignoring")

    # defaults
    cfg.setdefault("base_url", "")
    cfg.setdefault("default_headers", {})
    cfg.setdefault("timeout_sec", 30.0)
    cfg.setdefault("retries", 1)
    cfg.setdefault("backoff_base_sec", 0.5)
    cfg.setdefault("max_concurrency", 8)
    cfg.setdefault("verify_ssl", True)
    cfg.setdefault("cb_threshold", 10)        # consecutive step fails
    cfg.setdefault("cb_cooldown_sec", 20.0)   # cooldown when open
    # optional step assertion default for perf (can be overridden per step)
    cfg.setdefault("default_response_time_ms_lte", None)
    return cfg

# -------- Utilities --------
def _log_step_json(step_id: str, status: str, message: str, extra: Optional[Dict] = None):
    payload = {
        "step_id": step_id,
        "status": status,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if extra:
        payload.update(extra)
    out = STEP_LOG_DIR / f"{step_id}.json"
    try:
        with out.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
    except Exception:
        logger.debug("Failed writing step log %s", out, exc_info=True)

def _extract_dotted(obj: Any, dotted: str) -> Any:
    parts = [p for p in dotted.split(".") if p]
    cur = obj
    for p in parts:
        if isinstance(cur, list):
            try:
                idx = int(p)
            except Exception:
                return None
            if idx < 0 or idx >= len(cur):
                return None
            cur = cur[idx]
        elif isinstance(cur, dict):
            if p not in cur:
                return None
            cur = cur[p]
        else:
            return None
    return cur

def _render_template(value: Any, ctx: Dict[str, Any]) -> Any:
    if value is None:
        return None
    if isinstance(value, str):
        def replace(match: re.Match) -> str:
            k = match.group(1)
            v = ctx.get(k)
            return str(v) if v is not None else match.group(0)
        return re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", replace, value)
    if isinstance(value, dict):
        return {k: _render_template(v, ctx) for k, v in value.items()}
    if isinstance(value, list):
        return [_render_template(v, ctx) for v in value]
    return value

def _redact_headers(h: Optional[Dict[str, str]]) -> Dict[str, str]:
    h = h or {}
    out = {}
    for k, v in h.items():
        out[k] = "[REDACTED]" if k.lower() in _SENSITIVE_KEYS else v
    return out

# -------- Data types --------
@dataclass
class APIRequest:
    method: str
    url: str  # absolute or path; if path, joined with base_url by httpx
    headers: Dict[str, str] = field(default_factory=dict)
    json_body: Any = None
    data: Any = None
    params: Dict[str, Any] = field(default_factory=dict)

@dataclass
class APIExpect:
    status: Optional[int] = None
    json_path_equals: Dict[str, Any] = field(default_factory=dict)  # {"data.token": "abc"}
    json_contains: Dict[str, Any] = field(default_factory=dict)     # shallow subset
    header_equals: Dict[str, str] = field(default_factory=dict)     # case-insensitive
    header_contains: Dict[str, str] = field(default_factory=dict)   # substring match
    response_time_ms_lte: Optional[int] = None                      # performance assertion

@dataclass
class APIStep:
    request: Optional[APIRequest] = None
    expect: Optional[APIExpect] = None
    save: Dict[str, str] = field(default_factory=dict)  # {"token": "data.token"}
    name: Optional[str] = None  # optional friendly name

# -------- Heuristic string parsing (fallback) --------
_REQ_RE = re.compile(r"^\s*(GET|POST|PUT|PATCH|DELETE|HEAD|OPTIONS)\s+(\S+)\s*$", re.I)
_EXP_STATUS_RE = re.compile(r"^\s*expect\s+status\s+(\d{3})\s*$", re.I)

def _parse_step_any(step: Union[str, Dict[str, Any]]) -> APIStep:
    if isinstance(step, dict):
        req = step.get("request")
        exp = step.get("expect")
        save = step.get("save") or {}
        name = step.get("name")
        api_req = None
        api_exp = None
        if isinstance(req, dict):
            api_req = APIRequest(
                method=str(req.get("method") or "GET").upper(),
                url=str(req.get("url") or req.get("path") or ""),
                headers=dict(req.get("headers") or {}),
                json_body=req.get("json"),
                data=req.get("data"),
                params=dict(req.get("params") or {}),
            )
        if isinstance(exp, dict):
            api_exp = APIExpect(
                status=exp.get("status"),
                json_path_equals=dict(exp.get("json_path_equals") or {}),
                json_contains=dict(exp.get("json_contains") or {}),
                header_equals={k.lower(): v for k, v in (exp.get("header_equals") or {}).items()},
                header_contains={k.lower(): v for k, v in (exp.get("header_contains") or {}).items()},
                response_time_ms_lte=exp.get("response_time_ms_lte"),
            )
        return APIStep(request=api_req, expect=api_exp, save=save, name=name)
    s = str(step)
    m = _REQ_RE.match(s)
    if m:
        return APIStep(request=APIRequest(method=m.group(1).upper(), url=m.group(2)))
    m = _EXP_STATUS_RE.match(s)
    if m:
        return APIStep(expect=APIExpect(status=int(m.group(1))))
    return APIStep(name=s)

# -------- Assertions --------
def _assert_response(resp: httpx.Response, exp: APIExpect, elapsed_ms: Optional[int]) -> Tuple[bool, str]:
    if exp.status is not None and resp.status_code != int(exp.status):
        return False, f"status {resp.status_code} != {exp.status}"

    # headers
    if exp.header_equals:
        for k, v in exp.header_equals.items():
            if resp.headers.get(k) != v:
                return False, f"header {k} != {v!r} (got {resp.headers.get(k)!r})"
    if exp.header_contains:
        for k, needle in exp.header_contains.items():
            val = resp.headers.get(k)
            if val is None or str(needle) not in val:
                return False, f"header {k} missing substring {needle!r} (got {val!r})"

    # performance
    if exp.response_time_ms_lte is not None and elapsed_ms is not None:
        if elapsed_ms > int(exp.response_time_ms_lte):
            return False, f"slow response: {elapsed_ms}ms > {exp.response_time_ms_lte}ms"

    # json checks
    if exp.json_path_equals or exp.json_contains:
        try:
            data = resp.json()
        except Exception:
            return False, "response is not JSON"

        for jp, expected in exp.json_path_equals.items():
            actual = _extract_dotted(data, jp)
            if actual != expected:
                return False, f"json_path_equals {jp}: {actual!r} != {expected!r}"

        for k, v in exp.json_contains.items():
            if not isinstance(data, dict) or k not in data or data[k] != v:
                return False, f"json_contains mismatch at key={k!r}"
    return True, "OK"

# -------- Engine --------
class APITestEngine:
    def __init__(
        self,
        reports_dir: str = "reports",
        progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.reports_dir = Path(reports_dir)
        self.cfg = _load_api_config()
        self._progress_cb = progress_cb
        self._stop = False
        # circuit breaker state
        self._cb_failures = 0
        self._cb_open_until: Optional[float] = None

    # ---- public control ----
    def stop(self) -> None:
        """Ask the engine to stop after current step."""
        self._stop = True

    # ---- public API (sync wrapper) ----
    def run_suites(self, suites: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Sync wrapper returning a summary dict. Prefer run_suites_async from async contexts.
        """
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        if loop and loop.is_running():
            raise RuntimeError("run_suites() called inside a running event loop; use await run_suites_async()")

        return asyncio.run(self.run_suites_async(suites))

    # ---- public API (async) ----
    async def run_suites_async(self, suites: List[Dict[str, Any]]) -> Dict[str, Any]:
        start = time.perf_counter()
        timeout = httpx.Timeout(self.cfg["timeout_sec"])
        limits = httpx.Limits(max_connections=self.cfg["max_concurrency"])
        async with httpx.AsyncClient(
            base_url=self.cfg["base_url"],
            headers=self.cfg["default_headers"],
            timeout=timeout,
            limits=limits,
            verify=self.cfg["verify_ssl"],
            follow_redirects=True,
        ) as client:
            results: List[Dict[str, Any]] = []
            total = passed = failed = skipped = 0

            self._emit("run_start", suites=len(suites or []))

            for case in suites or []:
                if self._should_pause_on_circuit():
                    self._emit("circuit_open", until=self._cb_open_until)
                    await self._sleep_open_circuit()
                if self._stop:
                    break

                case_name = str(case.get("name") or "API Case").strip()
                steps_raw = case.get("steps") or []
                steps_parsed = [_parse_step_any(s) for s in steps_raw]

                self._emit("case_start", name=case_name, steps=len(steps_parsed))

                case_res = await self._run_case(client, case_name, steps_parsed)
                results.append(case_res)

                # accumulate counts
                c_total = len(case_res["steps"])
                c_failed = sum(1 for s in case_res["steps"] if s["status"] == "FAIL")
                c_skipped = sum(1 for s in case_res["steps"] if s["status"] == "SKIPPED")
                c_passed = c_total - c_failed - c_skipped

                total += c_total
                failed += c_failed
                passed += c_passed
                skipped += c_skipped

                self._emit("case_done", name=case_name, total=c_total, passed=c_passed, failed=c_failed, skipped=c_skipped)

            duration = round(time.perf_counter() - start, 2)
            summary = {
                "total": total,
                "passed": passed,
                "failed": failed,
                "skipped": skipped,
                "cases": results,
                "duration_s": duration,
            }
            self._emit("run_done", **summary)
            return summary

    # ---- internals ----
    def _emit(self, event: str, **data):
        if self._progress_cb:
            try:
                self._progress_cb({"event": event, **data})
            except Exception:
                logger.debug("progress_cb failed", exc_info=True)

    def _cb_reset_success(self):
        self._cb_failures = 0
        self._cb_open_until = None

    def _cb_record_fail(self):
        self._cb_failures += 1
        if self._cb_failures >= int(self.cfg["cb_threshold"]):
            self._cb_open_until = time.time() + float(self.cfg["cb_cooldown_sec"])
            logger.warning("⚠️ API circuit open for %.1fs (failures=%d)", self.cfg["cb_cooldown_sec"], self._cb_failures)

    def _should_pause_on_circuit(self) -> bool:
        return self._cb_open_until is not None and time.time() < self._cb_open_until

    async def _sleep_open_circuit(self):
        if self._cb_open_until is None:
            return
        remaining = max(0.0, self._cb_open_until - time.time())
        if remaining > 0:
            await asyncio.sleep(remaining)
        self._cb_reset_success()

    async def _run_case(self, client: httpx.AsyncClient, case_name: str, steps: List[APIStep]) -> Dict[str, Any]:
        ctx: Dict[str, Any] = {}
        out_steps: List[Dict[str, Any]] = []

        for idx, st in enumerate(steps, start=1):
            if self._stop:
                out_steps.append({"step": st.name or "stop_requested", "status": "SKIPPED"})
                continue

            step_id = f"api_{abs(hash((case_name, idx, st.name or '', st.request.method if st.request else ''))) % (10**10):010d}"

            # 1) Assertion-only step (uses last response)
            if not st.request and st.expect:
                resp: Optional[httpx.Response] = ctx.get("_last_response")
                if resp is None:
                    msg = "No previous response to assert against"
                    _log_step_json(step_id, "FAIL", msg, {"case": case_name})
                    out_steps.append({"step": st.name or "assert", "status": "FAIL"})
                    self._cb_record_fail()
                    continue
                elapsed_ms = ctx.get("_last_elapsed_ms")
                ok, why = _assert_response(resp, st.expect, elapsed_ms)
                if ok:
                    _log_step_json(step_id, "PASS", f"assertions passed: {why}", {"case": case_name})
                    out_steps.append({"step": st.name or "assert", "status": "PASS"})
                else:
                    _log_step_json(step_id, "FAIL", f"assertions failed: {why}", {"case": case_name})
                    out_steps.append({"step": st.name or "assert", "status": "FAIL"})
                    self._cb_record_fail()
                continue

            # 2) Comment-only step
            if not st.request and not st.expect:
                _log_step_json(step_id, "PASS", f"note: {st.name or ''}", {"case": case_name})
                out_steps.append({"step": st.name or "note", "status": "PASS"})
                continue

            # 3) Request step
            assert st.request is not None
            req = st.request
            rendered_url = _render_template(req.url or "", ctx)
            rendered_headers = _render_template(req.headers or {}, ctx)
            rendered_json = _render_template(req.json_body, ctx)
            rendered_data = _render_template(req.data, ctx)
            rendered_params = _render_template(req.params or {}, ctx)

            # retries/backoff
            retries = max(0, int(self.cfg["retries"]))
            backoff = float(self.cfg["backoff_base_sec"])

            last_exc: Optional[Exception] = None
            resp: Optional[httpx.Response] = None
            elapsed_ms: Optional[int] = None

            for attempt in range(retries + 1):
                try:
                    t0 = time.perf_counter()
                    resp = await client.request(
                        req.method.upper(),
                        rendered_url,
                        headers=rendered_headers or None,
                        json=rendered_json,
                        data=rendered_data,
                        params=rendered_params or None,
                    )
                    elapsed_ms = int((time.perf_counter() - t0) * 1000)
                    # retry only for 5xx
                    if resp.status_code >= 500 and attempt < retries:
                        await asyncio.sleep(backoff * (attempt + 1))
                        continue
                    break
                except Exception as e:
                    last_exc = e
                    if attempt < retries:
                        await asyncio.sleep(backoff * (attempt + 1))
                        continue
                    break

            ctx["_last_response"] = resp
            ctx["_last_elapsed_ms"] = elapsed_ms

            # outcome & assertions
            if resp is None:
                msg = f"request failed: {repr(last_exc)}"
                _log_step_json(step_id, "FAIL", msg, {
                    "case": case_name,
                    "url": rendered_url,
                    "headers": _redact_headers(rendered_headers)
                })
                out_steps.append({"step": f"{req.method} {rendered_url}", "status": "FAIL"})
                self._cb_record_fail()
                continue

            # Default response-time assertion if provided globally
            default_rtl = self.cfg.get("default_response_time_ms_lte")
            if default_rtl is not None:
                if st.expect is None:
                    st.expect = APIExpect()
                if st.expect.response_time_ms_lte is None:
                    st.expect.response_time_ms_lte = int(default_rtl)

            status_str = "PASS"
            if st.expect:
                ok, why = _assert_response(resp, st.expect, elapsed_ms)
                if not ok:
                    status_str = "FAIL"
                    self._cb_record_fail()
                    _log_step_json(step_id, status_str, f"assertions failed: {why}", {
                        "case": case_name,
                        "url": rendered_url,
                        "status_code": resp.status_code,
                        "elapsed_ms": elapsed_ms
                    })
                else:
                    _log_step_json(step_id, status_str, "assertions passed", {
                        "case": case_name,
                        "url": rendered_url,
                        "status_code": resp.status_code,
                        "elapsed_ms": elapsed_ms
                    })
            else:
                _log_step_json(step_id, status_str, "request completed", {
                    "case": case_name,
                    "url": rendered_url,
                    "status_code": resp.status_code,
                    "elapsed_ms": elapsed_ms
                })

            # saves (extract to context)
            if st.save:
                try:
                    data = None
                    if "application/json" in (resp.headers.get("content-type") or ""):
                        try:
                            data = resp.json()
                        except Exception:
                            data = None
                    for var, dotted in st.save.items():
                        val = None
                        if dotted == "$status":
                            val = resp.status_code
                        elif dotted.startswith("header."):
                            key = dotted[7:]
                            val = resp.headers.get(key)
                        elif dotted == "$elapsed_ms":
                            val = elapsed_ms
                        else:
                            if data is None:
                                try:
                                    data = resp.json()
                                except Exception:
                                    data = None
                            if data is not None:
                                val = _extract_dotted(data, dotted)
                        if val is not None:
                            ctx[var] = val
                except Exception:
                    logger.debug("save extraction failed", exc_info=True)

            out_steps.append({"step": f"{req.method} {rendered_url}", "status": status_str})

            # reset circuit breaker on success
            if status_str == "PASS":
                self._cb_reset_success()

        return {"name": case_name, "steps": out_steps}
