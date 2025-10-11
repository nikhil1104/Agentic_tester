# modules/reporter.py
"""
Reporter (Phase 3.10 — Atomic, Schema-Tolerant, CI-Friendly)
------------------------------------------------------------
Persists the unified run JSON and renders a lightweight HTML overview.
The rich dashboard should be produced by modules.report_dashboard.ReportDashboard.

Contract:
    Reporter().create_reports(results: dict) -> {"json": <path>, "html": <path>}
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from jinja2 import Environment, BaseLoader, select_autoescape

logger = logging.getLogger(__name__)

# Honor env var so containers/CI can redirect artifacts
REPORTS_DIR = Path(os.environ.get("RUNNER_REPORTS_DIR", "reports"))
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Minimal HTML overview (we keep it simple; the full dashboard is separate)
_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI QA — Run {{ eid }}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
  :root { --bg:#f7fafc; --fg:#111; --muted:#666; --card:#fff; --accent:#0b5fff; --ok:#1a7f37; --bad:#d00000; }
  body { font-family: Inter, 'Segoe UI', Roboto, Arial, sans-serif; background:var(--bg); color:var(--fg); margin:0; padding:20px; }
  .wrap { max-width: 980px; margin: 0 auto; }
  .card { background:var(--card); border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.06); padding:16px; margin-bottom:16px; }
  .grid { display:grid; grid-template-columns: 1fr 1fr; gap: 12px; }
  h1,h2,h3 { margin: 0 0 8px 0; }
  .muted { color: var(--muted); font-size: 13px; }
  .kv { display:grid; grid-template-columns: 200px 1fr; gap:6px; font-size:14px; }
  a { color: var(--accent); text-decoration:none; }
  .ok { color: var(--ok); } .bad { color: var(--bad); }
  code, pre { background:#f0f3f7; padding:10px; border-radius:8px; display:block; overflow:auto; }
  table { width:100%; border-collapse:collapse; font-size:14px; }
  th,td { padding:8px; border-bottom:1px solid #eee; text-align:left; }
  th { background:#eef4ff; }
</style>
</head>
<body>
<div class="wrap">
  <div class="card">
    <h1>AI QA — Execution {{ eid }}</h1>
    <div class="muted">Generated: {{ now }} (UTC)</div>
    {% if dashboard_href %}
      <div class="muted">Dashboard: <a href="{{ dashboard_href }}" target="_blank">{{ dashboard_href }}</a></div>
    {% endif %}
    {% if events_href %}
      <div class="muted">Events: <a href="{{ events_href }}" target="_blank">{{ events_href }}</a></div>
    {% endif %}
  </div>

  <div class="grid">
    <div class="card">
      <h3>Summary</h3>
      <div class="kv">
        <div>Overall:</div><div><strong class="{{ 'ok' if summary.overall_passed else 'bad' }}">{{ summary.overall_passed }}</strong></div>
        <div>UI:</div><div>{{ summary.ui_passed }}</div>
        <div>API:</div><div>{{ summary.api_passed }}</div>
        <div>Perf:</div><div>{{ summary.perf_passed }}</div>
      </div>
    </div>

    <div class="card">
      <h3>Execution Meta</h3>
      <div class="kv">
        <div>Start:</div><div>{{ meta.start_time or '-' }}</div>
        <div>End:</div><div>{{ meta.end_time or '-' }}</div>
        <div>Duration (s):</div><div>{{ meta.duration_sec or 0 }}</div>
        <div>Parallel:</div><div>{{ meta.parallel }}</div>
        <div>Max Workers:</div><div>{{ meta.max_workers }}</div>
        <div>Fail Fast:</div><div>{{ meta.fail_fast }}</div>
      </div>
    </div>
  </div>

  <div class="card">
    <h3>Stages</h3>
    <table>
      <thead><tr><th>Stage</th><th>Succeeded</th><th>Attempts</th></tr></thead>
      <tbody>
        {% for stage, info in stages.items() %}
          <tr>
            <td>{{ stage }}</td>
            <td class="{{ 'ok' if info.succeeded else 'bad' }}">{{ info.succeeded }}</td>
            <td>{{ info.attempts }}</td>
          </tr>
        {% endfor %}
      </tbody>
    </table>
  </div>

  {% if ui_artifacts %}
  <div class="card">
    <h3>UI Artifacts</h3>
    <div class="kv">
      <div>Raw Log:</div><div>{{ ui_artifacts.raw_log or '-' }}</div>
      <div>Playwright JSON:</div><div>{{ ui_artifacts.playwright_report or '-' }}</div>
    </div>
  </div>
  {% endif %}

  <div class="card">
    <h3>Raw Results (excerpt)</h3>
    <pre>{{ raw_excerpt }}</pre>
  </div>
</div>
</body>
</html>
"""

_env = Environment(
    loader=BaseLoader(),
    autoescape=select_autoescape(enabled_extensions=("html", "xml"), default_for_string=True),
    enable_async=False,
)


class Reporter:
    """Persist run JSON and render a slim HTML overview."""

    def __init__(self, reports_dir: Optional[str] = None) -> None:
        self.reports_dir = Path(reports_dir) if reports_dir else REPORTS_DIR
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ----------------------------- public API -----------------------------
    def create_reports(self, results: Dict[str, Any]) -> Dict[str, str]:
        """
        Save <execution_id>.json (atomic) and a small HTML overview.
        Returns dict of file paths.
        """
        eid = str(results.get("execution_id") or "run")
        json_path = self.reports_dir / f"{eid}.json"
        html_path = self.reports_dir / f"{eid}.html"

        # 1) JSON (atomic)
        self._atomic_json_dump(json_path, results)
        logger.info("Saved report JSON → %s", json_path)

        # 2) HTML overview (robust to differing shapes)
        summary = self._extract_summary(results)
        meta = results.get("execution_meta") or {}
        stages = (meta.get("stages") or {}) if isinstance(meta.get("stages"), dict) else {}

        # Optional links (may be created by other components later)
        dashboard_href = f"dashboards/{eid}.html"  # Runner usually generates this after Reporter
        events_path = self.reports_dir / f"{eid}.events.ndjson"
        events_href = events_path.name if events_path.exists() else None

        ui_artifacts = self._extract_ui_artifacts(results)

        # Compact excerpt of raw JSON to aid quick debugging in CI
        raw_excerpt = self._json_excerpt(results, max_chars=8000)

        html = _env.from_string(_HTML_TEMPLATE).render(
            eid=eid,
            now=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S"),
            summary=summary,
            meta=meta,
            stages=stages,
            dashboard_href=dashboard_href,
            events_href=events_href,
            ui_artifacts=ui_artifacts,
            raw_excerpt=raw_excerpt,
        )
        self._atomic_text_write(html_path, html)
        logger.info("Saved report HTML → %s", html_path)

        return {"json": str(json_path), "html": str(html_path)}

    # --------------------------- helpers (private) ------------------------
    @staticmethod
    def _atomic_text_write(path: Path, text: str) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _atomic_json_dump(path: Path, data: Any) -> None:
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)

    @staticmethod
    def _json_excerpt(obj: Any, max_chars: int = 8000) -> str:
        try:
            s = json.dumps(obj, indent=2)
        except Exception:
            s = str(obj)
        return (s[: max_chars - 3] + "...") if len(s) > max_chars else s

    @staticmethod
    def _extract_ui_artifacts(results: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Best-effort extraction of UI artifacts regardless of layout."""
        ui = (results.get("test_results") or {}).get("ui")
        if isinstance(ui, dict):
            arts = ui.get("artifacts") or {}
            if isinstance(arts, dict) and (arts.get("raw_log") or arts.get("playwright_report")):
                return {"raw_log": arts.get("raw_log"), "playwright_report": arts.get("playwright_report")}
        return None

    @staticmethod
    def _extract_summary(results: Dict[str, Any]) -> Dict[str, Any]:
        """Return a stable summary block. If missing, infer conservatively."""
        summ = results.get("summary")
        if isinstance(summ, dict) and "overall_passed" in summ:
            return summ

        # fallback inference
        ui_passed = Reporter._infer_ui_pass((results.get("test_results") or {}).get("ui"))
        api_passed = Reporter._infer_api_pass((results.get("test_results") or {}).get("api"))
        perf_passed = Reporter._infer_perf_pass((results.get("test_results") or {}).get("performance"))

        present = [x for x in (ui_passed, api_passed, perf_passed) if x is not None]
        overall = all(present) if present else True
        return {
            "overall_passed": overall,
            "ui_passed": ui_passed,
            "api_passed": api_passed,
            "perf_passed": perf_passed,
        }

    @staticmethod
    def _infer_ui_pass(ui_result: Any) -> Optional[bool]:
        if ui_result is None:
            return None
        if isinstance(ui_result, dict):
            if isinstance(ui_result.get("summary"), dict) and "passed" in ui_result["summary"]:
                return bool(ui_result["summary"]["passed"])
            # Some pipelines store succeeded flag
            if "succeeded" in ui_result:
                return bool(ui_result["succeeded"])
        # Unknown layout → don't guess
        return None

    @staticmethod
    def _infer_api_pass(api_results: Any) -> Optional[bool]:
        if api_results is None:
            return None
        if not isinstance(api_results, list):
            return None
        def ok(r: Dict[str, Any]) -> bool:
            if r.get("error"):
                return False
            for k in ("passed", "success"):
                if k in r:
                    return bool(r[k])
            status = str(r.get("status", "")).lower()
            if status in {"pass", "ok", "success"}:
                return True
            if isinstance(r.get("failures"), int):
                return r["failures"] == 0
            return True
        return all(ok(x) for x in api_results) if api_results else True

    @staticmethod
    def _infer_perf_pass(perf_results: Any) -> Optional[bool]:
        if perf_results is None:
            return None
        if not isinstance(perf_results, list):
            return None
        def ok(r: Dict[str, Any]) -> bool:
            if r.get("error"):
                return False
            sla = r.get("sla") or {}
            if "passed" in sla:
                return bool(sla["passed"])
            return True
        return all(ok(x) for x in perf_results) if perf_results else True
