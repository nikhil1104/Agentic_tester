# modules/report_dashboard.py
"""
Report Dashboard (Phase 3.9.8 — Hardened, CI-Friendly, Extensible)
-------------------------------------------------------------------
Generates interactive HTML dashboards from runner output or Playwright JSON.
- Reads: reports/<execution_id>.json  (primary, our unified report)
- Falls back to normalizing Playwright JSON shapes when needed
- Maintains rolling history (reports/history.json)
- Produces: reports/dashboards/<execution_id>.html  (+ optional index page)

Public API:
    rd = ReportDashboard(reports_dir="reports")
    rd.generate_dashboard(execution_id)  # returns path string
    rd.generate_index(limit=50)          # builds dashboards/index.html
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from datetime import datetime

JSON = Dict[str, Any]
logger = logging.getLogger(__name__)


class ReportDashboard:
    def __init__(self, reports_dir: str = "reports", *, history_limit: int = 50):
        """
        :param reports_dir: base directory where JSON reports and dashboards live
        :param history_limit: keep last N history entries
        """
        self.reports_dir = Path(reports_dir)
        self.dashboard_dir = self.reports_dir / "dashboards"
        self.history_file = self.reports_dir / "history.json"
        self.history_limit = max(1, history_limit)

        self.dashboard_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------------------
    # JSON IO (safe)
    # ---------------------------------------------------------------------
    def _load_json(self, path: Union[str, Path]) -> JSON:
        p = Path(path)
        if not p.exists():
            logger.warning("JSON not found: %s", p)
            return {}
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except Exception as e:
            logger.warning("Failed to parse JSON %s: %s", p, e)
            return {}

    def _write_json(self, path: Path, data: Any) -> None:
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception:
            logger.exception("Failed writing %s", path)

    # ---------------------------------------------------------------------
    # Shape detection & normalization
    # ---------------------------------------------------------------------
    @staticmethod
    def _is_playwright_report(data: JSON) -> bool:
        """Heuristic detection of Playwright report-like structures."""
        if not isinstance(data, dict):
            return False
        return any(k in data for k in ("suites", "projects", "tests", "entries"))

    def _normalize_playwright(self, data: JSON) -> JSON:
        """
        Convert Playwright JSON into our internal 'test_results' mapping:
        { suite_name: [ {name: str, steps: [ {step: str, status: PASS|FAIL|...} ]}, ... ], ... }
        """
        normalized: JSON = {}

        def _walk_suite(suite: JSON) -> List[JSON]:
            """Recursive helper for 'suites' style reports."""
            cases: List[JSON] = []
            tests = suite.get("tests") or suite.get("specs") or []
            if isinstance(tests, dict):
                tests = [tests]
            for tcase in tests:
                name = tcase.get("title") or tcase.get("name") or "test"
                steps: List[JSON] = []
                results = tcase.get("results") or []
                if isinstance(results, dict):
                    results = [results]
                for r in results:
                    status = str(r.get("status") or "UNKNOWN").upper()
                    steps.append({"step": name, "status": status})
                if not steps:
                    steps.append({"step": name, "status": "UNKNOWN"})
                cases.append({"name": name, "steps": steps})
            # children
            for child in suite.get("suites", []) or []:
                cases.extend(_walk_suite(child))
            return cases

        # v1 style: top-level "suites"
        suites = data.get("suites")
        if isinstance(suites, list) and suites:
            for s in suites:
                sname = s.get("title") or s.get("name") or "playwright_suite"
                normalized[sname] = _walk_suite(s)
            if normalized:
                return normalized

        # v2+ style: "entries" or sometimes "tests" structured by file
        entries = data.get("entries") or data.get("tests") or []
        if isinstance(entries, list) and entries:
            grouped: JSON = {}
            for e in entries:
                suite_name = e.get("file") or e.get("name") or "playwright"
                grouped.setdefault(suite_name, [])
                items = e.get("tests") or e.get("items") or []
                if isinstance(items, dict):
                    items = [items]
                for it in items:
                    tname = it.get("title") or it.get("name") or it.get("testName") or "test"
                    status = str(it.get("status") or "UNKNOWN").upper()
                    grouped[suite_name].append(
                        {"name": tname, "steps": [{"step": tname, "status": status}]}
                    )
            if grouped:
                return grouped

        # fallback: flat tests list
        flat_tests = data.get("tests")
        if isinstance(flat_tests, list) and flat_tests:
            normalized["playwright"] = []
            for tcase in flat_tests:
                tname = tcase.get("title") or tcase.get("name") or "test"
                status = str(tcase.get("status") or "UNKNOWN").upper()
                normalized["playwright"].append({"name": tname, "steps": [{"step": tname, "status": status}]})
            return normalized

        # ultimate fallback
        return {"playwright": [{"name": "unknown", "steps": [{"step": "unknown", "status": "UNKNOWN"}]}]}

    def _to_internal_shape(self, data: JSON) -> JSON:
        """
        Normalize any supported input into:
        {
          "test_results": { suite_name: [ {name, steps:[{step,status}, ...]}, ... ] },
          "execution_meta": {...}
        }
        """
        if not isinstance(data, dict):
            return {"test_results": {}, "execution_meta": {}}

        # Already in our shape?
        if "test_results" in data and isinstance(data["test_results"], dict):
            return {"test_results": data["test_results"], "execution_meta": data.get("execution_meta", {})}

        # Playwright?
        if self._is_playwright_report(data):
            test_results = self._normalize_playwright(data)
            meta = data.get("metadata") or data.get("meta") or {}
            return {"test_results": test_results, "execution_meta": meta}

        # Otherwise, assume top-level suites mapping
        assumed: JSON = {}
        for k, v in data.items():
            if k in ("execution_id", "execution_meta", "meta", "metadata"):
                continue
            if isinstance(v, list):
                assumed[k] = v
        return {"test_results": assumed, "execution_meta": data.get("execution_meta", {})}

    # ---------------------------------------------------------------------
    # Aggregation & history
    # ---------------------------------------------------------------------
    @staticmethod
    def _aggregate_results(model: JSON) -> JSON:
        """
        Build a summary of PASS/FAIL by suite and totals.
        """
        result = {"total": {"PASS": 0, "FAIL": 0}, "suites": []}
        test_results = model.get("test_results", {}) or {}

        for suite_name, cases in test_results.items():
            suite_sum = {"suite": suite_name, "PASS": 0, "FAIL": 0, "total": 0}
            for case in cases or []:
                steps = case.get("steps")
                if not isinstance(steps, list):
                    # handle degenerate cases {"name":..., "status":...}
                    status = str(case.get("status", "UNKNOWN")).upper()
                    suite_sum["total"] += 1
                    if status == "PASS":
                        suite_sum["PASS"] += 1
                    else:
                        suite_sum["FAIL"] += 1
                    continue
                for step in steps:
                    suite_sum["total"] += 1
                    status = str(step.get("status", "UNKNOWN")).upper()
                    if status == "PASS":
                        suite_sum["PASS"] += 1
                    else:
                        suite_sum["FAIL"] += 1
            result["suites"].append(suite_sum)
            result["total"]["PASS"] += suite_sum["PASS"]
            result["total"]["FAIL"] += suite_sum["FAIL"]
        return result

    def _update_history(self, execution_id: str, summary: JSON, duration_sec: float) -> List[JSON]:
        """
        Append the execution to history.json and keep only the last `history_limit` records.
        """
        history: List[JSON] = []
        if self.history_file.exists():
            try:
                loaded = self._load_json(self.history_file)
                if isinstance(loaded, list):
                    history = loaded
            except Exception:
                logger.warning("history.json unreadable; resetting")

        record = {
            "execution_id": execution_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "pass": summary["total"]["PASS"],
            "fail": summary["total"]["FAIL"],
            "duration_sec": float(duration_sec or 0.0),
        }
        history.append(record)
        history = history[-self.history_limit :]
        self._write_json(self.history_file, history)
        return history

    # ---------------------------------------------------------------------
    # HTML builders
    # ---------------------------------------------------------------------
    def _build_dashboard_html(
        self,
        execution_id: str,
        meta: JSON,
        summary: JSON,
        history: List[JSON],
        events_href: Optional[str],
    ) -> str:
        """Return final HTML string (self-contained except Chart.js CDN)."""
        duration = float(meta.get("duration_sec", 0) or 0)
        suites_rows = "".join(
            f"<tr><td>{s['suite']}</td>"
            f"<td style='color:#1a7f37'>{s['PASS']}</td>"
            f"<td style='color:#d00000'>{s['FAIL']}</td>"
            f"<td>{s['total']}</td></tr>"
            for s in summary["suites"]
        )
        events_link = (
            f"<div class='muted'>Events: <a href='{events_href}' target='_blank'>{events_href}</a></div>"
            if events_href else ""
        )
        meta_pretty = json.dumps(meta, indent=2)

        return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI QA Dashboard — {execution_id}</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  :root {{ --bg:#f7fafc; --fg:#111; --card:#fff; --muted:#666; --accent:#0b5fff; }}
  body {{ font-family: Inter, 'Segoe UI', Roboto, Arial, sans-serif; background:var(--bg); color:var(--fg); margin:0; padding:20px; }}
  .wrap {{ max-width:1100px; margin:0 auto; }}
  header {{ margin-bottom:16px; }}
  .grid {{ display:grid; grid-template-columns: 1fr 380px; gap:16px; }}
  .card {{ background:var(--card); padding:16px; border-radius:10px; box-shadow:0 6px 18px rgba(0,0,0,0.06); }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; }}
  th,td {{ padding:8px; border-bottom:1px solid #eee; text-align:left; }}
  th {{ background:var(--accent); color:#fff; }}
  .muted {{ color:var(--muted); font-size:13px; }}
  footer {{ margin-top:20px; font-size:12px; color:var(--muted); text-align:center; }}
</style>
</head>
<body>
<div class="wrap">
  <header>
    <h1>AI QA Execution Dashboard</h1>
    <div class="muted">
      Execution ID: <strong>{execution_id}</strong> • Duration: {duration:.2f}s • Generated: {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")}Z
    </div>
    {events_link}
  </header>

  <div class="grid">
    <div class="card">
      <h3>Suites Summary</h3>
      <table>
        <thead><tr><th>Suite</th><th>PASS</th><th>FAIL</th><th>Total</th></tr></thead>
        <tbody>{suites_rows}</tbody>
      </table>
    </div>

    <div class="card">
      <h3>Result Distribution</h3>
      <canvas id="pieChart" height="220"></canvas>
    </div>
  </div>

  <div style="margin-top:18px" class="grid">
    <div class="card">
      <h3>Execution History (Last {len(history)})</h3>
      <canvas id="trendChart" height="120"></canvas>
    </div>

    <div class="card">
      <h3>Run Metadata</h3>
      <pre style="font-size:13px;white-space:pre-wrap">{meta_pretty}</pre>
    </div>
  </div>

  <footer>AI QA Agent — Dashboard • {datetime.utcnow().year}</footer>
</div>

<script>
  // Pie chart
  const pieCtx = document.getElementById('pieChart').getContext('2d');
  new Chart(pieCtx, {{
    type: 'pie',
    data: {{
      labels: ['PASS','FAIL'],
      datasets: [{{
        data: [{summary['total']['PASS']}, {summary['total']['FAIL']}],
        backgroundColor: ['#1a7f37','#d00000']
      }}]
    }},
    options: {{ responsive: true, plugins: {{ legend: {{ position: 'bottom' }} }} }}
  }});

  // Trend chart (history)
  const trendCtx = document.getElementById('trendChart').getContext('2d');
  const history = {json.dumps(history)};
  new Chart(trendCtx, {{
    type: 'line',
    data: {{
      labels: history.map(h => h.execution_id),
      datasets: [
        {{ label: 'PASS', data: history.map(h => h.pass), borderColor: '#1a7f37', fill:false }},
        {{ label: 'FAIL', data: history.map(h => h.fail), borderColor: '#d00000', fill:false }}
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{ y: {{ beginAtZero: true, precision: 0 }} }}
    }}
  }});
</script>
</body>
</html>"""

    # ---------------------------------------------------------------------
    # Public: dashboard generation
    # ---------------------------------------------------------------------
    def generate_dashboard(self, execution_id: str) -> str:
        """
        Generate the dashboard HTML for a run (returns filesystem path).
        Reads reports/<execution_id>.json created by the Runner.
        """
        json_path = self.reports_dir / f"{execution_id}.json"
        raw = self._load_json(json_path)
        if not raw:
            raise FileNotFoundError(f"No valid report JSON at {json_path}")

        model = self._to_internal_shape(raw)
        summary = self._aggregate_results(model)
        meta = model.get("execution_meta", {}) or {}
        duration = float(meta.get("duration_sec", 0) or 0)

        # Append to history + get trimmed list
        history = self._update_history(execution_id, summary, duration)

        # Optional events stream link (Phase 6.4 NDJSON)
        ndjson = self.reports_dir / f"{execution_id}.events.ndjson"
        events_href = ndjson.name if ndjson.exists() else None

        html = self._build_dashboard_html(execution_id, meta, summary, history, events_href)
        out = self.dashboard_dir / f"{execution_id}.html"
        try:
            out.write_text(html, encoding="utf-8")
            logger.info("Dashboard generated → %s", out)
        except Exception:
            logger.exception("Failed writing dashboard HTML %s", out)
            raise
        return str(out)

    def generate_index(self, *, limit: Optional[int] = None) -> str:
        """
        Build dashboards/index.html listing the last N runs with quick stats.
        """
        limit = limit or self.history_limit
        history: List[JSON] = []
        if self.history_file.exists():
            loaded = self._load_json(self.history_file)
            if isinstance(loaded, list):
                history = loaded[-limit:]

        # rows
        row_html = ""
        for h in reversed(history):
            exec_id = h.get("execution_id", "")
            pass_ct = h.get("pass", 0)
            fail_ct = h.get("fail", 0)
            dur = float(h.get("duration_sec", 0) or 0)
            ts = h.get("timestamp", "")
            dash_path = f"./{exec_id}.html"
            row_html += (
                f"<tr>"
                f"<td><a href='{dash_path}'>{exec_id}</a></td>"
                f"<td>{ts}</td>"
                f"<td style='color:#1a7f37'>{pass_ct}</td>"
                f"<td style='color:#d00000'>{fail_ct}</td>"
                f"<td>{dur:.2f}s</td>"
                f"</tr>"
            )

        index_html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>AI QA — Dashboards Index</title>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<style>
  body {{ font-family: Inter, 'Segoe UI', Roboto, Arial, sans-serif; background:#f7fafc; color:#111; margin:0; padding:20px; }}
  .wrap {{ max-width:1000px; margin:0 auto; }}
  h1 {{ margin-bottom:16px; }}
  table {{ width:100%; border-collapse:collapse; font-size:14px; background:#fff; border-radius:10px; overflow:hidden; box-shadow:0 6px 18px rgba(0,0,0,0.06); }}
  th,td {{ padding:10px; border-bottom:1px solid #eee; text-align:left; }}
  th {{ background:#0b5fff; color:#fff; }}
  a {{ color:#0b5fff; text-decoration:none; }}
</style>
</head>
<body>
<div class="wrap">
  <h1>AI QA — Recent Dashboards</h1>
  <table>
    <thead><tr><th>Execution ID</th><th>Timestamp (UTC)</th><th>PASS</th><th>FAIL</th><th>Duration</th></tr></thead>
    <tbody>
      {row_html or "<tr><td colspan='5'>No history yet.</td></tr>"}
    </tbody>
  </table>
</div>
</body>
</html>"""

        out = self.dashboard_dir / "index.html"
        try:
            out.write_text(index_html, encoding="utf-8")
            logger.info("Dashboards index generated → %s", out)
        except Exception:
            logger.exception("Failed writing dashboards index %s", out)
            raise
        return str(out)
