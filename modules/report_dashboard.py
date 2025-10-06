"""
Report Dashboard (Phase 3.9.6 — Fully Documented & Maintainable)
-----------------------------------------------------------------
Purpose:
--------
This module generates an **interactive HTML dashboard** summarizing the results
of AI QA Agent executions. It reads the generated JSON reports, aggregates
pass/fail counts, and visualizes them using Chart.js.

Key Responsibilities:
---------------------
✅ Read & validate structured report JSONs (UI/API/Performance)
✅ Auto-detect schema type (Playwright or internal test_results)
✅ Compute suite-wise & total statistics
✅ Append execution data into trend history (for line charts)
✅ Render standalone HTML dashboard with charts and tables

Output:
-------
- /reports/dashboards/<execution_id>.html   ← Interactive report
- /reports/history.json                     ← Persistent execution trend log
"""

import os
import json
import datetime


class ReportDashboard:
    """
    Generates visual dashboards for test execution summaries.
    This class is modular — it can later integrate with RAG/LLM
    to add semantic insights (like top-failing selectors or frequent errors).
    """

    def __init__(self, reports_dir: str = "reports"):
        """
        Initialize dashboard paths.

        Args:
            reports_dir (str): Base directory containing JSON reports.
        """
        self.reports_dir = reports_dir
        self.dashboard_dir = os.path.join(reports_dir, "dashboards")
        self.history_file = os.path.join(reports_dir, "history.json")

        os.makedirs(self.dashboard_dir, exist_ok=True)

    # ----------------------------------------------------------------------
    # 1️⃣ Load JSON Report File
    # ----------------------------------------------------------------------
    def _load_json(self, path: str) -> dict:
        """
        Safely load a JSON file. Returns an empty dict on error.

        Args:
            path (str): Path to the JSON file.
        Returns:
            dict: Parsed JSON content or empty dictionary.
        """
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception as e:
            print(f"⚠️ Failed to load JSON from {path}: {e}")
            return {}

    # ----------------------------------------------------------------------
    # 2️⃣ Aggregate Pass/Fail Summary
    # ----------------------------------------------------------------------
    def _aggregate_results(self, data: dict) -> dict:
        """
        Build a summary of pass/fail stats per suite and globally.

        Args:
            data (dict): Raw JSON data from report file.
        Returns:
            dict: Summary with total and per-suite statistics.
        """
        summary = {"total": {"PASS": 0, "FAIL": 0}, "suites": []}

        # Detect root-level or nested "test_results"
        test_results = data.get("test_results", data)

        for suite_name, cases in test_results.items():
            suite_summary = {"suite": suite_name, "PASS": 0, "FAIL": 0, "total": 0}

            for case in cases:
                for step in case.get("steps", []):
                    suite_summary["total"] += 1
                    status = step.get("status", "UNKNOWN").upper()
                    if status == "PASS":
                        suite_summary["PASS"] += 1
                    else:
                        suite_summary["FAIL"] += 1

            summary["suites"].append(suite_summary)
            summary["total"]["PASS"] += suite_summary["PASS"]
            summary["total"]["FAIL"] += suite_summary["FAIL"]

        return summary

    # ----------------------------------------------------------------------
    # 3️⃣ Append Execution Data to Trend History
    # ----------------------------------------------------------------------
    def _update_history(self, execution_id: str, summary: dict, duration: float):
        """
        Append a new execution record to persistent history.json.

        Args:
            execution_id (str): Unique execution ID.
            summary (dict): Aggregated summary of results.
            duration (float): Duration in seconds.
        Returns:
            list: Updated history records (up to last 50 runs).
        """
        # Load existing history (if any)
        history = []
        if os.path.exists(self.history_file):
            history = self._load_json(self.history_file)
        if not isinstance(history, list):
            history = []

        record = {
            "execution_id": execution_id,
            "timestamp": datetime.datetime.now().isoformat(),
            "pass": summary["total"]["PASS"],
            "fail": summary["total"]["FAIL"],
            "duration_sec": duration,
        }

        history.append(record)

        # Keep only the last 50 runs
        with open(self.history_file, "w", encoding="utf-8") as f:
            json.dump(history[-50:], f, indent=2)

        return history

    # ----------------------------------------------------------------------
    # 4️⃣ Generate HTML Dashboard
    # ----------------------------------------------------------------------
    def generate_dashboard(self, execution_id: str):
        """
        Generate an interactive HTML dashboard summarizing the test results.

        Args:
            execution_id (str): Execution ID of the test run.
        Returns:
            str: Path to the generated HTML dashboard file.
        """
        # Load execution report JSON
        json_report_path = os.path.join(self.reports_dir, f"{execution_id}.json")
        data = self._load_json(json_report_path)
        if not data:
            raise ValueError(f"No valid report found at {json_report_path}")

        meta = data.get("execution_meta", {})
        duration = meta.get("duration_sec", 0)
        summary = self._aggregate_results(data)
        history = self._update_history(execution_id, summary, duration)

        # Dashboard output path
        html_path = os.path.join(self.dashboard_dir, f"{execution_id}.html")

        # ------------------------------------------------------------------
        # HTML Template: Self-contained dashboard using Chart.js
        # ------------------------------------------------------------------
        html = f"""
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>AI QA Dashboard – Execution {execution_id}</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
body {{
  font-family: 'Segoe UI', sans-serif;
  background:#f8f9fa;
  color:#333;
  margin:0; padding:20px;
}}
h1, h2 {{ color:#003366; }}
.container {{
  display:grid;
  grid-template-columns: 1fr 1fr;
  gap:20px;
}}
.card {{
  background:white;
  padding:20px;
  border-radius:10px;
  box-shadow:0 2px 8px rgba(0,0,0,0.1);
}}
.pass {{ color:green; }}
.fail {{ color:red; }}
footer {{
  margin-top:30px;
  font-size:12px;
  text-align:center;
  color:#777;
}}
table {{
  width:100%;
  border-collapse:collapse;
}}
td, th {{
  border:1px solid #ddd;
  padding:8px;
}}
th {{
  background:#003366;
  color:white;
}}
</style>
</head>
<body>
  <h1>AI QA Execution Dashboard</h1>
  <h2>Execution ID: {execution_id}</h2>
  <p><strong>Duration:</strong> {duration:.2f} seconds</p>

  <div class="container">
    <!-- Suite Summary Table -->
    <div class="card">
      <h3>Suite Summary</h3>
      <table>
        <tr><th>Suite</th><th>PASS</th><th>FAIL</th><th>Total</th></tr>
        {''.join(f"<tr><td>{s['suite']}</td><td class='pass'>{s['PASS']}</td><td class='fail'>{s['FAIL']}</td><td>{s['total']}</td></tr>" for s in summary['suites'])}
      </table>
    </div>

    <!-- Pie Chart for Distribution -->
    <div class="card">
      <h3>Overall Result Distribution</h3>
      <canvas id="pieChart" height="200"></canvas>
    </div>
  </div>

  <!-- Historical Trend -->
  <div class="card" style="margin-top:30px;">
    <h3>Execution History (Last 50 Runs)</h3>
    <canvas id="trendChart" height="120"></canvas>
  </div>

  <footer>
    <p>Generated on {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | AI QA Agent v3.9.6</p>
  </footer>

  <script>
  // Pie Chart: Current Run Summary
  const ctx = document.getElementById('pieChart');
  const chartData = {{
    labels: ['PASS', 'FAIL'],
    datasets: [{{
      label: 'Results',
      data: [{summary['total']['PASS']}, {summary['total']['FAIL']}],
      backgroundColor: ['#2b9348','#d00000']
    }}]
  }};
  new Chart(ctx, {{ type: 'pie', data: chartData }});

  // Line Chart: Execution Trend
  const trendCtx = document.getElementById('trendChart');
  const history = {json.dumps(history)};
  new Chart(trendCtx, {{
    type: 'line',
    data: {{
      labels: history.map(h => h.execution_id),
      datasets: [
        {{
          label: 'PASS',
          data: history.map(h => h.pass),
          borderColor: '#2b9348',
          fill: false
        }},
        {{
          label: 'FAIL',
          data: history.map(h => h.fail),
          borderColor: '#d00000',
          fill: false
        }}
      ]
    }},
    options: {{
      responsive: true,
      plugins: {{ legend: {{ position: 'top' }} }},
      scales: {{ y: {{ beginAtZero: true }} }}
    }}
  }});
  </script>
</body>
</html>
"""

        # Save HTML file
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"📊 Dashboard generated successfully → {html_path}")
        return html_path
