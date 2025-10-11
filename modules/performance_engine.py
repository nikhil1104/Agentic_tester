"""
PerformanceEngine (Phase 6.2+)
------------------------------
Runs performance / load tests using JMeter or Locust.

Key features:
 - Tool availability checks, timeouts, returncode propagation
 - Per-suite artifact directories under a workspace
 - Rich metrics (p50/p90/p95/p99, RPS, error %, success %)
 - CSV parsing via csv.DictReader; optional JMeter JSON JTL support
 - SLA evaluation with pass/fail + violation details
 - Predictable result schema for unified reporting
 - Centralised tool defaults (TOOLS) and run_all(plan) helper
"""

from __future__ import annotations

import csv
import json
import shutil
import logging
import datetime
from pathlib import Path
from statistics import median
from typing import Dict, Any, List, Optional, Tuple

logger = logging.getLogger(__name__)


class PerformanceEngine:
    TOOLS = {
        "jmeter": {
            "binary": "jmeter",
            "default_timeout_sec": 3600,
            "output_format": "csv",   # csv | json (json produces larger files)
            "args": [],
            "file_suffix": ".jmx",
        },
        "locust": {
            "binary": "locust",
            "default_timeout_sec": 3600,
            "args": [],
            "file_suffix": ".py",
        },
        # "k6": {...}  # future
    }

    def __init__(self, project: str, workspace: str = "performance_workspace", default_timeout_sec: int = 3600):
        self.project = project
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.default_timeout_sec = default_timeout_sec

    # --------------------- Utilities ---------------------
    def _ensure_tool(self, tool: str) -> None:
        if shutil.which(tool) is None:
            raise FileNotFoundError(f"Required tool '{tool}' not found on PATH.")

    def _safe_under_workspace(self, rel: str) -> Path:
        p = (self.workspace / rel).resolve()
        if self.workspace not in p.parents and p != self.workspace:
            raise ValueError(f"Refusing path outside workspace: {p}")
        return p

    @staticmethod
    def _percentiles(values: List[float], ps: List[int]) -> Dict[str, float]:
        if not values:
            return {f"p{p}": None for p in ps}
        sorted_v = sorted(values)
        out: Dict[str, float] = {}
        n = len(sorted_v)
        for p in ps:
            # nearest-rank style
            k = max(1, int(round(p / 100.0 * n)))
            out[f"p{p}"] = float(sorted_v[k - 1])
        return out

    def _eval_sla(self, metrics: Dict[str, Any], sla: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Return {'passed': bool, 'violations': {...}} based on optional SLA dict."""
        if not sla:
            return {"passed": True, "violations": {}}
        violations: Dict[str, Any] = {}
        p95 = metrics.get("p95_ms")
        if "p95_ms" in sla and (p95 is None or p95 > sla["p95_ms"]):
            violations["p95_ms"] = {"actual": p95, "limit": sla["p95_ms"]}
        err = metrics.get("error_pct")
        if "error_pct_max" in sla and (err is None or err > sla["error_pct_max"]):
            violations["error_pct_max"] = {"actual": err, "limit": sla["error_pct_max"]}
        rps = metrics.get("requests_per_sec") or metrics.get("throughput_rps")
        if "rps_min" in sla and (rps is None or rps < sla["rps_min"]):
            violations["rps_min"] = {"actual": rps, "limit": sla["rps_min"]}
        return {"passed": len(violations) == 0, "violations": violations}

    def _run(self, cmd: List[str], cwd: Path, timeout_sec: Optional[int]) -> Any:
        import subprocess
        completed = subprocess.run(
            cmd,
            cwd=str(cwd),
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            timeout=timeout_sec or self.default_timeout_sec,
        )
        if completed.returncode != 0:
            logger.warning("Command exited non-zero (code=%s): %s", completed.returncode, " ".join(cmd))
        return completed

    # --------------------- JMeter Runner ---------------------
    def _run_jmeter(
        self,
        test_file: str,
        suite_dir: Path,
        output_format: str = "csv",
        extra_args: Optional[List[str]] = None,
        timeout_sec: Optional[int] = None,
    ) -> Dict[str, Any]:
        tool_cfg = self.TOOLS["jmeter"]
        self._ensure_tool(tool_cfg["binary"])

        test_path = self._safe_under_workspace(test_file)
        if not test_path.exists() or test_path.suffix.lower() != tool_cfg["file_suffix"]:
            raise FileNotFoundError(f"JMeter test file not found or wrong suffix: {test_path}")

        results_jtl = suite_dir / f"results.jtl"
        html_report = suite_dir / "html_report"
        html_report.mkdir(parents=True, exist_ok=True)

        cmd = [
            tool_cfg["binary"], "-n",
            "-t", str(test_path),
            "-l", str(results_jtl),
            "-e", "-o", str(html_report),
            f"-Jjmeter.save.saveservice.output_format={output_format}",
        ]
        cmd.extend(tool_cfg.get("args", []))
        if extra_args:
            cmd.extend(extra_args)

        logger.info("Running JMeter: %s", " ".join(cmd))
        completed = self._run(cmd, cwd=suite_dir, timeout_sec=timeout_sec or tool_cfg["default_timeout_sec"])

        # Parse metrics
        if output_format == "json":
            metrics = self._parse_jmeter_json(results_jtl)
        else:
            metrics = self._parse_jmeter_csv(results_jtl)

        return {
            "tool": "jmeter",
            "cmd": " ".join(cmd),
            "cwd": str(suite_dir),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "metrics": metrics,
            "artifacts": {
                "primary": str(results_jtl if results_jtl.exists() else ""),
                "extras": [str(html_report)] if html_report.exists() else [],
            },
        }

    # --------------------- Locust Runner ---------------------
    def _run_locust(
        self,
        script: str,
        suite_dir: Path,
        users: int = 10,
        spawn: int = 2,
        run_time: str = "1m",
        host: Optional[str] = None,
        extra_args: Optional[List[str]] = None,
        timeout_sec: Optional[int] = None,
    ) -> Dict[str, Any]:
        tool_cfg = self.TOOLS["locust"]
        self._ensure_tool(tool_cfg["binary"])

        script_path = self._safe_under_workspace(script)
        if not script_path.exists() or script_path.suffix.lower() != tool_cfg["file_suffix"]:
            raise FileNotFoundError(f"Locust script not found or wrong suffix: {script_path}")

        csv_prefix = suite_dir / "locust"
        cmd = [
            tool_cfg["binary"], "-f", str(script_path), "--headless",
            "-u", str(users), "-r", str(spawn),
            "--run-time", run_time,
            "--csv", str(csv_prefix),
        ]
        cmd.extend(tool_cfg.get("args", []))
        if host:
            cmd.extend(["--host", host])
        if extra_args:
            cmd.extend(extra_args)

        logger.info("Running Locust: %s", " ".join(cmd))
        completed = self._run(cmd, cwd=suite_dir, timeout_sec=timeout_sec or tool_cfg["default_timeout_sec"])
        metrics = self._parse_locust_csvs(csv_prefix)

        return {
            "tool": "locust",
            "cmd": " ".join(cmd),
            "cwd": str(suite_dir),
            "returncode": completed.returncode,
            "stdout": completed.stdout,
            "stderr": completed.stderr,
            "metrics": metrics,
            "artifacts": {
                "primary": str(csv_prefix.with_name(csv_prefix.name + "_requests.csv")),
                "extras": [str(csv_prefix.with_name(csv_prefix.name + "_stats_history.csv"))],
            },
        }

    # --------------------- Parsers: JMeter ---------------------
    def _base_metrics(self) -> Dict[str, Any]:
        return {
            "samples": 0,
            "avg_latency_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "error_pct": None,
            "throughput_rps": None,
            "success_rate_pct": None,
        }

    def _parse_jmeter_csv(self, jtl_path: Path) -> Dict[str, Any]:
        """
        Expects CSV JTL with at least: timeStamp, elapsed, success
        Ensure JMeter output format is CSV via:
        -Jjmeter.save.saveservice.output_format=csv
        """
        metrics = self._base_metrics()
        if not jtl_path.exists():
            logger.error("JMeter JTL not found at %s", jtl_path)
            return metrics

        latencies: List[float] = []
        errors = 0
        first_ts: Optional[int] = None
        last_ts: Optional[int] = None

        with jtl_path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    elapsed = float(row.get("elapsed", "") or 0.0)
                    success = str(row.get("success", "")).strip().lower() == "true"
                    ts = int(row.get("timeStamp", "") or 0)
                except Exception:
                    continue
                latencies.append(elapsed)
                if not success:
                    errors += 1
                if ts:
                    first_ts = ts if first_ts is None else min(first_ts, ts)
                    last_ts = ts if last_ts is None else max(last_ts, ts)

        n = len(latencies)
        if n == 0:
            return metrics

        metrics["samples"] = n
        metrics["avg_latency_ms"] = sum(latencies) / n
        pct = self._percentiles(latencies, [50, 90, 95, 99])
        metrics["p50_ms"] = pct["p50"]
        metrics["p90_ms"] = pct["p90"]
        metrics["p95_ms"] = pct["p95"]
        metrics["p99_ms"] = pct["p99"]
        metrics["error_pct"] = (errors / n) * 100.0

        if first_ts and last_ts and last_ts > first_ts:
            duration_s = (last_ts - first_ts) / 1000.0
            metrics["throughput_rps"] = round(n / duration_s, 3)
        metrics["success_rate_pct"] = round(100.0 - metrics["error_pct"], 3)
        return metrics

    def _parse_jmeter_json(self, jtl_path: Path) -> Dict[str, Any]:
        """
        Basic JSON JTL support (JSONL or array). Looks for keys: elapsed, success, timeStamp.
        Note: JSON JTLs can be very large; consider CSV for lighter pipelines.
        """
        metrics = self._base_metrics()
        if not jtl_path.exists():
            logger.error("JMeter JSON JTL not found at %s", jtl_path)
            return metrics

        latencies: List[float] = []
        errors = 0
        first_ts: Optional[int] = None
        last_ts: Optional[int] = None

        # Try JSONL first
        try:
            with jtl_path.open("r", encoding="utf-8") as fh:
                first_char = fh.read(1)
                fh.seek(0)
                if first_char == "[":
                    # JSON array
                    data = json.load(fh)
                    iterable = data
                else:
                    # JSON lines
                    iterable = (json.loads(line) for line in fh if line.strip())

                for row in iterable:
                    elapsed = float(row.get("elapsed", 0.0))
                    success = str(row.get("success", "")).strip().lower() == "true"
                    ts = int(row.get("timeStamp", 0))
                    latencies.append(elapsed)
                    if not success:
                        errors += 1
                    if ts:
                        first_ts = ts if first_ts is None else min(first_ts, ts)
                        last_ts = ts if last_ts is None else max(last_ts, ts)
        except Exception as e:
            logger.exception("Failed to parse JMeter JSON JTL: %s", e)
            return metrics

        n = len(latencies)
        if n == 0:
            return metrics

        metrics["samples"] = n
        metrics["avg_latency_ms"] = sum(latencies) / n
        pct = self._percentiles(latencies, [50, 90, 95, 99])
        metrics["p50_ms"] = pct["p50"]
        metrics["p90_ms"] = pct["p90"]
        metrics["p95_ms"] = pct["p95"]
        metrics["p99_ms"] = pct["p99"]
        metrics["error_pct"] = (errors / n) * 100.0

        if first_ts and last_ts and last_ts > first_ts:
            duration_s = (last_ts - first_ts) / 1000.0
            metrics["throughput_rps"] = round(n / duration_s, 3)
        metrics["success_rate_pct"] = round(100.0 - metrics["error_pct"], 3)
        return metrics

    # --------------------- Parsers: Locust ---------------------
    def _parse_locust_csvs(self, csv_prefix: Path) -> Dict[str, Any]:
        """
        Parse Locust aggregate CSV (requests.csv) for latency percentiles and RPS.
        Columns vary by version; prefer header names.
        """
        metrics: Dict[str, Any] = {
            "avg_latency_ms": None,
            "p50_ms": None,
            "p90_ms": None,
            "p95_ms": None,
            "p99_ms": None,
            "requests_per_sec": None,
            "fail_ratio": None,
            "samples": 0,
        }
        requests_csv = csv_prefix.with_name(csv_prefix.name + "_requests.csv")
        if not requests_csv.exists():
            logger.error("Locust requests CSV not found at %s", requests_csv)
            return metrics

        total_reqs = 0
        total_fails = 0
        p50_list: List[float] = []
        p90_list: List[float] = []
        p95_list: List[float] = []
        p99_list: List[float] = []
        avg_list: List[float] = []
        combined_rps = 0.0

        with requests_csv.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                try:
                    reqs = int(row.get("# requests", "0") or 0)
                    fails = int(row.get("# failures", "0") or 0)
                    p50 = float(row.get("Median response time", "nan"))
                    avg = float(row.get("Average response time", "nan"))
                    p90 = float(row.get("90%", "nan"))
                    p95 = float(row.get("95%", "nan"))
                    p99 = float(row.get("99%", "nan"))
                    rps = float(row.get("Requests/s", "nan"))
                except Exception:
                    continue

                total_reqs += reqs
                total_fails += fails
                if p50 == p50:
                    p50_list.append(p50)
                if avg == avg:
                    avg_list.append(avg)
                if p90 == p90:
                    p90_list.append(p90)
                if p95 == p95:
                    p95_list.append(p95)
                if p99 == p99:
                    p99_list.append(p99)
                if rps == rps:
                    combined_rps += rps

        metrics["samples"] = total_reqs
        metrics["p50_ms"] = median(p50_list) if p50_list else None
        metrics["p90_ms"] = median(p90_list) if p90_list else None
        metrics["p95_ms"] = median(p95_list) if p95_list else None
        metrics["p99_ms"] = median(p99_list) if p99_list else None
        metrics["avg_latency_ms"] = (sum(avg_list) / len(avg_list)) if avg_list else None
        metrics["fail_ratio"] = (total_fails / total_reqs * 100.0) if total_reqs else None
        metrics["requests_per_sec"] = combined_rps or None
        return metrics

    # --------------------- Public API ---------------------
    def run_suite(self, suite: dict) -> Dict[str, Any]:
        """
        suite example:
        {
          "name": "Checkout Load Test",
          "tool": "jmeter",
          "file": "tests/checkout.jmx",
          "timeout_sec": 1800,
          "jmeter_output": "csv",           # or "json"
          "jmeter_args": ["-Jthreads=50"],  # extra engine args
          "locust": {"script": "locustfile.py", "users": 100, "spawn": 10, "run_time": "5m", "host": "https://api.example.com"},
          "sla": {"p95_ms": 500, "error_pct_max": 1, "rps_min": 50}
        }
        """
        name = suite.get("name", "Unnamed Performance Test")
        tool = suite.get("tool", "jmeter").lower()
        suite_dir = self.workspace / (name.replace(" ", "_") + "_" + datetime.datetime.now().strftime("%Y%m%dT%H%M%S"))
        suite_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Starting Performance Suite: %s (tool=%s)", name, tool)
        start = datetime.datetime.now()

        if tool == "jmeter":
            result = self._run_jmeter(
                test_file=suite.get("file", "test.jmx"),
                suite_dir=suite_dir,
                output_format=suite.get("jmeter_output", self.TOOLS["jmeter"]["output_format"]),
                extra_args=suite.get("jmeter_args"),
                timeout_sec=suite.get("timeout_sec"),
            )
        elif tool == "locust":
            loc = suite.get("locust", {}) or {}
            result = self._run_locust(
                script=loc.get("script", suite.get("script", "locustfile.py")),
                suite_dir=suite_dir,
                users=int(loc.get("users", suite.get("users", 10))),
                spawn=int(loc.get("spawn", suite.get("spawn", 2))),
                run_time=str(loc.get("run_time", suite.get("run_time", "1m"))),
                host=loc.get("host", suite.get("host")),
                extra_args=loc.get("args") or suite.get("locust_args"),
                timeout_sec=suite.get("timeout_sec"),
            )
        else:
            result = {
                "name": name,
                "tool": tool,
                "error": f"Unsupported tool: {tool}",
                "duration_sec": 0.0,
            }
            logger.error("Unsupported tool: %s", tool)
            return result

        duration = (datetime.datetime.now() - start).total_seconds()
        result["name"] = name
        result["duration_sec"] = duration

        # SLA evaluation (optional)
        sla_cfg = suite.get("sla")
        metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
        result["sla"] = self._eval_sla(metrics, sla_cfg)

        # CI-friendly one-liner
        logger.info(
            "Suite '%s' completed in %.2fs (tool=%s, SLA=%s)",
            name, duration, result.get("tool"), result.get("sla", {}).get("passed")
        )
        return result

    def run_all(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        plan example:
        {
          "suites": [
            { ...suite1... },
            { ...suite2... }
          ]
        }
        """
        results: List[Dict[str, Any]] = []
        for suite in plan.get("suites", []):
            try:
                results.append(self.run_suite(suite))
            except Exception as e:
                logger.exception("Suite crashed: %s", suite.get("name"))
                results.append({
                    "name": suite.get("name", "Unnamed Performance Test"),
                    "tool": suite.get("tool"),
                    "error": str(e),
                    "duration_sec": 0.0,
                })
        return results
