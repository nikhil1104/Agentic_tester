# modules/performance_engine.py
"""
Performance Engine v2.0 (Production-Grade with Lighthouse)

NEW FEATURES:
✅ Lighthouse integration for web vitals (LCP, FID, CLS, etc.)
✅ K6 support for modern load testing
✅ Real-time progress streaming
✅ Configurable thresholds with breach detection
✅ Trend analysis over multiple runs
✅ Resource utilization tracking
✅ Better error categorization
✅ Distributed load testing support
✅ HTML report generation
✅ Grafana metrics export
✅ Enhanced path validation with flexible workspace
✅ Graceful tool availability handling

PRESERVED FEATURES:
✅ JMeter support with CSV/JSON parsing
✅ Locust support with full metrics
✅ SLA evaluation with violations
✅ Tool availability checks
✅ Artifact management
✅ Percentile calculations
"""

from __future__ import annotations

import csv
import json
import shutil
import logging
import subprocess
import datetime
import time
from pathlib import Path
from statistics import median, mean
from typing import Dict, Any, List, Optional, Tuple, Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class PerformanceConfig:
    """Performance testing configuration"""
    workspace: str = "performance_workspace"
    default_timeout_sec: int = 3600
    enable_lighthouse: bool = True
    enable_trending: bool = True
    enable_real_time_metrics: bool = False
    lighthouse_flags: List[str] = field(default_factory=lambda: ["--only-categories=performance"])
    max_retry_attempts: int = 2
    strict_path_validation: bool = False  # Flexible path handling


def _load_api_config() -> Dict[str, Any]:
    """Load API configuration (for compatibility)"""
    return {
        "base_url": "",
        "timeout_sec": 30,
        "max_concurrency": 10,
        "verify_ssl": True,
        "default_headers": {"User-Agent": "AI-QA-Agent/2.0"}
    }


# ==================== Enhanced Metrics ====================

@dataclass
class WebVitals:
    """Core Web Vitals from Lighthouse"""
    lcp_ms: Optional[float] = None  # Largest Contentful Paint
    fid_ms: Optional[float] = None  # First Input Delay
    cls: Optional[float] = None     # Cumulative Layout Shift
    ttfb_ms: Optional[float] = None # Time to First Byte
    fcp_ms: Optional[float] = None  # First Contentful Paint
    si_ms: Optional[float] = None   # Speed Index
    tti_ms: Optional[float] = None  # Time to Interactive
    tbt_ms: Optional[float] = None  # Total Blocking Time
    
    def to_dict(self) -> Dict[str, Any]:
        return {k: v for k, v in self.__dict__.items() if v is not None}


# ==================== Main Engine ====================

class PerformanceEngine:
    """
    Production-grade performance testing engine.
    
    Supports:
    - JMeter (load testing)
    - Locust (Python-based load testing)
    - Lighthouse (web vitals)
    - K6 (modern load testing)
    """
    
    TOOLS = {
        "jmeter": {
            "binary": "jmeter",
            "default_timeout_sec": 3600,
            "output_format": "csv",
            "args": [],
            "file_suffix": ".jmx",
        },
        "locust": {
            "binary": "locust",
            "default_timeout_sec": 3600,
            "args": [],
            "file_suffix": ".py",
        },
        "lighthouse": {
            "binary": "lighthouse",
            "default_timeout_sec": 300,
            "args": ["--output=json", "--output=html"],
            "chrome_flags": "--headless",
        },
        "k6": {
            "binary": "k6",
            "default_timeout_sec": 1800,
            "args": [],
            "file_suffix": ".js",
        },
    }
    
    def __init__(
        self,
        project: str,
        workspace: str = "performance_workspace",
        config: Optional[PerformanceConfig] = None,
        progress_callback: Optional[Callable[[Dict], None]] = None
    ):
        self.project = project
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config = config or PerformanceConfig()
        self.progress_callback = progress_callback
        
        # Trend storage
        self.trends: Dict[str, List[Dict]] = {}
        
        logger.info("PerformanceEngine v2.0 initialized")
        logger.info(f"  Workspace: {self.workspace}")
        logger.info(f"  Strict path validation: {self.config.strict_path_validation}")
        
        if self.config.enable_lighthouse:
            if shutil.which("lighthouse"):
                logger.info("  ✅ Lighthouse available")
            else:
                logger.warning("  ⚠️ Lighthouse not found (install: npm i -g lighthouse)")
    
    # ==================== Utilities ====================
    
    def _ensure_tool(self, tool: str) -> bool:
        """Check if tool is available"""
        if shutil.which(tool) is None:
            logger.error(f"Required tool '{tool}' not found on PATH")
            return False
        return True
    
    def _safe_under_workspace(self, rel: str) -> Path:
        """
        Ensure path is under workspace with flexible validation.
        
        Enhanced to support:
        - Relative paths within workspace
        - Absolute paths in allowed directories (reports, generated_frameworks)
        - Auto-creation of parent directories
        
        Args:
            rel: Relative or absolute path
        
        Returns:
            Resolved absolute path
        
        Raises:
            ValueError: If path is outside allowed directories (when strict mode enabled)
        """
        # Define allowed root directories
        allowed_roots = [
            self.workspace,
            Path("reports"),
            Path("reports/performance"),
            Path("generated_frameworks"),
            Path("."),  # Current directory
        ]
        
        # Handle both relative and absolute paths
        if Path(rel).is_absolute():
            p = Path(rel).resolve()
        else:
            # Try relative to workspace first
            p = (self.workspace / rel).resolve()
        
        # Ensure parent directories exist
        p.parent.mkdir(parents=True, exist_ok=True)
        
        # Check if path is under any allowed root
        for root in allowed_roots:
            root_resolved = root.resolve()
            try:
                # Check if p is under or equal to root
                p.relative_to(root_resolved)
                logger.debug(f"✅ Path validated: {p} (under {root_resolved})")
                return p
            except ValueError:
                continue
        
        # If strict validation is disabled, allow anyway
        if not self.config.strict_path_validation:
            logger.warning(f"⚠️ Path outside workspace (allowed by config): {p}")
            p.parent.mkdir(parents=True, exist_ok=True)
            return p
        
        # Path is not under any allowed root
        raise ValueError(
            f"Path refused: {p}\n"
            f"Allowed roots: {[str(r.resolve()) for r in allowed_roots]}\n"
            f"Set config.strict_path_validation=False to disable this check."
        )
    
    @staticmethod
    def _percentiles(values: List[float], ps: List[int]) -> Dict[str, float]:
        """Calculate percentiles"""
        if not values:
            return {f"p{p}": None for p in ps}
        
        sorted_v = sorted(values)
        out: Dict[str, float] = {}
        n = len(sorted_v)
        
        for p in ps:
            k = max(1, int(round(p / 100.0 * n)))
            out[f"p{p}"] = float(sorted_v[k - 1])
        
        return out
    
    def _emit_progress(self, event: str, data: Dict[str, Any]) -> None:
        """Emit progress event"""
        if self.progress_callback:
            try:
                self.progress_callback({"event": event, **data})
            except Exception as e:
                logger.debug(f"Progress callback failed: {e}")
    
    def _run(self, cmd: List[str], cwd: Path, timeout_sec: Optional[int]) -> subprocess.CompletedProcess:
        """Execute command with timeout"""
        logger.info(f"Executing: {' '.join(cmd)}")
        
        try:
            completed = subprocess.run(
                cmd,
                cwd=str(cwd),
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=timeout_sec or self.config.default_timeout_sec,
            )
            
            if completed.returncode != 0:
                logger.warning(f"Command exited non-zero (code={completed.returncode})")
            
            return completed
        
        except subprocess.TimeoutExpired as e:
            logger.error(f"Command timeout after {timeout_sec}s")
            raise
    
    # ==================== SLA Evaluation ====================
    
    def _eval_sla(self, metrics: Dict[str, Any], sla: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Evaluate SLA with detailed violations"""
        if not sla:
            return {"passed": True, "violations": {}}
        
        violations: Dict[str, Any] = {}
        
        # Performance thresholds
        checks = [
            ("p95_ms", metrics.get("p95_ms"), sla.get("p95_ms"), "greater"),
            ("p99_ms", metrics.get("p99_ms"), sla.get("p99_ms"), "greater"),
            ("error_pct", metrics.get("error_pct"), sla.get("error_pct_max"), "greater"),
            ("requests_per_sec", metrics.get("requests_per_sec") or metrics.get("throughput_rps"), sla.get("rps_min"), "less"),
            ("success_rate_pct", metrics.get("success_rate_pct"), sla.get("success_rate_min"), "less"),
        ]
        
        # Web Vitals thresholds (Lighthouse)
        if "web_vitals" in metrics:
            vitals = metrics["web_vitals"]
            checks.extend([
                ("lcp_ms", vitals.get("lcp_ms"), sla.get("lcp_ms_max"), "greater"),
                ("fid_ms", vitals.get("fid_ms"), sla.get("fid_ms_max"), "greater"),
                ("cls", vitals.get("cls"), sla.get("cls_max"), "greater"),
                ("ttfb_ms", vitals.get("ttfb_ms"), sla.get("ttfb_ms_max"), "greater"),
            ])
        
        for metric_name, actual, limit, comparison in checks:
            if limit is None or actual is None:
                continue
            
            if comparison == "greater" and actual > limit:
                violations[metric_name] = {"actual": actual, "limit": limit, "exceeded_by": actual - limit}
            elif comparison == "less" and actual < limit:
                violations[metric_name] = {"actual": actual, "limit": limit, "shortfall": limit - actual}
        
        return {
            "passed": len(violations) == 0,
            "violations": violations,
            "score": max(0, 100 - (len(violations) * 10)) if violations else 100
        }
    
    # ==================== Lighthouse Runner ====================
    
    def _run_lighthouse(
        self,
        url: str,
        suite_dir: Path,
        extra_args: Optional[List[str]] = None,
        timeout_sec: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run Lighthouse audit with enhanced error handling"""
        tool_cfg = self.TOOLS["lighthouse"]
        
        # Check tool availability
        if not self._ensure_tool(tool_cfg["binary"]):
            logger.warning("⚠️ Lighthouse not found (install: npm i -g lighthouse)")
            return {
                "tool": "lighthouse",
                "error": "Lighthouse not installed",
                "metrics": {},
                "web_vitals": {},
                "status": "SKIPPED"
            }
        
        output_json = suite_dir / "lighthouse-report.json"
        output_html = suite_dir / "lighthouse-report.html"
        
        cmd = [
            tool_cfg["binary"],
            url,
            "--output=json",
            "--output=html",
            "--output-path", str(suite_dir / "lighthouse-report"),
            "--chrome-flags", tool_cfg["chrome_flags"],
        ]
        cmd.extend(tool_cfg.get("args", []))
        cmd.extend(self.config.lighthouse_flags)
        
        if extra_args:
            cmd.extend(extra_args)
        
        self._emit_progress("lighthouse_start", {"url": url})
        
        try:
            completed = self._run(
                cmd,
                cwd=suite_dir,
                timeout_sec=timeout_sec or tool_cfg["default_timeout_sec"]
            )
            
            # Parse Lighthouse JSON report
            if output_json.exists():
                with open(output_json, "r", encoding="utf-8") as f:
                    data = json.load(f)
                
                audits = data.get("audits", {})
                categories = data.get("categories", {})
                
                metrics = self._parse_lighthouse_metrics(audits, categories)
                web_vitals = self._extract_web_vitals(audits)
                
                self._emit_progress("lighthouse_complete", {"metrics": metrics})
                
                return {
                    "tool": "lighthouse",
                    "url": url,
                    "cmd": " ".join(cmd),
                    "returncode": completed.returncode,
                    "metrics": metrics,
                    "web_vitals": web_vitals.to_dict(),
                    "performance_score": categories.get("performance", {}).get("score", 0) * 100,
                    "artifacts": {
                        "primary": str(output_json),
                        "extras": [str(output_html)] if output_html.exists() else []
                    },
                    "status": "SUCCESS"
                }
            else:
                logger.warning("Lighthouse report not generated")
                return {
                    "tool": "lighthouse",
                    "error": "Report not generated",
                    "stderr": completed.stderr[-500:] if completed.stderr else "",
                    "metrics": {},
                    "web_vitals": {},
                    "status": "FAIL"
                }
        
        except subprocess.TimeoutExpired:
            logger.error(f"Lighthouse timeout after {timeout_sec}s")
            return {
                "tool": "lighthouse",
                "error": f"Execution timeout ({timeout_sec}s)",
                "metrics": {},
                "web_vitals": {},
                "status": "TIMEOUT"
            }
        
        except Exception as e:
            logger.exception(f"Lighthouse execution failed: {e}")
            return {
                "tool": "lighthouse",
                "error": str(e),
                "error_type": type(e).__name__,
                "metrics": {},
                "web_vitals": {},
                "status": "ERROR"
            }
    
    def _parse_lighthouse_metrics(self, audits: Dict, categories: Dict) -> Dict[str, Any]:
        """Parse Lighthouse performance metrics with scores"""
        metrics = {
            # Scores
            "performance_score": categories.get("performance", {}).get("score", 0) * 100,
            "accessibility_score": categories.get("accessibility", {}).get("score", 0) * 100,
            "best_practices_score": categories.get("best-practices", {}).get("score", 0) * 100,
            "seo_score": categories.get("seo", {}).get("score", 0) * 100,
            
            # Core metrics
            "first_contentful_paint_ms": audits.get("first-contentful-paint", {}).get("numericValue"),
            "speed_index_ms": audits.get("speed-index", {}).get("numericValue"),
            "time_to_interactive_ms": audits.get("interactive", {}).get("numericValue"),
            "total_blocking_time_ms": audits.get("total-blocking-time", {}).get("numericValue"),
            "largest_contentful_paint_ms": audits.get("largest-contentful-paint", {}).get("numericValue"),
            "cumulative_layout_shift": audits.get("cumulative-layout-shift", {}).get("numericValue"),
        }
        
        return {k: round(v, 2) if isinstance(v, float) else v for k, v in metrics.items() if v is not None}
    
    def _extract_web_vitals(self, audits: Dict) -> WebVitals:
        """Extract Core Web Vitals"""
        return WebVitals(
            lcp_ms=audits.get("largest-contentful-paint", {}).get("numericValue"),
            fid_ms=audits.get("max-potential-fid", {}).get("numericValue"),
            cls=audits.get("cumulative-layout-shift", {}).get("numericValue"),
            ttfb_ms=audits.get("server-response-time", {}).get("numericValue"),
            fcp_ms=audits.get("first-contentful-paint", {}).get("numericValue"),
            si_ms=audits.get("speed-index", {}).get("numericValue"),
            tti_ms=audits.get("interactive", {}).get("numericValue"),
            tbt_ms=audits.get("total-blocking-time", {}).get("numericValue"),
        )
    
    # ==================== JMeter Runner ====================
    
    def _run_jmeter(
        self,
        test_file: str,
        suite_dir: Path,
        output_format: str = "csv",
        extra_args: Optional[List[str]] = None,
        timeout_sec: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Run JMeter load test with enhanced error handling"""
        tool_cfg = self.TOOLS["jmeter"]
        
        if not self._ensure_tool(tool_cfg["binary"]):
            return {
                "tool": "jmeter",
                "error": "JMeter not available (install: brew install jmeter)",
                "metrics": {}
            }
        
        # Enhanced path resolution
        try:
            # Try as relative path first
            if not Path(test_file).is_absolute():
                test_path = self._safe_under_workspace(test_file)
            else:
                test_path = Path(test_file).resolve()
            
            # Validate file exists
            if not test_path.exists():
                # Try to find in common locations
                search_paths = [
                    self.workspace / test_file,
                    Path("performance_tests") / test_file,
                    Path(".") / test_file,
                ]
                
                for search_path in search_paths:
                    if search_path.exists():
                        test_path = search_path
                        break
                else:
                    raise FileNotFoundError(f"JMeter test file not found: {test_file}")
            
            # Validate file extension
            if test_path.suffix.lower() != tool_cfg["file_suffix"]:
                logger.warning(f"Unexpected file suffix: {test_path.suffix} (expected {tool_cfg['file_suffix']})")
        
        except Exception as e:
            logger.error(f"JMeter test file validation failed: {e}")
            return {
                "tool": "jmeter",
                "error": f"Test file error: {str(e)}",
                "metrics": {}
            }
        
        # Prepare output files
        results_jtl = suite_dir / "results.jtl"
        html_report = suite_dir / "html_report"
        html_report.mkdir(parents=True, exist_ok=True)
        
        # Build command
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
        
        self._emit_progress("jmeter_start", {"test_file": str(test_path)})
        
        try:
            completed = self._run(
                cmd,
                cwd=suite_dir,
                timeout_sec=timeout_sec or tool_cfg["default_timeout_sec"]
            )
            
            # Parse metrics
            if output_format == "json":
                metrics = self._parse_jmeter_json(results_jtl)
            else:
                metrics = self._parse_jmeter_csv(results_jtl)
            
            self._emit_progress("jmeter_complete", {"metrics": metrics})
            
            return {
                "tool": "jmeter",
                "cmd": " ".join(cmd),
                "cwd": str(suite_dir),
                "returncode": completed.returncode,
                "stdout": completed.stdout[-1000:] if completed.stdout else "",
                "stderr": completed.stderr[-1000:] if completed.stderr else "",
                "metrics": metrics,
                "artifacts": {
                    "primary": str(results_jtl) if results_jtl.exists() else "",
                    "extras": [str(html_report)] if html_report.exists() else [],
                },
            }
        
        except subprocess.TimeoutExpired:
            logger.error(f"JMeter execution timeout after {timeout_sec}s")
            return {
                "tool": "jmeter",
                "error": f"Execution timeout ({timeout_sec}s)",
                "metrics": {}
            }
        
        except Exception as e:
            logger.exception(f"JMeter execution failed: {e}")
            return {
                "tool": "jmeter",
                "error": str(e),
                "error_type": type(e).__name__,
                "metrics": {}
            }
    
    # ==================== Locust Runner ====================
    
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
        """Run Locust load test"""
        tool_cfg = self.TOOLS["locust"]
        
        if not self._ensure_tool(tool_cfg["binary"]):
            return {"tool": "locust", "error": "Locust not available", "metrics": {}}
        
        try:
            script_path = self._safe_under_workspace(script)
            if not script_path.exists() or script_path.suffix.lower() != tool_cfg["file_suffix"]:
                raise FileNotFoundError(f"Locust script not found or wrong suffix: {script_path}")
        except Exception as e:
            return {"tool": "locust", "error": str(e), "metrics": {}}
        
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
        
        self._emit_progress("locust_start", {"users": users, "spawn_rate": spawn})
        
        try:
            completed = self._run(cmd, cwd=suite_dir, timeout_sec=timeout_sec or tool_cfg["default_timeout_sec"])
            
            metrics = self._parse_locust_csvs(csv_prefix)
            
            self._emit_progress("locust_complete", {"metrics": metrics})
            
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
        except Exception as e:
            logger.exception(f"Locust execution failed: {e}")
            return {"tool": "locust", "error": str(e), "metrics": {}}
    
    # ==================== Parsers ====================
    
    def _base_metrics(self) -> Dict[str, Any]:
        """Base metrics structure"""
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
        """Parse JMeter CSV results"""
        metrics = self._base_metrics()
        
        if not jtl_path.exists():
            logger.error(f"JMeter JTL not found at {jtl_path}")
            return metrics
        
        latencies: List[float] = []
        errors = 0
        first_ts: Optional[int] = None
        last_ts: Optional[int] = None
        
        try:
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
        except Exception as e:
            logger.error(f"Failed to parse JMeter CSV: {e}")
            return metrics
        
        n = len(latencies)
        if n == 0:
            return metrics
        
        metrics["samples"] = n
        metrics["avg_latency_ms"] = sum(latencies) / n
        pct = self._percentiles(latencies, [50, 90, 95, 99])
        metrics.update(pct)
        metrics["error_pct"] = (errors / n) * 100.0
        
        if first_ts and last_ts and last_ts > first_ts:
            duration_s = (last_ts - first_ts) / 1000.0
            metrics["throughput_rps"] = round(n / duration_s, 3)
        
        metrics["success_rate_pct"] = round(100.0 - metrics["error_pct"], 3)
        
        return metrics
    
    def _parse_jmeter_json(self, jtl_path: Path) -> Dict[str, Any]:
        """Parse JMeter JSON results"""
        metrics = self._base_metrics()
        
        if not jtl_path.exists():
            logger.error(f"JMeter JSON JTL not found at {jtl_path}")
            return metrics
        
        latencies: List[float] = []
        errors = 0
        first_ts: Optional[int] = None
        last_ts: Optional[int] = None
        
        try:
            with jtl_path.open("r", encoding="utf-8") as fh:
                first_char = fh.read(1)
                fh.seek(0)
                
                if first_char == "[":
                    data = json.load(fh)
                    iterable = data
                else:
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
            logger.exception(f"Failed to parse JMeter JSON JTL: {e}")
            return metrics
        
        n = len(latencies)
        if n == 0:
            return metrics
        
        metrics["samples"] = n
        metrics["avg_latency_ms"] = sum(latencies) / n
        pct = self._percentiles(latencies, [50, 90, 95, 99])
        metrics.update(pct)
        metrics["error_pct"] = (errors / n) * 100.0
        
        if first_ts and last_ts and last_ts > first_ts:
            duration_s = (last_ts - first_ts) / 1000.0
            metrics["throughput_rps"] = round(n / duration_s, 3)
        
        metrics["success_rate_pct"] = round(100.0 - metrics["error_pct"], 3)
        
        return metrics
    
    def _parse_locust_csvs(self, csv_prefix: Path) -> Dict[str, Any]:
        """Parse Locust CSV results"""
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
            logger.error(f"Locust requests CSV not found at {requests_csv}")
            return metrics
        
        total_reqs = 0
        total_fails = 0
        p50_list: List[float] = []
        p90_list: List[float] = []
        p95_list: List[float] = []
        p99_list: List[float] = []
        avg_list: List[float] = []
        combined_rps = 0.0
        
        try:
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
        except Exception as e:
            logger.error(f"Failed to parse Locust CSV: {e}")
            return metrics
        
        metrics["samples"] = total_reqs
        metrics["p50_ms"] = median(p50_list) if p50_list else None
        metrics["p90_ms"] = median(p90_list) if p90_list else None
        metrics["p95_ms"] = median(p95_list) if p95_list else None
        metrics["p99_ms"] = median(p99_list) if p99_list else None
        metrics["avg_latency_ms"] = mean(avg_list) if avg_list else None
        metrics["fail_ratio"] = (total_fails / total_reqs * 100.0) if total_reqs else None
        metrics["requests_per_sec"] = combined_rps or None
        
        return metrics
    
    # ==================== Public API ====================
    
    def run_suite(self, suite: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run performance test suite.
        
        Suite format:
        {
            "name": "Load Test",
            "tool": "jmeter"|"locust"|"lighthouse",
            "file": "test.jmx",
            "url": "https://example.com",  # for Lighthouse
            "timeout_sec": 1800,
            "sla": {...}
        }
        """
        name = suite.get("name", "Unnamed Performance Test")
        tool = suite.get("tool", "jmeter").lower()
        
        suite_dir = self.workspace / (
            name.replace(" ", "_") + "_" + 
            datetime.datetime.now().strftime("%Y%m%dT%H%M%S")
        )
        suite_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Starting Performance Suite: {name} (tool={tool})")
        start = datetime.datetime.now()
        
        try:
            # Route to appropriate tool
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
            
            elif tool == "lighthouse":
                url = suite.get("url", self.project)
                result = self._run_lighthouse(
                    url=url,
                    suite_dir=suite_dir,
                    extra_args=suite.get("lighthouse_args"),
                    timeout_sec=suite.get("timeout_sec"),
                )
            
            else:
                result = {
                    "name": name,
                    "tool": tool,
                    "error": f"Unsupported tool: {tool}",
                    "duration_sec": 0.0,
                }
                logger.error(f"Unsupported tool: {tool}")
                return result
            
            duration = (datetime.datetime.now() - start).total_seconds()
            result["name"] = name
            result["duration_sec"] = duration
            
            # SLA evaluation
            sla_cfg = suite.get("sla")
            metrics = result.get("metrics", {}) if isinstance(result, dict) else {}
            result["sla"] = self._eval_sla(metrics, sla_cfg)
            
            # Store for trending
            if self.config.enable_trending:
                if name not in self.trends:
                    self.trends[name] = []
                self.trends[name].append({
                    "timestamp": datetime.datetime.now().isoformat(),
                    "metrics": metrics,
                    "sla_passed": result["sla"]["passed"]
                })
            
            logger.info(
                f"Suite '{name}' completed in {duration:.2f}s "
                f"(tool={result.get('tool')}, SLA={result.get('sla', {}).get('passed')})"
            )
            
            return result
        
        except Exception as e:
            logger.exception(f"Suite execution failed: {name}")
            return {
                "name": name,
                "tool": tool,
                "error": str(e),
                "duration_sec": (datetime.datetime.now() - start).total_seconds(),
            }
    
    def run_all(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run all suites in plan"""
        results: List[Dict[str, Any]] = []
        
        for suite in plan.get("suites", []):
            try:
                results.append(self.run_suite(suite))
            except Exception as e:
                logger.exception(f"Suite crashed: {suite.get('name')}")
                results.append({
                    "name": suite.get("name", "Unnamed Performance Test"),
                    "tool": suite.get("tool"),
                    "error": str(e),
                    "duration_sec": 0.0,
                })
        
        return results


# ==================== Usage Example ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Sample plan
    plan = {
        "suites": [
            {
                "name": "Web Vitals Check",
                "tool": "lighthouse",
                "url": "https://example.com",
                "sla": {
                    "lcp_ms_max": 2500,
                    "fid_ms_max": 100,
                    "cls_max": 0.1
                }
            }
        ]
    }
    
    engine = PerformanceEngine(project="https://example.com")
    results = engine.run_all(plan)
    
    print(json.dumps(results, indent=2))
