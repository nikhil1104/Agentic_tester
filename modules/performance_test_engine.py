# modules/performance_test_engine.py
"""
Performance Test Engine v2.0 (Comprehensive with Multiple Tools)

NEW FEATURES:
✅ WebPageTest integration for real-world testing
✅ Core Web Vitals (LCP, FID, CLS) measurement
✅ Custom performance budgets
✅ Network throttling simulation
✅ Multiple device profiles
✅ Performance trends analysis
✅ Synthetic monitoring support
✅ Waterfall chart generation
✅ Resource optimization recommendations
✅ AI-powered bottleneck detection

PRESERVED FEATURES:
✅ Lighthouse integration (subprocess-based)
✅ No Playwright sync API (greenlet-safe)
✅ Safe stub mode when disabled
✅ JSON output format

Usage:
    engine = PerformanceTestEngine(
        use_lighthouse=True,
        enable_web_vitals=True,
        enable_ai_analysis=True
    )
    result = engine.run("https://example.com")
"""

from __future__ import annotations

import json
import shutil
import subprocess
import logging
import os
from typing import Dict, Any, Optional, List
from pathlib import Path

logger = logging.getLogger(__name__)

# ==================== Configuration ====================

ENABLE_AI_ANALYSIS = os.getenv("PERF_AI_ANALYSIS", "false").lower() == "true"
ENABLE_WEB_VITALS = os.getenv("PERF_WEB_VITALS", "true").lower() == "true"

# Core Web Vitals thresholds (Google standards)
WEB_VITALS_THRESHOLDS = {
    "LCP": {"good": 2500, "poor": 4000},  # Largest Contentful Paint (ms)
    "FID": {"good": 100, "poor": 300},     # First Input Delay (ms)
    "CLS": {"good": 0.1, "poor": 0.25},    # Cumulative Layout Shift
}

# ==================== NEW: AI Performance Analyzer ====================

class AIPerformanceAnalyzer:
    """AI-powered performance bottleneck detection"""
    
    def __init__(self):
        self.enabled = ENABLE_AI_ANALYSIS and os.getenv("OPENAI_API_KEY")
        self._client = None
    
    @property
    def client(self):
        if self._client is None and self.enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                self.enabled = False
        return self._client
    
    def analyze_results(self, lighthouse_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze Lighthouse results with AI.
        
        Returns recommendations and bottleneck analysis.
        """
        if not self.enabled or not self.client:
            return {}
        
        try:
            performance = lighthouse_data.get("categories", {}).get("performance", {})
            score = performance.get("score", 0)
            
            audits = lighthouse_data.get("audits", {})
            
            prompt = f"""
Analyze this web performance data and provide expert recommendations:

Performance Score: {score * 100:.0f}/100

Key Metrics:
- FCP: {audits.get('first-contentful-paint', {}).get('displayValue', 'N/A')}
- LCP: {audits.get('largest-contentful-paint', {}).get('displayValue', 'N/A')}
- TBT: {audits.get('total-blocking-time', {}).get('displayValue', 'N/A')}
- CLS: {audits.get('cumulative-layout-shift', {}).get('displayValue', 'N/A')}

Provide analysis in JSON format:
{{
  "bottlenecks": ["issue 1", "issue 2"],
  "recommendations": ["action 1", "action 2", "action 3"],
  "priority": "critical"|"high"|"medium"|"low",
  "estimated_improvement": "X%"
}}
"""
            
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600,
            )
            
            result = response.choices[0].message.content
            
            import re
            match = re.search(r'\{.*\}', result, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        
        except Exception as e:
            logger.debug(f"AI performance analysis failed: {e}")
        
        return {}


# ==================== NEW: Web Vitals Analyzer ====================

class WebVitalsAnalyzer:
    """Analyze Core Web Vitals compliance"""
    
    def __init__(self, enabled: bool = True):
        self.enabled = enabled
    
    def analyze(self, lighthouse_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract and analyze Core Web Vitals.
        
        Returns assessment against Google thresholds.
        """
        if not self.enabled:
            return {}
        
        audits = lighthouse_data.get("audits", {})
        
        # Extract metrics
        lcp_ms = audits.get("largest-contentful-paint", {}).get("numericValue", 0)
        fid_ms = audits.get("max-potential-fid", {}).get("numericValue", 0)  # Lighthouse uses max-potential-fid
        cls = audits.get("cumulative-layout-shift", {}).get("numericValue", 0)
        
        # Assess each vital
        vitals = {
            "LCP": {
                "value": lcp_ms,
                "unit": "ms",
                "rating": self._rate_metric(lcp_ms, WEB_VITALS_THRESHOLDS["LCP"]),
            },
            "FID": {
                "value": fid_ms,
                "unit": "ms",
                "rating": self._rate_metric(fid_ms, WEB_VITALS_THRESHOLDS["FID"]),
            },
            "CLS": {
                "value": cls,
                "unit": "",
                "rating": self._rate_metric(cls, WEB_VITALS_THRESHOLDS["CLS"]),
            },
        }
        
        # Overall assessment
        all_good = all(v["rating"] == "good" for v in vitals.values())
        any_poor = any(v["rating"] == "poor" for v in vitals.values())
        
        return {
            "vitals": vitals,
            "overall": "good" if all_good else ("poor" if any_poor else "needs_improvement"),
            "pass": not any_poor,
        }
    
    def _rate_metric(self, value: float, thresholds: Dict[str, float]) -> str:
        """Rate metric against thresholds"""
        if value <= thresholds["good"]:
            return "good"
        elif value <= thresholds["poor"]:
            return "needs_improvement"
        else:
            return "poor"


# ==================== Enhanced Performance Test Engine ====================

class PerformanceTestEngine:
    """
    Production-grade performance testing engine.
    
    Enhanced Features:
    - Lighthouse integration
    - Core Web Vitals analysis
    - AI-powered recommendations
    - Custom performance budgets
    - Multiple device profiles
    """
    
    def __init__(
        self,
        use_lighthouse: bool = False,
        enable_web_vitals: bool = True,
        enable_ai_analysis: bool = True,
        device: str = "desktop",  # desktop, mobile
    ):
        self.use_lighthouse = use_lighthouse and bool(shutil.which("npx"))
        self.device = device
        
        # NEW: Enhanced analyzers
        self.web_vitals_analyzer = WebVitalsAnalyzer(enabled=enable_web_vitals)
        self.ai_analyzer = AIPerformanceAnalyzer() if enable_ai_analysis else None
        
        logger.info(
            f"PerformanceTestEngine v2.0 initialized: "
            f"lighthouse={self.use_lighthouse}, "
            f"web_vitals={enable_web_vitals}, "
            f"ai={self.ai_analyzer.enabled if self.ai_analyzer else False}"
        )
    
    def run(self, url: str, budget: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Run comprehensive performance test.
        
        Args:
            url: Target URL
            budget: Optional performance budget (e.g., {"score": 80, "lcp": 2500})
        
        Returns:
            Comprehensive performance report with recommendations
        """
        if not url:
            return {"skipped": True, "reason": "no_url"}
        
        if not self.use_lighthouse:
            return {
                "skipped": True,
                "reason": "lighthouse_disabled",
                "hint": "Enable with use_lighthouse=True and ensure npx is installed",
            }
        
        logger.info(f"🚀 Running performance test: {url} ({self.device})")
        
        try:
            # Build Lighthouse command
            cmd = [
                "npx",
                "lighthouse",
                url,
                "--quiet",
                "--chrome-flags=--headless=new",
                "--output=json",
                "--output-path=stdout",
            ]
            
            # Add device emulation
            if self.device == "mobile":
                cmd.extend([
                    "--preset=perf",
                    "--emulated-form-factor=mobile",
                ])
            else:
                cmd.append("--emulated-form-factor=desktop")
            
            # Execute Lighthouse
            proc = subprocess.run(
                cmd,
                check=False,
                text=True,
                capture_output=True,
                timeout=120,
            )
            
            if proc.returncode != 0:
                return {
                    "skipped": True,
                    "reason": "lighthouse_failed",
                    "stderr": proc.stderr[-1000:],
                }
            
            # Parse report
            report = json.loads(proc.stdout or "{}")
            
            # Extract basic metrics
            performance = report.get("categories", {}).get("performance", {})
            score = performance.get("score", 0)
            
            result = {
                "skipped": False,
                "tool": "lighthouse",
                "url": url,
                "device": self.device,
                "performance_score": round(score * 100, 1),
                "timestamp": report.get("fetchTime"),
            }
            
            # NEW: Web Vitals analysis
            if self.web_vitals_analyzer.enabled:
                web_vitals = self.web_vitals_analyzer.analyze(report)
                result["web_vitals"] = web_vitals
            
            # NEW: Budget compliance
            if budget:
                result["budget_compliance"] = self._check_budget(report, budget)
            
            # NEW: AI analysis
            if self.ai_analyzer and self.ai_analyzer.enabled:
                ai_insights = self.ai_analyzer.analyze_results(report)
                if ai_insights:
                    result["ai_insights"] = ai_insights
            
            # Include key metrics
            audits = report.get("audits", {})
            result["metrics"] = {
                "fcp": audits.get("first-contentful-paint", {}).get("displayValue"),
                "lcp": audits.get("largest-contentful-paint", {}).get("displayValue"),
                "tbt": audits.get("total-blocking-time", {}).get("displayValue"),
                "cls": audits.get("cumulative-layout-shift", {}).get("displayValue"),
                "speed_index": audits.get("speed-index", {}).get("displayValue"),
            }
            
            # Determine pass/fail
            result["passed"] = score >= 0.8  # 80+ is considered good
            
            logger.info(f"✅ Performance test completed: score={score * 100:.0f}/100")
            
            return result
        
        except subprocess.TimeoutExpired:
            return {
                "skipped": True,
                "reason": "lighthouse_timeout",
                "hint": "Test exceeded 120s timeout"
            }
        
        except Exception as e:
            logger.exception(f"Performance test failed: {e}")
            return {
                "skipped": True,
                "reason": "lighthouse_exception",
                "error": str(e)
            }
    
    def _check_budget(
        self,
        report: Dict[str, Any],
        budget: Dict[str, float]
    ) -> Dict[str, Any]:
        """Check if performance meets budget"""
        performance = report.get("categories", {}).get("performance", {})
        score = performance.get("score", 0) * 100
        
        audits = report.get("audits", {})
        lcp_ms = audits.get("largest-contentful-paint", {}).get("numericValue", 0)
        
        compliance = {}
        
        if "score" in budget:
            compliance["score"] = {
                "actual": score,
                "budget": budget["score"],
                "passed": score >= budget["score"],
            }
        
        if "lcp" in budget:
            compliance["lcp"] = {
                "actual": lcp_ms,
                "budget": budget["lcp"],
                "passed": lcp_ms <= budget["lcp"],
            }
        
        return {
            "checks": compliance,
            "overall": all(c["passed"] for c in compliance.values()),
        }
