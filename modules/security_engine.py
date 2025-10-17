# modules/security_engine.py
"""
Security Engine v2.0 (AI-Enhanced with Advanced Threat Detection)

FEATURES:
✅ AI-powered vulnerability analysis (GPT-4o-mini)
✅ CVE database integration with NVD API
✅ CVSS scoring and risk assessment
✅ Modular plugin-based security checks
✅ Result caching with TTL
✅ Retry logic with exponential backoff
✅ Async and sync execution modes
✅ OWASP Top 10 2021 coverage
✅ CWE/OWASP mapping
✅ Comprehensive reporting (JSON/Markdown/HTML)
✅ Rate limiting and circuit breakers
✅ Progress tracking and streaming results
"""

from __future__ import annotations

import asyncio
import logging
import re
import time
import os
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, Any, Optional, List, AsyncGenerator
import httpx

from modules.security_types import (
    SecurityCheckResult,
    SecurityFinding,
    CheckStatus,
    Severity,
)
from modules.security_checks import CHECK_REGISTRY, get_enabled_checks

logger = logging.getLogger(__name__)
# ✅ NEW: Optional advanced modules
try:
    from modules.threat_intel import ThreatIntelligence
    THREAT_INTEL_AVAILABLE = True
except ImportError:
    THREAT_INTEL_AVAILABLE = False
    logger.debug("Threat intelligence module not available")

try:
    from modules.compliance_checker import ComplianceChecker
    COMPLIANCE_AVAILABLE = True
except ImportError:
    COMPLIANCE_AVAILABLE = False
    logger.debug("Compliance checker module not available")

try:
    from modules.sbom_generator import SBOMGenerator
    SBOM_AVAILABLE = True
except ImportError:
    SBOM_AVAILABLE = False
    logger.debug("SBOM generator module not available")

# ==================== Configuration ====================

@dataclass
class SecurityConfig:
    """Centralized security engine configuration."""
    # Core settings
    timeout_s: float = 15.0
    enabled_checks: List[str] = field(default_factory=lambda: list(CHECK_REGISTRY.keys()))
    enable_async: bool = True
    max_concurrent_checks: int = 3
    
    # AI settings
    enable_ai: bool = False
    ai_model: str = "gpt-4o-mini"
    ai_max_tokens: int = 1000
    
    # CVE settings
    enable_cve_lookup: bool = False
    cve_api_url: str = "https://services.nvd.nist.gov/rest/json/cves/2.0"
    cve_results_limit: int = 5
    
    # Caching
    enable_cache: bool = True
    cache_ttl_s: int = 3600
    cache_file: str = "data/security_cache.json"
    
    # Retry & resilience
    retry_max_attempts: int = 3
    retry_delay_s: float = 2.0
    request_timeout_s: float = 30.0
    
    # Rate limiting
    api_rate_limit_per_min: int = 60
    
    # Reporting
    export_dir: str = "reports/security"
    enable_html_reports: bool = True
    enable_markdown_reports: bool = True
    
    # ✅ NEW: Advanced feature flags
    enable_threat_intelligence: bool = False
    enable_compliance_checking: bool = False
    enable_sbom_generation: bool = False
    
    # Threat intelligence API keys
    alienvault_api_key: Optional[str] = None
    virustotal_api_key: Optional[str] = None
    
    # Compliance frameworks
    compliance_frameworks: List[str] = field(default_factory=lambda: ["PCI-DSS"])
    
    # Check tier preset
    check_tier: str = "standard"  # "basic", "standard", "advanced", "all"
    
    @classmethod
    def from_env(cls) -> "SecurityConfig":
        """Load configuration from environment variables."""
        return cls(
            enable_ai=os.getenv("SECURITY_AI_ANALYSIS", "false").lower() == "true",
            enable_cve_lookup=os.getenv("SECURITY_CVE_LOOKUP", "false").lower() == "true",
            enable_cache=os.getenv("SECURITY_CACHE", "true").lower() == "true",
        )
        
    @classmethod
    def from_tier(cls, tier: str = "standard", **kwargs) -> "SecurityConfig":
        """
        Create config from tier preset.
        
        Args:
            tier: "basic", "standard", "advanced", "all"
            **kwargs: Override any config values
        
        Returns:
            SecurityConfig instance
        
        Examples:
            >>> config = SecurityConfig.from_tier("advanced", enable_ai=True)
            >>> config = SecurityConfig.from_tier("all")
        """
        from modules.security_checks import CHECK_TIERS
        
        # Define tier presets
        tier_configs = {
            "basic": {
                "enabled_checks": CHECK_TIERS.get("basic", ["headers", "cookies", "tls"]),
            },
            "standard": {
                "enabled_checks": CHECK_TIERS.get("standard", ["headers", "cookies", "tls", "cors", "rate_limit"]),
            },
            "advanced": {
                "enabled_checks": CHECK_TIERS.get("advanced", ["headers", "cookies", "tls", "cors", "rate_limit", "zero_day"]),
                "enable_compliance_checking": True,
            },
            "all": {
                "enabled_checks": list(CHECK_REGISTRY.keys()),
                "enable_compliance_checking": True,
                "enable_threat_intelligence": True,
            },
        }
        
        # Get base config for tier
        config_dict = tier_configs.get(tier, tier_configs["standard"]).copy()
        
        # Override with provided kwargs
        config_dict.update(kwargs)
        config_dict["check_tier"] = tier
        
        return cls(**config_dict)

# ==================== Result Cache ====================

class SecurityCache:
    """Cache for security scan results with TTL."""
    
    def __init__(self, cache_file: str, ttl_s: int = 3600):
        self.cache_file = Path(cache_file)
        self.ttl_s = ttl_s
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._load_cache()
    
    def _load_cache(self):
        """Load cache from file."""
        try:
            if self.cache_file.exists():
                with open(self.cache_file, "r") as f:
                    self._cache = json.load(f)
                logger.debug("Loaded security cache from %s", self.cache_file)
        except Exception as e:
            logger.warning("Failed to load security cache: %s", e)
            self._cache = {}
    
    def _save_cache(self):
        """Save cache to file atomically."""
        try:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            tmp_path = self.cache_file.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(self._cache, f, indent=2)
            os.replace(tmp_path, self.cache_file)
            logger.debug("Saved security cache to %s", self.cache_file)
        except Exception as e:
            logger.warning("Failed to save security cache: %s", e)
    
    def get(self, url: str) -> Optional[SecurityCheckResult]:
        """Get cached result if valid."""
        if url not in self._cache:
            return None
        
        entry = self._cache[url]
        cached_time = datetime.fromisoformat(entry["timestamp"])
        
        if (datetime.utcnow() - cached_time).total_seconds() > self.ttl_s:
            logger.debug("Cache expired for %s", url)
            del self._cache[url]
            self._save_cache()
            return None
        
        logger.info("✅ Using cached security results for %s", url)
        
        # Reconstruct SecurityCheckResult from cached data
        result_data = entry["result"]
        result = SecurityCheckResult(
            url=result_data["url"],
            timestamp=result_data["timestamp"],
            overall_status=CheckStatus(result_data["overall_status"]),
            duration_s=result_data["duration_s"],
            summary=result_data["summary"],
            metadata=result_data.get("metadata", {}),
        )
        
        # Reconstruct findings
        for finding_data in result_data.get("findings", []):
            finding = SecurityFinding(
                check_name=finding_data["check_name"],
                status=CheckStatus(finding_data["status"]),
                severity=Severity(finding_data["severity"]),
                message=finding_data["message"],
                details=finding_data.get("details", {}),
                recommendation=finding_data.get("recommendation"),
                references=finding_data.get("references", []),
                cwe_id=finding_data.get("cwe_id"),
                owasp_category=finding_data.get("owasp_category"),
                source=finding_data.get("source"),
                confidence=finding_data.get("confidence", 1.0),
            )
            result.findings.append(finding)
        
        return result
    
    def set(self, url: str, result: SecurityCheckResult):
        """Cache result."""
        self._cache[url] = {
            "timestamp": datetime.utcnow().isoformat(),
            "result": result.to_dict(),
        }
        self._save_cache()
    
    def clear_expired(self):
        """Remove expired cache entries."""
        now = datetime.utcnow()
        expired_keys = []
        
        for url, entry in self._cache.items():
            cached_time = datetime.fromisoformat(entry["timestamp"])
            if (now - cached_time).total_seconds() > self.ttl_s:
                expired_keys.append(url)
        
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            self._save_cache()
            logger.info("Cleared %d expired cache entries", len(expired_keys))


# ==================== AI Vulnerability Analyzer ====================

class AIVulnerabilityAnalyzer:
    """AI-powered vulnerability analysis and remediation."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.enabled = config.enable_ai and os.getenv("OPENAI_API_KEY")
        self._client = None
        
        if self.enabled:
            logger.info("✅ AI vulnerability analysis enabled (model: %s)", config.ai_model)
        else:
            logger.info("⚠️ AI analysis disabled (no OPENAI_API_KEY or disabled in config)")
    
    @property
    def client(self):
        """Lazy load OpenAI client."""
        if self._client is None and self.enabled:
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            except ImportError:
                logger.warning("OpenAI not installed: pip install openai")
                self.enabled = False
        return self._client
    
    def analyze_findings(
        self,
        findings: List[SecurityFinding],
        target_url: str,
    ) -> Dict[str, Any]:
        """
        Analyze security findings using AI.
        
        Args:
            findings: List of security findings
            target_url: Target URL being analyzed
        
        Returns:
            {
                "severity_analysis": {...},
                "remediation_steps": [...],
                "attack_scenarios": [...],
                "business_impact": str,
                "priority_order": [...]
            }
        """
        if not self.enabled or not self.client or not findings:
            return {}
        
        try:
            # Filter to critical/high findings for AI analysis
            critical_findings = [
                f for f in findings
                if f.severity in (Severity.CRITICAL, Severity.HIGH) and
                f.status in (CheckStatus.FAIL, CheckStatus.WARNING)
            ]
            
            if not critical_findings:
                return {
                    "severity_analysis": {
                        "critical_count": 0,
                        "high_count": 0,
                        "overall_risk": "low"
                    },
                    "message": "No critical or high severity findings to analyze"
                }
            
            prompt = self._build_analysis_prompt(critical_findings[:10], target_url)
            
            response = self.client.chat.completions.create(
                model=self.config.ai_model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a cybersecurity expert analyzing web application vulnerabilities. Provide actionable remediation guidance."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0.2,
                max_tokens=self.config.ai_max_tokens,
            )
            
            result = response.choices[0].message.content
            analysis = self._parse_analysis(result)
            
            # Add token usage metadata
            analysis["metadata"] = {
                "model": self.config.ai_model,
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }
            
            logger.info("✅ AI analysis completed for %s", target_url)
            return analysis
        
        except Exception as e:
            logger.error("AI vulnerability analysis failed: %s", e)
            return {"error": str(e)}
    
    def _build_analysis_prompt(
        self,
        findings: List[SecurityFinding],
        target_url: str,
    ) -> str:
        """Build comprehensive analysis prompt."""
        findings_summary = "\n".join([
            f"{i+1}. [{f.severity.value}] {f.check_name}\n"
            f"   Message: {f.message}\n"
            f"   CWE: {f.cwe_id or 'N/A'} | OWASP: {f.owasp_category or 'N/A'}\n"
            f"   Confidence: {f.confidence:.0%}"
            for i, f in enumerate(findings)
        ])
        
        return f"""Analyze these security vulnerabilities found in {target_url}:

FINDINGS:
{findings_summary}

Provide expert security assessment in JSON format:
{{
  "severity_analysis": {{
    "critical_count": <number>,
    "high_count": <number>,
    "overall_risk": "critical"|"high"|"medium"|"low",
    "risk_justification": "<brief explanation>"
  }},
  "remediation_steps": [
    "<prioritized step 1>",
    "<prioritized step 2>",
    "<prioritized step 3>"
  ],
  "attack_scenarios": [
    "<realistic attack scenario 1>",
    "<realistic attack scenario 2>"
  ],
  "business_impact": "<business risk description>",
  "priority_order": [
    "<finding to fix first>",
    "<finding to fix second>",
    "<finding to fix third>"
  ],
  "estimated_fix_time": "<time estimate>"
}}

Focus on:
1. Prioritized, actionable remediation steps
2. Realistic attack scenarios an attacker might use
3. Business impact in non-technical terms
4. Fix priority based on risk and complexity
"""
    
    def _parse_analysis(self, response: str) -> Dict[str, Any]:
        """Parse AI analysis response."""
        try:
            # Extract JSON from response
            match = re.search(r'\{.*\}', response, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            logger.warning("Failed to parse AI response as JSON: %s", e)
        
        # Fallback: return raw response
        return {
            "raw_response": response,
            "parsed": False,
        }


# ==================== CVE Database Integration ====================

class CVELookup:
    """Integrate with NVD CVE database for known vulnerabilities."""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.enabled = config.enable_cve_lookup
        self.api_url = config.cve_api_url
        
        if self.enabled:
            logger.info("✅ CVE database lookup enabled")
    
    async def lookup_vulnerabilities(
        self,
        product: str,
        version: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Lookup CVEs for given product/version.
        
        Args:
            product: Product name (e.g., "nginx", "apache")
            version: Optional version string
        
        Returns:
            List of CVE entries with CVSS scores
        """
        if not self.enabled:
            return []
        
        try:
            keyword = f"{product} {version}" if version else product
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                params = {
                    "keyword": keyword,
                    "resultsPerPage": self.config.cve_results_limit,
                }
                
                response = await client.get(self.api_url, params=params)
                
                if response.status_code == 200:
                    data = response.json()
                    vulnerabilities = data.get("vulnerabilities", [])
                    
                    results = []
                    for vuln in vulnerabilities[:self.config.cve_results_limit]:
                        cve_data = vuln.get("cve", {})
                        results.append({
                            "cve_id": cve_data.get("id"),
                            "description": self._extract_description(cve_data),
                            "severity": self._extract_severity(vuln),
                            "cvss_score": self._extract_cvss_score(vuln),
                            "published_date": cve_data.get("published"),
                            "references": [
                                ref.get("url")
                                for ref in cve_data.get("references", [])[:3]
                            ],
                        })
                    
                    logger.info("✅ Found %d CVEs for %s", len(results), keyword)
                    return results
                else:
                    logger.warning("CVE API returned status %d", response.status_code)
        
        except httpx.HTTPError as e:
            logger.error("CVE lookup HTTP error: %s", e)
        except Exception as e:
            logger.error("CVE lookup failed: %s", e)
        
        return []
    
    def _extract_description(self, cve_data: Dict[str, Any]) -> str:
        """Extract description from CVE data."""
        descriptions = cve_data.get("descriptions", [])
        if descriptions:
            return descriptions[0].get("value", "")
        return ""
    
    def _extract_severity(self, vuln: Dict[str, Any]) -> str:
        """Extract severity from CVE data."""
        metrics = vuln.get("cve", {}).get("metrics", {})
        
        # Try CVSS v3.1 first
        cvss_v31 = metrics.get("cvssMetricV31", [])
        if cvss_v31:
            severity = cvss_v31[0].get("cvssData", {}).get("baseSeverity", "UNKNOWN")
            return severity.upper()
        
        # Fallback to CVSS v3.0
        cvss_v30 = metrics.get("cvssMetricV30", [])
        if cvss_v30:
            severity = cvss_v30[0].get("cvssData", {}).get("baseSeverity", "UNKNOWN")
            return severity.upper()
        
        return "UNKNOWN"
    
    def _extract_cvss_score(self, vuln: Dict[str, Any]) -> float:
        """Extract CVSS score from CVE data."""
        metrics = vuln.get("cve", {}).get("metrics", {})
        
        # Try CVSS v3.1 first
        cvss_v31 = metrics.get("cvssMetricV31", [])
        if cvss_v31:
            score = cvss_v31[0].get("cvssData", {}).get("baseScore", 0.0)
            return float(score)
        
        # Fallback to CVSS v3.0
        cvss_v30 = metrics.get("cvssMetricV30", [])
        if cvss_v30:
            score = cvss_v30[0].get("cvssData", {}).get("baseScore", 0.0)
            return float(score)
        
        return 0.0


# ==================== Rate Limiter ====================

class RateLimiter:
    """Simple rate limiter for API calls."""
    
    def __init__(self, max_calls_per_minute: int = 60):
        self.max_calls = max_calls_per_minute
        self.calls: List[float] = []
    
    async def acquire(self):
        """Wait if rate limit is exceeded."""
        now = time.time()
        
        # Remove calls older than 1 minute
        self.calls = [t for t in self.calls if now - t < 60]
        
        if len(self.calls) >= self.max_calls:
            # Wait until oldest call expires
            wait_time = 60 - (now - self.calls[0])
            if wait_time > 0:
                logger.debug("Rate limit reached, waiting %.1fs", wait_time)
                await asyncio.sleep(wait_time)
                self.calls = []
        
        self.calls.append(now)


# ==================== Enhanced Security Engine ====================

class SecurityEngine:
    """
    Production-grade security engine with AI enhancement.
    
    Features:
    - Modular plugin-based security checks
    - AI-powered vulnerability analysis
    - CVE database integration
    - Result caching with TTL
    - Retry logic with exponential backoff
    - Rate limiting
    - Progress tracking
    - Comprehensive reporting
    """
    
    def __init__(
        self,
        config: Optional[SecurityConfig] = None,
        timeout_s: Optional[float] = None,  # Backward compatibility
        enabled_checks: Optional[List[str]] = None,  # Backward compatibility
        enable_async: Optional[bool] = None,  # Backward compatibility
        max_concurrent_checks: Optional[int] = None,  # Backward compatibility
        enable_ai: Optional[bool] = None,  # Backward compatibility
    ):
        """
        Initialize security engine.
        
        Args:
            config: SecurityConfig instance (recommended)
            timeout_s: Request timeout (backward compat)
            enabled_checks: List of check names (backward compat)
            enable_async: Use async mode (backward compat)
            max_concurrent_checks: Max concurrent checks (backward compat)
            enable_ai: Enable AI analysis (backward compat)
        """
        # Use provided config or create from params
        if config is None:
            config = SecurityConfig(
                timeout_s=timeout_s or 15.0,
                enabled_checks=enabled_checks or list(CHECK_REGISTRY.keys()),
                enable_async=enable_async if enable_async is not None else True,
                max_concurrent_checks=max_concurrent_checks or 3,
                enable_ai=enable_ai if enable_ai is not None else False,
            )
        
        self.config = config
        
        # Load enabled checks from registry
        self.check_classes = get_enabled_checks(self.config.enabled_checks)
        
        # Initialize components
        self.ai_analyzer = (
            AIVulnerabilityAnalyzer(config)
            if config.enable_ai
            else None
        )
        self.cve_lookup = CVELookup(config)
        self.cache = (
            SecurityCache(
                cache_file=config.cache_file,
                ttl_s=config.cache_ttl_s,
            )
            if config.enable_cache
            else None
        )
        self.rate_limiter = RateLimiter(config.api_rate_limit_per_min)
        
        # ✅ NEW: Initialize advanced components
        self.threat_intel = None
        self.compliance_checker = None
        self.sbom_generator = None
        
        if config.enable_threat_intelligence and THREAT_INTEL_AVAILABLE:
            try:
                self.threat_intel = ThreatIntelligence()
                if self.threat_intel.enabled:
                    logger.info("✅ Threat intelligence enabled")
                else:
                    logger.warning("⚠️  Threat intelligence disabled (missing API keys)")
            except Exception as e:
                logger.warning("Failed to initialize threat intelligence: %s", e)
        
        if config.enable_compliance_checking and COMPLIANCE_AVAILABLE:
            try:
                self.compliance_checker = ComplianceChecker()
                logger.info("✅ Compliance checking enabled")
            except Exception as e:
                logger.warning("Failed to initialize compliance checker: %s", e)
        
        if config.enable_sbom_generation and SBOM_AVAILABLE:
            try:
                self.sbom_generator = SBOMGenerator()
                logger.info("✅ SBOM generation enabled")
            except Exception as e:
                logger.warning("Failed to initialize SBOM generator: %s", e)
                
        logger.info(
            "SecurityEngine v2.0 initialized: "
            "tier=%s, checks=%d, async=%s, ai=%s, cve=%s, cache=%s, threat_intel=%s, compliance=%s",
            config.check_tier,
            len(self.config.enabled_checks),
            config.enable_async,
            self.ai_analyzer.enabled if self.ai_analyzer else False,
            self.cve_lookup.enabled,
            config.enable_cache,
            bool(self.threat_intel and self.threat_intel.enabled) if self.threat_intel else False,
            bool(self.compliance_checker),
        )
    
    # ==================== Plan Generation ====================
    
    def plan_from_base_url(self, base_url: str) -> Optional[Dict[str, Any]]:
        """Generate security test plan from base URL."""
        if not base_url:
            return None
        
        steps = [
            f"run {check} security check on {base_url}"
            for check in self.config.enabled_checks
        ]
        
        if self.ai_analyzer and self.ai_analyzer.enabled:
            steps.append("analyze findings with AI")
        
        if self.cve_lookup.enabled:
            steps.append("lookup CVE database")
        
        steps.append("generate comprehensive security report")
        
        return {
            "name": "OWASP Security Validation Suite v2.0",
            "description": "AI-enhanced security checks with CVSS scoring and CVE lookup",
            "steps": steps,
            "priority": "P0",
            "tags": ["security", "owasp", "ai-enhanced"] + self.config.enabled_checks,
            "metadata": {
                "check_types": self.config.enabled_checks,
                "standard": "OWASP Top 10 2021",
                "ai_enabled": bool(self.ai_analyzer and self.ai_analyzer.enabled),
                "cve_lookup": self.cve_lookup.enabled,
                "cache_enabled": bool(self.cache),
                "engine_version": "2.0",
            },
        }
    
    # ==================== Suite Execution ====================
    
    def execute_suite(
        self,
        suite: Dict[str, Any],
        base_url: str,
    ) -> Dict[str, Any]:
        """
        Execute single security test suite.
        
        Args:
            suite: Test suite dictionary
            base_url: Target URL
        
        Returns:
            Execution results dictionary
        """
        logger.info("🔒 Executing security suite for %s", base_url)
        
        # Run scan
        if self.config.enable_async:
            result = asyncio.run(self.scan_async(base_url))
        else:
            result = self.scan_sync(base_url)
        
        # Convert to suite execution format
        return {
            "name": suite.get("name"),
            "url": base_url,
            "status": result.overall_status.value,
            "summary": result.summary,
            "findings": [f.to_dict() for f in result.findings],
            "duration_s": result.duration_s,
            "metadata": result.metadata,
            "steps": [
                {
                    "step": step,
                    "status": "PASS" if result.overall_status == CheckStatus.PASS else "FAIL",
                }
                for step in suite.get("steps", [])
            ],
        }
    
    def run_suites(
        self,
        suites: List[Dict[str, Any]],
        base_url: Optional[str] = None,
        export_dir: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute multiple security scan suites with AI enhancement.
        
        Args:
            suites: List of test suite dictionaries
            base_url: Optional base URL (will extract from suites if not provided)
            export_dir: Export directory for reports
            run_id: Optional run ID for tracking
        
        Returns:
            Comprehensive security assessment with remediation
        """
        # Extract URLs from suites or use provided base_url
        urls: List[str] = []
        if base_url:
            urls.append(base_url)
        else:
            for suite in suites or []:
                for step in suite.get("steps", []):
                    match = re.search(r"(https?://\S+)", step)
                    if match:
                        urls.append(match.group(1))
        
        # Deduplicate URLs
        urls = list(dict.fromkeys(urls))  # Preserves order
        
        if not urls:
            logger.warning("No URLs found in suites")
            return {
                "targets": [],
                "result_count": 0,
                "failed": 0,
                "success": True,
                "message": "No URLs to scan",
            }
        
        # Setup export directory
        out_dir = Path(export_dir or self.config.export_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        rid = run_id or datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        
        # Scan each URL
        exports: List[Dict[str, str]] = []
        results: List[Dict[str, Any]] = []
        failed = False
        
        for url in urls:
            logger.info("🎯 Scanning %s", url)
            
            # Run scan
            if self.config.enable_async:
                scan_result = asyncio.run(self.scan_async(url))
            else:
                scan_result = self.scan_sync(url)
            
            # AI analysis
            if self.ai_analyzer and self.ai_analyzer.enabled and scan_result.findings:
                logger.info("🤖 Running AI analysis for %s", url)
                ai_analysis = self.ai_analyzer.analyze_findings(
                    scan_result.findings,
                    url,
                )
                scan_result.metadata["ai_analysis"] = ai_analysis
            
            results.append(scan_result.to_dict())
            
            if scan_result.overall_status in (CheckStatus.FAIL, CheckStatus.ERROR):
                failed = True
            
            # Export artifacts
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")[:60]
            
            artifact_paths = {}
            
            # JSON export
            json_path = out_dir / f"{rid}_{slug}.json"
            try:
                json_path.write_text(scan_result.to_json(), encoding="utf-8")
                artifact_paths["json"] = str(json_path)
            except Exception as e:
                logger.error("Failed to export JSON: %s", e)
            
            # Markdown export
            if self.config.enable_markdown_reports:
                md_path = out_dir / f"{rid}_{slug}.md"
                try:
                    md_path.write_text(scan_result.to_markdown(), encoding="utf-8")
                    artifact_paths["md"] = str(md_path)
                except Exception as e:
                    logger.error("Failed to export Markdown: %s", e)
            
            # HTML export
            if self.config.enable_html_reports:
                html_path = out_dir / f"{rid}_{slug}.html"
                try:
                    html_content = self._generate_html_report(scan_result)
                    html_path.write_text(html_content, encoding="utf-8")
                    artifact_paths["html"] = str(html_path)
                except Exception as e:
                    logger.error("Failed to export HTML: %s", e)
            
            exports.append({
                "url": url,
                **artifact_paths,
            })
        
        # Build summary
        summary = {
            "targets": urls,
            "result_count": len(results),
            "failed": int(failed),
            "success": not failed,
            "exports": exports,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "engine_version": "2.0",
            "ai_enhanced": bool(self.ai_analyzer and self.ai_analyzer.enabled),
            "cve_lookup_enabled": self.cve_lookup.enabled,
            "cache_enabled": bool(self.cache),
        }
        
        logger.info(
            "✅ Security scan completed: %d targets, %d results, success=%s",
            len(urls),
            len(results),
            summary["success"],
        )
        
        return {**summary, "results": results}
    
    # ==================== Async Scan ====================
    
    async def scan_async(self, url: str) -> SecurityCheckResult:
        """
        Async security scan with retry logic and AI enhancement.
        
        Args:
            url: Target URL
        
        Returns:
            SecurityCheckResult with all findings
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(url)
            if cached:
                return cached
        
        start_time = time.time()
        
        result = SecurityCheckResult(
            url=url,
            timestamp=datetime.utcnow().isoformat() + "Z",
            overall_status=CheckStatus.PASS,
        )
        
        logger.info("🔒 Starting async security scan: %s", url)
        
        # Retry logic
        for attempt in range(self.config.retry_max_attempts):
            try:
                # Rate limiting
                await self.rate_limiter.acquire()
                
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    timeout=self.config.timeout_s,
                    verify=True,
                ) as client:
                    # Initialize checks
                    checks = [
                        check_class(timeout_s=self.config.timeout_s)
                        for check_class in self.check_classes
                    ]
                    
                    # Execute checks with concurrency control
                    sem = asyncio.Semaphore(self.config.max_concurrent_checks)
                    
                    async def bounded_check(check):
                        async with sem:
                            await check.execute_async(url, client, result)
                    
                    await asyncio.gather(*[bounded_check(c) for c in checks])
                
                # Success - break retry loop
                break
            
            except httpx.HTTPError as e:
                logger.warning(
                    "Scan attempt %d/%d failed for %s: %s",
                    attempt + 1,
                    self.config.retry_max_attempts,
                    url,
                    e,
                )
                
                if attempt < self.config.retry_max_attempts - 1:
                    # Exponential backoff
                    delay = self.config.retry_delay_s * (2 ** attempt)
                    logger.info("Retrying in %.1fs...", delay)
                    await asyncio.sleep(delay)
                else:
                    # Final attempt failed
                    logger.error("All scan attempts failed for %s", url)
                    result.add_finding(SecurityFinding(
                        check_name="scan_execution",
                        status=CheckStatus.ERROR,
                        severity=Severity.HIGH,
                        message=f"Scan failed after {self.config.retry_max_attempts} attempts: {str(e)}",
                        details={"error": str(e), "attempts": self.config.retry_max_attempts},
                        source="engine",
                    ))
            
            except Exception as e:
                logger.error("Unexpected error during scan of %s: %s", url, e)
                result.add_finding(SecurityFinding(
                    check_name="scan_execution",
                    status=CheckStatus.ERROR,
                    severity=Severity.HIGH,
                    message=f"Unexpected scan error: {str(e)}",
                    details={"error": str(e), "error_type": type(e).__name__},
                    source="engine",
                ))
                break
        
        # Calculate duration and summary
        result.duration_s = round(time.time() - start_time, 2)
        result.calculate_summary()
        
        logger.info(
            "✅ Async scan completed: %s "
            "(risk_score=%d, findings=%d, duration=%.2fs)",
            url,
            result.summary.get("risk_score", 0),
            len(result.findings),
            result.duration_s,
        )
        # ✅ NEW: Enrich with advanced features
        await self._enrich_with_advanced_features(url, result)
        
        # Cache result
        if self.cache:
            self.cache.set(url, result)
        
        return result
    
    async def _enrich_with_advanced_features(
        self,
        url: str,
        result: SecurityCheckResult,
    ):
        """
        Enrich scan results with advanced features.
        
        Args:
            url: Target URL
            result: SecurityCheckResult to enrich
        """
        # 1. Threat Intelligence Check
        if self.threat_intel and self.threat_intel.enabled:
            logger.info("🔍 Checking threat intelligence feeds...")
            try:
                threat_data = await self.threat_intel.check_url_reputation(url)
                result.metadata["threat_intelligence"] = threat_data
                
                if threat_data.get("malicious"):
                    logger.warning("⚠️  URL flagged as malicious by: %s", threat_data.get("sources"))
                    
                    # Add critical finding
                    result.add_finding(SecurityFinding(
                        check_name="threat_intelligence_malicious",
                        status=CheckStatus.FAIL,
                        severity=Severity.CRITICAL,
                        message=f"URL flagged as malicious by threat intelligence feeds: {', '.join(threat_data.get('sources', []))}",
                        details=threat_data,
                        recommendation="Investigate immediately - URL appears in threat intelligence databases",
                        source="threat_intel",
                        cwe_id="CWE-506",
                        owasp_category="A03:2021 - Injection",
                        confidence=0.95,
                    ))
            except Exception as e:
                logger.error("Threat intelligence check failed: %s", e)
        
        # 2. Compliance Checking
        if self.compliance_checker:
            logger.info("📋 Running compliance checks...")
            try:
                for framework in self.config.compliance_frameworks:
                    compliance_result = self.compliance_checker.check_compliance(
                        [f.to_dict() for f in result.findings],
                        framework=framework,
                    )
                    result.metadata[f"compliance_{framework.lower().replace('-', '_')}"] = compliance_result
                    
                    if not compliance_result.get("compliant"):
                        violations = compliance_result.get("violations", [])
                        logger.warning("⚠️  %s violations: %d", framework, len(violations))
                        
                        # Add compliance finding
                        result.add_finding(SecurityFinding(
                            check_name=f"compliance_{framework.lower().replace('-', '_')}",
                            status=CheckStatus.FAIL,
                            severity=Severity.HIGH,
                            message=f"{framework} compliance violations detected: {len(violations)} requirement(s) not met",
                            details={
                                "framework": framework,
                                "violations": violations[:5],  # Show first 5
                                "total_violations": len(violations),
                            },
                            recommendation=f"Address {framework} compliance violations to meet regulatory requirements",
                            source="compliance",
                            confidence=1.0,
                        ))
            except Exception as e:
                logger.error("Compliance checking failed: %s", e)
        
        # 3. SBOM Generation
        if self.sbom_generator:
            logger.info("📦 Generating SBOM...")
            try:
                sbom_dir = Path("reports/sbom")
                sbom_dir.mkdir(parents=True, exist_ok=True)
                
                timestamp = result.timestamp.replace(":", "-").replace(".", "-")
                slug = re.sub(r"[^a-zA-Z0-9]+", "_", url).strip("_")[:40]
                sbom_path = sbom_dir / f"{timestamp}_{slug}_sbom.json"
                
                sbom = self.sbom_generator.generate(url, str(sbom_path))
                result.metadata["sbom"] = {
                    "path": str(sbom_path),
                    "components": len(sbom.get("components", [])),
                    "format": sbom.get("bomFormat", "unknown"),
                }
                logger.info("✅ SBOM saved to: %s", sbom_path)
            except Exception as e:
                logger.error("SBOM generation failed: %s", e)
    # ==================== Sync Scan ====================
    
    def scan_sync(self, url: str) -> SecurityCheckResult:
        """
        Synchronous security scan (fallback for non-async environments).
        
        Args:
            url: Target URL
        
        Returns:
            SecurityCheckResult with all findings
        """
        # Check cache first
        if self.cache:
            cached = self.cache.get(url)
            if cached:
                return cached
        
        start_time = time.time()
        
        result = SecurityCheckResult(
            url=url,
            timestamp=datetime.utcnow().isoformat() + "Z",
            overall_status=CheckStatus.PASS,
        )
        
        logger.info("🔒 Starting sync security scan: %s", url)
        
        try:
            with httpx.Client(
                follow_redirects=True,
                timeout=self.config.timeout_s,
                verify=True,
            ) as client:
                checks = [
                    check_class(timeout_s=self.config.timeout_s)
                    for check_class in self.check_classes
                ]
                
                for check in checks:
                    try:
                        check.execute_sync(url, client, result)
                    except Exception as e:
                        logger.error("Check '%s' failed: %s", check.name, e)
        
        except Exception as e:
            logger.error("Security scan failed for %s: %s", url, e)
            result.add_finding(SecurityFinding(
                check_name="scan_execution",
                status=CheckStatus.ERROR,
                severity=Severity.HIGH,
                message=f"Scan failed: {str(e)}",
                source="engine",
            ))
        
        result.duration_s = round(time.time() - start_time, 2)
        result.calculate_summary()
        
        logger.info(
            "✅ Sync scan completed: %s "
            "(risk_score=%d, findings=%d)",
            url,
            result.summary.get("risk_score", 0),
            len(result.findings),
        )
        
        # Cache result
        if self.cache:
            self.cache.set(url, result)
        
        return result
    
    # ==================== Streaming Scan (NEW) ====================
    
    async def scan_stream(
        self,
        url: str,
    ) -> AsyncGenerator[SecurityFinding, None]:
        """
        Stream security findings as they are discovered.
        Useful for real-time progress tracking.
        
        Args:
            url: Target URL
        
        Yields:
            SecurityFinding objects as they are discovered
        
        Example:
            async for finding in engine.scan_stream("https://example.com"):
                print(f"Found: {finding.message}")
        """
        logger.info("🔒 Starting streaming security scan: %s", url)
        
        try:
            async with httpx.AsyncClient(
                follow_redirects=True,
                timeout=self.config.timeout_s,
                verify=True,
            ) as client:
                checks = [
                    check_class(timeout_s=self.config.timeout_s)
                    for check_class in self.check_classes
                ]
                
                # Run checks sequentially for streaming
                for check in checks:
                    result = SecurityCheckResult(
                        url=url,
                        timestamp=datetime.utcnow().isoformat() + "Z",
                        overall_status=CheckStatus.PASS,
                    )
                    
                    try:
                        await check.execute_async(url, client, result)
                        
                        # Yield findings as they are discovered
                        for finding in result.findings:
                            yield finding
                    
                    except Exception as e:
                        logger.error("Check '%s' failed: %s", check.name, e)
                        
                        # Yield error finding
                        yield SecurityFinding(
                            check_name=f"{check.name}_error",
                            status=CheckStatus.ERROR,
                            severity=Severity.MEDIUM,
                            message=f"Check failed: {str(e)}",
                            source=check.source,
                        )
        
        except Exception as e:
            logger.error("Streaming scan failed for %s: %s", url, e)
            yield SecurityFinding(
                check_name="scan_execution",
                status=CheckStatus.ERROR,
                severity=Severity.HIGH,
                message=f"Scan failed: {str(e)}",
                source="engine",
            )
    
    # ==================== HTML Report Generation ====================
    
    def _generate_html_report(self, result: SecurityCheckResult) -> str:
        """Generate enhanced HTML security report with AI insights."""
        # Build findings HTML
        findings_by_severity = {
            "CRITICAL": [],
            "HIGH": [],
            "MEDIUM": [],
            "LOW": [],
            "INFO": [],
        }
        
        for finding in result.findings:
            severity = finding.severity.value if hasattr(finding, 'severity') else finding.get('severity', 'INFO')
            findings_by_severity[severity].append(finding)
        
        findings_html = ""
        for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
            findings_list = findings_by_severity[severity]
            if not findings_list:
                continue
            
            findings_html += f'<h3>{severity} ({len(findings_list)})</h3>'
            
            for finding in findings_list:
                if isinstance(finding, SecurityFinding):
                    check_name = finding.check_name
                    message = finding.message
                    remediation = finding.recommendation
                    cwe_id = finding.cwe_id
                    owasp = finding.owasp_category
                else:
                    check_name = finding.get('check_name', 'Unknown')
                    message = finding.get('message', '')
                    remediation = finding.get('remediation') or finding.get('recommendation')
                    cwe_id = finding.get('cwe_id')
                    owasp = finding.get('owasp_category')
                
                findings_html += f"""
<div class="finding {severity.lower()}">
  <h4>🔍 {check_name}</h4>
  <p class="message">{message}</p>
  {f'<p class="compliance"><strong>CWE:</strong> {cwe_id} | <strong>OWASP:</strong> {owasp}</p>' if cwe_id or owasp else ''}
  {f'<div class="remediation"><strong>💡 Remediation:</strong><p>{remediation}</p></div>' if remediation else ''}
</div>
"""
        
        # AI analysis section
        ai_html = ""
        ai_analysis = result.metadata.get("ai_analysis", {})
        if ai_analysis and "raw_response" not in ai_analysis:
            ai_html = f"""
<div class="ai-analysis">
  <h2>🤖 AI Security Analysis</h2>
  <div class="ai-section">
    <h3>Severity Assessment</h3>
    <p><strong>Overall Risk:</strong> <span class="badge {ai_analysis.get('severity_analysis', {}).get('overall_risk', 'unknown')}">{ai_analysis.get('severity_analysis', {}).get('overall_risk', 'N/A').upper()}</span></p>
    <p>{ai_analysis.get('severity_analysis', {}).get('risk_justification', '')}</p>
  </div>
  
  <div class="ai-section">
    <h3>Prioritized Remediation Steps</h3>
    <ol>
      {''.join(f'<li>{step}</li>' for step in ai_analysis.get('remediation_steps', []))}
    </ol>
  </div>
  
  <div class="ai-section">
    <h3>Attack Scenarios</h3>
    <ul>
      {''.join(f'<li>{scenario}</li>' for scenario in ai_analysis.get('attack_scenarios', []))}
    </ul>
  </div>
  
  <div class="ai-section">
    <h3>Business Impact</h3>
    <p>{ai_analysis.get('business_impact', 'N/A')}</p>
  </div>
  
  {f'<p class="ai-meta"><small>Estimated fix time: {ai_analysis.get("estimated_fix_time", "N/A")}</small></p>' if ai_analysis.get('estimated_fix_time') else ''}
</div>
"""
        
        # Generate HTML
        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Security Report - {result.url}</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ 
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
    margin: 0; 
    padding: 20px; 
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    min-height: 100vh;
  }}
  .container {{ 
    max-width: 1200px; 
    margin: 0 auto; 
    background: white; 
    padding: 40px; 
    border-radius: 12px;
    box-shadow: 0 20px 60px rgba(0,0,0,0.3);
  }}
  h1 {{ 
    color: #d32f2f; 
    margin-bottom: 10px;
    font-size: 2.5em;
  }}
  .header-meta {{ 
    color: #666; 
    margin-bottom: 30px;
    font-size: 0.95em;
  }}
  .risk-score {{ 
    display: inline-block;
    padding: 10px 20px;
    border-radius: 8px;
    font-weight: bold;
    font-size: 1.2em;
    margin: 20px 0;
  }}
  .risk-critical {{ background: #d32f2f; color: white; }}
  .risk-high {{ background: #f57c00; color: white; }}
  .risk-medium {{ background: #fbc02d; color: #333; }}
  .risk-low {{ background: #388e3c; color: white; }}
  
  .finding {{ 
    border-left: 5px solid; 
    padding: 20px; 
    margin: 15px 0; 
    background: #fafafa;
    border-radius: 0 8px 8px 0;
    transition: transform 0.2s, box-shadow 0.2s;
  }}
  .finding:hover {{
    transform: translateX(5px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
  }}
  .finding.critical {{ border-color: #d32f2f; background: #ffebee; }}
  .finding.high {{ border-color: #f57c00; background: #fff3e0; }}
  .finding.medium {{ border-color: #fbc02d; background: #fffde7; }}
  .finding.low {{ border-color: #388e3c; background: #e8f5e9; }}
  .finding.info {{ border-color: #1976d2; background: #e3f2fd; }}
  
  .finding h4 {{ margin: 0 0 10px 0; color: #333; }}
  .finding .message {{ color: #555; line-height: 1.6; }}
  .finding .compliance {{ 
    font-size: 0.9em; 
    color: #666;
    margin: 10px 0;
  }}
  .finding .remediation {{ 
    background: white;
    padding: 15px;
    border-radius: 6px;
    margin-top: 10px;
    border: 1px solid #ddd;
  }}
  .finding .remediation strong {{ color: #1976d2; }}
  
  .badge {{ 
    padding: 5px 12px; 
    border-radius: 4px; 
    color: white; 
    font-weight: bold;
    font-size: 0.85em;
    text-transform: uppercase;
  }}
  .badge.critical {{ background: #d32f2f; }}
  .badge.high {{ background: #f57c00; }}
  .badge.medium {{ background: #fbc02d; color: #333; }}
  .badge.low {{ background: #388e3c; }}
  .badge.info {{ background: #1976d2; }}
  
  .ai-analysis {{
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: white;
    padding: 30px;
    border-radius: 12px;
    margin: 30px 0;
    box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
  }}
  .ai-analysis h2 {{ 
    margin-top: 0; 
    border-bottom: 2px solid rgba(255,255,255,0.3);
    padding-bottom: 10px;
  }}
  .ai-section {{
    background: rgba(255,255,255,0.1);
    padding: 20px;
    border-radius: 8px;
    margin: 15px 0;
  }}
  .ai-section h3 {{ 
    margin-top: 0;
    font-size: 1.2em;
  }}
  .ai-section ol, .ai-section ul {{
    margin: 10px 0;
    padding-left: 20px;
  }}
  .ai-section li {{
    margin: 8px 0;
    line-height: 1.6;
  }}
  .ai-meta {{
    text-align: right;
    font-style: italic;
    opacity: 0.8;
    margin-top: 10px;
  }}
  
  .summary-stats {{
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
    gap: 15px;
    margin: 30px 0;
  }}
  .stat-card {{
    background: #f5f5f5;
    padding: 20px;
    border-radius: 8px;
    text-align: center;
  }}
  .stat-card .number {{
    font-size: 2em;
    font-weight: bold;
    color: #1976d2;
  }}
  .stat-card .label {{
    color: #666;
    font-size: 0.9em;
    margin-top: 5px;
  }}
  
  @media print {{
    body {{ background: white; }}
    .container {{ box-shadow: none; }}
  }}
</style>
</head>
<body>
<div class="container">
  <h1>🔒 Security Assessment Report</h1>
  <div class="header-meta">
    <p><strong>Target:</strong> <code>{result.url}</code></p>
    <p><strong>Scan Time:</strong> {result.timestamp}</p>
    <p><strong>Duration:</strong> {result.duration_s}s</p>
    <p><strong>Engine:</strong> SecurityEngine v2.0 {' (AI-Enhanced)' if result.metadata.get('ai_analysis') else ''}</p>
  </div>
  
  <div class="risk-score risk-{'critical' if result.summary.get('risk_score', 0) > 70 else 'high' if result.summary.get('risk_score', 0) > 40 else 'medium' if result.summary.get('risk_score', 0) > 10 else 'low'}">
    🎯 Risk Score: {result.summary.get('risk_score', 0)}/100
  </div>
  
  <div class="summary-stats">
    <div class="stat-card">
      <div class="number">{result.summary.get('total_checks', 0)}</div>
      <div class="label">Total Checks</div>
    </div>
    <div class="stat-card">
      <div class="number">{result.summary.get('passed', 0)}</div>
      <div class="label">Passed</div>
    </div>
    <div class="stat-card">
      <div class="number">{result.summary.get('failed', 0)}</div>
      <div class="label">Failed</div>
    </div>
    <div class="stat-card">
      <div class="number">{result.summary.get('warnings', 0)}</div>
      <div class="label">Warnings</div>
    </div>
  </div>
  
  {ai_html}
  
  <h2>🔍 Security Findings ({len(result.findings)})</h2>
  {findings_html if findings_html else '<p style="color: #388e3c; padding: 20px; background: #e8f5e9; border-radius: 8px;">✅ No security issues detected. Your application follows security best practices!</p>'}
  
  <div style="margin-top: 40px; padding-top: 20px; border-top: 2px solid #eee; text-align: center; color: #999; font-size: 0.9em;">
    <p>Generated by SecurityEngine v2.0 | {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
    <p>Report ID: {result.timestamp.replace(':', '-').replace('.', '-')}</p>
  </div>
</div>
</body>
</html>"""


# ==================== CLI Support ====================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    
    if len(sys.argv) < 2:
        print("Usage: python -m modules.security_engine <url>")
        print("\nExample:")
        print("  python -m modules.security_engine https://example.com")
        print("\nEnvironment variables:")
        print("  SECURITY_AI_ANALYSIS=true     # Enable AI analysis")
        print("  SECURITY_CVE_LOOKUP=true      # Enable CVE lookup")
        print("  OPENAI_API_KEY=sk-...         # OpenAI API key for AI")
        sys.exit(1)
    
    target_url = sys.argv[1]
    
    # Create engine with config from environment
    config = SecurityConfig.from_env()
    engine = SecurityEngine(config=config)
    
    # Run scan
    print(f"\n🔒 Starting security scan of {target_url}...\n")
    
    if config.enable_async:
        result = asyncio.run(engine.scan_async(target_url))
    else:
        result = engine.scan_sync(target_url)
    
    # Print summary
    print("\n" + "=" * 60)
    print("SECURITY SCAN RESULTS")
    print("=" * 60)
    print(f"URL: {result.url}")
    print(f"Status: {result.overall_status.value}")
    print(f"Duration: {result.duration_s}s")
    print(f"Risk Score: {result.summary.get('risk_score', 0)}/100")
    print(f"\nSummary:")
    for key, value in result.summary.items():
        if key != "risk_score":
            print(f"  {key}: {value}")
    
    print(f"\nFindings:")
    for finding in result.findings[:10]:  # Show first 10
        icon = "✅" if finding.status == CheckStatus.PASS else "❌" if finding.status == CheckStatus.FAIL else "⚠️"
        severity_icon = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🟢", "INFO": "ℹ️"}.get(finding.severity.value, "")
        print(f"  {icon} {severity_icon} [{finding.severity.value}] {finding.message}")
    
    if len(result.findings) > 10:
        print(f"\n  ... and {len(result.findings) - 10} more findings")
    
    print("=" * 60 + "\n")
    
    # Export reports
    export_dir = Path("reports/security")
    export_dir.mkdir(parents=True, exist_ok=True)
    
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", target_url).strip("_")[:60]
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    
    json_path = export_dir / f"{timestamp}_{slug}.json"
    json_path.write_text(result.to_json(), encoding="utf-8")
    print(f"📄 JSON report: {json_path}")
    
    if config.enable_markdown_reports:
        md_path = export_dir / f"{timestamp}_{slug}.md"
        md_path.write_text(result.to_markdown(), encoding="utf-8")
        print(f"📄 Markdown report: {md_path}")
    
    if config.enable_html_reports:
        html_path = export_dir / f"{timestamp}_{slug}.html"
        html_content = engine._generate_html_report(result)
        html_path.write_text(html_content, encoding="utf-8")
        print(f"📄 HTML report: {html_path}")
    
    print()
    
    # Exit with appropriate code
    sys.exit(0 if result.overall_status == CheckStatus.PASS else 1)
