# modules/security_checks/rate_limit.py
"""
Rate Limiting Detection Module

Tests for presence of rate limiting / throttling controls:
- Detects 429 (Too Many Requests) responses
- Checks for rate limit headers (X-RateLimit-*, Retry-After)
- Adaptive probing to avoid overwhelming servers
- Early termination when rate limiting detected

References:
- OWASP: https://owasp.org/API-Security/editions/2023/en/0xa4-unrestricted-resource-consumption/
- CWE-307: Improper Restriction of Excessive Authentication Attempts
- CWE-770: Allocation of Resources Without Limits or Throttling
"""

from __future__ import annotations
import asyncio
import logging
import time
from typing import Dict, List, Optional
import httpx

from modules.security_checks.base import AbstractSecurityCheck
from modules.security_types import (
    SecurityFinding,
    SecurityCheckResult,
    CheckStatus,
    Severity,
)

logger = logging.getLogger(__name__)


class RateLimitCheck(AbstractSecurityCheck):
    """
    Rate limiting detection check.
    
    Tests:
    1. Rapid request detection (429 responses)
    2. Rate limit headers (X-RateLimit-*, Retry-After)
    3. Adaptive probing (stops early if rate limiting detected)
    4. Response time degradation (soft rate limiting)
    """
    
    def __init__(
        self,
        timeout_s: float = 15.0,
        probe_count: int = 15,
        probe_delay_ms: int = 50,
        early_stop_threshold: int = 3,
    ):
        """
        Initialize rate limit check.
        
        Args:
            timeout_s: Request timeout
            probe_count: Number of rapid requests to send
            probe_delay_ms: Delay between requests (milliseconds)
            early_stop_threshold: Stop after N 429 responses
        """
        super().__init__(timeout_s=timeout_s)
        self.probe_count = probe_count
        self.probe_delay_ms = probe_delay_ms
        self.early_stop_threshold = early_stop_threshold
    
    @property
    def name(self) -> str:
        return "rate_limiting"
    
    @property
    def source(self) -> str:
        return "rate_limit"
    
    async def run_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """Execute async rate limit check."""
        try:
            probe_results = await self._probe_rate_limiting_async(url, client)
            self._analyze_rate_limiting(url, probe_results, result)
        
        except Exception as e:
            logger.warning("Rate limiting check failed for %s: %s", url, e)
            result.add_finding(SecurityFinding(
                check_name="rate_limit_error",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message=f"Rate limiting check failed: {str(e)}",
                source=self.source,
            ))
    
    def run_sync(
        self,
        url: str,
        client: httpx.Client,
        result: SecurityCheckResult,
    ) -> None:
        """Execute sync rate limit check."""
        try:
            probe_results = self._probe_rate_limiting_sync(url, client)
            self._analyze_rate_limiting(url, probe_results, result)
        
        except Exception as e:
            logger.warning("Rate limiting check failed for %s: %s", url, e)
    
    async def _probe_rate_limiting_async(
        self,
        url: str,
        client: httpx.AsyncClient,
    ) -> List[Dict[str, any]]:
        """Send rapid requests to test for rate limiting."""
        results = []
        consecutive_429s = 0
        
        logger.info("Probing rate limiting for %s (%d requests)", url, self.probe_count)
        
        for i in range(self.probe_count):
            start_time = time.time()
            
            try:
                response = await client.get(
                    url,
                    headers={
                        "User-Agent": "SecurityEngine/2.0 (Rate Limit Scanner)",
                        "X-Test-Probe": f"rate-limit-{i+1}",
                    }
                )
                
                duration_ms = (time.time() - start_time) * 1000
                
                # Extract rate limit headers
                rate_limit_headers = self._extract_rate_limit_headers(response.headers)
                
                result = {
                    "request_num": i + 1,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                    "rate_limit_headers": rate_limit_headers,
                }
                
                results.append(result)
                
                # Early stopping if rate limited
                if response.status_code == 429:
                    consecutive_429s += 1
                    if consecutive_429s >= self.early_stop_threshold:
                        logger.info("Rate limiting detected, stopping probe early")
                        break
                else:
                    consecutive_429s = 0
            
            except httpx.HTTPError as e:
                results.append({
                    "request_num": i + 1,
                    "status_code": 0,
                    "error": str(e),
                })
            
            # Delay between requests
            if i < self.probe_count - 1:
                await asyncio.sleep(self.probe_delay_ms / 1000)
        
        return results
    
    def _probe_rate_limiting_sync(
        self,
        url: str,
        client: httpx.Client,
    ) -> List[Dict[str, any]]:
        """Sync version of rate limiting probe."""
        results = []
        consecutive_429s = 0
        
        for i in range(self.probe_count):
            start_time = time.time()
            
            try:
                response = client.get(url)
                duration_ms = (time.time() - start_time) * 1000
                
                rate_limit_headers = self._extract_rate_limit_headers(response.headers)
                
                results.append({
                    "request_num": i + 1,
                    "status_code": response.status_code,
                    "duration_ms": round(duration_ms, 2),
                    "rate_limit_headers": rate_limit_headers,
                })
                
                if response.status_code == 429:
                    consecutive_429s += 1
                    if consecutive_429s >= self.early_stop_threshold:
                        break
                else:
                    consecutive_429s = 0
            
            except httpx.HTTPError as e:
                results.append({
                    "request_num": i + 1,
                    "status_code": 0,
                    "error": str(e),
                })
            
            if i < self.probe_count - 1:
                time.sleep(self.probe_delay_ms / 1000)
        
        return results
    
    def _extract_rate_limit_headers(self, headers: httpx.Headers) -> Dict[str, str]:
        """Extract rate limit related headers."""
        rate_limit_headers = {}
        
        # Common rate limit header patterns
        header_patterns = [
            "x-ratelimit-limit",
            "x-ratelimit-remaining",
            "x-ratelimit-reset",
            "x-rate-limit-limit",
            "x-rate-limit-remaining",
            "x-rate-limit-reset",
            "ratelimit-limit",
            "ratelimit-remaining",
            "ratelimit-reset",
            "retry-after",
        ]
        
        headers_lower = {k.lower(): v for k, v in headers.items()}
        
        for pattern in header_patterns:
            if pattern in headers_lower:
                rate_limit_headers[pattern] = headers_lower[pattern]
        
        return rate_limit_headers
    
    def _analyze_rate_limiting(
        self,
        url: str,
        probe_results: List[Dict[str, any]],
        result: SecurityCheckResult,
    ) -> None:
        """Analyze probe results to determine rate limiting presence."""
        if not probe_results:
            result.add_finding(SecurityFinding(
                check_name="rate_limit_check_failed",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message="Rate limiting check failed - no results",
                source=self.source,
            ))
            return
        
        # Count status codes
        status_codes = [r.get("status_code", 0) for r in probe_results]
        count_429 = sum(1 for s in status_codes if s == 429)
        count_200 = sum(1 for s in status_codes if s == 200)
        count_errors = sum(1 for s in status_codes if s == 0)
        
        # Check for rate limit headers
        has_rate_limit_headers = any(
            r.get("rate_limit_headers")
            for r in probe_results
        )
        
        # Analyze response time degradation
        durations = [r.get("duration_ms", 0) for r in probe_results if r.get("duration_ms")]
        has_degradation = False
        if len(durations) >= 5:
            first_half_avg = sum(durations[:len(durations)//2]) / (len(durations)//2)
            second_half_avg = sum(durations[len(durations)//2:]) / (len(durations) - len(durations)//2)
            if second_half_avg > first_half_avg * 2:  # 2x slower
                has_degradation = True
        
        # Decision logic
        if count_429 >= self.early_stop_threshold:
            # Strong rate limiting detected
            result.add_finding(SecurityFinding(
                check_name="rate_limiting_detected",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message=f"Rate limiting is active (detected {count_429} HTTP 429 responses)",
                details={
                    "total_requests": len(probe_results),
                    "status_429_count": count_429,
                    "status_200_count": count_200,
                    "has_rate_limit_headers": has_rate_limit_headers,
                    "sample_statuses": status_codes[:10],
                },
                source=self.source,
            ))
        
        elif has_rate_limit_headers:
            # Rate limiting present via headers
            sample_headers = next(
                (r["rate_limit_headers"] for r in probe_results if r.get("rate_limit_headers")),
                {}
            )
            
            result.add_finding(SecurityFinding(
                check_name="rate_limiting_headers_present",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message="Rate limiting headers detected (policy enforced)",
                details={
                    "rate_limit_headers": sample_headers,
                    "total_requests": len(probe_results),
                },
                source=self.source,
            ))
        
        elif has_degradation:
            # Soft rate limiting (response time degradation)
            result.add_finding(SecurityFinding(
                check_name="rate_limiting_soft_detected",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message="Soft rate limiting detected (response time degradation)",
                details={
                    "total_requests": len(probe_results),
                    "response_time_increase": "2x or more",
                },
                source=self.source,
            ))
        
        elif count_200 == len(probe_results):
            # No rate limiting detected
            result.add_finding(SecurityFinding(
                check_name="rate_limiting_not_detected",
                status=CheckStatus.WARNING,
                severity=Severity.MEDIUM,
                message=f"No rate limiting detected (all {count_200} rapid requests succeeded)",
                details={
                    "total_requests": len(probe_results),
                    "status_200_count": count_200,
                    "probe_delay_ms": self.probe_delay_ms,
                },
                recommendation="Implement rate limiting to prevent abuse and DoS attacks",
                source=self.source,
                cwe_id="CWE-307",
                owasp_category="A04:2021 - Insecure Design",
                confidence=0.8,
                references=[
                    "https://owasp.org/API-Security/editions/2023/en/0xa4-unrestricted-resource-consumption/",
                    "https://cheatsheetseries.owasp.org/cheatsheets/Denial_of_Service_Cheat_Sheet.html",
                ],
            ))
        
        else:
            # Mixed results
            result.add_finding(SecurityFinding(
                check_name="rate_limiting_unclear",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message="Rate limiting behavior is unclear (mixed responses)",
                details={
                    "total_requests": len(probe_results),
                    "status_codes": status_codes,
                    "count_429": count_429,
                    "count_200": count_200,
                    "count_errors": count_errors,
                },
                source=self.source,
            ))
