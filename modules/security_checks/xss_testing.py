# modules/security_checks/xss_testing.py
"""
XSS (Cross-Site Scripting) Testing Module

Tests for reflected XSS vulnerabilities using safe probe payloads.
"""

from __future__ import annotations
import logging
from urllib.parse import parse_qs, urlparse, urlencode
import httpx

from modules.security_checks.base import AbstractSecurityCheck
from modules.security_types import (
    SecurityFinding,
    SecurityCheckResult,
    CheckStatus,
    Severity,
)

logger = logging.getLogger(__name__)


class XSSCheck(AbstractSecurityCheck):
    """Test for reflected XSS vulnerabilities."""
    
    # XSS test payloads (safe, non-executing markers)
    TEST_PAYLOADS = [
        "<script>alert('XSS')</script>",
        "<img src=x onerror=alert('XSS')>",
        "javascript:alert('XSS')",
        "<svg/onload=alert('XSS')>",
        "'-alert('XSS')-'",
        "<iframe src=javascript:alert('XSS')>",
    ]
    
    def __init__(self, timeout_s: float = 15.0):
        super().__init__(timeout_s=timeout_s)
    
    @property
    def name(self) -> str:
        return "xss_testing"
    
    @property
    def source(self) -> str:
        return "active_testing"
    
    async def run_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ):
        """Test for reflected XSS vulnerabilities."""
        try:
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            
            if not params:
                result.add_finding(SecurityFinding(
                    check_name="xss_no_params",
                    status=CheckStatus.PASS,
                    severity=Severity.INFO,
                    message="No query parameters to test for XSS",
                    details={"note": "XSS testing requires URL parameters"},
                    source=self.source,
                ))
                return
            
            vulnerabilities_found = []
            
            for param_name in params.keys():
                for payload in self.TEST_PAYLOADS:
                    # Create test URL with payload
                    test_params = params.copy()
                    test_params[param_name] = [payload]
                    test_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{urlencode(test_params, doseq=True)}"
                    
                    try:
                        # Send request
                        response = await client.get(test_url)
                        
                        # Check if payload is reflected unescaped
                        if payload in response.text:
                            vulnerabilities_found.append({
                                "parameter": param_name,
                                "payload": payload,
                            })
                            
                            result.add_finding(SecurityFinding(
                                check_name=f"xss_{param_name}",
                                status=CheckStatus.FAIL,
                                severity=Severity.HIGH,
                                message=f"Reflected XSS vulnerability detected in parameter '{param_name}'",
                                details={
                                    "parameter": param_name,
                                    "payload": payload,
                                    "reflected": True,
                                    "test_url": test_url[:200],
                                },
                                recommendation="Sanitize and encode all user input before rendering in HTML. Use Content-Security-Policy header.",
                                source=self.source,
                                cwe_id="CWE-79",
                                owasp_category="A03:2021 - Injection",
                                confidence=0.85,
                                references=[
                                    "https://owasp.org/www-community/attacks/xss/",
                                    "https://cheatsheetseries.owasp.org/cheatsheets/Cross_Site_Scripting_Prevention_Cheat_Sheet.html",
                                ],
                            ))
                            break  # Stop testing this param once vulnerability found
                    
                    except httpx.HTTPError as e:
                        logger.debug("XSS test request failed for %s: %s", param_name, e)
                        continue
            
            if not vulnerabilities_found:
                result.add_finding(SecurityFinding(
                    check_name="xss_test",
                    status=CheckStatus.PASS,
                    severity=Severity.INFO,
                    message=f"No reflected XSS vulnerabilities detected in {len(params)} parameter(s)",
                    details={
                        "parameters_tested": list(params.keys()),
                        "payloads_tested": len(self.TEST_PAYLOADS),
                    },
                    source=self.source,
                ))
        
        except Exception as e:
            logger.error("XSS testing failed: %s", e)
            result.add_finding(SecurityFinding(
                check_name="xss_test_error",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message=f"XSS testing failed: {str(e)}",
                source=self.source,
            ))
    
    def run_sync(
        self,
        url: str,
        client: httpx.Client,
        result: SecurityCheckResult,
    ):
        """Sync version of XSS testing."""
        import asyncio
        asyncio.run(self.run_async(url, client, result))
