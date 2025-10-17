# modules/security_checks/headers.py
"""
Security headers validation check.
Implements OWASP Secure Headers Project recommendations.
"""

from __future__ import annotations
import logging
from typing import Dict, List
import httpx

from modules.security_checks.base import AbstractSecurityCheck
from modules.security_types import SecurityFinding, SecurityCheckResult, CheckStatus, Severity

logger = logging.getLogger(__name__)


class SecurityHeadersCheck(AbstractSecurityCheck):
    """Validates HTTP security headers."""
    
    def __init__(
        self,
        timeout_s: float = 15.0,
        required_headers: Optional[List[str]] = None,
        recommended_headers: Optional[List[str]] = None,
    ):
        super().__init__(timeout_s=timeout_s)
        
        self.required_headers = required_headers or [
            "content-security-policy",
            "x-content-type-options",
            "x-frame-options",
            "referrer-policy",
            "strict-transport-security",
            "permissions-policy",
        ]
        
        self.recommended_headers = recommended_headers or [
            "x-xss-protection",
            "cross-origin-opener-policy",
            "cross-origin-resource-policy",
            "cross-origin-embedder-policy",
        ]
    
    @property
    def name(self) -> str:
        return "security_headers"
    
    @property
    def source(self) -> str:
        return "headers"
    
    async def run_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """Execute async header check."""
        try:
            response = await client.get(url, headers={"User-Agent": "SecurityEngine/1.0"})
            self._analyze_headers(response, result)
        except httpx.HTTPError as e:
            logger.warning("Failed to check headers for %s: %s", url, e)
            raise
    
    def run_sync(
        self,
        url: str,
        client: httpx.Client,
        result: SecurityCheckResult,
    ) -> None:
        """Execute sync header check."""
        try:
            response = client.get(url, headers={"User-Agent": "SecurityEngine/1.0"})
            self._analyze_headers(response, result)
        except httpx.HTTPError as e:
            logger.warning("Failed to check headers for %s: %s", url, e)
            raise
    
    def _analyze_headers(self, response: httpx.Response, result: SecurityCheckResult):
        """Analyze security headers with CSP parsing."""
        headers_lower = {k.lower(): v for k, v in response.headers.items()}
        
        missing_required = []
        present_required = []
        
        # Check required headers
        for header in self.required_headers:
            if header in headers_lower and headers_lower[header]:
                present_required.append(header)
                
                # ✨ Enhanced CSP parsing
                if header == "content-security-policy":
                    self._analyze_csp(headers_lower[header], result)
                else:
                    result.add_finding(SecurityFinding(
                        check_name=f"header_{header}",
                        status=CheckStatus.PASS,
                        severity=Severity.INFO,
                        message=f"Security header '{header}' is present",
                        details={"header": header, "value": headers_lower[header][:100]},
                        source=self.source,
                        cwe_id="CWE-693",
                        owasp_category="A05:2021 - Security Misconfiguration",
                        references=[f"https://owasp.org/www-project-secure-headers/#{header}"],
                    ))
            else:
                missing_required.append(header)
                
                result.add_finding(SecurityFinding(
                    check_name=f"header_{header}_missing",
                    status=CheckStatus.FAIL,
                    severity=Severity.HIGH,
                    message=f"Required security header '{header}' is missing",
                    details={"header": header},
                    recommendation=self._get_header_recommendation(header),
                    source=self.source,
                    cwe_id="CWE-693",
                    owasp_category="A05:2021 - Security Misconfiguration",
                    references=["https://owasp.org/www-project-secure-headers/"],
                ))
        
        # Check recommended headers
        for header in self.recommended_headers:
            if header not in headers_lower or not headers_lower[header]:
                result.add_finding(SecurityFinding(
                    check_name=f"header_{header}_recommended",
                    status=CheckStatus.WARNING,
                    severity=Severity.MEDIUM,
                    message=f"Recommended header '{header}' is missing",
                    details={"header": header},
                    recommendation=self._get_header_recommendation(header),
                    source=self.source,
                    cwe_id="CWE-693",
                    owasp_category="A05:2021 - Security Misconfiguration",
                ))
    
    def _analyze_csp(self, csp_value: str, result: SecurityCheckResult):
        """
        ✨ Parse CSP and detect unsafe directives.
        
        Args:
            csp_value: Content-Security-Policy header value
            result: Result object to add findings to
        """
        csp_lower = csp_value.lower()
        
        # Check for report-only mode
        if "report-uri" in csp_lower or "report-to" in csp_lower:
            if not any(directive in csp_lower for directive in ["default-src", "script-src", "style-src"]):
                result.add_finding(SecurityFinding(
                    check_name="csp_report_only",
                    status=CheckStatus.WARNING,
                    severity=Severity.MEDIUM,
                    message="CSP appears to be in report-only mode",
                    details={"csp": csp_value[:200]},
                    recommendation="Switch from report-only to enforcement mode",
                    source=self.source,
                    cwe_id="CWE-693",
                    owasp_category="A05:2021 - Security Misconfiguration",
                ))
        
        # Check for unsafe directives
        unsafe_directives = []
        if "'unsafe-inline'" in csp_value:
            unsafe_directives.append("unsafe-inline")
        if "'unsafe-eval'" in csp_value:
            unsafe_directives.append("unsafe-eval")
        
        if unsafe_directives:
            result.add_finding(SecurityFinding(
                check_name="csp_unsafe_directives",
                status=CheckStatus.FAIL,
                severity=Severity.HIGH,
                message=f"CSP contains unsafe directives: {', '.join(unsafe_directives)}",
                details={"unsafe_directives": unsafe_directives, "csp": csp_value[:200]},
                recommendation="Remove unsafe-inline and unsafe-eval; use nonces or hashes",
                source=self.source,
                cwe_id="CWE-1336",
                owasp_category="A05:2021 - Security Misconfiguration",
                references=["https://content-security-policy.com/unsafe-inline/"],
            ))
        else:
            result.add_finding(SecurityFinding(
                check_name="csp_valid",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message="CSP is present without unsafe directives",
                details={"csp": csp_value[:200]},
                source=self.source,
            ))
    
    def _get_header_recommendation(self, header: str) -> str:
        """Get recommendation for missing header."""
        recommendations = {
            "content-security-policy": "Add: Content-Security-Policy: default-src 'self'; script-src 'self'; object-src 'none';",
            "x-content-type-options": "Add: X-Content-Type-Options: nosniff",
            "x-frame-options": "Add: X-Frame-Options: DENY or SAMEORIGIN",
            "referrer-policy": "Add: Referrer-Policy: strict-origin-when-cross-origin",
            "strict-transport-security": "Add: Strict-Transport-Security: max-age=31536000; includeSubDomains",
            "permissions-policy": "Add: Permissions-Policy: geolocation=(), microphone=(), camera=()",
        }
        return recommendations.get(header, f"Add '{header}' security header")
