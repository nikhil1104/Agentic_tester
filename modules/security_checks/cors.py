# modules/security_checks/cors.py
"""
CORS (Cross-Origin Resource Sharing) Security Check Module

Validates CORS configuration according to OWASP standards:
- Access-Control-Allow-Origin wildcard usage
- Access-Control-Allow-Credentials with wildcard
- Access-Control-Allow-Methods excessive permissions
- Access-Control-Allow-Headers validation
- CORS preflight responses

References:
- OWASP: https://owasp.org/www-community/attacks/CORS_OriginHeaderScrutiny
- CWE-942: Permissive Cross-domain Policy with Untrusted Domains
"""

from __future__ import annotations
import logging
from typing import Dict, List, Optional
from urllib.parse import urlparse
import httpx

from modules.security_checks.base import AbstractSecurityCheck
from modules.security_types import (
    SecurityFinding,
    SecurityCheckResult,
    CheckStatus,
    Severity,
)

logger = logging.getLogger(__name__)


class CORSSecurityCheck(AbstractSecurityCheck):
    """
    Comprehensive CORS security validation.
    
    Checks:
    1. Access-Control-Allow-Origin wildcard (*)
    2. Credentials with wildcard origin (security violation)
    3. Unsafe methods exposure (PUT, DELETE, TRACE)
    4. Overly permissive headers
    5. Missing CORS headers (secure default)
    6. Reflected origin (potential vulnerability)
    """
    
    # Unsafe HTTP methods that should be restricted
    UNSAFE_METHODS = ["PUT", "DELETE", "TRACE", "CONNECT"]
    
    # Sensitive headers that should be restricted
    SENSITIVE_HEADERS = [
        "authorization",
        "x-api-key",
        "x-auth-token",
        "cookie",
    ]
    
    def __init__(self, timeout_s: float = 15.0):
        super().__init__(timeout_s=timeout_s)
    
    @property
    def name(self) -> str:
        return "cors_security"
    
    @property
    def source(self) -> str:
        return "cors"
    
    async def run_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """Execute async CORS security check."""
        try:
            # Test 1: GET request (baseline)
            get_response = await client.get(
                url,
                headers={"User-Agent": "SecurityEngine/2.0 (CORS Scanner)"}
            )
            self._analyze_cors_headers(url, get_response, "GET", result)
            
            # Test 2: OPTIONS preflight (simulate cross-origin request)
            try:
                options_response = await client.options(
                    url,
                    headers={
                        "Origin": "https://evil.com",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type,x-custom-header",
                        "User-Agent": "SecurityEngine/2.0 (CORS Scanner)",
                    }
                )
                self._analyze_preflight(url, options_response, result)
            except httpx.HTTPError as e:
                logger.debug("OPTIONS preflight not supported: %s", e)
                # This is fine - many servers don't support OPTIONS
            
            # Test 3: Check for reflected origin vulnerability
            await self._test_reflected_origin(url, client, result)
        
        except httpx.HTTPError as e:
            logger.warning("Failed to check CORS for %s: %s", url, e)
            result.add_finding(SecurityFinding(
                check_name="cors_security_error",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message=f"Failed to check CORS policy: {str(e)}",
                source=self.source,
            ))
    
    def run_sync(
        self,
        url: str,
        client: httpx.Client,
        result: SecurityCheckResult,
    ) -> None:
        """Execute sync CORS security check."""
        try:
            # Test 1: GET request
            get_response = client.get(
                url,
                headers={"User-Agent": "SecurityEngine/2.0 (CORS Scanner)"}
            )
            self._analyze_cors_headers(url, get_response, "GET", result)
            
            # Test 2: OPTIONS preflight
            try:
                options_response = client.options(
                    url,
                    headers={
                        "Origin": "https://evil.com",
                        "Access-Control-Request-Method": "POST",
                        "Access-Control-Request-Headers": "content-type",
                    }
                )
                self._analyze_preflight(url, options_response, result)
            except httpx.HTTPError:
                pass  # OPTIONS not supported
            
            # Test 3: Reflected origin test (sync version)
            self._test_reflected_origin_sync(url, client, result)
        
        except httpx.HTTPError as e:
            logger.warning("Failed to check CORS for %s: %s", url, e)
    
    def _analyze_cors_headers(
        self,
        url: str,
        response: httpx.Response,
        method: str,
        result: SecurityCheckResult,
    ) -> None:
        """Analyze CORS headers from response."""
        headers = {k.lower(): v for k, v in response.headers.items()}
        
        # Extract CORS headers
        acao = headers.get("access-control-allow-origin")
        acac = headers.get("access-control-allow-credentials")
        acam = headers.get("access-control-allow-methods")
        acah = headers.get("access-control-allow-headers")
        acma = headers.get("access-control-max-age")
        
        # Case 1: No CORS headers (secure default)
        if not acao:
            result.add_finding(SecurityFinding(
                check_name="cors_not_enabled",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message="CORS is not enabled (secure default - no Access-Control-Allow-Origin header)",
                details={"cors_enabled": False},
                source=self.source,
            ))
            return
        
        # Case 2: Wildcard origin (*)
        if acao == "*":
            # Check if credentials are also allowed (CRITICAL vulnerability)
            if acac and acac.lower() == "true":
                result.add_finding(SecurityFinding(
                    check_name="cors_wildcard_with_credentials",
                    status=CheckStatus.FAIL,
                    severity=Severity.CRITICAL,
                    message="CORS allows wildcard origin (*) with credentials - CRITICAL VULNERABILITY",
                    details={
                        "acao": acao,
                        "acac": acac,
                        "impact": "Any website can make authenticated requests to this API",
                    },
                    recommendation="Remove 'Access-Control-Allow-Credentials: true' or restrict origins to specific domains",
                    source=self.source,
                    cwe_id="CWE-942",
                    owasp_category="A05:2021 - Security Misconfiguration",
                    confidence=1.0,
                    references=[
                        "https://owasp.org/www-community/attacks/CORS_OriginHeaderScrutiny",
                        "https://portswigger.net/web-security/cors",
                    ],
                ))
            else:
                result.add_finding(SecurityFinding(
                    check_name="cors_wildcard_origin",
                    status=CheckStatus.FAIL,
                    severity=Severity.HIGH,
                    message="CORS allows all origins (*) - potential security risk",
                    details={
                        "acao": acao,
                        "credentials_enabled": bool(acac),
                    },
                    recommendation="Restrict Access-Control-Allow-Origin to specific trusted domains",
                    source=self.source,
                    cwe_id="CWE-942",
                    owasp_category="A05:2021 - Security Misconfiguration",
                    confidence=0.9,
                    references=[
                        "https://owasp.org/www-community/attacks/CORS_OriginHeaderScrutiny",
                    ],
                ))
        
        # Case 3: Specific origin (good, but verify it's not reflected)
        else:
            result.add_finding(SecurityFinding(
                check_name="cors_specific_origin",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message=f"CORS restricted to specific origin: {acao}",
                details={"acao": acao, "credentials_enabled": acac == "true" if acac else False},
                source=self.source,
            ))
        
        # Check unsafe methods
        if acam:
            unsafe_methods = [m.strip().upper() for m in acam.split(",") if m.strip().upper() in self.UNSAFE_METHODS]
            
            if unsafe_methods:
                result.add_finding(SecurityFinding(
                    check_name="cors_unsafe_methods",
                    status=CheckStatus.FAIL,
                    severity=Severity.MEDIUM,
                    message=f"CORS allows unsafe HTTP methods: {', '.join(unsafe_methods)}",
                    details={
                        "acam": acam,
                        "unsafe_methods": unsafe_methods,
                    },
                    recommendation="Restrict Access-Control-Allow-Methods to safe methods (GET, POST, OPTIONS)",
                    source=self.source,
                    cwe_id="CWE-942",
                    owasp_category="A05:2021 - Security Misconfiguration",
                    confidence=0.85,
                ))
        
        # Check sensitive headers
        if acah:
            acah_lower = acah.lower()
            if "*" in acah:
                result.add_finding(SecurityFinding(
                    check_name="cors_wildcard_headers",
                    status=CheckStatus.FAIL,
                    severity=Severity.MEDIUM,
                    message="CORS allows all headers (*) via Access-Control-Allow-Headers",
                    details={"acah": acah},
                    recommendation="Specify explicit allowed headers instead of using wildcard",
                    source=self.source,
                    cwe_id="CWE-942",
                    owasp_category="A05:2021 - Security Misconfiguration",
                ))
            else:
                exposed_sensitive = [h for h in self.SENSITIVE_HEADERS if h in acah_lower]
                if exposed_sensitive:
                    result.add_finding(SecurityFinding(
                        check_name="cors_sensitive_headers",
                        status=CheckStatus.WARNING,
                        severity=Severity.MEDIUM,
                        message=f"CORS exposes sensitive headers: {', '.join(exposed_sensitive)}",
                        details={
                            "acah": acah,
                            "exposed_sensitive": exposed_sensitive,
                        },
                        recommendation="Restrict Access-Control-Allow-Headers to non-sensitive headers only",
                        source=self.source,
                        cwe_id="CWE-942",
                        owasp_category="A05:2021 - Security Misconfiguration",
                    ))
        
        # Check Max-Age (caching of preflight)
        if acma:
            try:
                max_age = int(acma)
                if max_age > 86400:  # > 24 hours
                    result.add_finding(SecurityFinding(
                        check_name="cors_excessive_max_age",
                        status=CheckStatus.WARNING,
                        severity=Severity.LOW,
                        message=f"CORS preflight cache duration is excessive: {max_age} seconds",
                        details={"acma": acma, "duration_hours": max_age / 3600},
                        recommendation="Set Access-Control-Max-Age to 24 hours or less",
                        source=self.source,
                    ))
            except ValueError:
                pass
    
    def _analyze_preflight(
        self,
        url: str,
        response: httpx.Response,
        result: SecurityCheckResult,
    ) -> None:
        """Analyze OPTIONS preflight response."""
        headers = {k.lower(): v for k, v in response.headers.items()}
        
        acao = headers.get("access-control-allow-origin")
        
        if acao:
            result.add_finding(SecurityFinding(
                check_name="cors_preflight_supported",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message="CORS preflight (OPTIONS) is supported",
                details={
                    "preflight_status": response.status_code,
                    "acao": acao,
                },
                source=self.source,
            ))
    
    async def _test_reflected_origin(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """Test if origin is reflected (potential vulnerability)."""
        try:
            test_origin = "https://attacker-controlled.com"
            
            response = await client.get(
                url,
                headers={
                    "Origin": test_origin,
                    "User-Agent": "SecurityEngine/2.0 (CORS Scanner)",
                }
            )
            
            headers = {k.lower(): v for k, v in response.headers.items()}
            acao = headers.get("access-control-allow-origin")
            
            if acao == test_origin:
                result.add_finding(SecurityFinding(
                    check_name="cors_reflected_origin",
                    status=CheckStatus.FAIL,
                    severity=Severity.HIGH,
                    message="CORS policy reflects arbitrary origins - SECURITY VULNERABILITY",
                    details={
                        "test_origin": test_origin,
                        "reflected_origin": acao,
                        "impact": "Attacker can bypass CORS restrictions by setting Origin header",
                    },
                    recommendation="Implement whitelist of allowed origins; do not reflect Origin header",
                    source=self.source,
                    cwe_id="CWE-942",
                    owasp_category="A05:2021 - Security Misconfiguration",
                    confidence=1.0,
                    references=[
                        "https://portswigger.net/web-security/cors#server-generated-acao-header-from-client-specified-origin-header",
                    ],
                ))
        
        except Exception as e:
            logger.debug("Reflected origin test failed: %s", e)
    
    def _test_reflected_origin_sync(
        self,
        url: str,
        client: httpx.Client,
        result: SecurityCheckResult,
    ) -> None:
        """Sync version of reflected origin test."""
        try:
            test_origin = "https://attacker-controlled.com"
            
            response = client.get(
                url,
                headers={"Origin": test_origin}
            )
            
            headers = {k.lower(): v for k, v in response.headers.items()}
            acao = headers.get("access-control-allow-origin")
            
            if acao == test_origin:
                result.add_finding(SecurityFinding(
                    check_name="cors_reflected_origin",
                    status=CheckStatus.FAIL,
                    severity=Severity.HIGH,
                    message="CORS policy reflects arbitrary origins",
                    source=self.source,
                    cwe_id="CWE-942",
                    owasp_category="A05:2021 - Security Misconfiguration",
                ))
        except Exception:
            pass
