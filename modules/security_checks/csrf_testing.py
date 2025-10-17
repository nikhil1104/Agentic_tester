# modules/security_checks/csrf_testing.py
"""
CSRF (Cross-Site Request Forgery) Testing Module

Production-grade CSRF vulnerability testing with comprehensive coverage:
- Form-based CSRF token validation
- Cookie SameSite attribute verification
- State-changing method protection
- Origin/Referer header validation

Standards Compliance:
- OWASP Top 10 2021: A01 (Broken Access Control)
- CWE-352: Cross-Site Request Forgery
- PCI DSS 6.5.9: Improper access control
"""

from __future__ import annotations

import logging
import re
import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from enum import Enum

import httpx
from httpx import TimeoutException, HTTPStatusError

from modules.security_checks.base import AbstractSecurityCheck
from modules.security_types import (
    SecurityFinding,
    SecurityCheckResult,
    CheckStatus,
    Severity,
)

logger = logging.getLogger(__name__)


# ✅ Constants (Industry Best Practice)
class CSRFConfig:
    """Configuration constants for CSRF testing."""
    
    # CSRF token field name patterns
    CSRF_TOKEN_PATTERNS: Set[str] = {
        'csrf', 'csrftoken', '_csrf', 'csrf_token',
        'authenticity_token', '_token', 'token',
        'xsrf', 'xsrftoken', '_xsrf',
    }
    
    # HTTP methods that should be protected
    STATE_CHANGING_METHODS: Set[str] = {'POST', 'PUT', 'DELETE', 'PATCH'}
    
    # Safe HTTP methods (don't need CSRF protection)
    SAFE_METHODS: Set[str] = {'GET', 'HEAD', 'OPTIONS'}
    
    # SameSite attribute values
    SAMESITE_VALUES: Set[str] = {'strict', 'lax', 'none'}
    
    # Timeout configurations
    REQUEST_TIMEOUT: float = 10.0
    MAX_RETRIES: int = 2


@dataclass
class FormInfo:
    """Structured form information."""
    form_number: int
    action: str
    method: str
    has_csrf_token: bool
    input_count: int
    form_id: Optional[str] = None


@dataclass
class CookieInfo:
    """Structured cookie information."""
    name: str
    secure: bool
    httponly: bool
    samesite: Optional[str]
    domain: Optional[str]
    path: str


class CSRFCheck(AbstractSecurityCheck):
    """
    Production-grade CSRF vulnerability testing.
    
    Features:
    - HTML form analysis with BeautifulSoup
    - Cookie attribute validation
    - State-changing method testing
    - Configurable timeout and retry logic
    - Comprehensive error handling
    
    Example:
        >>> check = CSRFCheck(timeout_s=15.0)
        >>> result = SecurityCheckResult(url="https://example.com", ...)
        >>> await check.run_async("https://example.com", client, result)
    """
    
    def __init__(self, timeout_s: float = 15.0):
        """
        Initialize CSRF checker.
        
        Args:
            timeout_s: Request timeout in seconds (default: 15.0)
        """
        super().__init__(timeout_s=timeout_s)
        self._timeout = httpx.Timeout(
            timeout=timeout_s,
            connect=5.0,
            read=timeout_s,
        )
    
    @property
    def name(self) -> str:
        """Check name identifier."""
        return "csrf_testing"
    
    @property
    def source(self) -> str:
        """Source identifier for findings."""
        return "csrf_protection"
    
    async def run_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """
        Execute CSRF vulnerability tests asynchronously.
        
        Args:
            url: Target URL to test
            client: Async HTTP client
            result: Result container to populate
            
        Raises:
            ValueError: If URL is invalid
            TimeoutError: If request exceeds timeout
        """
        # ✅ Input validation
        if not url or not url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL: {url}")
        
        logger.info("Starting CSRF testing for: %s", url)
        
        try:
            # Fetch page with retry logic
            response = await self._fetch_with_retry(url, client)
            
            # Run all CSRF tests
            await asyncio.gather(
                self._check_forms_for_csrf_tokens(url, response, result),
                self._check_cookie_samesite(response, result),
                self._check_state_changing_methods(url, client, result),
                return_exceptions=True,  # Don't fail entire test if one check fails
            )
            
            logger.info("CSRF testing completed for: %s", url)
            
        except TimeoutException as e:
            logger.error("CSRF testing timeout for %s: %s", url, e)
            result.add_finding(SecurityFinding(
                check_name="csrf_test_timeout",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message=f"CSRF testing timed out after {self.timeout_s}s",
                details={"error": str(e)},
                source=self.source,
            ))
        except HTTPStatusError as e:
            logger.error("HTTP error during CSRF testing for %s: %s", url, e)
            result.add_finding(SecurityFinding(
                check_name="csrf_test_http_error",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message=f"HTTP error during CSRF testing: {e.response.status_code}",
                details={"status_code": e.response.status_code},
                source=self.source,
            ))
        except Exception as e:
            logger.error("CSRF testing failed for %s: %s", url, e, exc_info=True)
            result.add_finding(SecurityFinding(
                check_name="csrf_test_error",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message=f"CSRF testing failed: {str(e)[:200]}",
                source=self.source,
            ))
    
    async def _fetch_with_retry(
        self,
        url: str,
        client: httpx.AsyncClient,
        max_retries: int = CSRFConfig.MAX_RETRIES,
    ) -> httpx.Response:
        """
        Fetch URL with exponential backoff retry.
        
        Args:
            url: Target URL
            client: HTTP client
            max_retries: Maximum retry attempts
            
        Returns:
            HTTP response
            
        Raises:
            httpx.HTTPError: If all retries fail
        """
        last_exception = None
        
        for attempt in range(max_retries + 1):
            try:
                return await client.get(
                    url,
                    timeout=self._timeout,
                    follow_redirects=True,
                )
            except (TimeoutException, httpx.ConnectError) as e:
                last_exception = e
                if attempt < max_retries:
                    wait_time = (2 ** attempt)  # Exponential backoff
                    logger.debug("Retry %d/%d after %ds for %s", 
                               attempt + 1, max_retries, wait_time, url)
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        raise last_exception
    
    async def _check_forms_for_csrf_tokens(
        self,
        url: str,
        response: httpx.Response,
        result: SecurityCheckResult,
    ) -> None:
        """
        Analyze HTML forms for CSRF token presence.
        
        Uses BeautifulSoup for robust HTML parsing.
        Only flags POST/PUT/DELETE/PATCH forms without tokens.
        """
        try:
            # ✅ Lazy import (only when needed)
            try:
                from bs4 import BeautifulSoup
            except ImportError:
                logger.warning("BeautifulSoup not installed, skipping form analysis")
                result.add_finding(SecurityFinding(
                    check_name="csrf_forms_skipped",
                    status=CheckStatus.SKIPPED,
                    severity=Severity.INFO,
                    message="Form analysis skipped (BeautifulSoup not installed)",
                    recommendation="Install BeautifulSoup: pip install beautifulsoup4",
                    source=self.source,
                ))
                return
            
            soup = BeautifulSoup(response.text, 'html.parser')
            forms = soup.find_all('form')
            
            if not forms:
                result.add_finding(SecurityFinding(
                    check_name="csrf_no_forms",
                    status=CheckStatus.PASS,
                    severity=Severity.INFO,
                    message="No HTML forms found to check for CSRF tokens",
                    details={"url": url},
                    source=self.source,
                ))
                return
            
            # Analyze each form
            vulnerable_forms: List[FormInfo] = []
            protected_forms: List[FormInfo] = []
            
            for i, form in enumerate(forms, 1):
                form_info = self._analyze_form(i, form)
                
                # Only flag state-changing forms
                if form_info.method in CSRFConfig.STATE_CHANGING_METHODS:
                    if not form_info.has_csrf_token:
                        vulnerable_forms.append(form_info)
                    else:
                        protected_forms.append(form_info)
            
            # Report findings
            if vulnerable_forms:
                self._report_vulnerable_forms(vulnerable_forms, len(forms), result)
            else:
                self._report_protected_forms(protected_forms, len(forms), result)
        
        except Exception as e:
            logger.error("Form CSRF analysis failed: %s", e, exc_info=True)
            result.add_finding(SecurityFinding(
                check_name="csrf_form_analysis_error",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message=f"Form analysis error: {str(e)[:100]}",
                source=self.source,
            ))
    
    def _analyze_form(self, form_number: int, form) -> FormInfo:
        """
        Extract structured information from HTML form.
        
        Args:
            form_number: Sequential form number
            form: BeautifulSoup form element
            
        Returns:
            FormInfo object with extracted data
        """
        action = form.get('action', '') or 'current_page'
        method = (form.get('method', 'GET') or 'GET').upper()
        form_id = form.get('id')
        
        # Check for CSRF tokens
        has_csrf_token = False
        inputs = form.find_all('input')
        input_count = len(inputs)
        
        for input_field in inputs:
            name = (input_field.get('name') or '').lower()
            input_type = (input_field.get('type') or '').lower()
            
            # Check for CSRF token patterns
            if any(pattern in name for pattern in CSRFConfig.CSRF_TOKEN_PATTERNS):
                has_csrf_token = True
                break
            
            # Hidden fields with long random values are likely tokens
            if input_type == 'hidden':
                value = input_field.get('value', '')
                if len(value) > 20 and not value.isdigit():
                    has_csrf_token = True
                    break
        
        return FormInfo(
            form_number=form_number,
            action=action,
            method=method,
            has_csrf_token=has_csrf_token,
            input_count=input_count,
            form_id=form_id,
        )
    
    def _report_vulnerable_forms(
        self,
        vulnerable_forms: List[FormInfo],
        total_forms: int,
        result: SecurityCheckResult,
    ) -> None:
        """Report forms without CSRF protection."""
        result.add_finding(SecurityFinding(
            check_name="csrf_missing_tokens",
            status=CheckStatus.FAIL,
            severity=Severity.HIGH,
            message=f"Found {len(vulnerable_forms)} form(s) without CSRF tokens (out of {total_forms} total)",
            details={
                "vulnerable_forms": [
                    {
                        "form_number": f.form_number,
                        "action": f.action,
                        "method": f.method,
                        "form_id": f.form_id,
                        "input_count": f.input_count,
                    }
                    for f in vulnerable_forms[:10]  # Limit to first 10
                ],
                "total_vulnerable": len(vulnerable_forms),
                "total_forms": total_forms,
            },
            recommendation=(
                "Implement CSRF tokens for all state-changing operations. "
                "Use framework-provided CSRF protection:\n"
                "- Django: {% csrf_token %}\n"
                "- Flask: flask_wtf.CSRFProtect\n"
                "- Express: csurf middleware\n"
                "- Spring: @EnableWebSecurity with CSRF"
            ),
            source=self.source,
            cwe_id="CWE-352",
            owasp_category="A01:2021 - Broken Access Control",
            confidence=0.85,
            references=[
                "https://owasp.org/www-community/attacks/csrf",
                "https://cheatsheetseries.owasp.org/cheatsheets/Cross-Site_Request_Forgery_Prevention_Cheat_Sheet.html",
                "https://cwe.mitre.org/data/definitions/352.html",
            ],
        ))
    
    def _report_protected_forms(
        self,
        protected_forms: List[FormInfo],
        total_forms: int,
        result: SecurityCheckResult,
    ) -> None:
        """Report forms with CSRF protection."""
        if protected_forms:
            result.add_finding(SecurityFinding(
                check_name="csrf_tokens_present",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message=f"All {len(protected_forms)} state-changing form(s) have CSRF protection (out of {total_forms} total forms)",
                details={
                    "protected_forms": len(protected_forms),
                    "total_forms": total_forms,
                },
                source=self.source,
            ))
    
    def _check_cookie_samesite(
        self,
        response: httpx.Response,
        result: SecurityCheckResult,
    ) -> None:
        """
        Validate SameSite cookie attributes.
        
        Checks if cookies have proper SameSite attribute to prevent CSRF.
        """
        cookies = response.cookies
        
        if not cookies:
            result.add_finding(SecurityFinding(
                check_name="csrf_no_cookies",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message="No cookies set by application",
                source=self.source,
            ))
            return
        
        weak_cookies: List[CookieInfo] = []
        
        for cookie in cookies.jar:
            cookie_info = self._analyze_cookie(cookie)
            
            # Flag cookies without SameSite attribute
            if not cookie_info.samesite:
                weak_cookies.append(cookie_info)
        
        if weak_cookies:
            result.add_finding(SecurityFinding(
                check_name="csrf_samesite_missing",
                status=CheckStatus.FAIL,
                severity=Severity.MEDIUM,
                message=f"Found {len(weak_cookies)} cookie(s) without SameSite attribute",
                details={
                    "cookies": [
                        {
                            "name": c.name,
                            "secure": c.secure,
                            "httponly": c.httponly,
                            "domain": c.domain,
                            "path": c.path,
                        }
                        for c in weak_cookies[:10]
                    ],
                    "total_weak_cookies": len(weak_cookies),
                },
                recommendation=(
                    "Set SameSite attribute on all cookies:\n"
                    "- SameSite=Strict: Strongest protection, blocks all cross-site requests\n"
                    "- SameSite=Lax: Balance between security and usability (recommended)\n"
                    "- SameSite=None: Only if cross-site cookies are required (must use Secure flag)\n\n"
                    "Example: Set-Cookie: session=abc; SameSite=Lax; Secure; HttpOnly"
                ),
                source=self.source,
                cwe_id="CWE-352",
                owasp_category="A01:2021 - Broken Access Control",
                confidence=0.95,
                references=[
                    "https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie/SameSite",
                    "https://web.dev/samesite-cookies-explained/",
                ],
            ))
        else:
            result.add_finding(SecurityFinding(
                check_name="csrf_samesite_present",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message=f"All {len(cookies)} cookie(s) have SameSite attribute",
                source=self.source,
            ))
    
    def _analyze_cookie(self, cookie) -> CookieInfo:
        """Extract structured cookie information."""
        # Extract SameSite attribute
        samesite = None
        if hasattr(cookie, '_rest') and cookie._rest:
            for key, value in cookie._rest.items():
                if key.lower() == 'samesite':
                    samesite = value.lower() if value else None
                    break
        
        return CookieInfo(
            name=cookie.name,
            secure=cookie.secure or False,
            httponly=cookie.has_nonstandard_attr('HttpOnly') or False,
            samesite=samesite,
            domain=cookie.domain,
            path=cookie.path or "/",
        )
    
    async def _check_state_changing_methods(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """
        Test if state-changing methods accept requests without CSRF protection.
        
        Sends test requests without Origin/Referer headers to check validation.
        """
        vulnerable_methods: List[Dict[str, Any]] = []
        
        for method in CSRFConfig.STATE_CHANGING_METHODS:
            try:
                # Send request without Origin/Referer headers
                headers = {
                    'Content-Type': 'application/x-www-form-urlencoded',
                }
                
                # Use a test payload
                test_data = {'test': 'csrf_probe'}
                
                request_kwargs = {
                    'url': url,
                    'headers': headers,
                    'data': test_data,
                    'timeout': self._timeout,
                    'follow_redirects': False,  # Don't follow redirects for testing
                }
                
                if method == 'POST':
                    response = await client.post(**request_kwargs)
                elif method == 'PUT':
                    response = await client.put(**request_kwargs)
                elif method == 'DELETE':
                    del request_kwargs['data']
                    response = await client.delete(**request_kwargs)
                elif method == 'PATCH':
                    response = await client.patch(**request_kwargs)
                
                # Method is supported if not 405 Method Not Allowed
                if response.status_code not in [405, 501]:
                    vulnerable_methods.append({
                        'method': method,
                        'status_code': response.status_code,
                        'accepts_requests': True,
                    })
            
            except (TimeoutException, httpx.HTTPError) as e:
                logger.debug("Method %s test failed for %s: %s", method, url, e)
                continue
        
        if vulnerable_methods:
            result.add_finding(SecurityFinding(
                check_name="csrf_methods_potentially_unprotected",
                status=CheckStatus.FAIL,
                severity=Severity.MEDIUM,
                message=(
                    f"State-changing HTTP methods accepted without validation: "
                    f"{', '.join(m['method'] for m in vulnerable_methods)}"
                ),
                details={
                    "methods": vulnerable_methods,
                    "note": "Methods accept requests without Origin/Referer headers",
                },
                recommendation=(
                    "Implement CSRF protection for all state-changing operations:\n"
                    "1. Validate Origin/Referer headers\n"
                    "2. Use CSRF tokens\n"
                    "3. Use SameSite cookies\n"
                    "4. Consider using custom request headers (X-Requested-With)"
                ),
                source=self.source,
                cwe_id="CWE-352",
                owasp_category="A01:2021 - Broken Access Control",
                confidence=0.65,  # Medium confidence as we can't fully verify without credentials
            ))
    
    def run_sync(
        self,
        url: str,
        client: httpx.Client,
        result: SecurityCheckResult,
    ) -> None:
        """
        Synchronous wrapper for async implementation.
        
        Args:
            url: Target URL
            client: Sync HTTP client
            result: Result container
        """
        asyncio.run(self.run_async(url, client, result))
