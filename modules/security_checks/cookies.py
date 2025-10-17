# modules/security_checks/cookies.py
"""
Cookie Security Check Module

Validates cookie security flags according to OWASP standards:
- Secure flag (HTTPS-only transmission)
- HttpOnly flag (JavaScript access prevention)
- SameSite attribute (CSRF protection)
- Cookie expiration and domain scope
- Sensitive data in cookies

References:
- OWASP: https://owasp.org/www-community/controls/SecureCookieAttribute
- CWE-614: Sensitive Cookie in HTTPS Session Without 'Secure' Attribute
- CWE-1004: Sensitive Cookie Without 'HttpOnly' Flag
"""

from __future__ import annotations
import logging
import re
from typing import Dict, List, Optional, Tuple
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


class CookieSecurityCheck(AbstractSecurityCheck):
    """
    Comprehensive cookie security validation.
    
    Checks:
    1. Secure flag presence (for HTTPS)
    2. HttpOnly flag presence
    3. SameSite attribute and value
    4. Cookie expiration (Max-Age/Expires)
    5. Domain and Path scope
    6. Sensitive data exposure
    """
    
    # Patterns for detecting sensitive data in cookies
    SENSITIVE_PATTERNS = {
        "password": r"(?i)(pass|pwd|password|passwd)",
        "token": r"(?i)(token|jwt|auth|session|secret)",
        "credit_card": r"\d{13,19}",
        "ssn": r"\d{3}-\d{2}-\d{4}",
        "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    }
    
    def __init__(self, timeout_s: float = 15.0):
        super().__init__(timeout_s=timeout_s)
    
    @property
    def name(self) -> str:
        return "cookie_security"
    
    @property
    def source(self) -> str:
        return "cookies"
    
    async def run_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """Execute async cookie security check."""
        try:
            response = await client.get(
                url,
                headers={"User-Agent": "SecurityEngine/2.0 (Cookie Scanner)"}
            )
            self._analyze_cookies(url, response, result)
        
        except httpx.HTTPError as e:
            logger.warning("Failed to check cookies for %s: %s", url, e)
            result.add_finding(SecurityFinding(
                check_name="cookie_security_error",
                status=CheckStatus.ERROR,
                severity=Severity.MEDIUM,
                message=f"Failed to retrieve cookies: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                source=self.source,
            ))
    
    def run_sync(
        self,
        url: str,
        client: httpx.Client,
        result: SecurityCheckResult,
    ) -> None:
        """Execute sync cookie security check."""
        try:
            response = client.get(
                url,
                headers={"User-Agent": "SecurityEngine/2.0 (Cookie Scanner)"}
            )
            self._analyze_cookies(url, response, result)
        
        except httpx.HTTPError as e:
            logger.warning("Failed to check cookies for %s: %s", url, e)
            result.add_finding(SecurityFinding(
                check_name="cookie_security_error",
                status=CheckStatus.ERROR,
                severity=Severity.MEDIUM,
                message=f"Failed to retrieve cookies: {str(e)}",
                source=self.source,
            ))
    
    def _analyze_cookies(
        self,
        url: str,
        response: httpx.Response,
        result: SecurityCheckResult,
    ) -> None:
        """Analyze cookie security from HTTP response."""
        # Extract Set-Cookie headers
        set_cookie_headers = self._extract_cookies(response)
        
        if not set_cookie_headers:
            result.add_finding(SecurityFinding(
                check_name="cookie_security_no_cookies",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message="No cookies set by application",
                details={"cookie_count": 0},
                source=self.source,
            ))
            return
        
        # Parse URL to check if HTTPS
        parsed_url = urlparse(url)
        is_https = parsed_url.scheme == "https"
        
        # Analyze each cookie
        total_cookies = len(set_cookie_headers)
        secure_cookies = 0
        insecure_cookies = []
        
        for cookie_header in set_cookie_headers:
            cookie_analysis = self._parse_cookie(cookie_header, is_https)
            
            if cookie_analysis["issues"]:
                insecure_cookies.append(cookie_analysis)
                self._create_cookie_finding(cookie_analysis, result)
            else:
                secure_cookies += 1
                result.add_finding(SecurityFinding(
                    check_name=f"cookie_secure_{cookie_analysis['name']}",
                    status=CheckStatus.PASS,
                    severity=Severity.INFO,
                    message=f"Cookie '{cookie_analysis['name']}' has proper security flags",
                    details=cookie_analysis,
                    source=self.source,
                ))
        
        # Overall cookie security assessment
        if insecure_cookies:
            result.add_finding(SecurityFinding(
                check_name="cookie_security_overall",
                status=CheckStatus.FAIL,
                severity=Severity.HIGH,
                message=f"Cookie security issues: {len(insecure_cookies)}/{total_cookies} cookies are insecure",
                details={
                    "total_cookies": total_cookies,
                    "secure_cookies": secure_cookies,
                    "insecure_cookies": len(insecure_cookies),
                    "insecure_cookie_names": [c["name"] for c in insecure_cookies],
                },
                recommendation="Fix cookie security flags: add Secure, HttpOnly, and SameSite attributes",
                source=self.source,
                cwe_id="CWE-614",
                owasp_category="A05:2021 - Security Misconfiguration",
                references=[
                    "https://owasp.org/www-community/controls/SecureCookieAttribute",
                    "https://developer.mozilla.org/en-US/docs/Web/HTTP/Cookies",
                ],
            ))
        else:
            result.add_finding(SecurityFinding(
                check_name="cookie_security_overall",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message=f"All {total_cookies} cookies are properly secured",
                details={"total_cookies": total_cookies, "secure_cookies": secure_cookies},
                source=self.source,
            ))
    
    def _extract_cookies(self, response: httpx.Response) -> List[str]:
        """Extract Set-Cookie headers from response."""
        # Try get_list first (httpx 0.24+)
        if hasattr(response.headers, "get_list"):
            return response.headers.get_list("set-cookie")
        
        # Fallback: parse from raw headers
        cookies = []
        for key, value in response.headers.raw:
            if key.lower() == b"set-cookie":
                cookies.append(value.decode("utf-8", errors="ignore"))
        
        return cookies
    
    def _parse_cookie(
        self,
        cookie_header: str,
        is_https: bool,
    ) -> Dict[str, any]:
        """
        Parse cookie header and identify security issues.
        
        Returns:
            {
                "name": str,
                "value": str (truncated),
                "flags": dict,
                "issues": list,
                "severity": str
            }
        """
        # Split cookie into name=value and attributes
        parts = [p.strip() for p in cookie_header.split(";")]
        
        if not parts:
            return {
                "name": "unknown",
                "issues": ["malformed_cookie"],
                "severity": "MEDIUM",
            }
        
        # Parse name=value
        name_value = parts[0]
        if "=" in name_value:
            name, value = name_value.split("=", 1)
        else:
            name = name_value
            value = ""
        
        # Parse attributes
        cookie_lower = cookie_header.lower()
        flags = {
            "secure": "secure" in cookie_lower,
            "httponly": "httponly" in cookie_lower,
            "samesite": None,
            "max_age": None,
            "expires": None,
            "domain": None,
            "path": None,
        }
        
        # Extract SameSite value
        samesite_match = re.search(r"samesite=(\w+)", cookie_lower)
        if samesite_match:
            flags["samesite"] = samesite_match.group(1)
        
        # Extract Max-Age
        max_age_match = re.search(r"max-age=(\d+)", cookie_lower)
        if max_age_match:
            flags["max_age"] = int(max_age_match.group(1))
        
        # Extract Expires
        expires_match = re.search(r"expires=([^;]+)", cookie_header, re.IGNORECASE)
        if expires_match:
            flags["expires"] = expires_match.group(1).strip()
        
        # Extract Domain
        domain_match = re.search(r"domain=([^;]+)", cookie_lower)
        if domain_match:
            flags["domain"] = domain_match.group(1).strip()
        
        # Extract Path
        path_match = re.search(r"path=([^;]+)", cookie_lower)
        if path_match:
            flags["path"] = path_match.group(1).strip()
        
        # Identify issues
        issues = []
        
        # Check Secure flag (required for HTTPS)
        if is_https and not flags["secure"]:
            issues.append("missing_secure_flag")
        
        # Check HttpOnly flag
        if not flags["httponly"]:
            issues.append("missing_httponly_flag")
        
        # Check SameSite attribute
        if not flags["samesite"]:
            issues.append("missing_samesite_attribute")
        elif flags["samesite"] not in ("strict", "lax"):
            issues.append("weak_samesite_value")
        
        # Check for sensitive data in cookie name/value
        sensitive_type = self._check_sensitive_data(name, value)
        if sensitive_type:
            issues.append(f"sensitive_data_{sensitive_type}")
        
        # Check expiration (session cookies are OK, but persistent should have reasonable expiry)
        if flags["max_age"] and flags["max_age"] > 31536000:  # > 1 year
            issues.append("excessive_expiration")
        
        # Determine severity
        severity = self._calculate_cookie_severity(issues)
        
        return {
            "name": name,
            "value": value[:20] + "..." if len(value) > 20 else value,
            "flags": flags,
            "issues": issues,
            "severity": severity,
        }
    
    def _check_sensitive_data(self, name: str, value: str) -> Optional[str]:
        """Check if cookie contains sensitive data."""
        combined = f"{name}={value}"
        
        for data_type, pattern in self.SENSITIVE_PATTERNS.items():
            if re.search(pattern, combined):
                return data_type
        
        return None
    
    def _calculate_cookie_severity(self, issues: List[str]) -> str:
        """Calculate severity based on issues found."""
        if not issues:
            return "INFO"
        
        critical_issues = [
            "sensitive_data_password",
            "sensitive_data_credit_card",
            "sensitive_data_ssn",
        ]
        
        high_issues = [
            "missing_secure_flag",
            "missing_httponly_flag",
            "sensitive_data_token",
        ]
        
        if any(issue in critical_issues for issue in issues):
            return "CRITICAL"
        elif any(issue in high_issues for issue in issues):
            return "HIGH"
        elif len(issues) >= 2:
            return "MEDIUM"
        else:
            return "LOW"
    
    def _create_cookie_finding(
        self,
        cookie_analysis: Dict[str, any],
        result: SecurityCheckResult,
    ) -> None:
        """Create security finding for insecure cookie."""
        name = cookie_analysis["name"]
        issues = cookie_analysis["issues"]
        severity = Severity[cookie_analysis["severity"]]
        
        # Build detailed message
        issue_messages = {
            "missing_secure_flag": "Missing Secure flag (cookies can be sent over HTTP)",
            "missing_httponly_flag": "Missing HttpOnly flag (vulnerable to XSS attacks)",
            "missing_samesite_attribute": "Missing SameSite attribute (vulnerable to CSRF)",
            "weak_samesite_value": "Weak SameSite value (should be 'Strict' or 'Lax')",
            "sensitive_data_password": "Contains password-related data",
            "sensitive_data_token": "Contains authentication token",
            "sensitive_data_credit_card": "Contains potential credit card number",
            "sensitive_data_ssn": "Contains potential SSN",
            "sensitive_data_email": "Contains email address",
            "excessive_expiration": "Expiration exceeds 1 year",
        }
        
        issue_descriptions = [issue_messages.get(issue, issue) for issue in issues]
        
        message = f"Cookie '{name}' has {len(issues)} security issue(s): {', '.join(issue_descriptions)}"
        
        # Build recommendation
        recommendations = []
        if "missing_secure_flag" in issues:
            recommendations.append("Add Secure flag")
        if "missing_httponly_flag" in issues:
            recommendations.append("Add HttpOnly flag")
        if "missing_samesite_attribute" in issues or "weak_samesite_value" in issues:
            recommendations.append("Add SameSite=Strict or SameSite=Lax")
        if any("sensitive_data" in issue for issue in issues):
            recommendations.append("Avoid storing sensitive data in cookies; use secure server-side sessions")
        
        recommendation = "; ".join(recommendations) if recommendations else None
        
        # Map to CWE
        cwe_id = "CWE-614"  # Default: Sensitive Cookie in HTTPS Session Without 'Secure' Attribute
        if "missing_httponly_flag" in issues:
            cwe_id = "CWE-1004"  # Sensitive Cookie Without 'HttpOnly' Flag
        if any("sensitive_data" in issue for issue in issues):
            cwe_id = "CWE-315"  # Cleartext Storage of Sensitive Information in a Cookie
        
        result.add_finding(SecurityFinding(
            check_name=f"cookie_insecure_{name}",
            status=CheckStatus.FAIL,
            severity=severity,
            message=message,
            details=cookie_analysis,
            recommendation=recommendation,
            source=self.source,
            cwe_id=cwe_id,
            owasp_category="A05:2021 - Security Misconfiguration",
            confidence=0.95,
            references=[
                "https://owasp.org/www-community/controls/SecureCookieAttribute",
                "https://owasp.org/www-community/HttpOnly",
                "https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie",
            ],
        ))
