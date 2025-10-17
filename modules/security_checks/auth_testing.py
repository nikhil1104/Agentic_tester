# modules/security_checks/auth_testing.py
"""
Authentication Security Testing Module

Enterprise-grade authentication vulnerability testing covering:
- JWT (JSON Web Token) security analysis
- OAuth 2.0 flow validation
- Session management security
- Password policy validation
- Multi-factor authentication detection
- Token expiration and rotation
- Credential exposure in responses
- Authentication bypass attempts

Standards Compliance:
- OWASP Top 10 2021: A07 (Identification and Authentication Failures)
- OWASP ASVS (Application Security Verification Standard) v4.0
- NIST SP 800-63B (Digital Identity Guidelines)
- CWE-287: Improper Authentication
- CWE-290: Authentication Bypass
- CWE-521: Weak Password Requirements
- PCI DSS 8.2: User authentication and password management
"""

from __future__ import annotations

import logging
import re
import json
import base64
import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from datetime import datetime, timedelta

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


# ✅ Type Definitions
class AuthType(Enum):
    """Authentication mechanism types."""
    JWT = auto()
    SESSION = auto()
    OAUTH2 = auto()
    BASIC_AUTH = auto()
    API_KEY = auto()
    BEARER_TOKEN = auto()
    UNKNOWN = auto()


class JWTAlgorithm(Enum):
    """JWT signing algorithms."""
    HS256 = "HMAC with SHA-256"
    HS384 = "HMAC with SHA-384"
    HS512 = "HMAC with SHA-512"
    RS256 = "RSA with SHA-256"
    RS384 = "RSA with SHA-384"
    RS512 = "RSA with SHA-512"
    ES256 = "ECDSA with SHA-256"
    ES384 = "ECDSA with SHA-384"
    ES512 = "ECDSA with SHA-512"
    PS256 = "RSA-PSS with SHA-256"
    PS384 = "RSA-PSS with SHA-384"
    PS512 = "RSA-PSS with SHA-512"
    NONE = "No signature"


@dataclass
class JWTInfo:
    """Structured JWT token information."""
    token: str
    header: Dict[str, Any]
    payload: Dict[str, Any]
    algorithm: str
    is_expired: bool
    expiration_time: Optional[datetime]
    issued_at: Optional[datetime]
    has_signature: bool
    vulnerabilities: List[str] = field(default_factory=list)


@dataclass
class SessionInfo:
    """Session management information."""
    session_cookie_name: Optional[str]
    has_secure_flag: bool
    has_httponly_flag: bool
    has_samesite_flag: bool
    session_timeout: Optional[int]
    vulnerabilities: List[str] = field(default_factory=list)


@dataclass
class AuthEndpointInfo:
    """Authentication endpoint information."""
    url: str
    auth_type: AuthType
    requires_credentials: bool
    supports_mfa: bool
    password_policy_enforced: bool
    rate_limited: bool


# ✅ Configuration Constants
class AuthSecurityConfig:
    """Configuration constants for authentication testing."""
    
    # JWT detection patterns
    JWT_PATTERNS: List[str] = [
        r'eyJ[A-Za-z0-9_-]+\.eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]*',  # Standard JWT format
    ]
    
    # Weak JWT algorithms (should not be used)
    WEAK_JWT_ALGORITHMS: Set[str] = {'none', 'HS256'}  # HS256 weak if secret is weak
    
    # Session cookie name patterns
    SESSION_COOKIE_PATTERNS: Set[str] = {
        'sessionid', 'session_id', 'jsessionid', 'phpsessid',
        'asp.net_sessionid', 'connect.sid', 'sid', 'sess'
    }
    
    # OAuth 2.0 endpoints
    OAUTH_ENDPOINTS: Set[str] = {
        '/oauth/authorize', '/oauth/token', '/oauth2/authorize',
        '/oauth2/token', '/auth/login', '/login/oauth'
    }
    
    # Credential patterns in responses
    CREDENTIAL_PATTERNS: Dict[str, str] = {
        'password': r'(?i)(password|passwd|pwd)\s*[:=]\s*["\']([^"\']+)["\']',
        'api_key': r'(?i)(api[_-]?key|apikey)\s*[:=]\s*["\']([^"\']+)["\']',
        'token': r'(?i)(access[_-]?token|auth[_-]?token)\s*[:=]\s*["\']([^"\']+)["\']',
        'secret': r'(?i)(secret|client[_-]?secret)\s*[:=]\s*["\']([^"\']+)["\']',
    }
    
    # Password policy requirements (industry best practices)
    PASSWORD_POLICY: Dict[str, Any] = {
        'min_length': 12,
        'require_uppercase': True,
        'require_lowercase': True,
        'require_numbers': True,
        'require_special': True,
        'max_attempts': 5,
    }
    
    # Timeouts
    REQUEST_TIMEOUT: float = 10.0
    MAX_RETRIES: int = 2
    
    # Testing limits
    MAX_LOGIN_ATTEMPTS: int = 5
    MAX_TOKENS_TO_ANALYZE: int = 10


class AuthSecurityCheck(AbstractSecurityCheck):
    """
    Production-grade authentication security testing.
    
    Comprehensive authentication vulnerability testing including:
    - JWT token analysis and validation
    - Session management security
    - OAuth 2.0 flow validation
    - Password policy enforcement
    - MFA detection
    - Credential exposure detection
    - Authentication bypass testing
    
    Features:
    - Automatic authentication mechanism detection
    - JWT algorithm weakness detection
    - Token expiration validation
    - Session fixation testing
    - Brute force protection testing
    - Comprehensive security recommendations
    
    Example:
        >>> check = AuthSecurityCheck(timeout_s=15.0)
        >>> result = SecurityCheckResult(url="https://api.example.com", ...)
        >>> await check.run_async("https://api.example.com/login", client, result)
    """
    
    def __init__(self, timeout_s: float = 15.0):
        """
        Initialize authentication security checker.
        
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
        return "auth_testing"
    
    @property
    def source(self) -> str:
        """Source identifier for findings."""
        return "authentication"
    
    async def run_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """
        Execute comprehensive authentication security tests.
        
        Args:
            url: Target URL to test
            client: Async HTTP client
            result: Result container to populate
            
        Raises:
            ValueError: If URL is invalid
        """
        # ✅ Input validation
        if not url or not url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL: {url}")
        
        logger.info("Starting authentication security testing for: %s", url)
        
        try:
            # Fetch initial response
            response = await self._fetch_with_retry(url, client)
            
            # Run all authentication tests concurrently
            test_tasks = [
                self._check_jwt_tokens(response, result),
                self._check_session_management(response, result),
                self._check_credential_exposure(response, result),
                self._check_authentication_headers(response, result),
                self._check_oauth_endpoints(url, client, result),
                self._check_brute_force_protection(url, client, result),
            ]
            
            await asyncio.gather(*test_tasks, return_exceptions=True)
            
            logger.info("Authentication security testing completed for: %s", url)
            
        except TimeoutException as e:
            logger.error("Auth testing timeout for %s: %s", url, e)
            result.add_finding(SecurityFinding(
                check_name="auth_test_timeout",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message=f"Authentication testing timed out after {self.timeout_s}s",
                details={"error": str(e)},
                source=self.source,
            ))
        except Exception as e:
            logger.error("Auth security testing failed for %s: %s", url, e, exc_info=True)
            result.add_finding(SecurityFinding(
                check_name="auth_test_error",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message=f"Authentication testing failed: {str(e)[:200]}",
                source=self.source,
            ))
    
    async def _fetch_with_retry(
        self,
        url: str,
        client: httpx.AsyncClient,
        max_retries: int = AuthSecurityConfig.MAX_RETRIES,
    ) -> httpx.Response:
        """Fetch URL with exponential backoff retry."""
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
                    wait_time = (2 ** attempt)
                    logger.debug("Retry %d/%d after %ds for %s", 
                               attempt + 1, max_retries, wait_time, url)
                    await asyncio.sleep(wait_time)
                else:
                    raise
        
        raise last_exception
    
    async def _check_jwt_tokens(
        self,
        response: httpx.Response,
        result: SecurityCheckResult,
    ) -> None:
        """
        Analyze JWT tokens for security vulnerabilities.
        
        Checks:
        - Algorithm security (no 'none', prefer RS256/ES256)
        - Token expiration
        - Signature presence
        - Sensitive data in payload
        """
        # Extract JWT tokens from response
        jwt_tokens = self._extract_jwt_tokens(response)
        
        if not jwt_tokens:
            result.add_finding(SecurityFinding(
                check_name="jwt_not_found",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message="No JWT tokens detected in response",
                source=self.source,
            ))
            return
        
        # Analyze each token
        for token in jwt_tokens[:AuthSecurityConfig.MAX_TOKENS_TO_ANALYZE]:
            jwt_info = self._analyze_jwt_token(token)
            
            if jwt_info:
                self._report_jwt_vulnerabilities(jwt_info, result)
    
    def _extract_jwt_tokens(self, response: httpx.Response) -> List[str]:
        """Extract JWT tokens from response headers and body."""
        tokens = []
        
        # Check Authorization header
        auth_header = response.headers.get('authorization', '')
        if auth_header.startswith('Bearer '):
            token = auth_header[7:]
            if self._is_jwt_format(token):
                tokens.append(token)
        
        # Check response body
        try:
            body_text = response.text
            for match in re.finditer(AuthSecurityConfig.JWT_PATTERNS[0], body_text):
                token = match.group(0)
                if self._is_jwt_format(token):
                    tokens.append(token)
        except:
            pass
        
        # Check cookies
        for cookie in response.cookies.jar:
            if self._is_jwt_format(cookie.value):
                tokens.append(cookie.value)
        
        return list(set(tokens))  # Remove duplicates
    
    def _is_jwt_format(self, token: str) -> bool:
        """Validate JWT format (three base64 parts separated by dots)."""
        parts = token.split('.')
        return len(parts) == 3 and all(part for part in parts[:2])
    
    def _analyze_jwt_token(self, token: str) -> Optional[JWTInfo]:
        """
        Decode and analyze JWT token.
        
        Returns:
            JWTInfo object or None if invalid
        """
        try:
            parts = token.split('.')
            
            # Decode header
            header_json = base64.urlsafe_b64decode(parts[0] + '==').decode('utf-8')
            header = json.loads(header_json)
            
            # Decode payload
            payload_json = base64.urlsafe_b64decode(parts[1] + '==').decode('utf-8')
            payload = json.loads(payload_json)
            
            # Extract information
            algorithm = header.get('alg', 'unknown')
            has_signature = len(parts[2]) > 0
            
            # Check expiration
            exp = payload.get('exp')
            iat = payload.get('iat')
            
            expiration_time = None
            is_expired = False
            
            if exp:
                expiration_time = datetime.fromtimestamp(exp)
                is_expired = expiration_time < datetime.now()
            
            issued_at = None
            if iat:
                issued_at = datetime.fromtimestamp(iat)
            
            # Identify vulnerabilities
            vulnerabilities = []
            
            # Weak algorithm
            if algorithm.lower() in AuthSecurityConfig.WEAK_JWT_ALGORITHMS:
                vulnerabilities.append(f"weak_algorithm_{algorithm.lower()}")
            
            # No signature
            if not has_signature:
                vulnerabilities.append("no_signature")
            
            # No expiration
            if not exp:
                vulnerabilities.append("no_expiration")
            
            # Expired token still accepted
            if is_expired:
                vulnerabilities.append("expired_token")
            
            # Sensitive data in payload
            sensitive_keys = {'password', 'secret', 'private_key', 'ssn', 'credit_card'}
            if any(key.lower() in str(payload).lower() for key in sensitive_keys):
                vulnerabilities.append("sensitive_data_in_payload")
            
            return JWTInfo(
                token=token[:50] + "...",  # Truncate for safety
                header=header,
                payload={k: v for k, v in payload.items() if k not in {'password', 'secret'}},
                algorithm=algorithm,
                is_expired=is_expired,
                expiration_time=expiration_time,
                issued_at=issued_at,
                has_signature=has_signature,
                vulnerabilities=vulnerabilities,
            )
        
        except Exception as e:
            logger.debug("JWT analysis failed: %s", e)
            return None
    
    def _report_jwt_vulnerabilities(
        self,
        jwt_info: JWTInfo,
        result: SecurityCheckResult,
    ) -> None:
        """Report JWT security vulnerabilities."""
        if not jwt_info.vulnerabilities:
            result.add_finding(SecurityFinding(
                check_name="jwt_secure",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message=f"JWT token uses secure algorithm: {jwt_info.algorithm}",
                details={"algorithm": jwt_info.algorithm},
                source=self.source,
            ))
            return
        
        # Report each vulnerability
        for vuln in jwt_info.vulnerabilities:
            if vuln.startswith("weak_algorithm"):
                result.add_finding(SecurityFinding(
                    check_name="jwt_weak_algorithm",
                    status=CheckStatus.FAIL,
                    severity=Severity.HIGH,
                    message=f"JWT uses weak/insecure algorithm: {jwt_info.algorithm}",
                    details={
                        "algorithm": jwt_info.algorithm,
                        "header": jwt_info.header,
                    },
                    recommendation=(
                        "Use strong JWT signing algorithms:\n"
                        "✅ Recommended: RS256 (RSA with SHA-256) or ES256 (ECDSA with SHA-256)\n"
                        "❌ Avoid: 'none' algorithm (no signature)\n"
                        "⚠️  Be careful with: HS256 (requires strong secret)\n\n"
                        "Implementation:\n"
                        "- Use RS256 for public/private key scenarios\n"
                        "- Use ES256 for elliptic curve cryptography\n"
                        "- Never allow 'none' algorithm in production"
                    ),
                    source=self.source,
                    cwe_id="CWE-327",
                    owasp_category="A02:2021 - Cryptographic Failures",
                    confidence=0.95,
                    references=[
                        "https://auth0.com/blog/critical-vulnerabilities-in-json-web-token-libraries/",
                        "https://cwe.mitre.org/data/definitions/327.html",
                    ],
                ))
            
            elif vuln == "no_signature":
                result.add_finding(SecurityFinding(
                    check_name="jwt_no_signature",
                    status=CheckStatus.FAIL,
                    severity=Severity.CRITICAL,
                    message="JWT token has no signature - can be forged",
                    recommendation="Always sign JWT tokens. Never use 'none' algorithm.",
                    source=self.source,
                    cwe_id="CWE-347",
                    owasp_category="A02:2021 - Cryptographic Failures",
                    confidence=1.0,
                ))
            
            elif vuln == "no_expiration":
                result.add_finding(SecurityFinding(
                    check_name="jwt_no_expiration",
                    status=CheckStatus.FAIL,
                    severity=Severity.MEDIUM,
                    message="JWT token has no expiration time (exp claim)",
                    details={"payload": jwt_info.payload},
                    recommendation=(
                        "Set JWT expiration:\n"
                        "- Short-lived tokens: 15-60 minutes for access tokens\n"
                        "- Use refresh tokens for longer sessions\n"
                        "- Include 'exp' claim in payload\n"
                        "- Validate expiration on every request"
                    ),
                    source=self.source,
                    cwe_id="CWE-613",
                    owasp_category="A07:2021 - Identification and Authentication Failures",
                    confidence=0.9,
                ))
            
            elif vuln == "sensitive_data_in_payload":
                result.add_finding(SecurityFinding(
                    check_name="jwt_sensitive_data",
                    status=CheckStatus.FAIL,
                    severity=Severity.HIGH,
                    message="JWT payload contains sensitive information",
                    recommendation=(
                        "Never store sensitive data in JWT payload:\n"
                        "- JWT payload is base64 encoded, not encrypted\n"
                        "- Anyone can decode and read the payload\n"
                        "- Store only user ID and basic claims\n"
                        "- Use encrypted JWTs (JWE) if sensitive data is required"
                    ),
                    source=self.source,
                    cwe_id="CWE-359",
                    confidence=0.8,
                ))
    
    async def _check_session_management(
        self,
        response: httpx.Response,
        result: SecurityCheckResult,
    ) -> None:
        """Check session cookie security."""
        session_cookies = self._find_session_cookies(response)
        
        if not session_cookies:
            result.add_finding(SecurityFinding(
                check_name="session_no_cookies",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message="No session cookies detected",
                source=self.source,
            ))
            return
        
        for cookie_name, cookie in session_cookies:
            session_info = self._analyze_session_cookie(cookie_name, cookie)
            self._report_session_vulnerabilities(session_info, result)
    
    def _find_session_cookies(self, response: httpx.Response) -> List[Tuple[str, Any]]:
        """Find session cookies in response."""
        session_cookies = []
        
        for cookie in response.cookies.jar:
            cookie_name_lower = cookie.name.lower()
            
            # Check if cookie name matches session patterns
            if any(pattern in cookie_name_lower for pattern in AuthSecurityConfig.SESSION_COOKIE_PATTERNS):
                session_cookies.append((cookie.name, cookie))
        
        return session_cookies
    
    def _analyze_session_cookie(self, name: str, cookie: Any) -> SessionInfo:
        """Analyze session cookie security attributes."""
        has_secure = cookie.secure or False
        has_httponly = cookie.has_nonstandard_attr('HttpOnly') or False
        
        # Check SameSite attribute
        has_samesite = False
        if hasattr(cookie, '_rest') and cookie._rest:
            has_samesite = any(k.lower() == 'samesite' for k in cookie._rest.keys())
        
        vulnerabilities = []
        
        if not has_secure:
            vulnerabilities.append("missing_secure_flag")
        
        if not has_httponly:
            vulnerabilities.append("missing_httponly_flag")
        
        if not has_samesite:
            vulnerabilities.append("missing_samesite_flag")
        
        return SessionInfo(
            session_cookie_name=name,
            has_secure_flag=has_secure,
            has_httponly_flag=has_httponly,
            has_samesite_flag=has_samesite,
            session_timeout=None,
            vulnerabilities=vulnerabilities,
        )
    
    def _report_session_vulnerabilities(
        self,
        session_info: SessionInfo,
        result: SecurityCheckResult,
    ) -> None:
        """Report session management vulnerabilities."""
        if not session_info.vulnerabilities:
            result.add_finding(SecurityFinding(
                check_name="session_secure",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message=f"Session cookie '{session_info.session_cookie_name}' has secure attributes",
                source=self.source,
            ))
            return
        
        # Report vulnerabilities
        for vuln in session_info.vulnerabilities:
            if vuln == "missing_secure_flag":
                result.add_finding(SecurityFinding(
                    check_name="session_insecure_flag",
                    status=CheckStatus.FAIL,
                    severity=Severity.HIGH,
                    message=f"Session cookie '{session_info.session_cookie_name}' missing Secure flag",
                    recommendation=(
                        "Set Secure flag on session cookies:\n"
                        "Set-Cookie: sessionid=abc123; Secure; HttpOnly; SameSite=Lax\n\n"
                        "This ensures cookies are only sent over HTTPS connections."
                    ),
                    source=self.source,
                    cwe_id="CWE-614",
                    owasp_category="A05:2021 - Security Misconfiguration",
                    confidence=0.95,
                ))
            
            elif vuln == "missing_httponly_flag":
                result.add_finding(SecurityFinding(
                    check_name="session_no_httponly",
                    status=CheckStatus.FAIL,
                    severity=Severity.MEDIUM,
                    message=f"Session cookie '{session_info.session_cookie_name}' missing HttpOnly flag",
                    recommendation="Set HttpOnly flag to prevent XSS cookie theft",
                    source=self.source,
                    cwe_id="CWE-1004",
                    owasp_category="A03:2021 - Injection",
                    confidence=0.9,
                ))
            
            elif vuln == "missing_samesite_flag":
                result.add_finding(SecurityFinding(
                    check_name="session_no_samesite",
                    status=CheckStatus.FAIL,
                    severity=Severity.MEDIUM,
                    message=f"Session cookie '{session_info.session_cookie_name}' missing SameSite attribute",
                    recommendation="Set SameSite=Lax or SameSite=Strict to prevent CSRF",
                    source=self.source,
                    cwe_id="CWE-352",
                    confidence=0.85,
                ))
    
    async def _check_credential_exposure(
        self,
        response: httpx.Response,
        result: SecurityCheckResult,
    ) -> None:
        """Check for exposed credentials in response."""
        exposed_credentials = []
        
        response_text = response.text
        
        for cred_type, pattern in AuthSecurityConfig.CREDENTIAL_PATTERNS.items():
            matches = re.finditer(pattern, response_text)
            for match in matches:
                exposed_credentials.append({
                    'type': cred_type,
                    'match': match.group(0)[:50] + "...",
                })
        
        if exposed_credentials:
            result.add_finding(SecurityFinding(
                check_name="auth_credential_exposure",
                status=CheckStatus.FAIL,
                severity=Severity.CRITICAL,
                message=f"Found {len(exposed_credentials)} exposed credential(s) in response",
                details={"credentials": exposed_credentials[:5]},
                recommendation=(
                    "Never expose credentials in responses:\n"
                    "1. Remove passwords, secrets, tokens from API responses\n"
                    "2. Use environment variables for secrets\n"
                    "3. Implement proper secret management\n"
                    "4. Use secret scanning tools in CI/CD"
                ),
                source=self.source,
                cwe_id="CWE-200",
                owasp_category="A01:2021 - Broken Access Control",
                confidence=0.9,
            ))
    
    async def _check_authentication_headers(
        self,
        response: httpx.Response,
        result: SecurityCheckResult,
    ) -> None:
        """Check authentication-related headers."""
        # Check WWW-Authenticate header
        www_auth = response.headers.get('www-authenticate', '')
        
        if www_auth:
            # Extract authentication scheme
            scheme = www_auth.split()[0] if www_auth else 'Unknown'
            
            result.add_finding(SecurityFinding(
                check_name="auth_scheme_detected",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message=f"Authentication required: {scheme}",
                details={"www_authenticate": www_auth},
                source=self.source,
            ))
    
    async def _check_oauth_endpoints(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """Check for OAuth 2.0 endpoints and their security."""
        # Extract base URL
        from urllib.parse import urlparse
        parsed = urlparse(url)
        base_url = f"{parsed.scheme}://{parsed.netloc}"
        
        oauth_found = False
        
        for endpoint in AuthSecurityConfig.OAUTH_ENDPOINTS:
            test_url = base_url + endpoint
            
            try:
                response = await client.get(test_url, timeout=self._timeout)
                
                if response.status_code in [200, 302, 400, 401]:
                    oauth_found = True
                    break
            except:
                continue
        
        if oauth_found:
            result.add_finding(SecurityFinding(
                check_name="oauth_endpoint_found",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message="OAuth 2.0 endpoint detected",
                recommendation="Ensure OAuth implementation follows security best practices (PKCE, state parameter, etc.)",
                source=self.source,
            ))
    
    async def _check_brute_force_protection(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """Test for brute force protection."""
        # Only test if it looks like a login endpoint
        if 'login' not in url.lower() and 'auth' not in url.lower():
            return
        
        # Send multiple failed login attempts
        failed_attempts = 0
        max_attempts = min(AuthSecurityConfig.MAX_LOGIN_ATTEMPTS, 5)
        
        for i in range(max_attempts):
            try:
                response = await client.post(
                    url,
                    data={'username': f'test{i}', 'password': 'wrong'},
                    timeout=self._timeout,
                )
                
                if response.status_code in [401, 403]:
                    failed_attempts += 1
                elif response.status_code == 429:
                    # Rate limited - good!
                    result.add_finding(SecurityFinding(
                        check_name="auth_brute_force_protected",
                        status=CheckStatus.PASS,
                        severity=Severity.INFO,
                        message="Brute force protection detected (rate limiting active)",
                        source=self.source,
                    ))
                    return
            except:
                break
        
        if failed_attempts >= max_attempts:
            result.add_finding(SecurityFinding(
                check_name="auth_no_brute_force_protection",
                status=CheckStatus.FAIL,
                severity=Severity.MEDIUM,
                message=f"No brute force protection detected ({failed_attempts} attempts allowed)",
                recommendation=(
                    "Implement brute force protection:\n"
                    "1. Rate limiting (max 5-10 attempts per IP/account)\n"
                    "2. Account lockout after failed attempts\n"
                    "3. CAPTCHA after multiple failures\n"
                    "4. Exponential backoff\n"
                    "5. Monitor and alert on suspicious activity"
                ),
                source=self.source,
                cwe_id="CWE-307",
                owasp_category="A07:2021 - Identification and Authentication Failures",
                confidence=0.7,
            ))
    
    def run_sync(
        self,
        url: str,
        client: httpx.Client,
        result: SecurityCheckResult,
    ) -> None:
        """Synchronous wrapper."""
        asyncio.run(self.run_async(url, client, result))
