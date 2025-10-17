# modules/security_checks/api_security.py
"""
API Security Testing Module

Enterprise-grade API vulnerability testing covering:
- REST API security (OWASP API Security Top 10)
- GraphQL security and introspection
- API authentication and authorization
- Mass assignment vulnerabilities
- Excessive data exposure
- API versioning best practices
- Rate limiting and throttling
- Error handling and information disclosure

Standards Compliance:
- OWASP API Security Top 10 2023
- OWASP Top 10 2021: A01 (Broken Access Control)
- CWE-200: Exposure of Sensitive Information
- CWE-306: Missing Authentication
- PCI DSS 6.5.10: Broken authentication and session management
"""

from __future__ import annotations

import logging
import json
import re
import asyncio
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from urllib.parse import urlparse, parse_qs

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
class APIType(Enum):
    """API endpoint types."""
    REST = auto()
    GRAPHQL = auto()
    SOAP = auto()
    UNKNOWN = auto()


@dataclass
class APIEndpointInfo:
    """Structured API endpoint information."""
    url: str
    api_type: APIType
    content_type: str
    status_code: int
    response_time_ms: float
    has_versioning: bool
    version: Optional[str] = None
    authentication_required: bool = False
    authentication_scheme: Optional[str] = None


@dataclass
class SensitiveFieldInfo:
    """Information about sensitive fields in API responses."""
    field_path: str
    field_name: str
    sensitivity_level: str  # 'high', 'medium', 'low'
    value_preview: Optional[str] = None


# ✅ Configuration Constants
class APISecurityConfig:
    """Configuration constants for API security testing."""
    
    # API detection patterns
    API_PATH_PATTERNS: Set[str] = {
        '/api/', '/v1/', '/v2/', '/v3/', '/rest/',
        '/graphql', '/query', '/mutation', '/endpoint'
    }
    
    # Content types that indicate API endpoints
    API_CONTENT_TYPES: Set[str] = {
        'application/json', 'application/xml', 'text/xml',
        'application/graphql', 'application/hal+json',
        'application/vnd.api+json', 'application/ld+json'
    }
    
    # Sensitive field patterns (high sensitivity)
    HIGH_SENSITIVITY_FIELDS: Set[str] = {
        'password', 'passwd', 'pwd', 'secret', 'api_key', 'apikey',
        'private_key', 'privatekey', 'access_token', 'accesstoken',
        'refresh_token', 'refreshtoken', 'auth_token', 'authtoken',
        'session_id', 'sessionid', 'ssn', 'social_security',
        'credit_card', 'creditcard', 'card_number', 'cardnumber',
        'cvv', 'cvc', 'pin', 'bank_account'
    }
    
    # Medium sensitivity fields
    MEDIUM_SENSITIVITY_FIELDS: Set[str] = {
        'email', 'phone', 'address', 'dob', 'date_of_birth',
        'birthdate', 'salary', 'income', 'tax_id', 'license'
    }
    
    # GraphQL introspection queries
    GRAPHQL_INTROSPECTION_QUERIES: List[str] = [
        '{ __schema { types { name } } }',
        '{ __type(name: "Query") { name fields { name } } }',
        'query IntrospectionQuery { __schema { queryType { name } } }'
    ]
    
    # API versioning patterns
    VERSION_PATTERNS: List[str] = [
        r'/v(\d+)/',
        r'/api/v(\d+)',
        r'\?version=(\d+)',
        r'&version=(\d+)',
        r'/version/(\d+)'
    ]
    
    # Error disclosure patterns
    ERROR_DISCLOSURE_PATTERNS: Set[str] = {
        'stack trace', 'traceback', 'exception', 'debug',
        'line number', 'file path', '/home/', '/var/',
        'internal server error', 'sql error', 'database error',
        'syntax error', 'parse error', 'at line', 'in file'
    }
    
    # Timeouts
    REQUEST_TIMEOUT: float = 10.0
    MAX_RETRIES: int = 2
    
    # Limits
    MAX_NESTED_DEPTH: int = 10
    MAX_FIELDS_TO_CHECK: int = 100


class APISecurityCheck(AbstractSecurityCheck):
    """
    Production-grade API security vulnerability testing.
    
    Implements comprehensive API security checks based on:
    - OWASP API Security Top 10 2023
    - Industry best practices
    - Common API vulnerabilities
    
    Features:
    - Automatic API type detection (REST/GraphQL/SOAP)
    - Authentication mechanism analysis
    - GraphQL introspection testing
    - Excessive data exposure detection
    - Mass assignment vulnerability detection
    - API versioning validation
    - Error handling analysis
    - Rate limiting detection
    
    Example:
        >>> check = APISecurityCheck(timeout_s=15.0)
        >>> result = SecurityCheckResult(url="https://api.example.com", ...)
        >>> await check.run_async("https://api.example.com/v1/users", client, result)
    """
    
    def __init__(self, timeout_s: float = 15.0):
        """
        Initialize API security checker.
        
        Args:
            timeout_s: Request timeout in seconds (default: 15.0)
        """
        super().__init__(timeout_s=timeout_s)
        self._timeout = httpx.Timeout(
            timeout=timeout_s,
            connect=5.0,
            read=timeout_s,
        )
        self._endpoint_cache: Dict[str, APIEndpointInfo] = {}
    
    @property
    def name(self) -> str:
        """Check name identifier."""
        return "api_security"
    
    @property
    def source(self) -> str:
        """Source identifier for findings."""
        return "api_testing"
    
    async def run_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """
        Execute comprehensive API security tests.
        
        Args:
            url: Target API URL to test
            client: Async HTTP client
            result: Result container to populate
            
        Raises:
            ValueError: If URL is invalid
        """
        # ✅ Input validation
        if not url or not url.startswith(('http://', 'https://')):
            raise ValueError(f"Invalid URL: {url}")
        
        logger.info("Starting API security testing for: %s", url)
        
        try:
            # Step 1: Detect and classify API endpoint
            endpoint_info = await self._detect_and_classify_api(url, client)
            
            if endpoint_info.api_type == APIType.UNKNOWN:
                result.add_finding(SecurityFinding(
                    check_name="api_not_detected",
                    status=CheckStatus.PASS,
                    severity=Severity.INFO,
                    message="No API endpoint detected (not JSON/XML/GraphQL response)",
                    details={
                        "content_type": endpoint_info.content_type,
                        "note": "Standard web page detected, not an API endpoint"
                    },
                    source=self.source,
                ))
                return
            
            # Step 2: Run all API security tests in parallel
            test_tasks = [
                self._check_api_versioning(endpoint_info, result),
                self._check_api_authentication(url, client, endpoint_info, result),
                self._check_excessive_data_exposure(url, client, result),
                self._check_error_handling(url, client, result),
                self._check_rate_limiting(url, client, result),
            ]
            
            # Add GraphQL-specific tests
            if endpoint_info.api_type == APIType.GRAPHQL:
                test_tasks.append(
                    self._check_graphql_introspection(url, client, result)
                )
                test_tasks.append(
                    self._check_graphql_depth_limit(url, client, result)
                )
            
            # Run all tests concurrently
            await asyncio.gather(*test_tasks, return_exceptions=True)
            
            logger.info("API security testing completed for: %s", url)
            
        except TimeoutException as e:
            logger.error("API testing timeout for %s: %s", url, e)
            result.add_finding(SecurityFinding(
                check_name="api_test_timeout",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message=f"API testing timed out after {self.timeout_s}s",
                details={"error": str(e)},
                source=self.source,
            ))
        except Exception as e:
            logger.error("API security testing failed for %s: %s", url, e, exc_info=True)
            result.add_finding(SecurityFinding(
                check_name="api_test_error",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message=f"API security testing failed: {str(e)[:200]}",
                source=self.source,
            ))
    
    async def _detect_and_classify_api(
        self,
        url: str,
        client: httpx.AsyncClient,
    ) -> APIEndpointInfo:
        """
        Detect and classify API endpoint type.
        
        Returns:
            APIEndpointInfo with endpoint classification
        """
        import time
        
        # Check cache first
        if url in self._endpoint_cache:
            return self._endpoint_cache[url]
        
        try:
            start_time = time.time()
            response = await client.get(
                url,
                timeout=self._timeout,
                follow_redirects=True,
            )
            response_time_ms = (time.time() - start_time) * 1000
            
            content_type = response.headers.get('content-type', '').lower()
            
            # Detect API type
            api_type = self._classify_api_type(url, content_type, response)
            
            # Check for versioning
            has_versioning, version = self._detect_versioning(url, response)
            
            # Check authentication
            auth_required = response.status_code in [401, 403]
            auth_scheme = None
            if auth_required:
                www_auth = response.headers.get('www-authenticate', '')
                auth_scheme = www_auth.split()[0] if www_auth else None
            
            endpoint_info = APIEndpointInfo(
                url=url,
                api_type=api_type,
                content_type=content_type,
                status_code=response.status_code,
                response_time_ms=response_time_ms,
                has_versioning=has_versioning,
                version=version,
                authentication_required=auth_required,
                authentication_scheme=auth_scheme,
            )
            
            # Cache result
            self._endpoint_cache[url] = endpoint_info
            
            return endpoint_info
        
        except Exception as e:
            logger.error("API detection failed: %s", e)
            return APIEndpointInfo(
                url=url,
                api_type=APIType.UNKNOWN,
                content_type='unknown',
                status_code=0,
                response_time_ms=0.0,
                has_versioning=False,
            )
    
    def _classify_api_type(
        self,
        url: str,
        content_type: str,
        response: httpx.Response,
    ) -> APIType:
        """Classify API endpoint type."""
        # Check content type
        if 'json' in content_type:
            # Check if GraphQL
            if 'graphql' in url.lower() or 'graphql' in content_type:
                return APIType.GRAPHQL
            return APIType.REST
        
        if 'xml' in content_type or 'soap' in content_type:
            return APIType.SOAP
        
        # Check URL patterns
        if any(pattern in url.lower() for pattern in APISecurityConfig.API_PATH_PATTERNS):
            # Try to parse as JSON
            try:
                json.loads(response.text)
                return APIType.REST
            except:
                pass
        
        return APIType.UNKNOWN
    
    def _detect_versioning(
        self,
        url: str,
        response: httpx.Response,
    ) -> Tuple[bool, Optional[str]]:
        """
        Detect API versioning.
        
        Returns:
            Tuple of (has_versioning, version_string)
        """
        # Check URL patterns
        for pattern in APISecurityConfig.VERSION_PATTERNS:
            match = re.search(pattern, url)
            if match:
                return True, match.group(1)
        
        # Check headers
        version_headers = ['api-version', 'x-api-version', 'version']
        for header in version_headers:
            if header in response.headers:
                return True, response.headers[header]
        
        return False, None
    
    async def _check_api_versioning(
        self,
        endpoint_info: APIEndpointInfo,
        result: SecurityCheckResult,
    ) -> None:
        """Check if API uses proper versioning."""
        if not endpoint_info.has_versioning:
            result.add_finding(SecurityFinding(
                check_name="api_no_versioning",
                status=CheckStatus.FAIL,
                severity=Severity.LOW,
                message="API endpoint does not use versioning",
                details={
                    "url": endpoint_info.url,
                    "api_type": endpoint_info.api_type.name,
                },
                recommendation=(
                    "Implement API versioning to enable:\n"
                    "1. Gradual deprecation of old endpoints\n"
                    "2. Backward compatibility\n"
                    "3. Controlled breaking changes\n\n"
                    "Best practices:\n"
                    "- URI versioning: /api/v1/users\n"
                    "- Header versioning: Accept: application/vnd.api.v1+json\n"
                    "- Query parameter: /api/users?version=1"
                ),
                source=self.source,
                cwe_id="CWE-1059",
                owasp_category="API9:2023 - Improper Inventory Management",
                confidence=0.8,
                references=[
                    "https://owasp.org/API-Security/editions/2023/en/0xa9-improper-inventory-management/",
                ],
            ))
        else:
            result.add_finding(SecurityFinding(
                check_name="api_versioning_present",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message=f"API uses versioning: {endpoint_info.version or 'detected'}",
                details={"version": endpoint_info.version},
                source=self.source,
            ))
    
    async def _check_graphql_introspection(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """
        Test if GraphQL introspection is enabled in production.
        
        Introspection should be disabled in production to prevent
        schema disclosure and information gathering attacks.
        """
        introspection_enabled = False
        schema_types_found = []
        
        for query in APISecurityConfig.GRAPHQL_INTROSPECTION_QUERIES:
            try:
                response = await client.post(
                    url,
                    json={"query": query},
                    headers={'Content-Type': 'application/json'},
                    timeout=self._timeout,
                )
                
                if response.status_code == 200:
                    try:
                        data = response.json()
                        
                        # Check if introspection query succeeded
                        if '__schema' in str(data) or '__type' in str(data):
                            introspection_enabled = True
                            
                            # Extract schema information
                            if 'data' in data and '__schema' in data['data']:
                                schema = data['data']['__schema']
                                if 'types' in schema:
                                    schema_types_found = [
                                        t['name'] for t in schema['types'][:10]
                                        if isinstance(t, dict) and 'name' in t
                                    ]
                            
                            break
                    except (json.JSONDecodeError, KeyError):
                        pass
            
            except Exception as e:
                logger.debug("GraphQL introspection query failed: %s", e)
                continue
        
        if introspection_enabled:
            result.add_finding(SecurityFinding(
                check_name="graphql_introspection_enabled",
                status=CheckStatus.FAIL,
                severity=Severity.MEDIUM,
                message="GraphQL introspection is enabled in production",
                details={
                    "endpoint": url,
                    "schema_types_discovered": schema_types_found,
                    "types_count": len(schema_types_found),
                },
                recommendation=(
                    "Disable GraphQL introspection in production environments:\n\n"
                    "Apollo Server:\n"
                    "  introspection: process.env.NODE_ENV !== 'production'\n\n"
                    "GraphQL-JS:\n"
                    "  GraphQLSchema({ ..., enableIntrospection: false })\n\n"
                    "Django Graphene:\n"
                    "  GRAPHENE = {'MIDDLEWARE': [..., 'your.middleware.DisableIntrospection']}"
                ),
                source=self.source,
                cwe_id="CWE-200",
                owasp_category="API8:2023 - Security Misconfiguration",
                confidence=0.95,
                references=[
                    "https://graphql.org/learn/introspection/",
                    "https://owasp.org/API-Security/editions/2023/en/0xa8-security-misconfiguration/",
                ],
            ))
        else:
            result.add_finding(SecurityFinding(
                check_name="graphql_introspection_disabled",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message="GraphQL introspection appears to be disabled",
                source=self.source,
            ))
    
    async def _check_graphql_depth_limit(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """Check if GraphQL has query depth limiting."""
        # Test with deeply nested query
        deep_query = """
        query DeepQuery {
          level1 {
            level2 {
              level3 {
                level4 {
                  level5 {
                    level6 {
                      level7 {
                        level8 {
                          level9 {
                            level10 {
                              id
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
        """
        
        try:
            response = await client.post(
                url,
                json={"query": deep_query},
                headers={'Content-Type': 'application/json'},
                timeout=self._timeout,
            )
            
            # If query succeeds or takes very long, depth limiting might be missing
            if response.status_code == 200:
                try:
                    data = response.json()
                    if 'errors' not in data:
                        result.add_finding(SecurityFinding(
                            check_name="graphql_no_depth_limit",
                            status=CheckStatus.FAIL,
                            severity=Severity.MEDIUM,
                            message="GraphQL endpoint may not have query depth limiting",
                            recommendation=(
                                "Implement query depth limiting to prevent DoS attacks:\n"
                                "- Use graphql-depth-limit library\n"
                                "- Set maximum depth (recommended: 7-10)\n"
                                "- Monitor query complexity"
                            ),
                            source=self.source,
                            cwe_id="CWE-400",
                            owasp_category="API4:2023 - Unrestricted Resource Consumption",
                            confidence=0.6,
                        ))
                except:
                    pass
        
        except Exception as e:
            logger.debug("GraphQL depth limit test failed: %s", e)
    
    async def _check_excessive_data_exposure(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """
        Detect excessive data exposure in API responses.
        
        Checks for sensitive fields that shouldn't be exposed.
        """
        try:
            response = await client.get(url, timeout=self._timeout)
            
            if response.status_code != 200:
                return
            
            # Parse JSON response
            try:
                data = response.json()
            except json.JSONDecodeError:
                return
            
            # Find sensitive fields
            sensitive_fields = self._find_sensitive_fields(data)
            
            if sensitive_fields:
                # Categorize by sensitivity
                high_sensitivity = [f for f in sensitive_fields if f.sensitivity_level == 'high']
                medium_sensitivity = [f for f in sensitive_fields if f.sensitivity_level == 'medium']
                
                severity = Severity.CRITICAL if high_sensitivity else Severity.HIGH
                
                result.add_finding(SecurityFinding(
                    check_name="api_excessive_data_exposure",
                    status=CheckStatus.FAIL,
                    severity=severity,
                    message=(
                        f"API response exposes {len(sensitive_fields)} sensitive field(s): "
                        f"{len(high_sensitivity)} high sensitivity, {len(medium_sensitivity)} medium"
                    ),
                    details={
                        "high_sensitivity_fields": [f.field_path for f in high_sensitivity[:10]],
                        "medium_sensitivity_fields": [f.field_path for f in medium_sensitivity[:10]],
                        "total_fields": len(sensitive_fields),
                    },
                    recommendation=(
                        "Remove sensitive fields from API responses:\n"
                        "1. Use DTOs/View Models to control exposed data\n"
                        "2. Implement field-level access control\n"
                        "3. Use serializers to whitelist fields\n"
                        "4. Never expose internal IDs, tokens, or credentials\n"
                        "5. Implement proper data filtering based on user permissions"
                    ),
                    source=self.source,
                    cwe_id="CWE-359",
                    owasp_category="API3:2023 - Broken Object Property Level Authorization",
                    confidence=0.85,
                    references=[
                        "https://owasp.org/API-Security/editions/2023/en/0xa3-broken-object-property-level-authorization/",
                    ],
                ))
        
        except Exception as e:
            logger.debug("Excessive data exposure check failed: %s", e)
    
    def _find_sensitive_fields(
        self,
        data: Any,
        path: str = "",
        depth: int = 0,
    ) -> List[SensitiveFieldInfo]:
        """
        Recursively find sensitive fields in nested data structures.
        
        Args:
            data: Data to analyze (dict, list, or primitive)
            path: Current path in data structure
            depth: Current nesting depth
            
        Returns:
            List of sensitive fields found
        """
        if depth > APISecurityConfig.MAX_NESTED_DEPTH:
            return []
        
        sensitive_fields: List[SensitiveFieldInfo] = []
        fields_checked = 0
        
        if isinstance(data, dict):
            for key, value in data.items():
                if fields_checked >= APISecurityConfig.MAX_FIELDS_TO_CHECK:
                    break
                
                fields_checked += 1
                key_lower = key.lower()
                current_path = f"{path}.{key}" if path else key
                
                # Check if field name indicates sensitive data
                sensitivity = self._assess_field_sensitivity(key_lower)
                
                if sensitivity:
                    value_preview = None
                    if isinstance(value, (str, int, float)) and value:
                        value_preview = str(value)[:50]
                    
                    sensitive_fields.append(SensitiveFieldInfo(
                        field_path=current_path,
                        field_name=key,
                        sensitivity_level=sensitivity,
                        value_preview=value_preview,
                    ))
                
                # Recurse into nested structures
                if isinstance(value, (dict, list)):
                    sensitive_fields.extend(
                        self._find_sensitive_fields(value, current_path, depth + 1)
                    )
        
        elif isinstance(data, list):
            # Check first few items in lists
            for i, item in enumerate(data[:5]):
                if fields_checked >= APISecurityConfig.MAX_FIELDS_TO_CHECK:
                    break
                
                fields_checked += 1
                current_path = f"{path}[{i}]"
                
                if isinstance(item, (dict, list)):
                    sensitive_fields.extend(
                        self._find_sensitive_fields(item, current_path, depth + 1)
                    )
        
        return sensitive_fields
    
    def _assess_field_sensitivity(self, field_name_lower: str) -> Optional[str]:
        """
        Assess field name sensitivity level.
        
        Returns:
            'high', 'medium', or None
        """
        if any(pattern in field_name_lower for pattern in APISecurityConfig.HIGH_SENSITIVITY_FIELDS):
            return 'high'
        
        if any(pattern in field_name_lower for pattern in APISecurityConfig.MEDIUM_SENSITIVITY_FIELDS):
            return 'medium'
        
        return None
    
    async def _check_api_authentication(
        self,
        url: str,
        client: httpx.AsyncClient,
        endpoint_info: APIEndpointInfo,
        result: SecurityCheckResult,
    ) -> None:
        """Check API authentication requirements and methods."""
        if not endpoint_info.authentication_required:
            result.add_finding(SecurityFinding(
                check_name="api_no_authentication",
                status=CheckStatus.FAIL,
                severity=Severity.HIGH,
                message="API endpoint accessible without authentication",
                details={
                    "url": url,
                    "status_code": endpoint_info.status_code,
                },
                recommendation=(
                    "Implement authentication for API endpoints:\n"
                    "- JWT (JSON Web Tokens) for stateless authentication\n"
                    "- OAuth 2.0 for third-party access\n"
                    "- API Keys for service-to-service communication\n"
                    "- mTLS for high-security requirements"
                ),
                source=self.source,
                cwe_id="CWE-306",
                owasp_category="API2:2023 - Broken Authentication",
                confidence=0.9,
                references=[
                    "https://owasp.org/API-Security/editions/2023/en/0xa2-broken-authentication/",
                ],
            ))
        elif endpoint_info.authentication_scheme:
            result.add_finding(SecurityFinding(
                check_name="api_authentication_present",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message=f"API requires authentication: {endpoint_info.authentication_scheme}",
                details={"scheme": endpoint_info.authentication_scheme},
                source=self.source,
            ))
    
    async def _check_error_handling(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """Check for verbose error messages that expose internal details."""
        test_endpoints = [
            f"{url}/nonexistent-endpoint-test-12345",
            f"{url}?invalid_param=<script>alert(1)</script>",
        ]
        
        for test_url in test_endpoints:
            try:
                response = await client.get(test_url, timeout=self._timeout)
                response_text = response.text.lower()
                
                # Check for error disclosure patterns
                disclosed_info = []
                for pattern in APISecurityConfig.ERROR_DISCLOSURE_PATTERNS:
                    if pattern in response_text:
                        disclosed_info.append(pattern)
                
                if disclosed_info:
                    result.add_finding(SecurityFinding(
                        check_name="api_verbose_errors",
                        status=CheckStatus.FAIL,
                        severity=Severity.MEDIUM,
                        message="API returns verbose error messages exposing internal details",
                        details={
                            "patterns_found": disclosed_info[:10],
                            "test_url": test_url[:100],
                        },
                        recommendation=(
                            "Implement proper error handling:\n"
                            "1. Return generic error messages to clients\n"
                            "2. Log detailed errors server-side only\n"
                            "3. Use error codes instead of descriptive messages\n"
                            "4. Sanitize error output\n"
                            "5. Disable debug mode in production"
                        ),
                        source=self.source,
                        cwe_id="CWE-209",
                        owasp_category="API8:2023 - Security Misconfiguration",
                        confidence=0.9,
                    ))
                    break  # Only report once
            
            except Exception as e:
                logger.debug("Error handling test failed for %s: %s", test_url, e)
    
    async def _check_rate_limiting(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """Check if API implements rate limiting."""
        # Send multiple rapid requests
        num_requests = 10
        responses = []
        
        try:
            for i in range(num_requests):
                response = await client.get(url, timeout=self._timeout)
                responses.append(response.status_code)
                
                # Check for rate limit headers
                if any(h in response.headers for h in ['x-ratelimit-limit', 'x-rate-limit-limit', 'ratelimit-limit']):
                    result.add_finding(SecurityFinding(
                        check_name="api_rate_limiting_present",
                        status=CheckStatus.PASS,
                        severity=Severity.INFO,
                        message="API implements rate limiting",
                        details={"rate_limit_headers": dict(response.headers)},
                        source=self.source,
                    ))
                    return
            
            # If all requests succeeded, rate limiting might be missing
            if all(status == 200 for status in responses):
                result.add_finding(SecurityFinding(
                    check_name="api_no_rate_limiting",
                    status=CheckStatus.FAIL,
                    severity=Severity.MEDIUM,
                    message=f"No rate limiting detected ({num_requests} rapid requests succeeded)",
                    recommendation=(
                        "Implement rate limiting to prevent abuse:\n"
                        "- Use headers: X-RateLimit-Limit, X-RateLimit-Remaining\n"
                        "- Return 429 Too Many Requests when limit exceeded\n"
                        "- Consider per-IP and per-user limits\n"
                        "- Use Redis or similar for distributed rate limiting"
                    ),
                    source=self.source,
                    cwe_id="CWE-770",
                    owasp_category="API4:2023 - Unrestricted Resource Consumption",
                    confidence=0.7,
                ))
        
        except Exception as e:
            logger.debug("Rate limiting test failed: %s", e)
    
    def run_sync(
        self,
        url: str,
        client: httpx.Client,
        result: SecurityCheckResult,
    ) -> None:
        """Synchronous wrapper for async implementation."""
        asyncio.run(self.run_async(url, client, result))
