# modules/security_checks/sql_injection.py
"""
SQL Injection Testing Module

Tests for SQL injection vulnerabilities using safe probe payloads.
Does NOT execute destructive commands.
"""

from __future__ import annotations
import logging
from typing import List
import httpx
from urllib.parse import urljoin, parse_qs, urlparse, urlencode

from modules.security_checks.base import AbstractSecurityCheck
from modules.security_types import (
    SecurityFinding,
    SecurityCheckResult,
    CheckStatus,
    Severity,
)

logger = logging.getLogger(__name__)


class SQLInjectionCheck(AbstractSecurityCheck):
    """Test for SQL injection vulnerabilities."""
    
    # Safe SQL injection payloads (read-only, non-destructive)
    TEST_PAYLOADS = [
        "'",
        "1' OR '1'='1",
        "1' AND '1'='2",
        "' OR 1=1--",
        "admin'--",
        "' UNION SELECT NULL--",
    ]
    
    # SQL error patterns in responses
    SQL_ERROR_PATTERNS = [
        r"SQL syntax.*MySQL",
        r"Warning.*mysql_.*",
        r"valid MySQL result",
        r"PostgreSQL.*ERROR",
        r"Warning.*pg_.*",
        r"valid PostgreSQL result",
        r"Microsoft OLE DB Provider for SQL Server",
        r"Unclosed quotation mark",
        r"ODBC SQL Server Driver",
        r"SQLServer JDBC Driver",
        r"Oracle error",
        r"ORA-\d{5}",
    ]
    
    @property
    def name(self) -> str:
        return "sql_injection"
    
    @property
    def source(self) -> str:
        return "active_testing"
    
    async def run_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ):
        """Test for SQL injection vulnerabilities."""
        import re
        
        try:
            # Parse URL to get query parameters
            parsed = urlparse(url)
            params = parse_qs(parsed.query)
            
            if not params:
                result.add_finding(SecurityFinding(
                    check_name="sql_injection_no_params",
                    status=CheckStatus.PASS,
                    severity=Severity.INFO,
                    message="No query parameters to test for SQL injection",
                    source=self.source,
                ))
                return
            
            # Test each parameter with SQL injection payloads
            vulnerabilities_found = []
            
            for param_name in params.keys():
                for payload in self.TEST_PAYLOADS:
                    # Create test URL with payload
                    test_params = params.copy()
                    test_params[param_name] = [payload]
                    test_url = f"{parsed.scheme}://{parsed.netloc}{parsed.path}?{urlencode(test_params, doseq=True)}"
                    
                    # Send request
                    response = await client.get(test_url)
                    
                    # Check for SQL errors in response
                    for pattern in self.SQL_ERROR_PATTERNS:
                        if re.search(pattern, response.text, re.IGNORECASE):
                            vulnerabilities_found.append({
                                "parameter": param_name,
                                "payload": payload,
                                "pattern": pattern,
                            })
                            
                            result.add_finding(SecurityFinding(
                                check_name=f"sql_injection_{param_name}",
                                status=CheckStatus.FAIL,
                                severity=Severity.CRITICAL,
                                message=f"SQL injection vulnerability detected in parameter '{param_name}'",
                                details={
                                    "parameter": param_name,
                                    "payload": payload,
                                    "error_pattern": pattern,
                                },
                                recommendation="Use parameterized queries or prepared statements. Never concatenate user input into SQL queries.",
                                source=self.source,
                                cwe_id="CWE-89",
                                owasp_category="A03:2021 - Injection",
                                confidence=0.9,
                                references=[
                                    "https://owasp.org/www-community/attacks/SQL_Injection",
                                    "https://cheatsheetseries.owasp.org/cheatsheets/SQL_Injection_Prevention_Cheat_Sheet.html",
                                ],
                            ))
                            break  # Stop testing this param once vulnerability found
            
            if not vulnerabilities_found:
                result.add_finding(SecurityFinding(
                    check_name="sql_injection_test",
                    status=CheckStatus.PASS,
                    severity=Severity.INFO,
                    message=f"No SQL injection vulnerabilities detected in {len(params)} parameter(s)",
                    details={
                        "parameters_tested": list(params.keys()),
                        "payloads_tested": len(self.TEST_PAYLOADS),
                    },
                    source=self.source,
                ))
        
        except Exception as e:
            logger.error("SQL injection testing failed: %s", e)
            result.add_finding(SecurityFinding(
                check_name="sql_injection_error",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message=f"SQL injection testing failed: {str(e)}",
                source=self.source,
            ))
    
    def run_sync(self, url: str, client: httpx.Client, result: SecurityCheckResult):
        """Sync version."""
        import asyncio
        asyncio.run(self.run_async(url, client, result))
