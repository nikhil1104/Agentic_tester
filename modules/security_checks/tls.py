# modules/security_checks/tls.py
"""
SSL/TLS and certificate validation check.
Includes certificate expiry, issuer, and TLS version validation.
"""

from __future__ import annotations
import logging
import ssl
import socket
from datetime import datetime, timedelta
from urllib.parse import urlsplit
import httpx

from modules.security_checks.base import AbstractSecurityCheck
from modules.security_types import SecurityFinding, SecurityCheckResult, CheckStatus, Severity

logger = logging.getLogger(__name__)


class TLSCheck(AbstractSecurityCheck):
    """Validates SSL/TLS configuration and certificate."""
    
    def __init__(
        self,
        timeout_s: float = 15.0,
        min_tls_version: str = "TLSv1.2",
        certificate_expiry_warning_days: int = 30,
    ):
        super().__init__(timeout_s=timeout_s)
        self.min_tls_version = min_tls_version
        self.certificate_expiry_warning_days = certificate_expiry_warning_days
    
    @property
    def name(self) -> str:
        return "tls_validation"
    
    @property
    def source(self) -> str:
        return "tls"
    
    async def run_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """Execute async TLS check."""
        parsed = urlsplit(url)
        
        if parsed.scheme != "https":
            result.add_finding(SecurityFinding(
                check_name="https_enabled",
                status=CheckStatus.FAIL,
                severity=Severity.CRITICAL,
                message="Site does not use HTTPS",
                details={"scheme": parsed.scheme},
                recommendation="Enable HTTPS with valid SSL/TLS certificate",
                source=self.source,
                cwe_id="CWE-319",
                owasp_category="A02:2021 - Cryptographic Failures",
                references=["https://owasp.org/www-community/vulnerabilities/Missing_Encryption_of_Sensitive_Data"],
            ))
            return
        
        # Validate certificate
        try:
            context = ssl.create_default_context()
            async with httpx.AsyncClient(verify=context, timeout=self.timeout_s) as test_client:
                await test_client.get(url)
            
            # ✨ Extract certificate details
            self._extract_certificate_details(parsed.netloc, result)
            
        except httpx.HTTPError as e:
            result.add_finding(SecurityFinding(
                check_name="tls_certificate_invalid",
                status=CheckStatus.FAIL,
                severity=Severity.CRITICAL,
                message=f"SSL/TLS certificate validation failed: {str(e)}",
                details={"error": str(e)},
                recommendation="Fix SSL/TLS certificate issues",
                source=self.source,
                cwe_id="CWE-295",
                owasp_category="A02:2021 - Cryptographic Failures",
                references=["https://www.ssllabs.com/ssltest/"],
            ))
    
    def run_sync(
        self,
        url: str,
        client: httpx.Client,
        result: SecurityCheckResult,
    ) -> None:
        """Execute sync TLS check."""
        parsed = urlsplit(url)
        
        if parsed.scheme != "https":
            result.add_finding(SecurityFinding(
                check_name="https_enabled",
                status=CheckStatus.FAIL,
                severity=Severity.CRITICAL,
                message="Site does not use HTTPS",
                details={"scheme": parsed.scheme},
                recommendation="Enable HTTPS",
                source=self.source,
                cwe_id="CWE-319",
                owasp_category="A02:2021 - Cryptographic Failures",
            ))
            return
        
        try:
            context = ssl.create_default_context()
            with httpx.Client(verify=context, timeout=self.timeout_s) as test_client:
                test_client.get(url)
            
            self._extract_certificate_details(parsed.netloc, result)
            
        except httpx.HTTPError as e:
            result.add_finding(SecurityFinding(
                check_name="tls_certificate_invalid",
                status=CheckStatus.FAIL,
                severity=Severity.CRITICAL,
                message=f"SSL/TLS certificate validation failed: {str(e)}",
                source=self.source,
                cwe_id="CWE-295",
                owasp_category="A02:2021 - Cryptographic Failures",
            ))
    
    def _extract_certificate_details(self, hostname: str, result: SecurityCheckResult):
        """
        ✨ Extract and validate certificate details.
        
        Args:
            hostname: Hostname to check
            result: Result object to add findings to
        """
        try:
            # Create SSL context
            context = ssl.create_default_context()
            
            # Connect and get certificate
            with socket.create_connection((hostname, 443), timeout=self.timeout_s) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    tls_version = ssock.version()
            
            # Validate TLS version
            if tls_version and tls_version < self.min_tls_version:
                result.add_finding(SecurityFinding(
                    check_name="tls_version_outdated",
                    status=CheckStatus.FAIL,
                    severity=Severity.HIGH,
                    message=f"TLS version {tls_version} is outdated (minimum: {self.min_tls_version})",
                    details={"tls_version": tls_version, "min_required": self.min_tls_version},
                    recommendation=f"Upgrade to {self.min_tls_version} or higher",
                    source=self.source,
                    cwe_id="CWE-327",
                    owasp_category="A02:2021 - Cryptographic Failures",
                ))
            else:
                result.add_finding(SecurityFinding(
                    check_name="tls_version_valid",
                    status=CheckStatus.PASS,
                    severity=Severity.INFO,
                    message=f"TLS version {tls_version} is acceptable",
                    details={"tls_version": tls_version},
                    source=self.source,
                ))
            
            # Check certificate expiry
            not_after_str = cert.get("notAfter")
            if not_after_str:
                not_after = datetime.strptime(not_after_str, "%b %d %H:%M:%S %Y %Z")
                days_until_expiry = (not_after - datetime.utcnow()).days
                
                if days_until_expiry < 0:
                    result.add_finding(SecurityFinding(
                        check_name="certificate_expired",
                        status=CheckStatus.FAIL,
                        severity=Severity.CRITICAL,
                        message="SSL certificate has expired",
                        details={"expiry_date": not_after_str, "days_expired": abs(days_until_expiry)},
                        recommendation="Renew SSL certificate immediately",
                        source=self.source,
                        cwe_id="CWE-295",
                        owasp_category="A02:2021 - Cryptographic Failures",
                    ))
                elif days_until_expiry < self.certificate_expiry_warning_days:
                    result.add_finding(SecurityFinding(
                        check_name="certificate_expiring_soon",
                        status=CheckStatus.WARNING,
                        severity=Severity.MEDIUM,
                        message=f"SSL certificate expires in {days_until_expiry} days",
                        details={"expiry_date": not_after_str, "days_remaining": days_until_expiry},
                        recommendation="Renew SSL certificate soon",
                        source=self.source,
                        cwe_id="CWE-295",
                        owasp_category="A02:2021 - Cryptographic Failures",
                    ))
                else:
                    result.add_finding(SecurityFinding(
                        check_name="certificate_valid",
                        status=CheckStatus.PASS,
                        severity=Severity.INFO,
                        message=f"SSL certificate is valid (expires in {days_until_expiry} days)",
                        details={"expiry_date": not_after_str, "days_remaining": days_until_expiry},
                        source=self.source,
                    ))
            
            # Check issuer
            issuer = dict(x[0] for x in cert.get("issuer", []))
            result.add_finding(SecurityFinding(
                check_name="certificate_issuer",
                status=CheckStatus.PASS,
                severity=Severity.INFO,
                message=f"Certificate issued by: {issuer.get('organizationName', 'Unknown')}",
                details={"issuer": issuer},
                source=self.source,
            ))
            
        except Exception as e:
            logger.warning("Failed to extract certificate details for %s: %s", hostname, e)
            result.add_finding(SecurityFinding(
                check_name="certificate_details_extraction_failed",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message=f"Could not extract certificate details: {str(e)}",
                source=self.source,
            ))
