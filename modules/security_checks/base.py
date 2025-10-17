# modules/security_checks/base.py
"""
Base class for security checks.
All security checks must inherit from AbstractSecurityCheck.
"""

from __future__ import annotations
import asyncio
import logging
from abc import ABC, abstractmethod
from typing import Optional
import httpx

from modules.security_types import (
    SecurityFinding,
    SecurityCheckResult,
    CheckStatus,
    Severity,
    CheckExecutionError,
)

logger = logging.getLogger(__name__)


class AbstractSecurityCheck(ABC):
    """
    Abstract base class for all security checks.
    Provides consistent interface and timeout handling.
    """
    
    def __init__(self, timeout_s: float = 15.0, check_timeout_s: float = 30.0):
        """
        Initialize security check.
        
        Args:
            timeout_s: HTTP request timeout
            check_timeout_s: Overall check timeout (prevents hung checks)
        """
        self.timeout_s = timeout_s
        self.check_timeout_s = check_timeout_s
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Check name (e.g., 'security_headers')."""
        pass
    
    @property
    @abstractmethod
    def source(self) -> str:
        """Check source category (e.g., 'headers', 'cookies', 'tls')."""
        pass
    
    @abstractmethod
    async def run_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """
        Execute check asynchronously.
        
        Args:
            url: Target URL
            client: Shared HTTP client
            result: Result object to add findings to
        """
        pass
    
    @abstractmethod
    def run_sync(
        self,
        url: str,
        client: httpx.Client,
        result: SecurityCheckResult,
    ) -> None:
        """
        Execute check synchronously.
        
        Args:
            url: Target URL
            client: Shared HTTP client
            result: Result object to add findings to
        """
        pass
    
    # ✨ Wrapper with timeout guard
    async def execute_async(
        self,
        url: str,
        client: httpx.AsyncClient,
        result: SecurityCheckResult,
    ) -> None:
        """
        Execute check with timeout guard and error handling.
        
        Args:
            url: Target URL
            client: Shared HTTP client
            result: Result object to add findings to
        """
        try:
            # ✨ Timeout guard prevents hung checks
            await asyncio.wait_for(
                self.run_async(url, client, result),
                timeout=self.check_timeout_s,
            )
            logger.debug("✅ Check '%s' completed", self.name)
            
        except asyncio.TimeoutError:
            logger.warning("⏱️ Check '%s' timed out after %ds", self.name, self.check_timeout_s)
            result.add_finding(SecurityFinding(
                check_name=f"{self.name}_timeout",
                status=CheckStatus.ERROR,
                severity=Severity.LOW,
                message=f"Check timed out after {self.check_timeout_s}s",
                source=self.source,
            ))
        
        except asyncio.CancelledError:
            logger.warning("❌ Check '%s' was cancelled", self.name)
            raise  # Propagate cancellation
        
        except Exception as e:
            logger.error("❌ Check '%s' failed: %s", self.name, e)
            result.add_finding(SecurityFinding(
                check_name=f"{self.name}_error",
                status=CheckStatus.ERROR,
                severity=Severity.MEDIUM,
                message=f"Check execution failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                source=self.source,
            ))
    
    def execute_sync(
        self,
        url: str,
        client: httpx.Client,
        result: SecurityCheckResult,
    ) -> None:
        """
        Execute check synchronously with error handling.
        
        Args:
            url: Target URL
            client: Shared HTTP client
            result: Result object to add findings to
        """
        try:
            self.run_sync(url, client, result)
            logger.debug("✅ Check '%s' completed", self.name)
            
        except Exception as e:
            logger.error("❌ Check '%s' failed: %s", self.name, e)
            result.add_finding(SecurityFinding(
                check_name=f"{self.name}_error",
                status=CheckStatus.ERROR,
                severity=Severity.MEDIUM,
                message=f"Check execution failed: {str(e)}",
                details={"error": str(e), "error_type": type(e).__name__},
                source=self.source,
            ))
