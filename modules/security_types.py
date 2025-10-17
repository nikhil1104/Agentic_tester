# modules/security_types.py
"""
Shared types, enums, and dataclasses for security engine.
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, Any, Optional, List


class CheckStatus(Enum):
    """Status of a security check."""
    PASS = "PASS"
    FAIL = "FAIL"
    WARNING = "WARNING"
    SKIPPED = "SKIPPED"
    ERROR = "ERROR"


class Severity(Enum):
    """Severity of a security finding."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


@dataclass
class SecurityFinding:
    """Individual security finding with compliance mapping."""
    check_name: str
    status: CheckStatus
    severity: Severity
    message: str
    details: Dict[str, Any] = field(default_factory=dict)
    recommendation: Optional[str] = None
    references: List[str] = field(default_factory=list)
    
    # ✨ NEW: Compliance mapping
    cwe_id: Optional[str] = None  # e.g., "CWE-693"
    owasp_category: Optional[str] = None  # e.g., "A05:2021 - Security Misconfiguration"
    source: Optional[str] = None  # e.g., "headers", "cookies", "tls"
    confidence: float = 1.0  # 0.0-1.0 for heuristic checks
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "status": self.status.value,
            "severity": self.severity.value,
        }


@dataclass
class SecurityCheckResult:
    """Result of a complete security check suite."""
    url: str
    timestamp: str
    overall_status: CheckStatus
    findings: List[SecurityFinding] = field(default_factory=list)
    summary: Dict[str, int] = field(default_factory=dict)
    duration_s: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_finding(self, finding: SecurityFinding):
        """Add a finding to results."""
        self.findings.append(finding)
        
    def calculate_summary(self):
        """Calculate summary with severity weighting."""
        self.summary = {
            "total_checks": len(self.findings),
            "passed": sum(1 for f in self.findings if f.status == CheckStatus.PASS),
            "failed": sum(1 for f in self.findings if f.status == CheckStatus.FAIL),
            "warnings": sum(1 for f in self.findings if f.status == CheckStatus.WARNING),
            "errors": sum(1 for f in self.findings if f.status == CheckStatus.ERROR),
            "skipped": sum(1 for f in self.findings if f.status == CheckStatus.SKIPPED),
        }
        
        # ✨ Weighted severity score (0-100)
        severity_weights = {
            Severity.CRITICAL: 20,
            Severity.HIGH: 10,
            Severity.MEDIUM: 5,
            Severity.LOW: 2,
            Severity.INFO: 0,
        }
        
        total_score = 0
        for finding in self.findings:
            if finding.status in (CheckStatus.FAIL, CheckStatus.WARNING):
                total_score += severity_weights.get(finding.severity, 0) * finding.confidence
        
        self.summary["risk_score"] = round(min(total_score, 100), 2)
        
        # Determine overall status
        if self.summary["failed"] > 0 or self.summary["errors"] > 0:
            self.overall_status = CheckStatus.FAIL
        elif self.summary["warnings"] > 0:
            self.overall_status = CheckStatus.WARNING
        else:
            self.overall_status = CheckStatus.PASS
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            **asdict(self),
            "findings": [f.to_dict() for f in self.findings],
            "overall_status": self.overall_status.value,
        }
    
    # ✨ NEW: Export methods
    def to_json(self) -> str:
        """Export to JSON string."""
        import json
        return json.dumps(self.to_dict(), indent=2)
    
    def to_markdown(self) -> str:
        """Export to Markdown report."""
        lines = [
            f"# Security Scan Report",
            f"",
            f"**URL:** {self.url}",
            f"**Timestamp:** {self.timestamp}",
            f"**Status:** {self.overall_status.value}",
            f"**Duration:** {self.duration_s}s",
            f"**Risk Score:** {self.summary.get('risk_score', 0)}/100",
            f"",
            f"## Summary",
            f"",
            f"| Metric | Count |",
            f"|--------|-------|",
        ]
        
        for key, value in self.summary.items():
            if key != "risk_score":
                lines.append(f"| {key.replace('_', ' ').title()} | {value} |")
        
        lines.extend([
            f"",
            f"## Findings",
            f"",
        ])
        
        for finding in self.findings:
            icon = "✅" if finding.status == CheckStatus.PASS else "❌" if finding.status == CheckStatus.FAIL else "⚠️"
            lines.append(f"### {icon} {finding.check_name}")
            lines.append(f"")
            lines.append(f"**Status:** {finding.status.value}")
            lines.append(f"**Severity:** {finding.severity.value}")
            lines.append(f"**Message:** {finding.message}")
            
            if finding.cwe_id or finding.owasp_category:
                lines.append(f"**Compliance:** {finding.cwe_id or ''} {finding.owasp_category or ''}")
            
            if finding.recommendation:
                lines.append(f"**Recommendation:** {finding.recommendation}")
            
            lines.append(f"")
        
        return "\n".join(lines)


class SecurityEngineError(Exception):
    """Base exception for security engine errors."""
    pass


class CheckExecutionError(SecurityEngineError):
    """Raised when a security check fails to execute."""
    def __init__(self, check_name: str, original_error: Exception):
        self.check_name = check_name
        self.original_error = original_error
        super().__init__(f"Check '{check_name}' execution failed: {original_error}")
