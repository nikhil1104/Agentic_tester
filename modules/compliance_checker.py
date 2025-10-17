# modules/compliance_checker.py
"""
Compliance checking for PCI-DSS, HIPAA, SOC2.
Maps security findings to compliance requirements.
"""

from typing import Dict, Any, List
from dataclasses import dataclass


@dataclass
class ComplianceRequirement:
    """Single compliance requirement."""
    framework: str  # "PCI-DSS", "HIPAA", "SOC2"
    requirement_id: str  # e.g., "PCI-DSS 6.5.1"
    description: str
    related_cwe: List[str]
    related_owasp: List[str]


class ComplianceChecker:
    """
    Map security findings to compliance requirements.
    """
    
    def __init__(self):
        self.requirements = self._load_requirements()
    
    def _load_requirements(self) -> List[ComplianceRequirement]:
        """Load compliance requirements database."""
        return [
            ComplianceRequirement(
                framework="PCI-DSS",
                requirement_id="6.5.1",
                description="Injection flaws (SQL, OS command, LDAP)",
                related_cwe=["CWE-89", "CWE-78", "CWE-90"],
                related_owasp=["A03:2021"],
            ),
            ComplianceRequirement(
                framework="PCI-DSS",
                requirement_id="6.5.7",
                description="Cross-site scripting (XSS)",
                related_cwe=["CWE-79"],
                related_owasp=["A03:2021"],
            ),
            ComplianceRequirement(
                framework="HIPAA",
                requirement_id="164.312(a)(1)",
                description="Access control - encryption in transit",
                related_cwe=["CWE-319", "CWE-327"],
                related_owasp=["A02:2021"],
            ),
            ComplianceRequirement(
                framework="SOC2",
                requirement_id="CC6.1",
                description="Logical and physical access controls",
                related_cwe=["CWE-284"],
                related_owasp=["A01:2021"],
            ),
            # Add more requirements...
        ]
    
    def check_compliance(
        self,
        findings: List[Dict[str, Any]],
        framework: str = "PCI-DSS",
    ) -> Dict[str, Any]:
        """
        Check compliance against framework.
        
        Returns:
            {
                "framework": "PCI-DSS",
                "compliant": False,
                "violations": [...],
                "coverage": 0.85
            }
        """
        framework_requirements = [
            req for req in self.requirements
            if req.framework == framework
        ]
        
        violations = []
        
        for finding in findings:
            cwe_id = finding.get("cwe_id")
            owasp = finding.get("owasp_category")
            
            # Check if finding violates any requirement
            for req in framework_requirements:
                if (cwe_id in req.related_cwe or 
                    any(owasp_cat in str(owasp) for owasp_cat in req.related_owasp)):
                    
                    violations.append({
                        "requirement": req.requirement_id,
                        "description": req.description,
                        "finding": finding.get("message"),
                        "severity": finding.get("severity"),
                    })
        
        return {
            "framework": framework,
            "compliant": len(violations) == 0,
            "violations": violations,
            "total_requirements": len(framework_requirements),
            "violated_requirements": len(set(v["requirement"] for v in violations)),
            "coverage": 1.0 - (len(violations) / max(len(framework_requirements), 1)),
        }
