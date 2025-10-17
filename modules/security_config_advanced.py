# modules/security_config_advanced.py
"""
Advanced security configuration with feature flags.
"""

from dataclasses import dataclass, field
from typing import List, Optional
from modules.security_engine import SecurityConfig


@dataclass
class AdvancedSecurityConfig(SecurityConfig):
    """
    Extended security configuration with advanced features.
    """
    
    # Tier selection
    check_tier: str = "standard"  # "basic", "standard", "advanced", "all"
    
    # Advanced feature flags
    enable_zero_day_detection: bool = False
    enable_threat_intelligence: bool = False
    enable_compliance_checking: bool = False
    enable_sbom_generation: bool = False
    
    # Threat intelligence API keys
    alienvault_api_key: Optional[str] = None
    virustotal_api_key: Optional[str] = None
    
    # Compliance frameworks to check
    compliance_frameworks: List[str] = field(default_factory=lambda: ["PCI-DSS"])
    
    # SBOM settings
    sbom_format: str = "cyclonedx"  # "cyclonedx" or "spdx"
    sbom_tool: str = "auto"  # "auto", "syft", "trivy", "manual"
    
    # Performance settings
    enable_parallel_advanced_checks: bool = True
    advanced_check_timeout_s: float = 60.0
    
    @classmethod
    def from_tier(cls, tier: str = "standard", **kwargs) -> "AdvancedSecurityConfig":
        """
        Create config from tier name.
        
        Args:
            tier: "basic", "standard", "advanced", or "all"
            **kwargs: Additional config overrides
        
        Returns:
            AdvancedSecurityConfig instance
        """
        from modules.security_checks import get_checks_by_tier
        
        config = cls(
            check_tier=tier,
            enabled_checks=get_checks_by_tier(tier),
            **kwargs
        )
        
        # Auto-enable advanced features for "advanced" and "all" tiers
        if tier in ("advanced", "all"):
            config.enable_zero_day_detection = kwargs.get("enable_zero_day_detection", True)
        
        return config
    
    @classmethod
    def from_env(cls) -> "AdvancedSecurityConfig":
        """Load configuration from environment variables."""
        import os
        
        tier = os.getenv("SECURITY_TIER", "standard")
        
        return cls.from_tier(
            tier=tier,
            enable_ai=os.getenv("SECURITY_AI_ANALYSIS", "false").lower() == "true",
            enable_cve_lookup=os.getenv("SECURITY_CVE_LOOKUP", "false").lower() == "true",
            enable_zero_day_detection=os.getenv("SECURITY_ZERO_DAY", "false").lower() == "true",
            enable_threat_intelligence=os.getenv("SECURITY_THREAT_INTEL", "false").lower() == "true",
            enable_compliance_checking=os.getenv("SECURITY_COMPLIANCE", "false").lower() == "true",
            enable_sbom_generation=os.getenv("SECURITY_SBOM", "false").lower() == "true",
            alienvault_api_key=os.getenv("ALIENVAULT_API_KEY"),
            virustotal_api_key=os.getenv("VIRUSTOTAL_API_KEY"),
        )
