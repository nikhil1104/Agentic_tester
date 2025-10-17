# modules/security_engine_advanced.py
"""
Advanced Security Engine with threat intelligence and compliance.
Extends base SecurityEngine with additional capabilities.
"""

from __future__ import annotations
import asyncio
import logging
from typing import Dict, Any, Optional, List

from modules.security_engine import SecurityEngine
from modules.security_config_advanced import AdvancedSecurityConfig
from modules.security_types import SecurityCheckResult

logger = logging.getLogger(__name__)


class AdvancedSecurityEngine(SecurityEngine):
    """
    Production-grade security engine with advanced features.
    
    New capabilities:
    - Zero-day threat detection
    - Threat intelligence feed integration
    - Compliance checking (PCI-DSS, HIPAA, SOC2)
    - SBOM generation
    """
    
    def __init__(self, config: Optional[AdvancedSecurityConfig] = None):
        """
        Initialize advanced security engine.
        
        Args:
            config: AdvancedSecurityConfig instance
        """
        # Initialize base engine
        if config is None:
            config = AdvancedSecurityConfig.from_tier("standard")
        
        super().__init__(config=config)
        
        self.advanced_config = config
        
        # Initialize advanced components
        self.threat_intel = None
        self.compliance_checker = None
        self.sbom_generator = None
        
        if config.enable_threat_intelligence:
            self._init_threat_intelligence()
        
        if config.enable_compliance_checking:
            self._init_compliance_checker()
        
        if config.enable_sbom_generation:
            self._init_sbom_generator()
        
        logger.info(
            "AdvancedSecurityEngine initialized: tier=%s, advanced_features=%s",
            config.check_tier,
            {
                "threat_intel": bool(self.threat_intel),
                "compliance": bool(self.compliance_checker),
                "sbom": bool(self.sbom_generator),
            }
        )
    
    def _init_threat_intelligence(self):
        """Initialize threat intelligence integration."""
        try:
            from modules.threat_intel import ThreatIntelligence
            self.threat_intel = ThreatIntelligence()
            if self.threat_intel.enabled:
                logger.info("✅ Threat intelligence enabled")
            else:
                logger.warning("⚠️  Threat intelligence disabled (missing API keys)")
        except ImportError as e:
            logger.warning("Threat intelligence module not available: %s", e)
    
    def _init_compliance_checker(self):
        """Initialize compliance checker."""
        try:
            from modules.compliance_checker import ComplianceChecker
            self.compliance_checker = ComplianceChecker()
            logger.info("✅ Compliance checking enabled")
        except ImportError as e:
            logger.warning("Compliance checker module not available: %s", e)
    
    def _init_sbom_generator(self):
        """Initialize SBOM generator."""
        try:
            from modules.sbom_generator import SBOMGenerator
            self.sbom_generator = SBOMGenerator()
            logger.info("✅ SBOM generation enabled")
        except ImportError as e:
            logger.warning("SBOM generator module not available: %s", e)
    
    async def scan_async(self, url: str) -> SecurityCheckResult:
        """
        Enhanced async scan with advanced features.
        
        Args:
            url: Target URL
        
        Returns:
            SecurityCheckResult with additional enrichment
        """
        # Run base scan
        result = await super().scan_async(url)
        
        # Enrich with advanced features
        await self._enrich_with_advanced_features(url, result)
        
        return result
    
    async def _enrich_with_advanced_features(
        self,
        url: str,
        result: SecurityCheckResult,
    ):
        """Enrich scan results with advanced features."""
        
        # 1. Threat Intelligence Check
        if self.threat_intel and self.threat_intel.enabled:
            logger.info("🔍 Checking threat intelligence feeds...")
            try:
                threat_data = await self.threat_intel.check_url_reputation(url)
                result.metadata["threat_intelligence"] = threat_data
                
                if threat_data.get("malicious"):
                    logger.warning("⚠️  URL flagged as malicious by: %s", threat_data.get("sources"))
            except Exception as e:
                logger.error("Threat intelligence check failed: %s", e)
        
        # 2. Compliance Checking
        if self.compliance_checker:
            logger.info("📋 Running compliance checks...")
            try:
                for framework in self.advanced_config.compliance_frameworks:
                    compliance_result = self.compliance_checker.check_compliance(
                        [f.to_dict() for f in result.findings],
                        framework=framework,
                    )
                    result.metadata[f"compliance_{framework.lower()}"] = compliance_result
                    
                    if not compliance_result.get("compliant"):
                        logger.warning(
                            "⚠️  %s compliance violations: %d",
                            framework,
                            len(compliance_result.get("violations", []))
                        )
            except Exception as e:
                logger.error("Compliance checking failed: %s", e)
        
        # 3. SBOM Generation
        if self.sbom_generator:
            logger.info("📦 Generating SBOM...")
            try:
                sbom_path = f"reports/sbom/{result.timestamp.replace(':', '-')}_sbom.json"
                sbom = self.sbom_generator.generate(url, sbom_path)
                result.metadata["sbom"] = {
                    "path": sbom_path,
                    "components": len(sbom.get("components", [])),
                }
            except Exception as e:
                logger.error("SBOM generation failed: %s", e)
