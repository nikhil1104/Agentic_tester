# scripts/test_advanced_security.py
"""
Test script for advanced security features.
Usage:
    python -m scripts.test_advanced_security https://google.com --tier advanced
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import asyncio
import logging
from modules.security_engine_advanced import AdvancedSecurityEngine
from modules.security_config_advanced import AdvancedSecurityConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def test_advanced_features(target_url: str, tier: str = "advanced"):
    """Test advanced security features."""
    
    print(f"\n{'='*60}")
    print(f"🔒 ADVANCED SECURITY TESTING")
    print(f"{'='*60}")
    print(f"🎯 URL: {target_url}")
    print(f"🏆 Tier: {tier}")
    print(f"{'='*60}\n")
    
    # Create advanced config
    config = AdvancedSecurityConfig.from_tier(
        tier=tier,
        enable_ai=False,  # Set to True if you have OpenAI API key
        enable_cve_lookup=False,  # Set to True for CVE checks
        enable_zero_day_detection=True,
        enable_threat_intelligence=False,  # Requires API keys
        enable_compliance_checking=True,
        enable_sbom_generation=False,  # Requires external tools
        compliance_frameworks=["PCI-DSS", "HIPAA"],
    )
    
    # Create advanced engine
    engine = AdvancedSecurityEngine(config=config)
    
    # Run scan
    print("🔍 Running advanced security scan...\n")
    result = await engine.scan_async(target_url)
    
    # Display results
    print(f"\n{'='*60}")
    print(f"📊 RESULTS")
    print(f"{'='*60}")
    print(f"Status: {result.overall_status.value}")
    print(f"Duration: {result.duration_s}s")
    print(f"Risk Score: {result.summary.get('risk_score', 0)}/100")
    print(f"Total Findings: {len(result.findings)}")
    
    # Show advanced features results
    if "threat_intelligence" in result.metadata:
        print(f"\n🔍 Threat Intelligence:")
        threat_data = result.metadata["threat_intelligence"]
        if threat_data.get("malicious"):
            print(f"   ⚠️  MALICIOUS - Flagged by: {threat_data.get('sources')}")
        else:
            print(f"   ✅ Clean - No threats detected")
    
    if "compliance_pci-dss" in result.metadata:
        print(f"\n📋 PCI-DSS Compliance:")
        comp = result.metadata["compliance_pci-dss"]
        if comp.get("compliant"):
            print(f"   ✅ Compliant")
        else:
            print(f"   ❌ Non-compliant - {len(comp.get('violations', []))} violations")
    
    if "compliance_hipaa" in result.metadata:
        print(f"\n📋 HIPAA Compliance:")
        comp = result.metadata["compliance_hipaa"]
        if comp.get("compliant"):
            print(f"   ✅ Compliant")
        else:
            print(f"   ❌ Non-compliant - {len(comp.get('violations', []))} violations")
    
    print(f"\n{'='*60}\n")
    
    return result


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Advanced security testing")
    parser.add_argument("url", help="URL to test")
    parser.add_argument(
        "--tier",
        choices=["basic", "standard", "advanced", "all"],
        default="advanced",
        help="Security check tier"
    )
    
    args = parser.parse_args()
    
    asyncio.run(test_advanced_features(args.url, args.tier))
