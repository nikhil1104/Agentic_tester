# scripts/test_security_checks.py
"""
Test script for security checks with dynamic URL, report generation, and advanced features.

Usage:
    # Basic scan
    python -m scripts.test_security_checks https://google.com
    
    # Advanced scan with compliance
    python -m scripts.test_security_checks https://google.com --tier advanced --enable-compliance
    
    # Penetration testing (active SQL injection, XSS, CSRF tests)
    python -m scripts.test_security_checks https://google.com --tier penetration
    
    # Comprehensive testing (all modules)
    python -m scripts.test_security_checks https://google.com --tier comprehensive
    
    # Maximum security scan
    python -m scripts.test_security_checks https://google.com --tier all --enable-ai --enable-cve --enable-compliance
    
    # Interactive mode
    python -m scripts.test_security_checks
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# ✅ Load .env file
from dotenv import load_dotenv
load_dotenv()  # This loads .env from project root

# ✅ ALL REQUIRED IMPORTS
import asyncio
import logging
import re
from datetime import datetime
from modules.security_engine import SecurityEngine, SecurityConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


async def test_security_checks(target_url: str, config: SecurityConfig):
    """Test security checks on a target URL."""
    
    # Validate and normalize URL
    if not target_url.startswith(("http://", "https://")):
        target_url = f"https://{target_url}"
    
    print(f"\n🎯 Target URL: {target_url}")
    
    # Create engine with provided config
    engine = SecurityEngine(config=config)
    
    print(f"\n{'='*60}")
    print(f"🔒 Testing Security Checks on: {target_url}")
    print(f"{'='*60}\n")
    
    # Run async scan
    result = await engine.scan_async(target_url)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"📊 DETAILED SECURITY REPORT")
    print(f"{'='*60}")
    print(f"🎯 URL: {target_url}")
    print(f"📊 Status: {result.overall_status.value}")
    print(f"⏱️  Duration: {result.duration_s}s")
    print(f"🎯 Risk Score: {result.summary.get('risk_score', 0)}/100")
    
    print(f"\n{'='*60}")
    print(f"📋 ALL FINDINGS ({len(result.findings)} total)")
    print(f"{'='*60}\n")
    
    # Group findings by severity
    by_severity = {}
    for finding in result.findings:
        severity = finding.severity.value
        by_severity.setdefault(severity, []).append(finding)
    
    # ✅ Display ALL findings (not just first 5)
    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        findings = by_severity.get(severity, [])
        if findings:
            print(f"\n{'─'*60}")
            print(f"{severity} SEVERITY ({len(findings)} findings)")
            print(f"{'─'*60}")
            
            for i, finding in enumerate(findings, 1):
                status_icon = {
                    "PASS": "✅",
                    "FAIL": "❌",
                    "WARNING": "⚠️",
                    "ERROR": "🔴",
                    "SKIPPED": "⏭️",
                }.get(finding.status.value, "❓")
                
                print(f"\n{i}. {status_icon} {finding.message}")
                
                # Show additional details
                if finding.recommendation:
                    print(f"   💡 Recommendation: {finding.recommendation}")
                
                if finding.cwe_id or finding.owasp_category:
                    compliance = []
                    if finding.cwe_id:
                        compliance.append(finding.cwe_id)
                    if finding.owasp_category:
                        compliance.append(finding.owasp_category)
                    print(f"   🔖 Compliance: {' | '.join(compliance)}")
                
                if finding.references:
                    print(f"   📚 References:")
                    for ref in finding.references[:2]:  # Show first 2 refs
                        print(f"      • {ref}")
    
    print(f"\n{'='*60}\n")
    
    return result


def generate_detailed_text_report(result) -> str:
    """Generate detailed plain text report."""
    lines = []
    
    lines.append("=" * 80)
    lines.append("SECURITY SCAN REPORT - DETAILED")
    lines.append("=" * 80)
    lines.append(f"\nURL: {result.url}")
    lines.append(f"Scan Time: {result.timestamp}")
    lines.append(f"Duration: {result.duration_s}s")
    lines.append(f"Overall Status: {result.overall_status.value}")
    lines.append(f"Risk Score: {result.summary.get('risk_score', 0)}/100")
    
    lines.append(f"\n{'='*80}")
    lines.append("SUMMARY STATISTICS")
    lines.append("=" * 80)
    for key, value in result.summary.items():
        lines.append(f"{key.replace('_', ' ').title():.<40} {value}")
    
    # ✅ Add advanced features results
    if result.metadata:
        lines.append(f"\n{'='*80}")
        lines.append("ADVANCED FEATURES")
        lines.append("=" * 80)
        
        if "threat_intelligence" in result.metadata:
            threat = result.metadata["threat_intelligence"]
            lines.append(f"\nThreat Intelligence:")
            lines.append(f"  Malicious: {threat.get('malicious', False)}")
            if threat.get("sources"):
                lines.append(f"  Sources: {', '.join(threat.get('sources', []))}")
        
        for key, value in result.metadata.items():
            if key.startswith("compliance_"):
                framework = key.replace("compliance_", "").replace("_", "-").upper()
                lines.append(f"\n{framework} Compliance:")
                lines.append(f"  Compliant: {value.get('compliant', False)}")
                lines.append(f"  Violations: {len(value.get('violations', []))}")
        
        if "sbom" in result.metadata:
            sbom = result.metadata["sbom"]
            lines.append(f"\nSBOM:")
            lines.append(f"  Path: {sbom.get('path', 'N/A')}")
            lines.append(f"  Components: {sbom.get('components', 0)}")
    
    lines.append(f"\n{'='*80}")
    lines.append(f"DETAILED FINDINGS ({len(result.findings)} total)")
    lines.append("=" * 80)
    
    # Group by severity
    by_severity = {}
    for finding in result.findings:
        severity = finding.severity.value
        by_severity.setdefault(severity, []).append(finding)
    
    # Print all findings
    for severity in ["CRITICAL", "HIGH", "MEDIUM", "LOW", "INFO"]:
        findings = by_severity.get(severity, [])
        if not findings:
            continue
        
        lines.append(f"\n{'-'*80}")
        lines.append(f"{severity} SEVERITY ({len(findings)} findings)")
        lines.append("-" * 80)
        
        for i, finding in enumerate(findings, 1):
            lines.append(f"\n[{i}] {finding.check_name}")
            lines.append(f"    Status: {finding.status.value}")
            lines.append(f"    Message: {finding.message}")
            
            if finding.recommendation:
                lines.append(f"    Recommendation: {finding.recommendation}")
            
            if finding.cwe_id:
                lines.append(f"    CWE: {finding.cwe_id}")
            
            if finding.owasp_category:
                lines.append(f"    OWASP: {finding.owasp_category}")
            
            if finding.confidence < 1.0:
                lines.append(f"    Confidence: {finding.confidence:.0%}")
            
            if finding.details:
                lines.append(f"    Details: {finding.details}")
            
            if finding.references:
                lines.append(f"    References:")
                for ref in finding.references:
                    lines.append(f"      - {ref}")
    
    lines.append(f"\n{'='*80}")
    lines.append("END OF REPORT")
    lines.append("=" * 80)
    
    return "\n".join(lines)


def save_reports(result, target_url: str):
    """Save security reports to multiple formats."""
    
    # Create reports directory
    reports_dir = Path("reports/security")
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate filename slug from URL
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", target_url).strip("_")[:60]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"{timestamp}_{slug}"
    
    saved_files = {}
    
    try:
        # 1. Save JSON (full details)
        json_path = reports_dir / f"{base_filename}.json"
        json_path.write_text(result.to_json(), encoding="utf-8")
        saved_files["json"] = str(json_path)
        print(f"📄 JSON Report: {json_path}")
    except Exception as e:
        print(f"⚠️  Failed to save JSON: {e}")
    
    try:
        # 2. Save Markdown (readable format)
        md_path = reports_dir / f"{base_filename}.md"
        md_path.write_text(result.to_markdown(), encoding="utf-8")
        saved_files["markdown"] = str(md_path)
        print(f"📄 Markdown Report: {md_path}")
    except Exception as e:
        print(f"⚠️  Failed to save Markdown: {e}")
    
    try:
        # 3. Save HTML (beautiful report)
        engine = SecurityEngine()
        html_content = engine._generate_html_report(result)
        html_path = reports_dir / f"{base_filename}.html"
        html_path.write_text(html_content, encoding="utf-8")
        saved_files["html"] = str(html_path)
        print(f"📄 HTML Report: {html_path}")
    except Exception as e:
        print(f"⚠️  Failed to save HTML: {e}")
    
    try:
        # 4. Save detailed text report
        txt_path = reports_dir / f"{base_filename}.txt"
        txt_content = generate_detailed_text_report(result)
        txt_path.write_text(txt_content, encoding="utf-8")
        saved_files["text"] = str(txt_path)
        print(f"📄 Text Report: {txt_path}")
    except Exception as e:
        print(f"⚠️  Failed to save Text: {e}")
    
    return saved_files


if __name__ == "__main__":
    # ✅ Enhanced CLI with tier and feature flags
    parser = argparse.ArgumentParser(
        description="Security testing tool with advanced features",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic scan
  python -m scripts.test_security_checks https://google.com
  
  # Advanced scan with compliance
  python -m scripts.test_security_checks https://google.com --tier advanced --enable-compliance
  
  # Penetration testing (active SQL injection, XSS, CSRF tests)
  python -m scripts.test_security_checks https://google.com --tier penetration
  
  # Comprehensive testing (all modules)
  python -m scripts.test_security_checks https://google.com --tier comprehensive
  
  # Maximum security scan with all features
  python -m scripts.test_security_checks https://google.com --tier all --enable-ai --enable-cve --enable-compliance --enable-threat-intel
  
  # Custom compliance frameworks
  python -m scripts.test_security_checks https://google.com --enable-compliance --compliance-frameworks PCI-DSS HIPAA SOC2
        """
    )
    
    # URL argument
    parser.add_argument(
        "url",
        nargs="?",
        help="URL to test (e.g., https://google.com)"
    )
    
    # ✅ FIXED: Single --tier definition (removed duplicate)
    parser.add_argument(
        "--tier",
        choices=["basic", "standard", "advanced", "penetration", "comprehensive", "all"],
        default="standard",
        help=(
            "Security check tier: "
            "basic (passive), standard (default), advanced (zero-day), "
            "penetration (active SQL/XSS/CSRF), comprehensive (full audit), all (everything)"
        )
    )
    
    # Feature flags
    parser.add_argument(
        "--enable-ai",
        action="store_true",
        help="Enable AI vulnerability analysis (requires OPENAI_API_KEY)"
    )
    parser.add_argument(
        "--enable-cve",
        action="store_true",
        help="Enable CVE database lookup"
    )
    parser.add_argument(
        "--enable-compliance",
        action="store_true",
        help="Enable compliance checking (PCI-DSS, HIPAA, SOC2)"
    )
    parser.add_argument(
        "--enable-threat-intel",
        action="store_true",
        help="Enable threat intelligence feeds (requires API keys)"
    )
    parser.add_argument(
        "--enable-sbom",
        action="store_true",
        help="Enable SBOM generation (requires syft/trivy)"
    )
    parser.add_argument(
        "--compliance-frameworks",
        nargs="+",
        default=["PCI-DSS"],
        help="Compliance frameworks to check (default: PCI-DSS)"
    )
    parser.add_argument(
        "--no-reports",
        action="store_true",
        help="Skip saving reports to files (show console output only)"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Get URL
    if args.url:
        target_url = args.url
    else:
        print("\n" + "="*60)
        print("🔒 SECURITY TESTING TOOL")
        print("="*60)
        print("\nExamples:")
        print("  • https://google.com")
        print("  • https://github.com")
        print("  • authentication.liveperson.net")
        print()
        
        target_url = input("🌐 Enter URL to test: ").strip()
        
        if not target_url:
            print("❌ No URL provided")
            parser.print_help()
            sys.exit(1)
    
    # Validate URL
    if not target_url.startswith(("http://", "https://")):
        target_url = f"https://{target_url}"
    
    print(f"\n🚀 Starting security check tests")
    print(f"   URL: {target_url}")
    print(f"   Tier: {args.tier}")
    print()
    
    try:
        # ✅ Create config based on tier and flags
        config = SecurityConfig.from_tier(
            tier=args.tier,
            enable_ai=args.enable_ai,
            enable_cve_lookup=args.enable_cve,
            enable_compliance_checking=args.enable_compliance,
            enable_threat_intelligence=args.enable_threat_intel,
            enable_sbom_generation=args.enable_sbom,
            compliance_frameworks=args.compliance_frameworks,
            enable_cache=False,
        )
        
        # Show enabled features
        print("🔧 Enabled features:")
        print(f"   • Checks: {', '.join(config.enabled_checks)}")
        if args.enable_ai:
            print(f"   • AI Analysis: ✅")
        if args.enable_cve:
            print(f"   • CVE Lookup: ✅")
        if args.enable_compliance:
            print(f"   • Compliance: ✅ ({', '.join(args.compliance_frameworks)})")
        if args.enable_threat_intel:
            print(f"   • Threat Intelligence: ✅")
        if args.enable_sbom:
            print(f"   • SBOM Generation: ✅")
        print()
        
        # Run scan with detailed output
        result = asyncio.run(test_security_checks(target_url, config))
        
        # Save reports (unless disabled)
        if not args.no_reports:
            print(f"\n{'='*60}")
            print("💾 SAVING REPORTS")
            print(f"{'='*60}\n")
            
            saved_files = save_reports(result, target_url)
        else:
            saved_files = {}
        
        # Summary
        print("\n" + "="*60)
        print("✅ ALL TESTS COMPLETED")
        print("="*60)
        print(f"📊 Overall Status: {result.overall_status.value}")
        print(f"⏱️  Duration: {result.duration_s}s")
        print(f"🎯 Risk Score: {result.summary.get('risk_score', 0)}/100")
        
        # ✅ Show advanced features results
        if "threat_intelligence" in result.metadata:
            threat_data = result.metadata["threat_intelligence"]
            print(f"\n🔍 Threat Intelligence:")
            if threat_data.get("malicious"):
                print(f"   ⚠️  MALICIOUS - Flagged by: {', '.join(threat_data.get('sources', []))}")
            else:
                print(f"   ✅ Clean - No threats detected")
        
        for framework in args.compliance_frameworks:
            key = f"compliance_{framework.lower().replace('-', '_')}"
            if key in result.metadata:
                comp = result.metadata[key]
                print(f"\n📋 {framework} Compliance:")
                if comp.get("compliant"):
                    print(f"   ✅ Compliant")
                else:
                    violations = len(comp.get("violations", []))
                    print(f"   ❌ Non-compliant - {violations} violation(s)")
                    if violations > 0:
                        print(f"   📋 Sample violations:")
                        for v in comp.get("violations", [])[:3]:
                            print(f"      • {v.get('requirement')}: {v.get('description', '')[:60]}...")
        
        if "sbom" in result.metadata:
            sbom_data = result.metadata["sbom"]
            print(f"\n📦 SBOM Generated:")
            print(f"   Path: {sbom_data.get('path')}")
            print(f"   Components: {sbom_data.get('components', 0)}")
            print(f"   Format: {sbom_data.get('format', 'unknown')}")
        
        if saved_files:
            print(f"\n📁 Reports saved to:")
            for format_type, filepath in saved_files.items():
                print(f"   • {format_type.upper()}: {filepath}")
        
        print("="*60 + "\n")
        
        # Exit with appropriate code
        sys.exit(0 if result.overall_status.value == "PASS" else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
