# verify_features.py
import sys
sys.path.insert(0, '.')

from modules.security_engine import SecurityEngine
from modules.security_checks import CHECK_REGISTRY

print("=" * 60)
print("SECURITY ENGINE FEATURE VERIFICATION")
print("=" * 60)

# Check 1: What's in CHECK_REGISTRY?
print("\n1. Registered Security Checks:")
for check_name in CHECK_REGISTRY.keys():
    print(f"   ✅ {check_name}")

print("\n2. Missing Advanced Features:")
missing = [
    "zero_day",
    "threat_intel", 
    "compliance",
    "baseline",
    "trending",
    "sbom"
]
for feature in missing:
    if feature not in CHECK_REGISTRY:
        print(f"   ❌ {feature} - NOT FOUND")

# Check 2: What does SecurityEngine have?
engine = SecurityEngine()
print("\n3. SecurityEngine Attributes:")
if hasattr(engine, 'ai_analyzer'):
    print(f"   ✅ ai_analyzer - EXISTS")
if hasattr(engine, 'cve_lookup'):
    print(f"   ✅ cve_lookup - EXISTS")
    
missing_attrs = [
    'threat_intel',
    'compliance',
    'baseline_profiler',
    'trending_analyzer',
    'sbom_generator'
]
for attr in missing_attrs:
    if not hasattr(engine, attr):
        print(f"   ❌ {attr} - MISSING")

print("\n" + "=" * 60)
print("CONCLUSION: Only AI analysis and CVE lookup are implemented")
print("=" * 60)
