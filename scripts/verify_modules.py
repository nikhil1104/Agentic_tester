# scripts/verify_modules.py
"""
Verify all security modules are properly installed and registered.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from modules.security_checks import (
    CHECK_REGISTRY,
    CHECK_TIERS,
    get_all_check_names,
    validate_check_names,
)


def main():
    """Verify module registration."""
    print("\n" + "=" * 60)
    print("🔍 SECURITY MODULE VERIFICATION")
    print("=" * 60 + "\n")
    
    # 1. List all registered modules
    all_checks = get_all_check_names()
    print(f"📋 Total Registered Modules: {len(all_checks)}\n")
    
    for i, check_name in enumerate(all_checks, 1):
        check_class = CHECK_REGISTRY[check_name]
        print(f"  {i:2d}. ✅ {check_name:20s} → {check_class.__name__}")
    
    # 2. Show tiers
    print("\n" + "=" * 60)
    print("🎯 CHECK TIERS")
    print("=" * 60 + "\n")
    
    for tier_name, checks in CHECK_TIERS.items():
        print(f"{tier_name.upper():15s} ({len(checks):2d} checks):")
        print(f"  {', '.join(checks)}\n")
    
    # 3. Validate new modules
    print("=" * 60)
    print("🆕 NEW MODULES VERIFICATION")
    print("=" * 60 + "\n")
    
    new_modules = ["csrf", "api_security", "auth_testing", "file_upload"]
    valid, invalid = validate_check_names(new_modules)
    
    for module in new_modules:
        if module in valid:
            print(f"  ✅ {module:20s} REGISTERED")
        else:
            print(f"  ❌ {module:20s} NOT FOUND")
    
    # 4. Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    expected_total = 12  # Update based on your modules
    actual_total = len(all_checks)
    
    print(f"\n  Expected modules: {expected_total}")
    print(f"  Registered modules: {actual_total}")
    
    if actual_total == expected_total:
        print("\n  ✅ All modules registered successfully!\n")
        return 0
    else:
        print(f"\n  ⚠️  Missing {expected_total - actual_total} module(s)\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
