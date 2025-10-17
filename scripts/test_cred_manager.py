#!/usr/bin/env python3
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("🔍 Checking credential manager dependencies...\n")

# Check cryptography
try:
    from cryptography.fernet import Fernet
    print("✅ cryptography installed - Vault features enabled")
except ImportError:
    print("⚠️  cryptography NOT installed - Vault features disabled")
    print("   Install: pip install cryptography\n")

# Check keyring
try:
    import keyring
    print("✅ keyring installed - OS keyring support enabled")
except ImportError:
    print("⚠️  keyring NOT installed - OS keyring disabled")
    print("   Install: pip install keyring\n")

# Test credential manager
try:
    from modules.credential_manager import CredentialManager
    manager = CredentialManager()
    print(f"\n✅ CredentialManager initialized")
    print(f"   Vault enabled: {manager.enable_vault}")
    print(f"   Keyring enabled: {manager.enable_keyring}")
except Exception as e:
    print(f"\n❌ Error: {e}")
