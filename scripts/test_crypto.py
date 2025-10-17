# Create file: test_crypto.py
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

print("🔍 Testing cryptography and credential manager...\n")

# Test 1: Direct cryptography import
print("1. Testing cryptography import...")
try:
    from cryptography.fernet import Fernet
    print("   ✅ cryptography imports successfully")
except ImportError as e:
    print(f"   ❌ cryptography import failed: {e}")
    sys.exit(1)

# Test 2: Check credential_manager module
print("\n2. Testing credential_manager import...")
try:
    from modules.credential_manager import credential_manager, ENCRYPTION_AVAILABLE
    print(f"   ✅ CredentialManager imported")
    print(f"   ENCRYPTION_AVAILABLE = {ENCRYPTION_AVAILABLE}")
except ImportError as e:
    print(f"   ❌ Import failed: {e}")
    sys.exit(1)

# Test 3: Initialize CredentialManager
print("\n3. Testing CredentialManager initialization...")
try:
    mgr = CredentialManager()
    print(f"   ✅ CredentialManager initialized")
    print(f"   enable_vault = {mgr.enable_vault}")
    print(f"   enable_keyring = {mgr.enable_keyring}")
except Exception as e:
    print(f"   ❌ Initialization failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Try storing a credential
print("\n4. Testing credential storage...")
try:
    success = mgr.store_credential(
        alias="test_credential",
        username="test_user",
        password="test_pass",
        service="test"
    )
    print(f"   {'✅' if success else '❌'} Store credential: {success}")
    
    # Clean up
    if success:
        mgr.delete_credential("test_credential")
        print("   ✅ Cleanup successful")
except Exception as e:
    print(f"   ❌ Storage failed: {e}")
    import traceback
    traceback.print_exc()

print("\n✅ All tests passed!" if success else "\n❌ Tests failed")
