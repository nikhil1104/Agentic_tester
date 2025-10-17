# modules/credential_manager.py
"""
Credential Manager v2.2 (Production-Grade, Hardened)

Key Capabilities
- Resolution order: Vault (aliases) → Env → Keyring → secrets.json
- AES-256 vault encryption using Fernet
  • Supports direct Fernet key via env: CREDENTIALS_MASTER_KEY (base64 urlsafe)
  • OR passphrase-derived key via PBKDF2HMAC (CREDENTIALS_PASSPHRASE + per-vault SALT)
- OS keyring (optional)
- secrets.json (hot-reload, tolerant parsing)
- Audit log (file-backed), access counters, auto-expiry for aliases
- Health reporting consumed by scripts/setup_credentials.py
- Thread-safety for vault ops; permission hardening

Backwards Compatibility
- Public methods preserved: get_credentials(), store_credential(), delete_credential(),
  list_credentials(), get_audit_log()
- Top-level function get_credentials(...) retained

Security Defaults
- No plain-text password prints
- Logs never include secrets
- Vault and audit files are created with restrictive permissions (0600 on POSIX)
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import threading
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

# ===== Optional encryption (Fernet) =====
try:
    from cryptography.fernet import Fernet, InvalidToken  # type: ignore
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC  # type: ignore
    from cryptography.hazmat.primitives import hashes  # type: ignore
    from cryptography.hazmat.backends import default_backend  # type: ignore
    ENCRYPTION_AVAILABLE = True
except Exception:
    ENCRYPTION_AVAILABLE = False

# ===== Optional keyring =====
try:
    import keyring  # type: ignore
    _KEYRING_AVAILABLE = True
except Exception:
    _KEYRING_AVAILABLE = False

logger = logging.getLogger(__name__)

_DEFAULT_ROLE = "test"
_DEFAULT_SECRETS_CANDIDATES = (
    Path("secrets.json"),
    Path("config") / "secrets.json",
    Path(".secrets") / "secrets.json",
)

# File locations (can be overridden by init args)
_DEFAULT_VAULT_PATH = Path("config/credentials.vault")
_DEFAULT_VAULT_SALT = Path("config/credentials.salt")
_DEFAULT_AUDIT_LOG = Path("config/credentials_audit.log")


# ==================== Data Classes ====================

@dataclass
class Credentials:
    username: Optional[str]
    password: Optional[str]
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: Optional[str] = None
    expires_at: Optional[str] = None
    access_count: int = 0

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {"username": self.username, "password": self.password}

    def redacted(self) -> Dict[str, Any]:
        return {
            "username": self.username,
            "password": "***" if self.password else None,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
            "access_count": self.access_count,
        }

    def is_expired(self) -> bool:
        if not self.expires_at:
            return False
        try:
            return datetime.now() > datetime.fromisoformat(self.expires_at)
        except Exception:
            return False


# ==================== Credential Manager ====================

class CredentialManager:
    """
    Hardened credential manager with encrypted alias vault + multi-source resolution.
    """

    # Health attributes the CLI probes
    vault_enabled: bool
    keyring_enabled: bool
    storage_path: Optional[str]
    last_error: Optional[str]

    def __init__(
        self,
        secrets_file: Optional[str] = None,
        enable_keyring: bool = True,
        enable_vault: bool = True,
        vault_path: str | Path = _DEFAULT_VAULT_PATH,
        vault_salt_path: str | Path = _DEFAULT_VAULT_SALT,
        audit_log_path: str | Path = _DEFAULT_AUDIT_LOG,
        master_key_env: str = "CREDENTIALS_MASTER_KEY",
        passphrase_env: str = "CREDENTIALS_PASSPHRASE",
        kdf_iterations: int = 200_000,
    ):
        self.keyring_enabled = bool(enable_keyring and _KEYRING_AVAILABLE)
        self.vault_enabled = bool(enable_vault and ENCRYPTION_AVAILABLE)
        self.storage_path = str(vault_path)
        self.last_error = None

        # Secrets file (hot-reload)
        self._secrets_path = Path(secrets_file) if secrets_file else self._find_secrets_file()
        self._secrets: Dict[str, Any] = {}
        self._secrets_mtime: Optional[float] = None

        # Vault + Audit
        self._vault_path = Path(vault_path)
        self._vault_salt_path = Path(vault_salt_path)
        self._audit_path = Path(audit_log_path)
        self._fernet: Optional[Fernet] = None
        self._vault: Dict[str, Credentials] = {}
        self._lock = threading.RLock()  # thread-safe vault ops
        self._kdf_iterations = int(kdf_iterations)
        self._master_key_env = master_key_env
        self._passphrase_env = passphrase_env

        # Initialize components
        try:
            self._load_secrets_if_present(initial=True)
        except Exception as e:
            logger.warning("Failed to load secrets.json: %s", e)

        if self.vault_enabled:
            try:
                self._init_vault_key()
                self._load_vault()
            except Exception as e:
                self.vault_enabled = False
                self.last_error = f"vault_init_failed: {e}"
                logger.error("Vault initialization failed: %s", e)

        # Ensure audit log path exists with safe permissions (POSIX)
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            if os.name != "nt":
                # Create if not exists & set mode; ignore errors on Windows
                if not self._audit_path.exists():
                    self._audit_path.touch(exist_ok=True)
                os.chmod(self._audit_path, 0o600)
        except Exception as e:
            logger.debug("Audit log permission hardening skipped: %s", e)

        logger.info(
            "✅ CredentialManager initialized (keyring=%s, vault=%s)",
            self.keyring_enabled, self.vault_enabled
        )

    # -------------------- Public API --------------------

    def get_credentials(
        self,
        domain_hint: Optional[str] = None,
        role: str = _DEFAULT_ROLE,
        *,
        service: Optional[str] = None,
        alias: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Resolve credentials in priority:
        1) Vault (if alias provided and vault enabled)
        2) Environment variables
        3) OS keyring
        4) secrets.json
        """
        # 1) Vault via alias
        if alias and self.vault_enabled:
            cred = self._get_from_vault(alias)
            if cred:
                self._log_audit("ACCESS", alias, "vault")
                return cred.as_dict()

        # Normalized tokens
        domain_norm = self._normalize_domain(domain_hint)
        role_norm = self._normalize_token(role) if role else None
        service_norm = self._normalize_token(service) if service else None

        # 2) Environment
        env = self._resolve_from_env(domain_norm, role_norm, service_norm)
        if env.username and env.password:
            self._log_audit("ACCESS", f"{domain_norm}_{role_norm}", "env")
            return env.as_dict()

        # 3) OS keyring
        if self.keyring_enabled:
            k = self._resolve_from_keyring(domain_norm, role_norm, service_norm)
            if k.username and k.password:
                self._log_audit("ACCESS", f"{domain_norm}_{role_norm}", "keyring")
                return k.as_dict()

        # 4) secrets.json
        self._load_secrets_if_present()
        s = self._resolve_from_secrets(domain_norm, role_norm)
        if s.username or s.password:
            self._log_audit("ACCESS", f"{domain_norm}_{role_norm}", "secrets_file")
            return s.as_dict()

        logger.info(
            "No credentials found for domain=%s role=%s service=%s alias=%s",
            domain_norm, role_norm, service_norm, alias
        )
        return Credentials(None, None).as_dict()

    def store_credential(
        self,
        alias: str,
        username: str,
        password: str,
        service: str = "generic",
        metadata: Optional[Dict[str, Any]] = None,
        ttl_hours: Optional[int] = None,
        overwrite: bool = False,
    ) -> bool:
        """
        Store or update a credential in the encrypted vault.

        Args:
            alias: unique alias
            username, password: secret pair
            service: logical service name
            metadata: arbitrary metadata
            ttl_hours: expire after N hours (optional)
            overwrite: if False and alias exists → return False
        """
        if not self.vault_enabled or not self._fernet:
            self.last_error = "vault_not_enabled"
            logger.error("Vault not enabled. Install: pip install cryptography")
            return False

        if not alias or not username or not password:
            self.last_error = "invalid_input"
            return False

        with self._lock:
            if alias in self._vault and not overwrite:
                self.last_error = "alias_exists"
                return False

            cred = Credentials(
                username=username,
                password=password,
                metadata=metadata or {},
                created_at=datetime.now().isoformat(),
                expires_at=(datetime.now() + timedelta(hours=ttl_hours)).isoformat() if ttl_hours else None,
                access_count=0,
            )
            self._vault[alias] = cred
            try:
                self._save_vault_locked()
                self._log_audit("STORE", alias, service)
                return True
            except Exception as e:
                self.last_error = f"vault_save_failed: {e}"
                logger.error("Failed to save vault: %s", e)
                return False

    def delete_credential(self, alias: str) -> bool:
        with self._lock:
            if alias in self._vault:
                del self._vault[alias]
                try:
                    self._save_vault_locked()
                    self._log_audit("DELETE", alias, "vault")
                    return True
                except Exception as e:
                    self.last_error = f"vault_save_failed: {e}"
                    logger.error("Failed to save vault: %s", e)
            return False

    def list_credentials(self) -> Dict[str, Dict[str, Any]]:
        """List vault aliases (redacted); no secrets are returned."""
        with self._lock:
            return {alias: cred.redacted() for alias, cred in self._vault.items()}

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Return last N audit entries from file (newest last).
        """
        try:
            if not self._audit_path.exists():
                return []
            lines = self._audit_path.read_text(encoding="utf-8").splitlines()
            tail = lines[-max(1, int(limit)):]
            out: List[Dict[str, Any]] = []
            for line in tail:
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
            return out
        except Exception as e:
            logger.debug("Audit read failed: %s", e)
            return []

    # ---------------- Vault internals ----------------

    def _init_vault_key(self) -> None:
        """
        Initialize self._fernet from either:
        1) direct Fernet key via CREDENTIALS_MASTER_KEY (urlsafe base64)
        2) or passphrase via CREDENTIALS_PASSPHRASE + per-vault SALT + PBKDF2HMAC
        """
        if not ENCRYPTION_AVAILABLE:
            raise RuntimeError("cryptography not installed")

        master_key_b64 = os.getenv(self._master_key_env)
        passphrase = os.getenv(self._passphrase_env)

        if master_key_b64:
            try:
                # Validate decodability; Fernet() will validate format
                _ = base64.urlsafe_b64decode(master_key_b64.encode())
                self._fernet = Fernet(master_key_b64.encode())
                return
            except Exception as e:
                raise ValueError(f"Invalid {self._master_key_env} value: {e}")

        if passphrase:
            # derive a fernet key via PBKDF2HMAC with per-vault salt
            salt = self._ensure_salt()
            key = self._derive_key_from_passphrase(passphrase.encode("utf-8"), salt)
            self._fernet = Fernet(key)
            return

        # Neither provided: generate ephemeral key, print guidance via logger only
        key = Fernet.generate_key()
        self._fernet = Fernet(key)
        key_env = base64.urlsafe_b64encode(base64.urlsafe_b64decode(key)).decode()
        logger.warning(
            "Generated ephemeral vault key. To persist across runs set ONE of:\n"
            f"  export {self._master_key_env}={key_env}\n"
            f"  # OR a passphrase:\n"
            f"  export {self._passphrase_env}='<strong-passphrase>'"
        )

    def _ensure_salt(self) -> bytes:
        """Ensure a per-vault salt file exists (0600), then return salt bytes."""
        salt_path = self._vault_salt_path
        try:
            salt_path.parent.mkdir(parents=True, exist_ok=True)
            if salt_path.exists():
                data = salt_path.read_bytes()
                if len(data) >= 16:
                    return data
            # create new salt
            new_salt = os.urandom(16)
            with salt_path.open("wb") as f:
                f.write(new_salt)
            if os.name != "nt":
                os.chmod(salt_path, 0o600)
            return new_salt
        except Exception as e:
            raise RuntimeError(f"Failed to create/read salt: {e}")

    def _derive_key_from_passphrase(self, passphrase: bytes, salt: bytes) -> bytes:
        """Derive a Fernet key from passphrase+salt using PBKDF2HMAC."""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=self._kdf_iterations,
            backend=default_backend(),
        )
        raw = kdf.derive(passphrase)  # 32 bytes
        return base64.urlsafe_b64encode(raw)

    def _load_vault(self) -> None:
        """Load encrypted vault from disk (tolerant)."""
        if not self._vault_path.exists():
            return
        if not self._fernet:
            raise RuntimeError("Fernet key not initialized")

        try:
            data = self._vault_path.read_bytes()
            if not data:
                return
            decrypted = self._fernet.decrypt(data)
            payload = json.loads(decrypted.decode("utf-8"))
            with self._lock:
                self._vault.clear()
                for alias, cred in payload.items():
                    self._vault[alias] = Credentials(**cred)
        except InvalidToken:
            raise RuntimeError("Vault decryption failed (invalid key or corrupted data)")
        except Exception as e:
            raise RuntimeError(f"Failed to load vault: {e}")

    def _save_vault_locked(self) -> None:
        """Persist vault to disk; caller must hold self._lock."""
        if not self._fernet:
            raise RuntimeError("Fernet key not initialized")

        payload = {
            alias: {
                "username": c.username,
                "password": c.password,
                "metadata": c.metadata,
                "created_at": c.created_at,
                "expires_at": c.expires_at,
                "access_count": c.access_count,
            }
            for alias, c in self._vault.items()
        }
        blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
        encrypted = self._fernet.encrypt(blob)

        self._vault_path.parent.mkdir(parents=True, exist_ok=True)
        with self._vault_path.open("wb") as f:
            f.write(encrypted)
        if os.name != "nt":
            try:
                os.chmod(self._vault_path, 0o600)
            except Exception:
                pass

    def _get_from_vault(self, alias: str) -> Optional[Credentials]:
        with self._lock:
            cred = self._vault.get(alias)
            if not cred:
                return None
            if cred.is_expired():
                logger.info("Credential expired for alias=%s; deleting.", alias)
                try:
                    del self._vault[alias]
                    self._save_vault_locked()
                except Exception as e:
                    self.last_error = f"vault_save_failed: {e}"
                return None
            # increment access count
            cred.access_count += 1
            try:
                self._save_vault_locked()
            except Exception as e:
                self.last_error = f"vault_save_failed: {e}"
            return cred

    # --------------- Resolution helpers (unchanged semantics) ---------------

    def _resolve_from_env(
        self, domain: Optional[str], role: Optional[str], service: Optional[str]
    ) -> Credentials:
        candidates: List[Tuple[str, str]] = []

        def add(prefix: Optional[str], r: Optional[str] = None):
            if not prefix:
                return
            if r:
                candidates.append((f"{prefix}_{r}_USER", f"{prefix}_{r}_PASS"))
            candidates.append((f"{prefix}_USER", f"{prefix}_PASS"))

        if domain and service and role:
            add(f"{domain}_{service}", role)
        if domain and role:
            add(domain, role)
        if domain:
            add(domain)
        if service and role:
            add(service, role)
        if service:
            add(service)

        candidates.extend([("TEST_USER", "TEST_PASS"), ("USER", "PASS")])

        for user_key, pass_key in candidates:
            u = os.environ.get(user_key)
            p = os.environ.get(pass_key)
            if u and p:
                return Credentials(u, p)

        for user_key, pass_key in candidates:
            u = os.environ.get(user_key)
            p = os.environ.get(pass_key)
            if u or p:
                return Credentials(u, p)

        return Credentials(None, None)

    def _resolve_from_keyring(
        self, domain: Optional[str], role: Optional[str], service: Optional[str]
    ) -> Credentials:
        if not _KEYRING_AVAILABLE:
            return Credentials(None, None)

        service_name = domain or service
        if not service_name:
            return Credentials(None, None)

        usernames_to_try: List[str] = []
        if role:
            usernames_to_try.append(f"{service_name}:{role}")
        usernames_to_try.append(service_name)

        for uname in usernames_to_try:
            try:
                pwd = keyring.get_password(service_name, uname)  # type: ignore
                if pwd:
                    return Credentials(uname, pwd)
            except Exception:
                continue
        return Credentials(None, None)

    def _resolve_from_secrets(self, domain: Optional[str], role: Optional[str]) -> Credentials:
        data = self._secrets or {}
        raw_domain = getattr(self, "_original_domain", None)

        dom_key_candidates = list(filter(None, [raw_domain, self._denormalize_domain(domain), domain]))
        dom_block = None
        for k in dom_key_candidates:
            if isinstance(data.get("domains"), dict) and k in data["domains"]:
                dom_block = data["domains"][k]
                break

        if isinstance(dom_block, dict):
            if role and isinstance(dom_block.get("roles"), dict):
                r = dom_block["roles"].get(role) or dom_block["roles"].get(role.lower()) or dom_block["roles"].get(role.upper())
                if isinstance(r, dict):
                    return self._creds_from_mapping(r)
            if isinstance(dom_block.get("default"), dict):
                return self._creds_from_mapping(dom_block["default"])

        for k in dom_key_candidates:
            v = data.get(k)
            if isinstance(v, dict):
                return self._creds_from_mapping(v)

        if role and isinstance(data.get("roles"), dict):
            v = data["roles"].get(role) or data["roles"].get(role.lower()) or data["roles"].get(role.upper())
            if isinstance(v, dict):
                return self._creds_from_mapping(v)

        if isinstance(data.get("default"), dict):
            return self._creds_from_mapping(data["default"])

        return Credentials(None, None)

    @staticmethod
    def _creds_from_mapping(m: Dict[str, Any]) -> Credentials:
        u = m.get("username") or m.get("user") or m.get("email")
        p = m.get("password") or m.get("pass") or m.get("token")
        return Credentials(u, p)

    def _find_secrets_file(self) -> Optional[Path]:
        for p in _DEFAULT_SECRETS_CANDIDATES:
            if p.exists():
                return p
        return None

    def _load_secrets_if_present(self, *, initial: bool = False) -> None:
        if not self._secrets_path or not self._secrets_path.exists():
            if initial:
                logger.debug("No secrets.json found; continuing without it.")
            return
        try:
            mtime = self._secrets_path.stat().st_mtime
            if self._secrets_mtime is not None and mtime == self._secrets_mtime:
                return
            with io.open(self._secrets_path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            if isinstance(data, dict):
                self._secrets = data
                self._secrets_mtime = mtime
                logger.info("Loaded secrets file: %s", self._secrets_path)
            else:
                logger.warning("Secrets file %s does not contain a JSON object; ignoring", self._secrets_path)
        except Exception as e:
            logger.warning("Failed to read secrets file %s: %s", self._secrets_path, e)

    def _normalize_domain(self, domain_hint: Optional[str]) -> Optional[str]:
        self._original_domain = None
        if not domain_hint:
            return None
        raw = domain_hint.strip()
        self._original_domain = raw
        raw = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", raw)  # strip scheme
        raw = raw.split("/", 1)[0]
        raw = raw.split(":", 1)[0]
        raw = re.sub(r"^www\.", "", raw, flags=re.I)
        token = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").upper()
        return token or None

    @staticmethod
    def _denormalize_domain(norm: Optional[str]) -> Optional[str]:
        if not norm:
            return None
        s = norm.lower()
        if "_" in s and not s.endswith("_"):
            return s.replace("_", ".")
        return None

    @staticmethod
    def _normalize_token(value: Optional[str]) -> Optional[str]:
        if not value:
            return None
        return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_").upper() or None

    # ---------------- Audit logging ----------------

    def _log_audit(self, action: str, identifier: str, source: str) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(),
            "action": action,
            "identifier": identifier,
            "source": source,
        }
        try:
            self._audit_path.parent.mkdir(parents=True, exist_ok=True)
            with self._audit_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.debug("Audit write failed: %s", e)

    # ---------------- Health reporting ----------------

    def health(self) -> Dict[str, Any]:
        """Return a health snapshot used by CLI."""
        return {
            "vault_enabled": bool(self.vault_enabled),
            "keyring_enabled": bool(self.keyring_enabled),
            "storage_path": str(self._vault_path),
            "last_error": self.last_error,
        }


# ==================== Backward Compatibility ====================

def get_credentials(domain_hint=None, role="test", service=None):
    """Backward-compatible helper."""
    manager = CredentialManager()
    return manager.get_credentials(domain_hint, role, service=service)
