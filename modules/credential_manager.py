# modules/credential_manager.py
"""
Credential Manager (Phase 6 — Hardened & Extensible)
----------------------------------------------------
Resolves credentials from multiple sources with clear precedence:

1) Environment variables (most specific → least specific)
   - <DOMAIN>_<SERVICE>_<ROLE>_USER / _PASS
   - <DOMAIN>_<ROLE>_USER / _PASS
   - <DOMAIN>_USER / _PASS
   - <SERVICE>_<ROLE>_USER / _PASS
   - <SERVICE>_USER / _PASS
   - TEST_USER / TEST_PASS
   - USER / PASS    (back-compat; use TEST_* in new setups)

   Where <DOMAIN>, <SERVICE>, <ROLE> are uppercased and non-alnum replaced by '_'.
   DOMAIN examples:
     "https://app.example.com:8443" -> "APP_EXAMPLE_COM"
     "example.com"                   -> "EXAMPLE_COM"

2) OS keyring (optional; if 'keyring' is installed)
   - Looks up entries by keys similar to env names; service name can be provided.

3) Secrets file (JSON) — searched in:
   - ./secrets.json
   - ./config/secrets.json
   - ./.secrets/secrets.json
   Supported shapes (all optional):
     {
       "example.com": {"username": "...", "password": "..."},
       "roles": {"test": {"username": "...", "password": "..."}},
       "domains": {
         "example.com": {
            "default": {"username": "...", "password": "..."},
            "roles": {"admin": {"username": "...", "password": "..."}}
         }
       },
       "default": {"username": "...", "password": "..."}
     }

Return shape (always):
  {"username": Optional[str], "password": Optional[str]}

Notes:
- Never logs secrets. Redaction helper provided.
- Secrets file is hot-reloaded if mtime changes.
- Backward-compatible with prior get_credentials(domain_hint, role="test").
"""

from __future__ import annotations

import io
import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple, List

logger = logging.getLogger(__name__)

# ----------------------------
# Constants / Defaults
# ----------------------------
_DEFAULT_ROLE = "test"
_DEFAULT_SECRETS_CANDIDATES = (
    Path("secrets.json"),
    Path("config") / "secrets.json",
    Path(".secrets") / "secrets.json",
)

# Optional keyring support (no hard dependency)
try:
    import keyring  # type: ignore
    _KEYRING_AVAILABLE = True
except Exception:
    _KEYRING_AVAILABLE = False


@dataclass
class Credentials:
    username: Optional[str]
    password: Optional[str]

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {"username": self.username, "password": self.password}

    def redacted(self) -> Dict[str, Optional[str]]:
        return {"username": self.username, "password": "***" if self.password else None}


class CredentialManager:
    """
    Resolve credentials from env, optional keyring, and secrets.json (in that order).

    Args:
      secrets_file: Optional explicit path to secrets.json; if None, common candidates are searched.
      enable_keyring: If True, attempt OS keyring lookups when env is missing.
    """

    def __init__(self, secrets_file: Optional[str] = None, enable_keyring: bool = True) -> None:
        self.enable_keyring = enable_keyring and _KEYRING_AVAILABLE
        self._secrets_path = Path(secrets_file) if secrets_file else self._find_secrets_file()
        self._secrets: Dict = {}
        self._secrets_mtime: Optional[float] = None
        self._load_secrets_if_present(initial=True)

    # ------------------------------------------------------------------
    # Public API (backward compatible)
    # ------------------------------------------------------------------
    def get_credentials(
        self,
        domain_hint: Optional[str] = None,
        role: str = _DEFAULT_ROLE,
        *,
        service: Optional[str] = None,
    ) -> Dict[str, Optional[str]]:
        """
        Resolve credentials using the configured precedence. Returns dict.

        domain_hint: e.g., "https://app.example.com" or "example.com"
        role:        logical role ("test", "admin", etc.)
        service:     optional service namespace (e.g., "BACKEND", "GATEWAY")

        Example env combos checked (highest priority first):
          EXAMPLE_COM_BACKEND_ADMIN_USER/PASS
          EXAMPLE_COM_ADMIN_USER/PASS
          EXAMPLE_COM_USER/PASS
          BACKEND_ADMIN_USER/PASS
          BACKEND_USER/PASS
          TEST_USER/TEST_PASS
          USER/PASS
        """
        domain_norm = self._normalize_domain(domain_hint)
        role_norm = self._normalize_token(role) if role else None
        service_norm = self._normalize_token(service) if service else None

        # 1) Environment variables
        creds = self._resolve_from_env(domain_norm, role_norm, service_norm)
        if creds.username and creds.password:
            logger.debug("Resolved credentials from ENV for domain=%s role=%s service=%s", domain_norm, role_norm, service_norm)
            return creds.as_dict()

        # 2) Keyring (optional)
        if self.enable_keyring:
            kcreds = self._resolve_from_keyring(domain_norm, role_norm, service_norm)
            if kcreds.username and kcreds.password:
                logger.debug("Resolved credentials from KEYRING for domain=%s role=%s service=%s", domain_norm, role_norm, service_norm)
                return kcreds.as_dict()

        # 3) Secrets file
        self._load_secrets_if_present()
        screds = self._resolve_from_secrets(domain_norm, role_norm)
        if screds.username or screds.password:
            logger.debug("Resolved credentials from SECRETS for domain=%s role=%s", domain_norm, role_norm)
            return screds.as_dict()

        # Nothing found: return empty placeholders (never raise in production flows)
        logger.info("No credentials found for domain=%s role=%s service=%s", domain_norm, role_norm, service_norm)
        return Credentials(None, None).as_dict()

    # ------------------------------------------------------------------
    # Environment resolution
    # ------------------------------------------------------------------
    def _resolve_from_env(
        self,
        domain: Optional[str],
        role: Optional[str],
        service: Optional[str],
    ) -> Credentials:
        """
        Try increasingly less specific env var names.
        Returns the first pair where both USER & PASS are present.
        """
        candidates: List[Tuple[str, str]] = []

        def add(prefix: Optional[str], r: Optional[str] = None):
            if not prefix:
                return
            if r:
                candidates.append((f"{prefix}_{r}_USER", f"{prefix}_{r}_PASS"))
            candidates.append((f"{prefix}_USER", f"{prefix}_PASS"))

        # Most specific: DOMAIN_SERVICE_ROLE, DOMAIN_ROLE, DOMAIN
        if domain and service and role:
            add(f"{domain}_{service}", role)
        if domain and role:
            add(domain, role)
        if domain:
            add(domain)

        # SERVICE_ROLE, SERVICE
        if service and role:
            add(service, role)
        if service:
            add(service)

        # Legacy / global fallbacks
        candidates.extend([
            ("TEST_USER", "TEST_PASS"),
            ("USER", "PASS"),  # backward-compat; prefer TEST_USER/PASS going forward
        ])

        for user_key, pass_key in candidates:
            u = os.environ.get(user_key)
            p = os.environ.get(pass_key)
            if u and p:
                return Credentials(u, p)

        # Partial matches (username only) — allow if at least something is provided
        for user_key, pass_key in candidates:
            u = os.environ.get(user_key)
            p = os.environ.get(pass_key)
            if u or p:
                return Credentials(u, p)

        return Credentials(None, None)

    # ------------------------------------------------------------------
    # Keyring resolution (optional)
    # ------------------------------------------------------------------
    def _resolve_from_keyring(
        self,
        domain: Optional[str],
        role: Optional[str],
        service: Optional[str],
    ) -> Credentials:
        """
        Attempts OS keyring lookups using common keys:
          service_name = domain or service (prefer domain)
          username_key = f"{service_name}:{role}" or service_name
        """
        if not _KEYRING_AVAILABLE:
            return Credentials(None, None)

        service_name = domain or service
        if not service_name:
            return Credentials(None, None)

        # Try role-scoped username first
        usernames_to_try = []
        if role:
            usernames_to_try.append(f"{service_name}:{role}")
        usernames_to_try.append(service_name)

        for uname in usernames_to_try:
            try:
                pwd = keyring.get_password(service_name, uname)  # type: ignore
                if pwd:
                    return Credentials(uname, pwd)
            except Exception:
                # Keyring backend may be unavailable in containers; ignore quietly.
                continue

        return Credentials(None, None)

    # ------------------------------------------------------------------
    # Secrets file resolution
    # ------------------------------------------------------------------
    def _resolve_from_secrets(self, domain: Optional[str], role: Optional[str]) -> Credentials:
        """
        Supports flexible shapes (see module docstring).
        Resolution order:
          - top-level "domains" → domain → roles → <role>
          - top-level "domains" → domain → "default"
          - top-level by domain ("example.com": {...})
          - top-level "roles" → <role>
          - top-level "default"
        """
        data = self._secrets or {}
        raw_domain = self._original_domain or None

        # 1) domains block (normalized domain key OR raw domain key)
        dom_key_candidates = list(filter(None, [raw_domain, self._denormalize_domain(domain), domain]))
        dom_block = None
        for k in dom_key_candidates:
            if isinstance(data.get("domains"), dict) and k in data["domains"]:
                dom_block = data["domains"][k]
                break

        if isinstance(dom_block, dict):
            # roles → role
            if role and isinstance(dom_block.get("roles"), dict):
                r = dom_block["roles"].get(role) or dom_block["roles"].get(role.lower()) or dom_block["roles"].get(role.upper())
                if isinstance(r, dict):
                    return self._creds_from_mapping(r)

            # default under domain
            if isinstance(dom_block.get("default"), dict):
                return self._creds_from_mapping(dom_block["default"])

        # 2) top-level domain entry
        for k in dom_key_candidates:
            v = data.get(k)
            if isinstance(v, dict):
                return self._creds_from_mapping(v)

        # 3) top-level roles → role
        if role and isinstance(data.get("roles"), dict):
            v = data["roles"].get(role) or data["roles"].get(role.lower()) or data["roles"].get(role.upper())
            if isinstance(v, dict):
                return self._creds_from_mapping(v)

        # 4) top-level default
        if isinstance(data.get("default"), dict):
            return self._creds_from_mapping(data["default"])

        return Credentials(None, None)

    @staticmethod
    def _creds_from_mapping(m: Dict) -> Credentials:
        u = m.get("username") or m.get("user") or m.get("email")
        p = m.get("password") or m.get("pass") or m.get("token")
        return Credentials(u, p)

    # ------------------------------------------------------------------
    # Secrets file loading / hot reload
    # ------------------------------------------------------------------
    def _find_secrets_file(self) -> Optional[Path]:
        for p in _DEFAULT_SECRETS_CANDIDATES:
            if p.exists():
                return p
        return None

    def _load_secrets_if_present(self, *, initial: bool = False) -> None:
        """
        Loads secrets JSON if path configured and mtime changed.
        Accepts UTF-8 and UTF-8 with BOM.
        """
        if not self._secrets_path or not self._secrets_path.exists():
            if initial:
                logger.debug("No secrets.json found; continuing without it.")
            return

        try:
            mtime = self._secrets_path.stat().st_mtime
            if self._secrets_mtime is not None and mtime == self._secrets_mtime:
                return  # unchanged
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

    # ------------------------------------------------------------------
    # Normalization helpers
    # ------------------------------------------------------------------
    def _normalize_domain(self, domain_hint: Optional[str]) -> Optional[str]:
        """
        Convert a domain or URL into ENV-friendly token:
          "https://app.example.com:8443" -> "APP_EXAMPLE_COM"
          "example.com"                  -> "EXAMPLE_COM"
        Stores original for secrets-file lookups.
        """
        self._original_domain = None  # used in secrets resolution
        if not domain_hint:
            return None
        raw = domain_hint.strip()
        self._original_domain = raw

        # strip scheme
        raw = re.sub(r"^[a-zA-Z][a-zA-Z0-9+.-]*://", "", raw)
        # drop path/query/fragment
        raw = raw.split("/", 1)[0]
        # drop port
        raw = raw.split(":", 1)[0]
        # strip www.
        raw = re.sub(r"^www\.", "", raw, flags=re.I)

        token = re.sub(r"[^A-Za-z0-9]+", "_", raw).strip("_").upper()
        return token or None

    @staticmethod
    def _denormalize_domain(norm: Optional[str]) -> Optional[str]:
        # For secrets lookups we also try lowercased dotted version if the token looks like a domain.
        # EXAMPLE_COM -> example.com
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
