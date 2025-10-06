"""
Credential Manager (Phase 5.2 hook)
----------------------------------
Simple credential loader supporting:
 - environment variables
 - optional local secrets.json for roles/domains
 - easy future extension to Vault integrations
"""
import os
import json
import re
from typing import Dict

DEFAULT_SECRETS = "secrets.json"


class CredentialManager:
    def __init__(self, secrets_file: str = DEFAULT_SECRETS):
        self.secrets_file = secrets_file
        self.secrets = {}
        if os.path.exists(self.secrets_file):
            try:
                with open(self.secrets_file, "r", encoding="utf-8") as f:
                    self.secrets = json.load(f)
            except Exception:
                self.secrets = {}

    def get_credentials(self, domain_hint: str = None, role: str = "test") -> Dict[str, str]:
        """
        Lookup order:
          1) env vars (DOMAIN_USER / DOMAIN_PASS or TEST_USER / TEST_PASS)
          2) secrets.json keyed by domain or role
          3) fallback to None entries
        """
        username = os.environ.get("TEST_USER") or os.environ.get("USER")
        password = os.environ.get("TEST_PASS") or os.environ.get("PASS")

        if domain_hint:
            key_prefix = re.sub(r"[^A-Za-z0-9]", "_", domain_hint).upper()
            username = os.environ.get(f"{key_prefix}_USER") or username
            password = os.environ.get(f"{key_prefix}_PASS") or password

        # secrets.json fallback
        if (not username or not password) and self.secrets:
            entry = None
            if domain_hint and domain_hint in self.secrets:
                entry = self.secrets.get(domain_hint)
            elif role and role in self.secrets:
                entry = self.secrets.get(role)
            if entry:
                username = username or entry.get("username") or entry.get("email")
                password = password or entry.get("password")

        return {"username": username, "password": password}
