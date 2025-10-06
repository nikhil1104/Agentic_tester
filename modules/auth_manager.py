"""
Auth Manager (Phase 5.4.1 — Intelligent Login Detection & Session Validation)
---------------------------------------------------------------------------
Responsibilities:
- Load auth configuration (config/auth_config.json)
- Perform form-based login and save Playwright storageState
- Validate an existing storageState to see if session is still valid
- Provide helper to create a browser context using saved storageState
- Designed to be non-blocking if no config is present (graceful skip)

Notes / Thumb rules:
- Keep auth_config minimal: login_url, username_selector, password_selector, submit_selector,
  username, password, optional post_login_check_selector (a selector that is present after login)
- storage state is saved to `session_file` (default auth/session_state.json)
- Validation is best-effort: we load a page with the storage state and check for a post-login indicator
"""

import os
import json
import time
from typing import Optional

# Try importing Playwright; if not available we'll still allow the module to be imported
try:
    from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
    PLAYWRIGHT_AVAILABLE = True
except Exception:
    PLAYWRIGHT_AVAILABLE = False


class AuthManager:
    def __init__(self,
                 config_path: str = "config/auth_config.json",
                 session_file: str = "auth/session_state.json",
                 validate_timeout_ms: int = 8000):
        """
        :param config_path: JSON config with login details (optional)
        :param session_file: Where to persist storageState (relative path)
        :param validate_timeout_ms: timeout while validating session (ms)
        """
        self.config_path = config_path
        self.session_file = session_file
        self.validate_timeout_ms = validate_timeout_ms
        self.auth_config = self._load_config()

    # ---------------------------------------------------------------
    # Load Auth Configuration
    # ---------------------------------------------------------------
    def _load_config(self) -> dict:
        """Read auth config if present. Returns {} if missing or invalid."""
        if not os.path.exists(self.config_path):
            print(f"⚠️ No auth_config.json found at {self.config_path}, skipping auth setup.")
            return {}
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            # Basic validation / defaults
            return data if isinstance(data, dict) else {}
        except Exception as e:
            print(f"⚠️ Error reading auth config: {e}")
            return {}

    # ---------------------------------------------------------------
    # Perform Login and Save Storage State
    # ---------------------------------------------------------------
    def login_and_save_session(self, force: bool = False) -> Optional[str]:
        """
        Perform login as per config and save storageState. If a valid session already exists
        and force is False, this will return the existing session path.

        Returns:
            path to session file, or None if login skipped/failed
        """
        if not self.auth_config:
            return None

        # If session exists and appears valid, skip unless forced
        if not force and os.path.exists(self.session_file):
            try:
                if self.is_session_valid(self.session_file):
                    print(f"🔐 Reusing existing valid session: {self.session_file}")
                    return self.session_file
                else:
                    print("⚠️ Existing session invalid/expired — will attempt login.")
            except Exception:
                # Fall back to performing login
                print("⚠️ Session validation failed unexpectedly — will attempt login.")

        if not PLAYWRIGHT_AVAILABLE:
            print("⚠️ Playwright not available — cannot perform login programmatically.")
            return None

        login_url = self.auth_config.get("login_url")
        username_selector = self.auth_config.get("username_selector")
        password_selector = self.auth_config.get("password_selector")
        submit_selector = self.auth_config.get("submit_selector")
        username = self.auth_config.get("username")
        password = self.auth_config.get("password")
        post_login_check = self.auth_config.get("post_login_check_selector")  # optional selector present after login
        wait_after_login_ms = int(self.auth_config.get("wait_after_login_ms", 3000))

        if not all([login_url, username_selector, password_selector, submit_selector, username, password]):
            print("⚠️ Incomplete auth configuration (login_url/username/password/selectors). Skipping login.")
            return None

        print(f"🔐 Performing login at: {login_url}")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                context = browser.new_context(ignore_https_errors=True)
                page = context.new_page()

                # Navigate to login page (networkidle to allow JS)
                try:
                    page.goto(login_url, wait_until="networkidle", timeout=self.validate_timeout_ms)
                except PlaywrightTimeoutError:
                    # Try a softer load if networkidle times out
                    page.goto(login_url, wait_until="domcontentloaded", timeout=self.validate_timeout_ms)

                # Fill credentials safely (selectors expected to be Playwright-compatible)
                page.fill(username_selector, username)
                page.fill(password_selector, password)
                page.click(submit_selector)

                # Allow for post-login redirects and JS setup
                page.wait_for_timeout(wait_after_login_ms)

                # Optional: if post_login_check is provided, ensure it exists
                if post_login_check:
                    try:
                        page.wait_for_selector(post_login_check, timeout=5000)
                    except PlaywrightTimeoutError:
                        print("⚠️ Post-login check selector not found — login may have failed.")

                # Save storage state
                os.makedirs(os.path.dirname(self.session_file), exist_ok=True)
                context.storage_state(path=self.session_file)
                print(f"✅ Session saved successfully → {self.session_file}")

                browser.close()
                return self.session_file
        except Exception as e:
            print(f"❌ Login attempt failed: {e}")
            return None

    # ---------------------------------------------------------------
    # Validate existing session storage state
    # ---------------------------------------------------------------
    def is_session_valid(self, session_file: str, validate_url: Optional[str] = None) -> bool:
        """
        Best-effort validation of a saved storageState.

        Strategy:
          - Launch a browser context with storageState
          - Navigate to either validate_url (if provided) or the login_url from config
          - Determine login state by absence of login form or presence of post_login_check_selector

        Returns True if session appears valid.
        """
        if not PLAYWRIGHT_AVAILABLE:
            # Can't validate without Playwright; assume invalid so the caller may elect to re-login.
            return False

        if not os.path.exists(session_file):
            return False

        # Choose page to load for validation
        target_url = validate_url or self.auth_config.get("validate_url") or self.auth_config.get("login_url") or "about:blank"
        post_login_check = self.auth_config.get("post_login_check_selector")

        try:
            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                # use storage state to create context
                context = browser.new_context(storage_state=session_file, ignore_https_errors=True)
                page = context.new_page()
                try:
                    page.goto(target_url, wait_until="domcontentloaded", timeout=self.validate_timeout_ms)
                except PlaywrightTimeoutError:
                    # keep going even if a strict wait failed
                    pass

                # If post_login_check selector is provided — presence implies valid login
                if post_login_check:
                    try:
                        sel = page.locator(post_login_check)
                        if sel.count() and sel.first().is_visible():
                            browser.close()
                            return True
                    except Exception:
                        # fall through to heuristic
                        pass

                # Heuristic: check whether login form exists on page (if login form present -> invalid)
                try:
                    username_sel = self.auth_config.get("username_selector")
                    password_sel = self.auth_config.get("password_selector")
                    # If both selectors are present on page, assume not logged in
                    found_username = False
                    found_password = False
                    if username_sel:
                        try:
                            found_username = page.locator(username_sel).count() > 0
                        except Exception:
                            found_username = False
                    if password_sel:
                        try:
                            found_password = page.locator(password_sel).count() > 0
                        except Exception:
                            found_password = False

                    if found_username or found_password:
                        # login form still present -> likely invalid
                        browser.close()
                        return False
                except Exception:
                    pass

                # If no clear failure signals, assume session is valid
                browser.close()
                return True
        except Exception:
            return False

    # ---------------------------------------------------------------
    # Helper: create a new browser context with session injected (if present)
    # ---------------------------------------------------------------
    def create_context_with_session(self, browser, force_fresh: bool = False):
        """
        Given a Playwright browser instance, return a new context:
         - If a valid session file exists and !force_fresh -> context with storage_state
         - Otherwise -> fresh context without storage_state
        """
        try:
            if not force_fresh and os.path.exists(self.session_file) and self.is_session_valid(self.session_file):
                print(f"🔄 Reusing saved session from {self.session_file}")
                return browser.new_context(storage_state=self.session_file, ignore_https_errors=True)
        except Exception:
            # fallback to fresh context
            pass
        print("ℹ️ Creating fresh context (no session or invalid).")
        return browser.new_context(ignore_https_errors=True)
