# modules/auth_manager.py
"""
Auth Manager (Async, greenlet-safe)
-----------------------------------
- No playwright.sync_api anywhere.
- Uses playwright.async_api inside a dedicated asyncio loop running on a
  private thread so callers can be sync OR async without deadlocking.
- Atomic storageState saves; robust timeouts; optional MFA waits.

Public API (same signatures):
- login_and_save_session(force: bool = False) -> Optional[str]
- is_session_valid(session_file: str, validate_url: Optional[str] = None) -> bool
- create_context_with_session(browser, force_fresh: bool = False)  # unchanged (Node PW users)
"""

from __future__ import annotations

import os
import json
import time
import logging
import threading
import asyncio
from typing import Optional, Dict

logger = logging.getLogger(__name__)


def _coerce_bool(val) -> bool:
    return str(val).strip().lower() in {"1", "true", "yes", "on"}


class _AsyncLoopThread:
    """
    Runs a private asyncio loop in a background thread and lets us
    submit coroutines from ANY context (sync or async) safely.
    """

    def __init__(self):
        self._loop = None
        self._thread = None
        self._ready = threading.Event()
        self._start()

    def _runner(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        self._loop.run_forever()

    def _start(self):
        t = threading.Thread(target=self._runner, name="AuthManager-EventLoop", daemon=True)
        t.start()
        self._thread = t
        self._ready.wait()

    def submit(self, coro, timeout: Optional[float] = None):
        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return fut.result(timeout=timeout)

    def stop(self):
        if self._loop and self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)


class AuthManager:
    def __init__(
        self,
        config_path: str = "config/auth_config.json",
        session_file: str = "auth/session_state.json",
        validate_timeout_ms: int = 30000,
    ):
        self.config_path = config_path
        self.session_file = session_file
        self.validate_timeout_ms = int(validate_timeout_ms)
        self.auth_config = self._load_config()
        # private loop thread
        self._loop = _AsyncLoopThread()

    # ---------------- Config ----------------
    def _load_config(self) -> Dict:
        cfg: Dict = {}
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, dict):
                    cfg.update(data)
            except Exception as e:
                logger.warning("Error reading auth config at %s: %s", self.config_path, e, exc_info=True)
        else:
            logger.info("auth_config.json not found at %s; auth is optional.", self.config_path)

        env_map = {
            "login_url": os.environ.get("AUTH_LOGIN_URL"),
            "username_selector": os.environ.get("AUTH_USERNAME_SELECTOR"),
            "password_selector": os.environ.get("AUTH_PASSWORD_SELECTOR"),
            "submit_selector": os.environ.get("AUTH_SUBMIT_SELECTOR"),
            "username": os.environ.get("AUTH_USERNAME"),
            "password": os.environ.get("AUTH_PASSWORD"),
            "post_login_check_selector": os.environ.get("AUTH_POST_LOGIN_CHECK_SELECTOR"),
            "wait_after_login_ms": os.environ.get("AUTH_WAIT_AFTER_LOGIN_MS"),
            "headless": os.environ.get("AUTH_HEADLESS"),
            "validate_url": os.environ.get("AUTH_VALIDATE_URL"),
            "mfa_wait_ms": os.environ.get("AUTH_MFA_WAIT_MS"),
            "mfa_selector": os.environ.get("AUTH_MFA_SELECTOR"),
        }
        for k, v in env_map.items():
            if v is None:
                continue
            if k in {"wait_after_login_ms", "mfa_wait_ms"}:
                try:
                    cfg[k] = int(v)
                except Exception:
                    logger.warning("Invalid int for %s env override: %r", k, v)
            elif k == "headless":
                cfg[k] = _coerce_bool(v)
            else:
                cfg[k] = v

        cfg.setdefault("wait_after_login_ms", 3000)
        cfg.setdefault("headless", True)
        return cfg

    # ---------------- Public API ----------------
    def login_and_save_session(self, force: bool = False) -> Optional[str]:
        """Synchronous facade; runs the async login on our private loop."""
        if not self.auth_config:
            logger.info("No auth config provided; skipping login.")
            return None

        if not force and os.path.exists(self.session_file):
            try:
                if self.is_session_valid(self.session_file):
                    logger.info("Reusing existing valid session: %s", self.session_file)
                    return self.session_file
            except Exception:
                logger.warning("Session validation raised; attempting fresh login.", exc_info=True)

        cfg = self.auth_config
        required = ["login_url", "username_selector", "password_selector", "submit_selector", "username", "password"]
        missing = [k for k in required if not cfg.get(k)]
        if missing:
            logger.warning("Incomplete auth configuration (missing: %s). Skipping login.", ", ".join(missing))
            return None

        attempts, backoff = 2, 1.5
        last_exc = None
        for i in range(attempts):
            try:
                return self._loop.submit(self._login_and_save_async(cfg), timeout=(self.validate_timeout_ms/1000)*2 + 10)
            except Exception as e:
                last_exc = e
                logger.exception("Login attempt %d failed", i + 1)
                if i < attempts - 1:
                    time.sleep(backoff)

        logger.error("All login attempts failed; no session created.")
        return None

    def is_session_valid(self, session_file: str, validate_url: Optional[str] = None) -> bool:
        """Synchronous facade calling the async validator on our private loop."""
        if not os.path.exists(session_file):
            return False
        try:
            return self._loop.submit(self._is_session_valid_async(session_file, validate_url), timeout=(self.validate_timeout_ms/1000)*2 + 10)
        except Exception:
            logger.debug("Validation raised.", exc_info=True)
            return False

    # keep for Node Playwright callers
    def create_context_with_session(self, browser, force_fresh: bool = False):
        """
        For Node Playwright (Runner UI stage) we still allow injecting storageState
        into a spawned browser context (that stage runs in a separate subprocess).
        """
        try:
            if (not force_fresh) and os.path.exists(self.session_file) and self.is_session_valid(self.session_file):
                logger.info("Reusing saved session from %s", self.session_file)
                return browser.new_context(storage_state=self.session_file, ignore_https_errors=True)
        except Exception:
            logger.debug("create_context_with_session: reuse probe failed", exc_info=True)
        logger.info("Creating fresh context (no valid session).")
        return browser.new_context(ignore_https_errors=True)

    # ---------------- Async guts ----------------
    async def _login_and_save_async(self, cfg: Dict) -> Optional[str]:
        from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError  # lazy import

        login_url = cfg.get("login_url")
        username_selector = cfg.get("username_selector")
        password_selector = cfg.get("password_selector")
        submit_selector = cfg.get("submit_selector")
        username = cfg.get("username")
        password = cfg.get("password")
        post_login_check = cfg.get("post_login_check_selector")
        wait_after_login_ms = int(cfg.get("wait_after_login_ms", 3000))
        headless = bool(cfg.get("headless", True))
        mfa_wait_ms = int(cfg.get("mfa_wait_ms", 0))
        mfa_selector = cfg.get("mfa_selector")

        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context(ignore_https_errors=True)
        page = await context.new_page()

        try:
            page.set_default_timeout(self.validate_timeout_ms)
            try:
                await page.goto(login_url, wait_until="networkidle", timeout=self.validate_timeout_ms)
            except PlaywrightTimeoutError:
                await page.goto(login_url, wait_until="domcontentloaded", timeout=self.validate_timeout_ms)

            await page.fill(username_selector, str(username))
            await page.fill(password_selector, str(password))
            await page.click(submit_selector)

            if mfa_selector:
                try:
                    await page.locator(mfa_selector).wait_for(
                        state="visible",
                        timeout=mfa_wait_ms or self.validate_timeout_ms
                    )
                except Exception:
                    pass
            elif mfa_wait_ms > 0:
                await page.wait_for_timeout(mfa_wait_ms)

            try:
                await page.wait_for_load_state("networkidle", timeout=min(self.validate_timeout_ms, 10_000))
            except Exception:
                pass
            await page.wait_for_timeout(wait_after_login_ms)

            if post_login_check:
                try:
                    await page.locator(post_login_check).wait_for(
                        state="visible", timeout=int(self.validate_timeout_ms * 0.6)
                    )
                except Exception:
                    # best-effort
                    pass

            # save storage state atomically
            os.makedirs(os.path.dirname(self.session_file) or ".", exist_ok=True)
            tmp_path = self.session_file + ".tmp"
            await context.storage_state(path=tmp_path)
            os.replace(tmp_path, self.session_file)
            return self.session_file
        finally:
            try:
                await context.close()
            except Exception:
                pass
            try:
                await browser.close()
            except Exception:
                pass
            try:
                await pw.stop()
            except Exception:
                pass

    async def _is_session_valid_async(self, session_file: str, validate_url: Optional[str]) -> bool:
        from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeoutError  # lazy import

        cfg = self.auth_config
        target_url = validate_url or cfg.get("validate_url") or cfg.get("login_url") or "about:blank"
        post_login_check = cfg.get("post_login_check_selector")
        username_sel = cfg.get("username_selector")
        password_sel = cfg.get("password_selector")
        headless = bool(cfg.get("headless", True))

        pw = await async_playwright().start()
        browser = await pw.chromium.launch(headless=headless)
        context = await browser.new_context(storage_state=session_file, ignore_https_errors=True)
        page = await context.new_page()

        try:
            page.set_default_timeout(self.validate_timeout_ms)
            try:
                await page.goto(target_url, wait_until="networkidle", timeout=self.validate_timeout_ms)
            except PlaywrightTimeoutError:
                await page.goto(target_url, wait_until="domcontentloaded", timeout=self.validate_timeout_ms)

            if post_login_check:
                try:
                    if await page.locator(post_login_check).count() > 0:
                        return True
                except Exception:
                    pass

            try:
                if (username_sel and await page.locator(username_sel).count() > 0) or \
                   (password_sel and await page.locator(password_sel).count() > 0):
                    return False
            except Exception:
                pass

            return True
        finally:
            try:
                await context.close()
            except Exception:
                pass
            try:
                await browser.close()
            except Exception:
                pass
            try:
                await pw.stop()
            except Exception:
                pass
