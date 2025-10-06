"""
Self-Healing Locator Engine (Phase 5.1)
--------------------------------------
Purpose:
 - Provide helper functions and heuristics to robustly locate elements.
 - Attempt best-effort locators in order of confidence and provide
   evidence/logging when retries or fallbacks are used.

How to use:
 - From Python-run Playwright tests you can import these helpers.
 - From the JS/TS generator (semantic engine) we generate actions that
   use locator styles compatible with Playwright (text=..., [id='...'], etc.)
 - This module focuses on producing locator suggestions and Python helpers.
"""

import re
import time
import json
import os
import uuid
from typing import List, Optional, Dict

STEP_LOG_DIR = "reports/step_logs"
os.makedirs(STEP_LOG_DIR, exist_ok=True)


def _log_step_json(step_id: str, status: str, message: str, extra: Optional[Dict] = None):
    """Write a small JSON file per step for easy bug-reporting and dashboard ingestion."""
    payload = {
        "step_id": step_id,
        "status": status,
        "message": message,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime())
    }
    if extra:
        payload.update(extra)
    path = os.path.join(STEP_LOG_DIR, f"{step_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


class SelfHealing:
    """
    SelfHealing helper exposes methods that try multiple locator strategies.
    Note: These are Python helpers that can be used by runner or tests executed in Python.
    For JS/TS generated tests we generate similar multi-strategy code strings.
    """

    def __init__(self, page):
        """
        :param page: Playwright Page object (sync or async depending on your runner)
        """
        self.page = page

    # ----- Locator scoring helpers -----
    def _candidate_locators(self, hint: str) -> List[str]:
        """
        Given a hint (label, visible text or attribute-like string)
        produce a list of candidate selectors in order of preference.
        """
        h = hint.strip()
        candidates = []
        # exact id/name/aria-style attempts (these assume hint may be attribute)
        candidates.append(f"[id='{h}']")
        candidates.append(f"[name='{h}']")
        candidates.append(f"[aria-label='{h}']")
        candidates.append(f"[placeholder='{h}']")
        # text-based (buttons/links)
        candidates.append(f"text=\"{h}\"")
        # more fuzzy: contains class, or partial text
        candidates.append(f"[class*='{h}']")
        candidates.append(f"css=*:has-text(\"{h}\")")
        # last resort: plain text selector
        candidates.append(f"text={h}")
        # dedupe while preserving order
        seen = set()
        out = []
        for c in candidates:
            if c not in seen:
                seen.add(c)
                out.append(c)
        return out

    # ----- Primary operations -----
    def find_and_click(self, hint: str, timeout: int = 5000, retries: int = 1) -> bool:
        """
        Try to click using candidate locators. Returns True on success.
        Writes step JSON logs and saves a screenshot on failure.
        """
        step_id = str(uuid.uuid4())[:8]
        candidates = self._candidate_locators(hint)
        last_err = None
        for attempt in range(retries + 1):
            for loc in candidates:
                try:
                    # Use Playwright locator with a short timeout
                    self.page.locator(loc).click(timeout=timeout)
                    _log_step_json(step_id, "PASS", f"Clicked using {loc}", {"locator": loc})
                    return True
                except Exception as e:
                    last_err = e
                    # continue trying
            # optional wait & retry
            time.sleep(0.5)
        # final failure: capture screenshot and log
        try:
            img_path = os.path.join(STEP_LOG_DIR, f"{step_id}_fail.png")
            self.page.screenshot(path=img_path, full_page=True)
        except Exception:
            img_path = None
        _log_step_json(step_id, "FAIL", f"Could not click '{hint}': {repr(last_err)}", {"screenshot": img_path})
        return False

    def find_and_fill(self, hint: str, value: str, timeout: int = 5000, retries: int = 1) -> bool:
        """
        Try fill (type) operations using candidate locators.
        """
        step_id = str(uuid.uuid4())[:8]
        candidates = self._candidate_locators(hint)
        last_err = None
        for attempt in range(retries + 1):
            for loc in candidates:
                try:
                    self.page.fill(loc, value, timeout=timeout)
                    _log_step_json(step_id, "PASS", f"Filled using {loc}", {"locator": loc, "value": value})
                    return True
                except Exception as e:
                    last_err = e
            time.sleep(0.5)
        try:
            img_path = os.path.join(STEP_LOG_DIR, f"{step_id}_fill_fail.png")
            self.page.screenshot(path=img_path, full_page=True)
        except Exception:
            img_path = None
        _log_step_json(step_id, "FAIL", f"Could not fill '{hint}' with '{value}': {repr(last_err)}", {"screenshot": img_path})
        return False

    def resolve_best_locator(self, hint: str, html: Optional[str] = None) -> str:
        """
        Optional: return the best candidate locator as string.
        If html is provided, you may parse & pick the best (future enhancement).
        """
        candidates = self._candidate_locators(hint)
        return candidates[0]  # best effort
