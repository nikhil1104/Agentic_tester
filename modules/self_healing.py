# modules/self_healing.py
"""
Self-Healing Locator Engine (Phase 5.1.4 — Selects & Waits)
-----------------------------------------------------------
Public API (sync Playwright):
  - find_and_click(hint, ...)
  - find_and_fill(hint, value, ...)
  - find_and_select(hint, label=None, value=None, index=None, ...)
  - wait_for_visible(hint, ...)
  - wait_for_hidden(hint, ...)

Features:
  - get_by_* first, CSS/text fallbacks after
  - Exponential backoff + jitter
  - Optional NDJSON event stream + atomic JSON step logs
  - Optional cancel propagation via threading.Event
"""

from __future__ import annotations

import json
import logging
import os
import random
import time
import uuid
from pathlib import Path
from threading import Event, Lock
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Paths / Config
# ---------------------------------------------------------------------
_BASE_REPORTS = Path(os.environ.get("RUNNER_REPORTS_DIR", "reports"))
_DEFAULT_STEP_LOG_DIR = _BASE_REPORTS / "step_logs"
_DEFAULT_EVENT_DIR = _BASE_REPORTS / "events"

# ---------------------------------------------------------------------
# File utils (atomic/thread-safe)
# ---------------------------------------------------------------------
_write_lock = Lock()

def _utc_now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _atomic_write_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with _write_lock:
        tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
        tmp.replace(path)

def _append_ndjson(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(obj, separators=(",", ":")) + "\n"
    with _write_lock:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)

def _log_step_json(step_log_dir: Path, step_id: str, status: str, message: str, extra: Optional[Dict] = None):
    payload = {
        "step_id": step_id,
        "status": status,
        "message": message,
        "timestamp": _utc_now_iso(),
    }
    if extra:
        payload.update(extra)
    _atomic_write_json(step_log_dir / f"{step_id}.json", payload)

# ---------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------
class SelfHealing:
    def __init__(
        self,
        page: Any,
        step_log_dir: Optional[Path] = None,
        ndjson_path: Optional[Path] = None,
        stop_event: Optional[Event] = None,
        screenshot_on_fail: bool = True,
    ) -> None:
        self.page = page
        self.step_log_dir = step_log_dir or _DEFAULT_STEP_LOG_DIR
        self.ndjson_path = ndjson_path  # e.g. reports/events/<exec>/steps.ndjson
        self.stop_event = stop_event
        self.screenshot_on_fail = screenshot_on_fail
        self.step_log_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------ Public ops ------------------------------
    def find_and_click(
        self,
        hint: str,
        timeout_ms: int = 5_000,
        retries: int = 2,
        base_interval: float = 0.6,
        max_interval: float = 2.5,
    ) -> bool:
        step_id = str(uuid.uuid4())[:8]
        last_err: Optional[Exception] = None
        self._emit_event("step_start", {"step_id": step_id, "op": "click", "hint": hint})

        for attempt in range(retries + 1):
            if self._cancel_requested(step_id, "click", hint):
                return False
            for method, locator, confidence in self._locator_sequence(hint):
                try:
                    self._click(method, locator, timeout_ms)
                    msg = f"Clicked '{hint}' using {method}:{locator} (conf={confidence:.2f})"
                    _log_step_json(self.step_log_dir, step_id, "PASS", msg, {"locator": locator, "confidence": confidence})
                    self._emit_event("step_end", {"step_id": step_id, "op": "click", "ok": True, "locator": locator, "confidence": confidence})
                    logger.info("Click PASS • %s", msg)
                    return True
                except Exception as e:
                    last_err = e
                    logger.debug("Click try failed via %s:%s • %s", method, locator, repr(e))
            self._sleep_with_backoff(attempt, base_interval, max_interval)

        screenshot = self._maybe_screenshot(step_id, "click_fail")
        msg = f"Could not click '{hint}' after {retries} retries: {repr(last_err)}"
        _log_step_json(self.step_log_dir, step_id, "FAIL", msg, {"screenshot": screenshot})
        self._emit_event("step_end", {"step_id": step_id, "op": "click", "ok": False, "error": str(last_err), "screenshot": screenshot})
        logger.warning("Click FAIL • %s", msg)
        return False

    def find_and_fill(
        self,
        hint: str,
        value: str,
        timeout_ms: int = 5_000,
        retries: int = 2,
        base_interval: float = 0.6,
        max_interval: float = 2.5,
        clear: bool = True,
    ) -> bool:
        step_id = str(uuid.uuid4())[:8]
        last_err: Optional[Exception] = None
        self._emit_event("step_start", {"step_id": step_id, "op": "fill", "hint": hint})

        for attempt in range(retries + 1):
            if self._cancel_requested(step_id, "fill", hint):
                return False
            for method, locator, confidence in self._locator_sequence(hint, prefer_textbox=True):
                try:
                    self._fill(method, locator, value, timeout_ms, clear=clear)
                    msg = f"Filled '{hint}' using {method}:{locator} (conf={confidence:.2f})"
                    _log_step_json(self.step_log_dir, step_id, "PASS", msg, {"locator": locator, "value": value, "confidence": confidence})
                    self._emit_event("step_end", {"step_id": step_id, "op": "fill", "ok": True, "locator": locator, "confidence": confidence})
                    logger.info("Fill PASS • %s", msg)
                    return True
                except Exception as e:
                    last_err = e
                    logger.debug("Fill try failed via %s:%s • %s", method, locator, repr(e))
            self._sleep_with_backoff(attempt, base_interval, max_interval)

        screenshot = self._maybe_screenshot(step_id, "fill_fail")
        msg = f"Could not fill '{hint}' after {retries} retries: {repr(last_err)}"
        _log_step_json(self.step_log_dir, step_id, "FAIL", msg, {"value": value, "screenshot": screenshot})
        self._emit_event("step_end", {"step_id": step_id, "op": "fill", "ok": False, "error": str(last_err), "screenshot": screenshot})
        logger.warning("Fill FAIL • %s", msg)
        return False

    def find_and_select(
        self,
        hint: str,
        *,
        label: Optional[str] = None,
        value: Optional[str] = None,
        index: Optional[int] = None,
        timeout_ms: int = 5_000,
        retries: int = 2,
        base_interval: float = 0.6,
        max_interval: float = 2.5,
    ) -> bool:
        """
        Robust option selection for native <select> and ARIA combobox/listbox UIs.
        At least one of (label, value, index) should be provided; otherwise defaults to index=0.
        """
        step_id = str(uuid.uuid4())[:8]
        last_err: Optional[Exception] = None
        if label is None and value is None and index is None:
            index = 0  # sensible default

        self._emit_event("step_start", {"step_id": step_id, "op": "select", "hint": hint,
                                        "label": label, "value": value, "index": index})

        for attempt in range(retries + 1):
            if self._cancel_requested(step_id, "select", hint):
                return False

            for method, locator, confidence in self._locator_sequence(hint, prefer_select=True):
                try:
                    self._select(method, locator, label=label, value=value, index=index, timeout_ms=timeout_ms)
                    msg = f"Selected on '{hint}' using {method}:{locator} (conf={confidence:.2f}) [label={label} value={value} index={index}]"
                    _log_step_json(self.step_log_dir, step_id, "PASS", msg, {"locator": locator, "confidence": confidence})
                    self._emit_event("step_end", {"step_id": step_id, "op": "select", "ok": True, "locator": locator, "confidence": confidence})
                    logger.info("Select PASS • %s", msg)
                    return True
                except Exception as e:
                    last_err = e
                    logger.debug("Select try failed via %s:%s • %s", method, locator, repr(e))

            self._sleep_with_backoff(attempt, base_interval, max_interval)

        screenshot = self._maybe_screenshot(step_id, "select_fail")
        msg = f"Could not select on '{hint}' after {retries} retries: {repr(last_err)}"
        _log_step_json(self.step_log_dir, step_id, "FAIL", msg, {"screenshot": screenshot})
        self._emit_event("step_end", {"step_id": step_id, "op": "select", "ok": False, "error": str(last_err), "screenshot": screenshot})
        logger.warning("Select FAIL • %s", msg)
        return False

    def wait_for_visible(
        self,
        hint: str,
        timeout_ms: int = 5_000,
        retries: int = 0,
        base_interval: float = 0.4,
        max_interval: float = 1.5,
    ) -> bool:
        """Healed wait until element is visible."""
        step_id = str(uuid.uuid4())[:8]
        self._emit_event("step_start", {"step_id": step_id, "op": "wait_visible", "hint": hint})
        last_err: Optional[Exception] = None

        for attempt in range(retries + 1):
            if self._cancel_requested(step_id, "wait_visible", hint):
                return False
            for method, locator, confidence in self._locator_sequence(hint):
                try:
                    self._wait(method, locator, state="visible", timeout_ms=timeout_ms)
                    msg = f"Visible: '{hint}' via {method}:{locator} (conf={confidence:.2f})"
                    _log_step_json(self.step_log_dir, step_id, "PASS", msg, {"locator": locator, "confidence": confidence})
                    self._emit_event("step_end", {"step_id": step_id, "op": "wait_visible", "ok": True, "locator": locator})
                    logger.info("Wait visible PASS • %s", msg)
                    return True
                except Exception as e:
                    last_err = e
                    logger.debug("Wait visible failed via %s:%s • %s", method, locator, repr(e))
            self._sleep_with_backoff(attempt, base_interval, max_interval)

        msg = f"Element not visible: '{hint}' after {retries} retries: {repr(last_err)}"
        _log_step_json(self.step_log_dir, step_id, "FAIL", msg)
        self._emit_event("step_end", {"step_id": step_id, "op": "wait_visible", "ok": False, "error": str(last_err)})
        logger.warning("Wait visible FAIL • %s", msg)
        return False

    def wait_for_hidden(
        self,
        hint: str,
        timeout_ms: int = 5_000,
        retries: int = 0,
        base_interval: float = 0.4,
        max_interval: float = 1.5,
    ) -> bool:
        """Healed wait until element is hidden/detached."""
        step_id = str(uuid.uuid4())[:8]
        self._emit_event("step_start", {"step_id": step_id, "op": "wait_hidden", "hint": hint})
        last_err: Optional[Exception] = None

        for attempt in range(retries + 1):
            if self._cancel_requested(step_id, "wait_hidden", hint):
                return False
            for method, locator, confidence in self._locator_sequence(hint):
                try:
                    self._wait(method, locator, state="hidden", timeout_ms=timeout_ms)
                    msg = f"Hidden: '{hint}' via {method}:{locator} (conf={confidence:.2f})"
                    _log_step_json(self.step_log_dir, step_id, "PASS", msg, {"locator": locator, "confidence": confidence})
                    self._emit_event("step_end", {"step_id": step_id, "op": "wait_hidden", "ok": True, "locator": locator})
                    logger.info("Wait hidden PASS • %s", msg)
                    return True
                except Exception as e:
                    last_err = e
                    logger.debug("Wait hidden failed via %s:%s • %s", method, locator, repr(e))
            self._sleep_with_backoff(attempt, base_interval, max_interval)

        msg = f"Element not hidden: '{hint}' after {retries} retries: {repr(last_err)}"
        _log_step_json(self.step_log_dir, step_id, "FAIL", msg)
        self._emit_event("step_end", {"step_id": step_id, "op": "wait_hidden", "ok": False, "error": str(last_err)})
        logger.warning("Wait hidden FAIL • %s", msg)
        return False

    # ------------------------------ Internals ------------------------------
    def _cancel_requested(self, step_id: str, op: str, hint: str) -> bool:
        if self.stop_event and self.stop_event.is_set():
            msg = f"Cancelled before {op}('{hint}')"
            _log_step_json(self.step_log_dir, step_id, "CANCELLED", msg)
            self._emit_event("step_cancelled", {"step_id": step_id, "op": op, "hint": hint})
            logger.info(msg)
            return True
        return False

    def _sleep_with_backoff(self, attempt: int, base: float, cap: float) -> None:
        delay = min(cap, base * (2 ** attempt)) * (0.8 + 0.4 * random.random())
        logger.debug("Retry backoff sleeping for %.2fs", delay)
        time.sleep(delay)

    def _maybe_screenshot(self, step_id: str, label: str) -> Optional[str]:
        if not self.screenshot_on_fail:
            return None
        try:
            path = self.step_log_dir / f"{step_id}_{label}.png"
            self.page.screenshot(path=str(path), full_page=True)
            return str(path)
        except Exception:
            logger.debug("Screenshot capture failed", exc_info=True)
            return None

    # Prefer select/combobox when requested
    def _locator_sequence(
        self,
        hint: str,
        *,
        prefer_textbox: bool = False,
        prefer_select: bool = False
    ) -> List[Tuple[str, str, float]]:
        h = (hint or "").strip()
        seq: List[Tuple[str, str, float]] = []

        if prefer_select:
            seq.extend([
                ("get_by_role[combobox]", h, 1.00),
                ("get_by_role[listbox]", h, 0.98),
                ("get_by_label", h, 0.96),
            ])
        elif prefer_textbox:
            seq.extend([
                ("get_by_label", h, 1.00),
                ("get_by_placeholder", h, 0.98),
                ("get_by_role[textbox]", h, 0.96),
            ])
        else:
            seq.extend([
                ("get_by_role[button]", h, 1.00),
                ("get_by_role[link]", h, 0.98),
                ("get_by_label", h, 0.95),
            ])

        seq.append(("get_by_text_exact", h, 0.92))
        seq.append(("get_by_text_fuzzy", h, 0.90))

        seq.extend([
            ("css", f"[id='{h}']", 0.88),
            ("css", f"[name='{h}']", 0.86),
            ("css", f"[aria-label='{h}']", 0.84),
            ("css", f"[placeholder='{h}']", 0.82),
            ("css", f"[class*='{h}']", 0.72),
            ("css", f"*:has-text(\"{h}\")", 0.66),
            ("text", h, 0.60),
        ])

        seen = set()
        out: List[Tuple[str, str, float]] = []
        for m, loc, conf in seq:
            key = (m, loc)
            if key not in seen:
                seen.add(key)
                out.append((m, loc, conf))
        return out

    # ------------------------------ Low-level ops ------------------------------
    def _click(self, method: str, locator: str, timeout_ms: int):
        if method == "get_by_role[button]":
            self.page.get_by_role("button", name=locator).click(timeout=timeout_ms)
        elif method == "get_by_role[link]":
            self.page.get_by_role("link", name=locator).click(timeout=timeout_ms)
        elif method == "get_by_role[textbox]":
            self.page.get_by_role("textbox", name=locator).click(timeout=timeout_ms)
        elif method == "get_by_label":
            self.page.get_by_label(locator).click(timeout=timeout_ms)
        elif method == "get_by_placeholder":
            self.page.get_by_placeholder(locator).click(timeout=timeout_ms)
        elif method == "get_by_text_exact":
            self.page.get_by_text(locator, exact=True).click(timeout=timeout_ms)
        elif method == "get_by_text_fuzzy":
            self.page.get_by_text(locator).click(timeout=timeout_ms)
        elif method == "css":
            self.page.locator(locator).click(timeout=timeout_ms)
        elif method == "text":
            self.page.locator(f"text={locator}").click(timeout=timeout_ms)
        else:
            self.page.locator(locator).click(timeout=timeout_ms)

    def _fill(self, method: str, locator: str, value: str, timeout_ms: int, clear: bool = True):
        if method in {"get_by_label", "get_by_placeholder", "get_by_text_exact", "get_by_text_fuzzy"}:
            loc = (
                self.page.get_by_label(locator) if method == "get_by_label" else
                self.page.get_by_placeholder(locator) if method == "get_by_placeholder" else
                self.page.get_by_text(locator, exact=True) if method == "get_by_text_exact" else
                self.page.get_by_text(locator)
            )
            if clear:
                try: loc.fill("", timeout=timeout_ms)
                except Exception: pass
            loc.fill(value, timeout=timeout_ms)
            return
        if method == "get_by_role[textbox]":
            loc = self.page.get_by_role("textbox", name=locator)
            if clear:
                try: loc.fill("", timeout=timeout_ms)
                except Exception: pass
            loc.fill(value, timeout=timeout_ms)
            return
        if method == "css":
            loc = self.page.locator(locator)
            if clear:
                try: loc.fill("", timeout=timeout_ms)
                except Exception: pass
            loc.fill(value, timeout=timeout_ms)
            return
        if method == "text":
            loc = self.page.locator(f"text={locator}")
            if clear:
                try: loc.fill("", timeout=timeout_ms)
                except Exception: pass
            loc.fill(value, timeout=timeout_ms)
            return
        self.page.locator(locator).fill(value, timeout=timeout_ms)

    def _select(
        self,
        method: str,
        locator: str,
        *,
        label: Optional[str],
        value: Optional[str],
        index: Optional[int],
        timeout_ms: int,
    ) -> None:
        # Try native selectOption first when possible
        def _try_native(loc):
            opts: Dict[str, Any] = {}
            if label is not None: opts["label"] = label
            if value is not None: opts["value"] = value
            if index is not None: opts["index"] = index
            if not opts:
                opts["index"] = 0
            loc.select_option(opts, timeout=timeout_ms)

        # Locator resolution
        if method == "get_by_role[combobox]":
            loc = self.page.get_by_role("combobox", name=locator)
            try:
                _try_native(loc)
                return
            except Exception:
                # Fallback: open, pick option by role/name
                loc.click(timeout=timeout_ms)
                if label:
                    self.page.get_by_role("option", name=label).click(timeout=timeout_ms)
                    return
                if value:
                    # try option[value=value]
                    self.page.locator(f"option[value='{value}']").click(timeout=timeout_ms)
                    return
                # else first option
                self.page.get_by_role("option").first.click(timeout=timeout_ms)
                return

        if method == "get_by_role[listbox]":
            # listbox often requires opening its trigger
            # heuristic: click nearest button with same accessible name
            try:
                self.page.get_by_role("button", name=locator).click(timeout=timeout_ms)
            except Exception:
                pass
            if label:
                self.page.get_by_role("option", name=label).click(timeout=timeout_ms)
                return
            if value:
                self.page.locator(f"[role='option'][data-value='{value}'], [role='option'][value='{value}']").first.click(timeout=timeout_ms)
                return
            self.page.get_by_role("option").first.click(timeout=timeout_ms)
            return

        if method == "get_by_label":
            loc = self.page.get_by_label(locator)
            try:
                _try_native(loc)
                return
            except Exception:
                # open & pick by option text
                loc.click(timeout=timeout_ms)
                if label:
                    self.page.get_by_role("option", name=label).click(timeout=timeout_ms)
                    return
                if value:
                    self.page.locator(f"option[value='{value}']").click(timeout=timeout_ms)
                    return
                self.page.get_by_role("option").first.click(timeout=timeout_ms)
                return

        if method in {"get_by_text_exact", "get_by_text_fuzzy"}:
            loc = self.page.get_by_text(locator, exact=(method == "get_by_text_exact"))
            try:
                _try_native(loc)
                return
            except Exception:
                loc.click(timeout=timeout_ms)
                if label:
                    self.page.get_by_role("option", name=label).click(timeout=timeout_ms)
                    return
                if value:
                    self.page.locator(f"option[value='{value}']").click(timeout=timeout_ms)
                    return
                self.page.get_by_role("option").first.click(timeout=timeout_ms)
                return

        if method in {"css", "text"}:
            loc = self.page.locator(locator if method == "css" else f"text={locator}")
            _try_native(loc)
            return

        # Final fallback
        self.page.locator(locator).select_option(
            {"label": label} if label is not None else
            {"value": value} if value is not None else
            {"index": index if index is not None else 0},
            timeout=timeout_ms,
        )

    def _wait(self, method: str, locator: str, *, state: str, timeout_ms: int) -> None:
        if method == "get_by_role[button]":
            self.page.get_by_role("button", name=locator).wait_for(state=state, timeout=timeout_ms)
            return
        if method == "get_by_role[link]":
            self.page.get_by_role("link", name=locator).wait_for(state=state, timeout=timeout_ms)
            return
        if method == "get_by_role[textbox]":
            self.page.get_by_role("textbox", name=locator).wait_for(state=state, timeout=timeout_ms)
            return
        if method == "get_by_role[combobox]":
            self.page.get_by_role("combobox", name=locator).wait_for(state=state, timeout=timeout_ms)
            return
        if method == "get_by_role[listbox]":
            self.page.get_by_role("listbox", name=locator).wait_for(state=state, timeout=timeout_ms)
            return
        if method == "get_by_label":
            self.page.get_by_label(locator).wait_for(state=state, timeout=timeout_ms)
            return
        if method == "get_by_placeholder":
            self.page.get_by_placeholder(locator).wait_for(state=state, timeout=timeout_ms)
            return
        if method == "get_by_text_exact":
            self.page.get_by_text(locator, exact=True).wait_for(state=state, timeout=timeout_ms)
            return
        if method == "get_by_text_fuzzy":
            self.page.get_by_text(locator).wait_for(state=state, timeout=timeout_ms)
            return
        if method == "css":
            self.page.locator(locator).wait_for(state=state, timeout=timeout_ms)
            return
        if method == "text":
            self.page.locator(f"text={locator}").wait_for(state=state, timeout=timeout_ms)
            return
        self.page.locator(locator).wait_for(state=state, timeout=timeout_ms)

    # ------------------------------ NDJSON ------------------------------
    def _emit_event(self, typ: str, payload: Dict[str, Any]) -> None:
        if not self.ndjson_path:
            return
        ev = {"ts": _utc_now_iso(), "type": typ, "payload": payload}
        try:
            _append_ndjson(self.ndjson_path, ev)
        except Exception:
            logger.debug("Failed to append NDJSON", exc_info=True)
