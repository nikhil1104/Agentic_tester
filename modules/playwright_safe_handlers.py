# modules/playwright_safe_handlers.py
"""
Safe, synchronous Playwright response capture for sync API.

Why this exists:
- Using threads/async from Playwright's sync callbacks triggers greenlet
  errors: "Cannot switch to a different thread".
- This helper attaches a purely synchronous, size-guarded JSON body reader.

How to use:
    from modules.playwright_safe_handlers import attach_safe_response_handler

    def capture_api(entry: dict) -> None:
        # Your existing function that records the API call (append to list, HAR, etc.)
        api_calls.append(entry)

    attach_safe_response_handler(page, capture_api, max_body_bytes=512_000)

Notes:
- We only read body for JSON-ish responses (content-type contains 'json').
- We cap body size to avoid memory spikes.
- We NEVER spawn threads or use asyncio here.
"""

from __future__ import annotations

import json
import logging
from typing import Callable, Optional

logger = logging.getLogger(__name__)


def attach_safe_response_handler(
    page,
    on_json_entry: Callable[[dict], None],
    *,
    max_body_bytes: int = 512_000,
) -> None:
    """
    Attach a synchronous response handler to a Playwright sync 'page'.

    Args:
        page: Playwright sync Page
        on_json_entry: callback that will receive a dict with minimal response info
        max_body_bytes: max bytes to read from the response body
    """

    def _on_response(resp) -> None:
        try:
            url = resp.url
            status = resp.status
            headers = dict(resp.headers or {})
            ctype = (headers.get("content-type") or "").lower()

            # Only peek JSON; avoid downloading big binaries or HTML.
            if "json" not in ctype:
                return

            # Skip huge payloads by content-length if available
            cl = headers.get("content-length")
            if cl and cl.isdigit() and int(cl) > max_body_bytes:
                logger.debug("Skip body read (content-length %s > cap %s): %s", cl, max_body_bytes, url)
                entry = {
                    "url": url,
                    "status": status,
                    "headers": headers,
                    "json_truncated": True,
                    "body_bytes": 0,
                }
                on_json_entry(entry)
                return

            # Read body synchronously, no threads / no asyncio
            raw = resp.body()
            if not raw:
                entry = {
                    "url": url,
                    "status": status,
                    "headers": headers,
                    "json": None,
                    "body_bytes": 0,
                }
                on_json_entry(entry)
                return

            # Cap to max size
            if len(raw) > max_body_bytes:
                raw = raw[:max_body_bytes]
                truncated = True
            else:
                truncated = False

            # Try to parse JSON; if not, store as text
            try:
                payload = json.loads(raw.decode("utf-8", "replace"))
                entry = {
                    "url": url,
                    "status": status,
                    "headers": headers,
                    "json": payload,
                    "body_bytes": len(raw),
                    "json_truncated": truncated,
                }
            except Exception:
                entry = {
                    "url": url,
                    "status": status,
                    "headers": headers,
                    "text": raw.decode("utf-8", "replace"),
                    "body_bytes": len(raw),
                    "json_truncated": truncated,
                }

            on_json_entry(entry)

        except Exception as e:
            # Swallow; never rethrow inside Playwright callback
            logger.debug("safe response handler skipped: %s", e, exc_info=True)

    page.on("response", _on_response)
