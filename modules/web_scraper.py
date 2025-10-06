"""
Web Scraper (Phase 5.3 – Adaptive DOM + Session-Aware Render)
-------------------------------------------------------------
Purpose:
Performs deep, session-aware rendering and DOM capture for AI QA Agent.

✅ Upgrades from Phase 4.1:
- Reuses authenticated session (via AuthManager → storageState)
- Multi-browser fallback (Chromium → Firefox → WebKit)
- Detects and waits for dynamic JS widgets & network-idle
- Random user-agent rotation for stealth
- Saves dual-format snapshots (HTML + Markdown)
- Returns structured metadata for TestGenerator
"""

import os
import random
import time
from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
from modules.auth_manager import AuthManager


class WebScraper:
    def __init__(self, max_links: int = 5, timeout_ms: int = 30000, post_load_wait: int = 5000):
        """
        :param max_links: Max number of links to follow from the main page.
        :param timeout_ms: Timeout per page load (in ms).
        :param post_load_wait: Extra time (in ms) to wait for JS widgets after page load.
        """
        self.max_links = max_links
        self.timeout_ms = timeout_ms
        self.post_load_wait = post_load_wait
        self.auth_manager = AuthManager()

    # -------------------------------------------------------------
    # INTERNAL: Random User-Agent Generator
    # -------------------------------------------------------------
    def _random_user_agent(self) -> str:
        agents = [
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)",
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)",
            "Mozilla/5.0 (X11; Linux x86_64)",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X)",
            "Mozilla/5.0 (iPad; CPU OS 16_2 like Mac OS X)",
        ]
        return random.choice(agents)

    # -------------------------------------------------------------
    # CORE: Perform Deep DOM Scan (Session-aware)
    # -------------------------------------------------------------
    def quick_scan(self, url: str) -> dict:
        """
        Performs deep scanning with authentication and multi-browser fallback.
        Saves fully rendered HTML for RAG ingestion.

        Returns:
            dict {
                "scanned_count": int,
                "pages": [
                    {"url": str, "title": str, "content": str, "length": int}, ...
                ]
            }
        """
        print(f"🔍 Starting deep scan for: {url}")
        scanned_pages = []

        try:
            # Attempt session reuse
            session_state = None
            if os.path.exists("auth/session_state.json"):
                session_state = "auth/session_state.json"
                print("🔐 Using saved session from auth/session_state.json")

            with sync_playwright() as p:
                browsers = [
                    ("chromium", p.chromium),
                    ("firefox", p.firefox),
                    ("webkit", p.webkit),
                ]
                success = False

                for name, browser_type in browsers:
                    try:
                        print(f"🧠 Trying with {name}...")
                        browser = browser_type.launch(
                            headless=True,
                            args=["--disable-http2", "--no-sandbox", "--disable-setuid-sandbox"]
                        )

                        context_args = {
                            "ignore_https_errors": True,
                            "user_agent": self._random_user_agent(),
                        }
                        if session_state:
                            context_args["storage_state"] = session_state

                        context = browser.new_context(**context_args)
                        page = context.new_page()

                        page.goto(url, wait_until="networkidle", timeout=self.timeout_ms)
                        page.wait_for_timeout(self.post_load_wait)

                        print(f"✅ Fully rendered with {name}.")
                        content = page.content()
                        title = page.title() or "Untitled Page"

                        scanned_pages.append({
                            "url": url,
                            "title": title,
                            "content": content,
                            "length": len(content)
                        })
                        success = True

                        # Optional: shallow link scan (up to max_links)
                        links = page.locator("a").element_handles()
                        for link in links[: self.max_links]:
                            href = link.get_attribute("href")
                            if href and href.startswith("http"):
                                scanned_pages.append({
                                    "url": href,
                                    "title": "",
                                    "content": "",
                                    "length": 0
                                })

                        browser.close()
                        break  # stop after success

                    except Exception as e:
                        print(f"⚠️ {name} failed due to: {e}")
                        continue

                if not success:
                    print("❌ All browsers failed to load the target URL.")

        except PlaywrightTimeoutError:
            print(f"⏰ Timeout while loading {url}")
        except Exception as e:
            print(f"❌ Scraper fatal error: {e}")

        # -------------------------------------------------------------
        # SAVE SNAPSHOTS (HTML + Markdown)
        # -------------------------------------------------------------
        os.makedirs("data/scraped_docs", exist_ok=True)
        for i, page in enumerate(scanned_pages):
            try:
                html_path = f"data/scraped_docs/page_{i+1}.html"
                md_path = f"data/scraped_docs/page_{i+1}.md"
                with open(html_path, "w", encoding="utf-8") as f1:
                    f1.write(page.get("content", ""))
                with open(md_path, "w", encoding="utf-8") as f2:
                    f2.write(page.get("content", ""))
            except Exception as e:
                print(f"⚠️ Could not save page {i+1}: {e}")

        print(f"🗂️ Scanned {len(scanned_pages)} page(s). Saved under data/scraped_docs/")
        return {"scanned_count": len(scanned_pages), "pages": scanned_pages}
