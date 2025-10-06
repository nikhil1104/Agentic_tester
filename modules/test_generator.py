"""
Test Generator (Phase 4.7 – Granular Dropdown & Dynamic Field Coverage)
-----------------------------------------------------------------------
This module generates a full, structured AI-driven test plan using:
 - User intent (UI/API/Performance)
 - Deep DOM extraction via SemanticActionEngine

🚀 New in this version:
1. Dropdown & Language elements now generate **per-option verification steps**
   → ensures every selectable option is validated.
2. Detects dynamic dropdowns (AJAX-loaded or missing <option>) and adds
   fallback verification steps to assert they load properly.
3. Smarter page grouping and naming — one suite per page, modular for future multi-page expansion.
4. Still backward-compatible with previous Phase 4.6.

Example output:
{
  "project": "https://site.com",
  "suites": {
    "ui": [
      {"name": "dropdown – language verification", "steps": [
         "verify dropdown 'Language' includes ['English','French']",
         "verify 'English' option visible",
         "verify 'French' option visible"
      ]}
    ]
  }
}
"""

import os
import json
from typing import Dict, Any, List
from modules.semantic_action_engine import SemanticActionEngine


class TestGenerator:
    """
    Generates intelligent, multi-suite test plans using scanned HTML
    and SemanticActionEngine’s DOM analysis.
    """

    def __init__(self, html_cache_dir: str = "data/scraped_docs"):
        self.engine = SemanticActionEngine(html_cache_dir=html_cache_dir)
        self.html_cache_dir = html_cache_dir

    # ======================================================================
    # 🧠 CORE FUNCTION: Generate Test Plan
    # ======================================================================
    def generate_plan(self, req: Dict[str, Any], scan_res: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a structured test plan from DOM analysis.
        Args:
            req: parsed requirement dict (includes "intent" + "details.url")
            scan_res: results from WebScraper.quick_scan()
        Returns:
            dict → structured plan {"project": url, "suites": {...}}
        """
        project_url = req.get("details", {}).get("url", "").strip()
        intent = req.get("intent", [])

        print("🧠 Building intelligent test plan from DOM and scanned data...")

        # Try to load last saved HTML snapshot for analysis
        try:
            html = self.engine._load_latest_html()
        except Exception as e:
            print(f"⚠️ Could not load cached HTML: {e}")
            html = None

        # If HTML loaded, extract testable elements
        parsed = self.engine.extract_testable_elements(html) if html else {
            "inputs": [], "selects": [], "buttons": [],
            "links": [], "images": [], "headings": [],
            "languages": [], "custom_clickables": []
        }

        suites: Dict[str, List[Dict[str, Any]]] = {}

        # ===============================================================
        # 🧩 UI Testing
        # ===============================================================
        if "ui_testing" in intent or "ui" in intent:
            ui_suites = []

            # -----------------------------------------------------------
            # 1️⃣ Form Field Tests
            # -----------------------------------------------------------
            if parsed.get("inputs"):
                steps = [
                    f"type sample data into '{inp.get('name') or inp.get('id') or inp.get('placeholder') or 'input'}' field"
                    for inp in parsed["inputs"]
                ]
                if steps:
                    ui_suites.append({
                        "name": "form fields – input typing & validation",
                        "steps": steps
                    })

            # -----------------------------------------------------------
            # 2️⃣ Dropdowns (Granular Option Checks)
            # -----------------------------------------------------------
            if parsed.get("selects"):
                for sel in parsed["selects"]:
                    label = sel.get("name") or sel.get("id") or "dropdown"
                    opts = sel.get("options", [])
                    steps = []

                    # Base dropdown check
                    if opts:
                        opts_str = ", ".join([f"'{o}'" for o in opts])
                        steps.append(f"verify dropdown '{label}' includes [{opts_str}]")

                        # Individual option validation
                        for o in opts:
                            steps.append(f"verify '{o}' option visible in dropdown '{label}'")

                    else:
                        # Dynamic dropdown fallback
                        steps.append(f"verify dropdown '{label}' dynamically loads options at runtime")

                    ui_suites.append({
                        "name": f"dropdown – {label} verification",
                        "steps": steps
                    })

            # -----------------------------------------------------------
            # 3️⃣ Language Selectors (Special Handling)
            # -----------------------------------------------------------
            if parsed.get("languages"):
                for lang in parsed["languages"]:
                    label = lang.get("name") or lang.get("text") or "language"
                    opts = lang.get("options", [])
                    steps = []
                    if opts:
                        steps.append(f"verify language dropdown '{label}' includes {opts}")
                        for o in opts:
                            steps.append(f"verify '{o}' language option visible")
                    else:
                        steps.append(f"verify '{label}' language selector is visible and clickable")
                    ui_suites.append({
                        "name": f"language selector – {label}",
                        "steps": steps
                    })

            # -----------------------------------------------------------
            # 4️⃣ Buttons and Clickables
            # -----------------------------------------------------------
            click_steps = []
            for btn in parsed.get("buttons", []):
                text = btn.get("text") or "button"
                click_steps.append(f"click '{text}' button")
            for c in parsed.get("custom_clickables", []):
                text = c.get("text") or "clickable"
                click_steps.append(f"click '{text}' element")
            if click_steps:
                ui_suites.append({
                    "name": "buttons & clickables – interactions",
                    "steps": click_steps
                })

            # -----------------------------------------------------------
            # 5️⃣ Headings Verification
            # -----------------------------------------------------------
            if parsed.get("headings"):
                head_texts = parsed["headings"]
                if head_texts:
                    ui_suites.append({
                        "name": "headings – visibility checks",
                        "steps": [f"verify headings visible: {head_texts}"]
                    })

            # -----------------------------------------------------------
            # 6️⃣ Image Accessibility
            # -----------------------------------------------------------
            if parsed.get("images"):
                img_steps = []
                for img in parsed["images"]:
                    alt = img.get("alt")
                    if alt:
                        img_steps.append(f"verify image '{alt}' has alt text")
                    else:
                        img_steps.append("verify image without alt is handled gracefully")
                ui_suites.append({
                    "name": "images – accessibility & alt validation",
                    "steps": img_steps
                })

            # -----------------------------------------------------------
            # 7️⃣ Links
            # -----------------------------------------------------------
            if parsed.get("links"):
                link_steps = []
                for link in parsed["links"]:
                    text = link.get("text") or "link"
                    href = link.get("href") or ""
                    if href:
                        link_steps.append(f"verify link '{text}' navigates to '{href}'")
                    else:
                        link_steps.append(f"click '{text}' link")
                if link_steps:
                    ui_suites.append({
                        "name": "links – href verification",
                        "steps": link_steps
                    })

            # -----------------------------------------------------------
            # 8️⃣ Smoke Suite (always added)
            # -----------------------------------------------------------
            ui_suites.append({
                "name": "smoke – page title visible",
                "steps": [f"goto {project_url}", "verify page title is visible"]
            })

            suites["ui"] = ui_suites

        # ===============================================================
        # 🔗 API / Performance Placeholders
        # ===============================================================
        if "api_testing" in intent:
            suites["api"] = [{
                "name": "API Health Check",
                "steps": ["GET /health", "expect status 200"]
            }]

        if "performance_testing" in intent:
            suites["performance"] = [{
                "name": "Performance Baseline",
                "steps": ["measure page load time", "assert < 3s"]
            }]

        # ===============================================================
        # ✅ Final Plan
        # ===============================================================
        plan = {"project": project_url, "suites": suites}
        print(f"✅ Deep test plan generated: {len(suites.get('ui', []))} UI suites.")
        return plan
