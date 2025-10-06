"""
Demo Runner (Phase 2 – Final, Clean Architecture)
-------------------------------------------------
Runs the complete AI QA Agent demo pipeline:
1. Conversational parsing of user request
2. Website quick scan (multi-browser support)
3. AI-driven test plan generation
4. Automated Playwright framework generation + execution
5. Report creation (HTML + JSON)

Future-ready for:
- RAG-based DOM enrichment
- Voice/MCP interface
- Self-healing test actions
"""

import json
import os
from modules.conversational_agent import ConversationalAgent
from modules.web_scraper import WebScraper
from modules.test_generator import TestGenerator
from modules.runner import Runner


def demo():
    print("\n🧠 Welcome to the AI QA Agent Demo!")
    print("Enter a requirement (e.g. 'Test https://example.com login and checkout'):")
    user_input = input().strip()

    # --- Initialize core modules ---
    agent = ConversationalAgent()
    scraper = WebScraper()
    generator = TestGenerator()
    runner = Runner()

    # --- Step 1: Parse the requirement into structured intent ---
    print("\n🤖 Understanding your requirement...")
    req = agent.parse_requirement(user_input)
    print("Structured requirement:\n", json.dumps(req, indent=2))

    # --- Step 2: Run quick website scan ---
    print("\n🔍 Running quick scan...")
    res = scraper.quick_scan(req["details"]["url"])

    # Display scan summary
    print(f"\n🕸️ Scanned {res['scanned_count']} page(s):")
    for p in res["pages"]:
        print(f"  → {p['url']}")

    # --- Step 3: Generate AI-driven test plan ---
    print("\n🧩 Generating test plan...")
    plan = generator.generate_plan(req, res)
    print("Generated plan:\n", json.dumps(plan, indent=2))

    # --- Step 4: Execute the generated plan ---
    print("\n🚀 Executing test plan...")
    results = runner.run_plan(plan)
    print("\n✅ Execution completed! Results:\n", json.dumps(results, indent=2))

    # --- Step 5: Generate and save reports ---
    os.makedirs("reports", exist_ok=True)
    json_report = f"reports/{results['execution_id']}.json"
    html_report = f"reports/{results['execution_id']}.html"

    with open(json_report, "w") as f:
        json.dump(results, f, indent=2)

    with open(html_report, "w") as f:
        f.write("<html><body><pre>" + json.dumps(results, indent=2) + "</pre></body></html>")

    print("\n📊 Reports created successfully:")
    print(json.dumps({"json": json_report, "html": html_report}, indent=2))

    print("\n🎯 Demo finished. You can open the Playwright HTML report for UI details.\n")


if __name__ == "__main__":
    demo()
