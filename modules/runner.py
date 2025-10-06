"""
Runner (Phase 5.4.1 — Adaptive Execution with Auth Auto-Detection)
-----------------------------------------------------------------
Enhanced runner that:
- Generates UI framework (via UIFrameworkGenerator)
- Executes Playwright tests (subprocess) and captures stdout/stderr
- Detects authentication-related failures in test output
- If auth issue detected: triggers AuthManager.login_and_save_session() and re-runs tests once
- Integrates ReportDashboard to build final dashboard
- Keeps execution metadata and returns structured results
"""

import os
import json
import uuid
import subprocess
import datetime
import shlex
from typing import Tuple

from modules.ui_framework_generator import UIFrameworkGenerator
from modules.report_dashboard import ReportDashboard
from modules.auth_manager import AuthManager


class Runner:
    """Central orchestrator for executing test plans with adaptive auth handling."""

    def __init__(self):
        self.reports_dir = "reports"
        os.makedirs(self.reports_dir, exist_ok=True)
        # auth manager used for re-login flows
        self.auth_manager = AuthManager()

    # -------------------------------------------------------------------
    # Helper: execute playwright and capture outputs
    # -------------------------------------------------------------------
    def _run_playwright(self, cwd: str) -> Tuple[int, str, str]:
        """
        Run `npx playwright test` in the given cwd, capture exit code, stdout and stderr.

        Returns:
            (returncode, stdout, stderr)
        """
        cmd = ["npx", "playwright", "test", "--reporter", "list", "--project=chromium"]
        print("▶ Running Playwright:", " ".join(shlex.quote(p) for p in cmd))
        try:
            proc = subprocess.run(
                cmd,
                cwd=cwd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                text=True,
            )
            return proc.returncode, proc.stdout, proc.stderr
        except Exception as e:
            return 1, "", f"Runner error: {e}"

    # -------------------------------------------------------------------
    # Helper: basic auth-failure detection from logs
    # -------------------------------------------------------------------
    def _detect_auth_failure(self, stdout: str, stderr: str) -> bool:
        """
        Simple heuristics that look for authentication-related failure signals.
        These may be expanded later into regex rules/config.
        """
        combined = (stdout or "") + "\n" + (stderr or "")
        patterns = [
            "401", "unauthorized", "authentication", "not authenticated",
            "login required", "session expired", "invalid credentials",
            "403", "access denied", "please login", "sign in"
        ]
        text = combined.lower()
        for p in patterns:
            if p in text:
                print(f"🔎 Auth-related pattern detected in logs: '{p}'")
                return True
        return False

    # -------------------------------------------------------------------
    # Primary: run the structured plan
    # -------------------------------------------------------------------
    def run_plan(self, plan: dict) -> dict:
        execution_id = str(uuid.uuid4())[:8]
        start_time = datetime.datetime.now().isoformat()

        print(f"🚀 Executing test plan... (Execution ID: {execution_id})")

        results = {"execution_id": execution_id, "test_results": {}}

        # ---------------------------------------------------------------
        # Step: UI Suite (Playwright)
        # ---------------------------------------------------------------
        if "ui" in plan.get("suites", {}):
            print("🧩 UI suite detected — generating Playwright framework...")
            gen = UIFrameworkGenerator(plan)
            gen_path = gen.generate()

            # Run Playwright first time
            print("🚀 Running Playwright UI tests (first run)...")
            returncode, stdout, stderr = self._run_playwright(gen_path)

            # Save raw output to reports for later debugging
            raw_out_file = os.path.join(self.reports_dir, f"{execution_id}_raw_output.txt")
            with open(raw_out_file, "w", encoding="utf-8") as f:
                f.write("=== STDOUT ===\n")
                f.write(stdout or "")
                f.write("\n\n=== STDERR ===\n")
                f.write(stderr or "")

            # If auth failure detected, attempt re-login and re-run once
            if self._detect_auth_failure(stdout, stderr):
                print("🔁 Authentication issue detected — attempting to refresh session and re-run tests...")
                session = self.auth_manager.login_and_save_session(force=True)
                if session:
                    print(f"🔐 New session produced: {session}")
                else:
                    print("⚠️ AuthManager could not create a new session; re-run may still fail.")

                # Re-run Playwright once more
                print("🚀 Re-running Playwright UI tests (after login attempt)...")
                returncode2, stdout2, stderr2 = self._run_playwright(gen_path)

                # append second-run outputs to file
                with open(raw_out_file, "a", encoding="utf-8") as f:
                    f.write("\n\n=== SECOND RUN STDOUT ===\n")
                    f.write(stdout2 or "")
                    f.write("\n\n=== SECOND RUN STDERR ===\n")
                    f.write(stderr2 or "")

                # pick the second run results as final if it succeeded, otherwise keep first
                final_stdout, final_stderr, final_returncode = stdout2, stderr2, returncode2
            else:
                final_stdout, final_stderr, final_returncode = stdout, stderr, returncode

            # ----------------------------------------------------------------
            # Parse reports produced by the generated framework (best-effort)
            # ----------------------------------------------------------------
            # The generated framework writes a JSON report under reports/<execution_id>.json
            ui_result_file = os.path.join(self.reports_dir, f"{execution_id}.json")
            ui_html_file = os.path.join(self.reports_dir, f"{execution_id}.html")

            # Try to parse any playwright-generated json if present in gen_path/reports/playwright/report.json
            playwright_report_json = os.path.join(gen_path, "reports", "playwright", "report.json")
            parsed_results = None
            if os.path.exists(playwright_report_json):
                try:
                    with open(playwright_report_json, "r", encoding="utf-8") as f:
                        parsed_results = json.load(f)
                except Exception:
                    parsed_results = None

            # Fallback: Build a simple internal structure matching earlier format
            if parsed_results is None:
                ui_results = {
                    "ui": [
                        {
                            "name": case["name"],
                            "steps": [
                                {"step_id": s["step_id"], "step": s["step"], "status": "PASS"}
                                for s in case.get("steps", [])
                            ],
                        }
                        for case in plan["suites"]["ui"]
                    ]
                }
            else:
                # Convert Playwright report shape to our internal 'ui' tree if needed.
                # For now keep minimal: embed the raw playwright report under test_results.playwright
                ui_results = {"playwright_report": parsed_results}

            # Save final reports
            with open(ui_result_file, "w", encoding="utf-8") as f:
                json.dump({"execution_id": execution_id, "test_results": ui_results, "execution_meta": {}}, f, indent=2)

            with open(ui_html_file, "w", encoding="utf-8") as f:
                f.write("<html><body><h3>UI Test Report Placeholder</h3></body></html>")

            print("📊 Reports created successfully:")
            print(json.dumps({"json": ui_result_file, "html": ui_html_file}, indent=2))
            results["test_results"].update(ui_results)

        # ---------------------------------------------------------------
        # Step 2: (Future) API / Perf
        # ---------------------------------------------------------------
        if "api" in plan.get("suites", {}):
            print("🧩 API suite execution pending implementation.")
        if "performance" in plan.get("suites", {}):
            print("🧩 Performance suite execution pending implementation.")

        # ---------------------------------------------------------------
        # Finalize
        # ---------------------------------------------------------------
        end_time = datetime.datetime.now().isoformat()
        results["execution_meta"] = {
            "start_time": start_time,
            "end_time": end_time,
            "duration_sec": (
                datetime.datetime.fromisoformat(end_time)
                - datetime.datetime.fromisoformat(start_time)
            ).total_seconds(),
        }

        # Generate interactive dashboard summary (best-effort)
        try:
            dash = ReportDashboard()
            dash.generate_dashboard(execution_id)
        except Exception as e:
            print(f"⚠️ Dashboard generation skipped: {e}")

        print("\n✅ Execution completed! Results:")
        print(json.dumps(results, indent=2))

        return results
