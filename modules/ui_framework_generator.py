# modules/ui_framework_generator.py
"""
UI Framework Generator (Phase 5.4.5 — CI-Ready, Workspace/Shared Caches, Fast Bootstrap)
---------------------------------------------------------------------------------------
Generates an isolated Playwright workspace from an AI-generated plan.

Highlights:
- Structured logging (no prints)
- Stable step IDs (uuid5) for cross-run diffing
- Optional per-case splitting for parallelism
- Env/plan-driven config (headless, baseURL, timeouts, browsers, workers, retries)
- Optional JUnit reporter for CI
- Robust npm bootstrap with workspace-local or shared caches to avoid EACCES in ~/.npm
- Skips reinstall when node_modules/@playwright/test already present
- Atomic JSON writes, deterministic metadata hashing
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple

from jinja2 import Template

from modules.semantic_action_engine import SemanticActionEngine
from modules.auth_manager import AuthManager

# ------------------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------------------
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Defaults (env-overridable)
# ------------------------------------------------------------------------------
ROOT_OUT = Path(os.environ.get("UI_OUT_ROOT", "generated_ui_framework"))
DEFAULT_SESSION_RELPATH = "auth/session_state.json"
PLAYWRIGHT_TEST_VERSION = os.environ.get("PLAYWRIGHT_TEST_VERSION", "^1.40.0")

BROWSERS_ENV = os.environ.get("BROWSERS", "chromium,firefox,webkit")
HEADLESS_DEFAULT = os.environ.get("PW_HEADLESS", "true").strip().lower() == "true"
PW_TEST_TIMEOUT_MS = int(os.environ.get("PW_TEST_TIMEOUT_MS", "45000"))
PW_EXPECT_TIMEOUT_MS = int(os.environ.get("PW_EXPECT_TIMEOUT_MS", "8000"))
PW_WORKERS = os.environ.get("PW_WORKERS")  # e.g. "4"
PW_RETRIES = os.environ.get("PW_RETRIES")  # e.g. "2"
PW_FULLY_PARALLEL = os.environ.get("PW_FULLY_PARALLEL", "false").lower() == "true"
PW_FORBID_ONLY = os.environ.get("PW_FORBID_ONLY", "true").lower() == "true"
PW_ENABLE_JUNIT = os.environ.get("PW_ENABLE_JUNIT", "false").lower() == "true"

NPM_TIMEOUT_SEC = int(os.environ.get("NPM_TIMEOUT_SEC", "360"))
SKIP_INSTALL_DEFAULT = os.environ.get("SKIP_NPM_INSTALL", "false").lower() == "true"

# caches
UI_CACHE_ROOT = Path(os.environ.get("UI_CACHE_ROOT", str(Path.cwd() / ".tooling_cache")))
UI_SHARED_CACHES = os.environ.get("UI_SHARED_CACHES", "true").lower() == "true"

# ------------------------------------------------------------------------------
# Templates
# ------------------------------------------------------------------------------
PLAYWRIGHT_CONFIG_TEMPLATE = """
// @ts-check
const { defineConfig } = require('@playwright/test');
const path = require('path');

const browserMatrix = (process.env.BROWSERS || '{{ browsers_csv }}')
  .split(',')
  .map(b => b.trim())
  .filter(Boolean);

const projects = browserMatrix.map(browserName => ({
  name: browserName,
  use: { browserName }
}));

const reporters = [
  ['html', { outputFolder: path.resolve(__dirname, './reports/playwright'), open: 'never' }],
  ['json', { outputFile: path.resolve(__dirname, './reports/playwright/report.json') }],
  {{ junit_line }}
];

module.exports = defineConfig({
  testDir: './tests',
  timeout: {{ test_timeout }},
  expect: { timeout: {{ expect_timeout }} },
  reporter: reporters.filter(Boolean),
  projects,
  use: {
    baseURL: '{{ base_url }}',
    {{ storage_state_line }}
    trace: 'retain-on-failure',
    video: 'retain-on-failure',
    screenshot: 'only-on-failure',
    headless: {{ headless }},
  },
  {{ workers_line }}
  {{ retries_line }}
  fullyParallel: {{ fully_parallel }},
  forbidOnly: {{ forbid_only }},
});
""".strip()

TEST_TEMPLATE_TS = """
import { test, expect } from '@playwright/test';
import fs from 'fs';
import path from 'path';

const LOG_DIR = path.resolve(process.cwd(), 'reports/step_logs');
if (!fs.existsSync(LOG_DIR)) fs.mkdirSync(LOG_DIR, { recursive: true });

async function logStep(page, stepId, description, status, takeScreenshot = false) {
  const timestamp = new Date().toISOString();
  console.log(`📘 [${timestamp}] [${status}] [STEP ${stepId}] ${description}`);
  fs.writeFileSync(path.join(LOG_DIR, `${stepId}_${status}.txt`), `[${timestamp}] ${status}: ${description}`);
  if (takeScreenshot) {
    try {
      await page.screenshot({ path: path.join(LOG_DIR, `${stepId}_${status}.png`), fullPage: true });
    } catch (e) {
      console.warn('Screenshot failed', e);
    }
  }
}

test.describe('{{ suite_name }} suite', () => {
{% for c in cases %}
  test('{{ c.safe_name }}', async ({ page }) => {
{% for s in c.steps %}
    // ---- STEP {{ s.step_id }} ----
    {
      const stepDesc = `{{ s.step_safe }}`;
      const stepId = '{{ s.step_id }}';
      try {
        {{ s.js_action }}
        await logStep(page, stepId, stepDesc, 'PASS', true);
      } catch (e) {
        console.error(`Step {{ s.step_id }} failed:`, e);
        await logStep(page, stepId, stepDesc, 'FAIL', true);
        // retry once
        try {
          {{ s.js_action }}
          await logStep(page, stepId, stepDesc, 'RETRY_PASS', true);
        } catch (retryErr) {
          console.error(`Retry failed for {{ s.step_id }}:`, retryErr);
          await logStep(page, stepId, stepDesc, 'RETRY_FAIL', true);
        }
      }
    }
{% endfor %}
  });
{% endfor %}
});
""".strip()

TEST_TEMPLATE_JS = TEST_TEMPLATE_TS  # identical semantics; saved as .js

TS_CONFIG = {
    "compilerOptions": {
        "target": "ES2020",
        "module": "commonjs",
        "lib": ["ES2020", "DOM"],
        "strict": False,
        "moduleResolution": "node",
        "esModuleInterop": True,
        "skipLibCheck": True,
        "forceConsistentCasingInFileNames": True,
        "outDir": "dist"
    },
    "include": ["tests/**/*.ts", "pages/**/*.ts"]
}

README_TEMPLATE = """# Generated Playwright Workspace
This workspace was generated by the AI QA Agent.
- Run tests: `npx playwright test`
- Install deps: `npm ci` (if package-lock.json present) or `npm install`
- StorageState (session): {{ session_relpath or 'NOT_PROVIDED' }}

## Notes
- Caches are local to the workspace by default. To share caches across runs (CI):
  export UI_SHARED_CACHES=true
  export UI_CACHE_ROOT="$(pwd)/.tooling_cache"
"""

# ------------------------------------------------------------------------------
# Utilities
# ------------------------------------------------------------------------------
def _escape_for_template(text: str) -> str:
    if not text:
        return ""
    text = text.replace("`", "\\`")
    text = text.replace("\\", "\\\\").replace("'", "\\'").replace('"', '\\"')
    text = re.sub(r"[\r\n]+", " ", text)
    return text.strip()


def _safe_name(s: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9_.-]+", "_", s or "").strip("_")
    return s or "case"


def _stable_step_id(case_name: str, step_text: str) -> str:
    """Stable across runs: deterministic based on content (uuid5)."""
    base = f"{case_name}::{step_text}".strip().lower()
    return str(uuid.uuid5(uuid.NAMESPACE_URL, base))[:8]


def _node_available() -> bool:
    return shutil.which("npm") is not None and shutil.which("npx") is not None


def _atomic_write_json(path: Path, data: dict) -> None:
    """Write JSON atomically to avoid partial files on abrupt exits."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _hash_plan_fragment(obj: Any) -> str:
    """Short SHA1 hash for traceability in workspace metadata."""
    try:
        blob = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    except Exception:
        blob = repr(obj).encode("utf-8", errors="ignore")
    return hashlib.sha1(blob).hexdigest()[:10]


# ------------------------------------------------------------------------------
# Main Class
# ------------------------------------------------------------------------------
class UIFrameworkGenerator:
    """
    Generate a Playwright workspace for UI tests.

    Args:
      plan: structured test plan with suites.ui
      html_cache_dir: folder with scraped HTML snapshots (for semantic locators)
      output_type: 'ts' or 'js' (default 'ts')
      execution_id: optional ID to embed in workspace name
      split_cases: if True, writes one spec per case (better parallelism)
      skip_install: skip npm bootstrap (env override if None)
    """
    def __init__(
        self,
        plan: Dict[str, Any],
        html_cache_dir: str = "data/scraped_docs",
        output_type: str = "ts",
        execution_id: Optional[str] = None,
        split_cases: bool = False,
        skip_install: Optional[bool] = None,
    ):
        assert output_type in ("ts", "js"), "output_type must be 'ts' or 'js'"
        self.plan = plan
        self.semantic_engine = SemanticActionEngine(html_cache_dir=html_cache_dir)
        self.html_cache_dir = Path(html_cache_dir)
        self.output_type = output_type
        self.execution_id = execution_id
        self.split_cases = split_cases
        self.skip_install = SKIP_INSTALL_DEFAULT if skip_install is None else skip_install

    # ---------------------- HTML snapshot load ----------------------
    def _load_latest_html(self) -> str:
        if not self.html_cache_dir.exists():
            logger.warning("html_cache_dir %s not found", self.html_cache_dir)
            return ""
        files = sorted([p for p in self.html_cache_dir.iterdir() if p.suffix in (".html", ".md")])
        if not files:
            return ""
        latest = files[-1]
        try:
            return latest.read_text(encoding="utf-8")
        except Exception:
            logger.exception("Failed to read snapshot %s", latest)
            return ""

    # ---------------------- Session copy ----------------------
    def _copy_session_into_workspace(self, session_src: Optional[str], workspace_dir: Path) -> Optional[str]:
        if not session_src:
            return None
        src = Path(session_src)
        if not src.exists():
            alt = Path.cwd() / session_src
            if alt.exists():
                src = alt
            else:
                logger.warning("session file not found: %s", session_src)
                return None

        target_dir = workspace_dir / "auth"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_path = target_dir / Path(DEFAULT_SESSION_RELPATH).name
        try:
            shutil.copyfile(src, target_path)
            rel = target_path.relative_to(workspace_dir).as_posix()
            return rel
        except Exception:
            logger.exception("Could not copy session into workspace")
            return None

    # ---------------------- Playwright config helpers ----------------------
    def _render_config(
        self,
        base_url: str,
        storage_state_line: str,
        browsers_csv: str,
        headless: bool,
        test_timeout: int,
        expect_timeout: int,
        workers: Optional[int],
        retries: Optional[int],
        fully_parallel: bool,
        forbid_only: bool,
        enable_junit: bool,
    ) -> str:
        junit_line = "['junit', { outputFile: path.resolve(__dirname, './reports/playwright/results.xml') }]"
        if not enable_junit:
            junit_line = "null"

        workers_line = f"workers: {workers}," if workers is not None else ""
        retries_line = f"retries: {retries}," if retries is not None else ""

        return Template(PLAYWRIGHT_CONFIG_TEMPLATE).render(
            base_url=base_url,
            storage_state_line=storage_state_line,
            headless="true" if headless else "false",
            test_timeout=test_timeout,
            expect_timeout=expect_timeout,
            browsers_csv=browsers_csv,
            junit_line=junit_line,
            workers_line=workers_line,
            retries_line=retries_line,
            fully_parallel=str(fully_parallel).lower(),
            forbid_only=str(forbid_only).lower(),
        )

    # ---------------------- JSON writer ----------------------
    def _write_json(self, path: Path, data: dict):
        _atomic_write_json(path, data)

    # ---------------------- NPM bootstrap ----------------------
    def _already_bootstrapped(self, workspace: Path) -> bool:
        node_modules = workspace / "node_modules"
        playwright_pkg = node_modules / "@playwright" / "test"
        return node_modules.exists() and playwright_pkg.exists()

    def _bootstrap_workspace(self, workspace: Path) -> None:
        """
        Installs workspace dependencies using a *local or shared* npm cache and
        Playwright browsers path to avoid EACCES in ~/.npm. Skips if already bootstrapped.
        """
        if self.skip_install:
            logger.info("SKIP_NPM_INSTALL=true → skipping npm bootstrap")
            return
        if not _node_available():
            logger.warning("npm/npx not found on PATH; skipping workspace bootstrap")
            return
        if self._already_bootstrapped(workspace):
            logger.info("node_modules/@playwright/test present → skipping npm install")
            return

        # Choose caches (shared or per-workspace)
        if UI_SHARED_CACHES:
            npm_cache = UI_CACHE_ROOT / "npm-cache"
            pw_browsers = UI_CACHE_ROOT / "pw-browsers"
        else:
            npm_cache = workspace / ".npm-cache"
            pw_browsers = workspace / ".pw-browsers"
        npm_cache.mkdir(parents=True, exist_ok=True)
        pw_browsers.mkdir(parents=True, exist_ok=True)

        # Force npm & Playwright to use our paths
        env = os.environ.copy()
        env["NPM_CONFIG_CACHE"] = str(npm_cache)
        env["npm_config_fund"] = "false"
        env["npm_config_audit"] = "false"
        env["npm_config_loglevel"] = "error"
        env["PLAYWRIGHT_BROWSERS_PATH"] = str(pw_browsers)
        # HOME controls where npx may persist stuff; pin to workspace for isolation
        env["HOME"] = str(workspace)

        try:
            lock = workspace / "package-lock.json"
            if lock.exists():
                subprocess.run(
                    ["npm", "ci", "--no-fund", "--no-audit"],
                    cwd=workspace, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=NPM_TIMEOUT_SEC, env=env
                )
            else:
                subprocess.run(
                    ["npm", "install", "--no-fund", "--no-audit"],
                    cwd=workspace, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=NPM_TIMEOUT_SEC, env=env
                )

            # ensure @playwright/test present (idempotent)
            node_mod_dir = workspace / "node_modules" / "@playwright" / "test"
            if not node_mod_dir.exists():
                subprocess.run(
                    ["npm", "install", "--save-dev", f"@playwright/test@{PLAYWRIGHT_TEST_VERSION}", "--no-fund", "--no-audit"],
                    cwd=workspace, check=False, timeout=NPM_TIMEOUT_SEC, env=env
                )

            # best-effort browser install
            try:
                subprocess.run(
                    ["npx", "playwright", "install", "--with-deps"],
                    cwd=workspace, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    timeout=NPM_TIMEOUT_SEC, env=env
                )
            except Exception:
                logger.debug("playwright browser install had issues (ignored)", exc_info=True)

            logger.info("Workspace bootstrap completed")
        except subprocess.TimeoutExpired:
            logger.warning("npm bootstrap timed out after %ss (continuing)", NPM_TIMEOUT_SEC)
        except Exception:
            logger.exception("npm bootstrap error (continuing)")

    # ---------------------- Main entry ----------------------
    def generate(self) -> Optional[str]:
        """Create the Playwright workspace and return its path (or None if no UI suite)."""
        if "ui" not in self.plan.get("suites", {}):
            logger.info("No UI suite in plan.")
            return None

        base_url = (self.plan.get("project") or "").strip()

        # 1) Acquire/reuse session (best-effort)
        session_src: Optional[str] = None
        try:
            session_src = AuthManager().login_and_save_session()
            if session_src:
                logger.info("Session created/reused at %s", session_src)
        except Exception:
            logger.debug("AuthManager skipped/failed", exc_info=True)

        # 2) Create workspace (timestamp + optional execution_id)
        ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        suffix = f"_{self.execution_id}" if self.execution_id else ""
        workspace = ROOT_OUT / f"run_{ts}{suffix}"
        (workspace / "tests").mkdir(parents=True, exist_ok=True)
        (workspace / "pages").mkdir(parents=True, exist_ok=True)
        (workspace / "reports" / "step_logs").mkdir(parents=True, exist_ok=True)
        (workspace / "reports" / "playwright").mkdir(parents=True, exist_ok=True)

        # 3) Session copy
        session_relpath = self._copy_session_into_workspace(session_src, workspace)
        storage_state_line = (
            f"storageState: process.env.STORAGE_STATE || '{session_relpath}',"
            if session_relpath else ""
        )

        # 4) Render Playwright config (env/plan overrides allowed)
        ui_cfg = (self.plan.get("ui_config") or {})
        browsers_csv = str(ui_cfg.get("browsers") or BROWSERS_ENV)
        headless = bool(ui_cfg.get("headless", HEADLESS_DEFAULT))
        test_timeout = int(ui_cfg.get("test_timeout_ms", PW_TEST_TIMEOUT_MS))
        expect_timeout = int(ui_cfg.get("expect_timeout_ms", PW_EXPECT_TIMEOUT_MS))
        workers = int(ui_cfg["workers"]) if "workers" in ui_cfg else (int(PW_WORKERS) if PW_WORKERS else None)
        retries = int(ui_cfg["retries"]) if "retries" in ui_cfg else (int(PW_RETRIES) if PW_RETRIES else None)
        fully_parallel = bool(ui_cfg.get("fully_parallel", PW_FULLY_PARALLEL))
        forbid_only = bool(ui_cfg.get("forbid_only", PW_FORBID_ONLY))
        enable_junit = bool(ui_cfg.get("enable_junit", PW_ENABLE_JUNIT))

        cfg_text = self._render_config(
            base_url=base_url,
            storage_state_line=storage_state_line,
            browsers_csv=browsers_csv,
            headless=headless,
            test_timeout=test_timeout,
            expect_timeout=expect_timeout,
            workers=workers,
            retries=retries,
            fully_parallel=fully_parallel,
            forbid_only=forbid_only,
            enable_junit=enable_junit,
        )
        (workspace / "playwright.config.js").write_text(cfg_text, encoding="utf-8")

        # 5) Build tests
        page_html = self._load_latest_html()
        suites = self.plan.get("suites", {}).get("ui", [])
        if not isinstance(suites, list):
            logger.warning("plan.suites.ui is not a list; coercing")
            suites = list(suites)

        written_files: List[Path] = []
        if self.split_cases:
            for case in suites:
                prepped_cases = [self._prep_case(case, page_html)]
                rendered, suffix_ext = self._render_tests("ui", prepped_cases)
                file_name = _safe_name(case.get("name") or "ui_case")
                test_file = workspace / "tests" / f"{file_name}.spec.{suffix_ext}"
                test_file.write_text(rendered, encoding="utf-8")
                written_files.append(test_file)
        else:
            prepped_cases = [self._prep_case(c, page_html) for c in suites]
            rendered, suffix_ext = self._render_tests("ui", prepped_cases)
            test_file = workspace / "tests" / f"ui_suite.spec.{suffix_ext}"
            test_file.write_text(rendered, encoding="utf-8")
            written_files.append(test_file)

        # 6) Base page & tsconfig
        if self.output_type == "ts":
            base_page = """\
import { Page, expect } from '@playwright/test';

export class BasePage {
  constructor(public page: Page) {}

  async goto(url: string) {
    await this.page.goto(url);
    await expect(this.page).toHaveTitle(/.*/);
  }

  async verifyTitle() {
    const title = await this.page.title();
    console.log('Title:', title);
    expect(title.length).toBeGreaterThan(0);
  }

  async clickElement(selector: string) {
    await this.page.locator(selector).click();
  }

  async typeInto(selector: string, value: string) {
    await this.page.fill(selector, value);
  }
}
"""
            (workspace / "pages" / "base.page.ts").write_text(base_page, encoding="utf-8")
            self._write_json(workspace / "tsconfig.json", TS_CONFIG)
        else:
            base_page_js = """\
/* Simple JS helpers (playwright test) */
exports.BasePage = class BasePage {
  constructor(page) { this.page = page; }
  async goto(url) { await this.page.goto(url); }
  async verifyTitle() { const t = await this.page.title(); console.log('Title', t); }
};
"""
            (workspace / "pages" / "base.page.js").write_text(base_page_js, encoding="utf-8")

        # 7) package.json (+ optional lock)
        pkg = {
            "name": workspace.name,
            "version": "1.0.0",
            "description": "AI-generated Playwright workspace",
            "scripts": {"test": "npx playwright test"},
            "devDependencies": {"@playwright/test": PLAYWRIGHT_TEST_VERSION},
            "type": "commonjs",
        }
        self._write_json(workspace / "package.json", pkg)

        # 8) README & metadata (incl. plan hash for traceability)
        (workspace / "README.md").write_text(
            Template(README_TEMPLATE).render(session_relpath=session_relpath),
            encoding="utf-8",
        )
        meta = {
            "generated_at": datetime.utcnow().isoformat() + "Z",
            "project": base_url,
            "session_relpath": session_relpath,
            "output_type": self.output_type,
            "workspace": str(workspace),
            "split_cases": self.split_cases,
            "browsers": browsers_csv,
            "workers": workers,
            "retries": retries,
            "fully_parallel": fully_parallel,
            "forbid_only": forbid_only,
            "plan_hash_ui": _hash_plan_fragment(suites),
            "shared_caches": UI_SHARED_CACHES,
            "cache_root": str(UI_CACHE_ROOT) if UI_SHARED_CACHES else str(workspace / ".npm-cache"),
        }
        self._write_json(workspace / "workspace_metadata.json", meta)

        # 9) Bootstrap (best-effort) — with local/shared caches to avoid EACCES
        self._bootstrap_workspace(workspace)

        logger.info("✅ Generated Playwright Framework → %s", workspace)
        for f in written_files:
            logger.info("   • Test file: %s", f)
        logger.info("   • Reports:   %s", workspace / 'reports')
        logger.info("   • Session:   %s", session_relpath or "NONE")

        return str(workspace)

    # ---------------------- Helpers ----------------------
    def _prep_case(self, case: Dict[str, Any], page_html: str) -> Dict[str, Any]:
        """Return a case object with safe name and enriched steps."""
        name = str(case.get("name") or "UI Case").strip()
        safe_name = _safe_name(name)
        steps_raw: List[str] = list(case.get("steps", []))
        steps_out: List[Dict[str, Any]] = []
        for step_str in steps_raw:
            safe = _escape_for_template(step_str)
            js_action = self.semantic_engine.generate_js_action(step_str, page_html)
            steps_out.append({
                "step_id": _stable_step_id(safe_name, step_str),
                "step": step_str,
                "step_safe": safe,
                "js_action": js_action,
            })
        return {"name": name, "safe_name": safe_name, "steps": steps_out}

    def _render_tests(self, suite_name: str, cases: List[Dict[str, Any]]) -> Tuple[str, str]:
        """Render tests as TS or JS; returns (source, suffix)."""
        tpl = TEST_TEMPLATE_TS if self.output_type == "ts" else TEST_TEMPLATE_JS
        rendered = Template(tpl).render(suite_name=suite_name, cases=cases)
        return rendered, ("ts" if self.output_type == "ts" else "js")
