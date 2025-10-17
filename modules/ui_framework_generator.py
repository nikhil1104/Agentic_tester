# modules/ui_framework_generator.py
"""
UI Framework Generator (Phase 5.3) — Production-Ready

Generates a Playwright (TypeScript) test framework with:
- Smoke tests (happy path)
- Negative tests
- Edge-case tests
- Boundary value analysis
- Data-driven tests
- Page Object Model (POM)
- Test utilities and custom fixtures
- Reporters (list, html, json, junit)
- CI/CD (GitHub Actions)
- Artifacts directories (screenshots/videos/traces)
- Project scaffolding with tsconfig, package.json, .gitignore, README

Hardening & Best Practices:
- Safe multi-line content builders (avoid accidental triple-quote issues)
- Robust baseURL extraction using urllib.parse
- UTF-8 writes for all files
- Type-safe JSON emission via json.dumps
- Defensive guards and structured logging
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

logger = logging.getLogger(__name__)

# ==================== Data Models ====================

@dataclass
class TestCase:
    """Enhanced test case with full metadata"""
    name: str
    type: str  # "smoke", "negative", "edge", "boundary", "performance"
    priority: str  # "P0", "P1", "P2", "P3"
    steps: List[str]
    expected_result: str
    preconditions: Optional[List[str]] = None
    test_data: Optional[Dict[str, Any]] = None
    tags: Optional[List[str]] = None
    timeout_ms: Optional[int] = None
    retry_count: Optional[int] = None


@dataclass
class PageObjectModel:
    """Page Object Model definition"""
    name: str
    url: str
    selectors: Dict[str, str]
    methods: List[Dict[str, Any]]


@dataclass
class TestSuite:
    """Complete test suite definition"""
    name: str
    description: str
    tests: List[TestCase] = field(default_factory=list)
    page_objects: List[PageObjectModel] = field(default_factory=list)
    shared_data: Optional[Dict[str, Any]] = None
    hooks: Optional[Dict[str, List[str]]] = None


# ==================== Main Generator ====================

class UIFrameworkGenerator:
    """
    Production-grade Playwright framework generator.
    """

    def __init__(self, plan: Dict[str, Any]):
        self.plan = plan
        self.project_name = (plan.get("project") or "test_project").strip().replace("://", "_").replace("/", "_")
        self.base_url = self._extract_base_url(plan)
        self.ui_config = plan.get("ui_config", {})
        self.suites = plan.get("suites", {}).get("ui", [])
        self.scan_data = plan.get("scan_data", {})

        # Feature flags (keep existing functionality)
        self.generate_page_objects = True
        self.generate_fixtures = True
        self.generate_utils = True
        self.generate_data_files = True

    # ==================== Helpers ====================

    def _extract_base_url(self, plan: Dict[str, Any]) -> str:
        """Extract base URL from stable sources; fall back sanely."""
        # 1) plan["project"] if it looks like a URL
        proj = plan.get("project")
        if isinstance(proj, str) and proj.startswith(("http://", "https://")):
            return self._origin(proj)

        # 2) plan["base_url"]
        if isinstance(plan.get("base_url"), str):
            return self._origin(plan["base_url"])

        # 3) scan_data["start_url"]
        start = plan.get("scan_data", {}).get("start_url")
        if isinstance(start, str):
            return self._origin(start)

        # 4) search within UI suite steps for the first URL, reduce to origin
        for suite in plan.get("suites", {}).get("ui", []):
            for step in suite.get("steps", []):
                urls = re.findall(r'https?://[^\s\'"]+', str(step))
                if urls:
                    return self._origin(urls[0])

        # 5) default local
        return "http://localhost:3000"

    def _origin(self, url: str) -> str:
        """Return the scheme://host[:port] portion of a URL, fallback to the url if parsing fails."""
        try:
            p = urlparse(url)
            if p.scheme and p.netloc:
                return f"{p.scheme}://{p.netloc}"
        except Exception:
            pass
        return url

    # ==================== Entry ====================

    def generate(self) -> Optional[str]:
        """
        Generate complete Playwright framework.

        Returns:
            str: Path to generated workspace, or None on failure
        """
        workspace = Path(f"generated_frameworks/{self.project_name}_ui")
        workspace.mkdir(parents=True, exist_ok=True)

        logger.info(f"🏗️ Generating framework at {workspace} (baseURL={self.base_url})")

        try:
            # 1. Setup structure
            self._setup_structure(workspace)

            # 2. Generate suites in full
            full_suites = self._generate_full_test_suites(self.suites)

            # 3. Write tests
            for suite in full_suites:
                self._write_test_file(workspace, suite)

            # 4. POM
            if self.generate_page_objects:
                self._generate_page_objects(workspace, full_suites)

            # 5. Utils
            if self.generate_utils:
                self._generate_utils(workspace)

            # 6. Fixtures
            if self.generate_fixtures:
                self._generate_fixtures(workspace)

            # 7. Data
            if self.generate_data_files:
                self._generate_test_data(workspace, full_suites)

            # 8. Config & metadata
            self._write_playwright_config(workspace)
            self._write_package_json(workspace)
            self._write_tsconfig(workspace)
            self._write_gitignore(workspace)
            self._write_editorconfig(workspace)
            self._write_nvmrc(workspace)
            self._write_readme(workspace)

            # 9. CI workflows
            self._generate_github_actions(workspace)

            logger.info(f"✅ Framework generated successfully at {workspace}")
            return str(workspace)

        except Exception as e:
            logger.error(f"❌ Framework generation failed: {e}", exc_info=True)
            return None

    # ==================== Structure ====================

    def _setup_structure(self, workspace: Path) -> None:
        """Create complete project structure"""
        directories = [
            "tests",
            "pages",
            "utils",
            "fixtures",
            "data",
            "reports",
            "reports/screenshots",
            "reports/videos",
            "reports/traces",
            "reports/playwright",
            "reports/junit",
            ".github/workflows",
        ]
        for d in directories:
            (workspace / d).mkdir(parents=True, exist_ok=True)

    # ==================== Suite Generation ====================

    def _generate_full_test_suites(self, suites: List[Dict[str, Any]]) -> List[TestSuite]:
        """Generate smoke, negative, edge, boundary tests for each suite."""
        full_suites: List[TestSuite] = []

        for suite_data in suites:
            suite_name = suite_data.get("name", "test_suite")
            suite_desc = suite_data.get("description", "")
            steps = suite_data.get("steps", []) or []
            priority = suite_data.get("priority", "P1")
            tags = suite_data.get("tags", []) or []

            suite = TestSuite(name=suite_name, description=suite_desc)

            # 1. Smoke
            suite.tests.extend(self._generate_smoke_tests(suite_name, steps, priority, tags))
            # 2. Negative
            suite.tests.extend(self._generate_negative_tests(suite_name, steps, tags))
            # 3. Edge
            suite.tests.extend(self._generate_edge_cases(suite_name, steps, tags))
            # 4. Boundary
            suite.tests.extend(self._generate_boundary_tests(suite_name, steps, tags))

            # POM extraction
            if self.generate_page_objects:
                suite.page_objects.extend(self._extract_page_objects(suite_name, steps))

            full_suites.append(suite)

        total_tests = sum(len(s.tests) for s in full_suites)
        logger.info(f"📦 Generated {len(full_suites)} test suites with {total_tests} total tests")
        return full_suites

    # ---- Smoke ----

    def _generate_smoke_tests(self, suite_name: str, steps: List[str], priority: str, tags: List[str]) -> List[TestCase]:
        tests: List[TestCase] = []

        for step in steps:
            step_lower = step.lower()

            if any(kw in step_lower for kw in ["login", "sign in", "signin"]):
                tests.append(TestCase(
                    name="test_login_successful",
                    type="smoke",
                    priority="P0",
                    steps=[
                        "Navigate to login page",
                        "Fill username with valid data",
                        "Fill password with valid data",
                        "Click submit button",
                        "Wait for navigation to complete",
                        "Verify user is logged in (dashboard visible)"
                    ],
                    expected_result="User successfully logged in and redirected to dashboard",
                    test_data={"username": "testuser@example.com", "password": "Test@123456"},
                    tags=tags + ["login", "authentication"]
                ))

            elif any(kw in step_lower for kw in ["register", "signup", "sign up"]):
                tests.append(TestCase(
                    name="test_registration_successful",
                    type="smoke",
                    priority="P0",
                    steps=[
                        "Navigate to registration page",
                        "Fill email with valid format",
                        "Fill password meeting requirements",
                        "Fill confirm password (matching)",
                        "Accept terms and conditions",
                        "Click register button",
                        "Verify success message or redirect"
                    ],
                    expected_result="User successfully registered",
                    test_data={
                        "email": "newuser@example.com",
                        "password": "SecurePass@123",
                        "confirm_password": "SecurePass@123",
                        "first_name": "Test",
                        "last_name": "User"
                    },
                    tags=tags + ["registration", "signup"]
                ))

            elif "search" in step_lower:
                tests.append(TestCase(
                    name="test_search_with_results",
                    type="smoke",
                    priority="P1",
                    steps=[
                        "Navigate to search page",
                        "Enter valid search query",
                        "Click search button or press Enter",
                        "Wait for results to load",
                        "Verify results are displayed",
                        "Verify result count is shown"
                    ],
                    expected_result="Search results displayed successfully",
                    test_data={"query": "test product"},
                    tags=tags + ["search"]
                ))

            elif any(kw in step_lower for kw in ["add to cart", "cart", "add item"]):
                tests.append(TestCase(
                    name="test_add_to_cart_successful",
                    type="smoke",
                    priority="P0",
                    steps=[
                        "Navigate to product page",
                        "Verify product details are visible",
                        "Click 'Add to Cart' button",
                        "Wait for confirmation",
                        "Verify cart count increases",
                        "Navigate to cart",
                        "Verify product is in cart"
                    ],
                    expected_result="Product successfully added to cart",
                    tags=tags + ["cart", "ecommerce"]
                ))

            elif any(kw in step_lower for kw in ["checkout", "purchase", "payment"]):
                tests.append(TestCase(
                    name="test_checkout_complete_flow",
                    type="smoke",
                    priority="P0",
                    steps=[
                        "Ensure user is logged in",
                        "Ensure cart has items",
                        "Navigate to checkout",
                        "Fill shipping information",
                        "Select shipping method",
                        "Fill payment information (test card)",
                        "Review order details",
                        "Click place order",
                        "Verify order confirmation page"
                    ],
                    expected_result="Order placed successfully",
                    preconditions=["User must be logged in", "Cart must have at least one item"],
                    test_data={
                        "shipping": {"address": "123 Test St", "city": "Test City", "zip": "12345"},
                        "payment": {"card_number": "4242424242424242", "expiry": "12/25", "cvv": "123"}
                    },
                    tags=tags + ["checkout", "payment", "ecommerce"],
                    timeout_ms=60000
                ))

            elif any(kw in step_lower for kw in ["submit", "form", "contact"]):
                tests.append(TestCase(
                    name="test_form_submission_successful",
                    type="smoke",
                    priority="P1",
                    steps=[
                        "Navigate to form page",
                        "Fill all required fields with valid data",
                        "Click submit button",
                        "Wait for submission to complete",
                        "Verify success message displayed"
                    ],
                    expected_result="Form submitted successfully",
                    test_data={"name": "Test User", "email": "test@example.com", "message": "This is a test message"},
                    tags=tags + ["form"]
                ))

        return tests

    # ---- Negative ----

    def _generate_negative_tests(self, suite_name: str, steps: List[str], tags: List[str]) -> List[TestCase]:
        tests: List[TestCase] = []

        for step in steps:
            step_lower = step.lower()

            if "login" in step_lower:
                tests.append(TestCase(
                    name="test_login_invalid_credentials",
                    type="negative",
                    priority="P1",
                    steps=[
                        "Navigate to login page",
                        "Fill username with invalid value",
                        "Fill password with invalid value",
                        "Click submit",
                        "Verify error message is displayed",
                        "Verify user remains on login page"
                    ],
                    expected_result="Error message: 'Invalid username or password'",
                    test_data={"username": "invalid@example.com", "password": "wrongpassword"},
                    tags=tags + ["login", "negative"]
                ))

                tests.append(TestCase(
                    name="test_login_empty_fields",
                    type="negative",
                    priority="P1",
                    steps=[
                        "Navigate to login page",
                        "Leave username field empty",
                        "Leave password field empty",
                        "Click submit",
                        "Verify validation errors are shown",
                        "Verify submit button is disabled or errors prevent submission"
                    ],
                    expected_result="Validation errors displayed for empty fields",
                    test_data={"username": "", "password": ""},
                    tags=tags + ["login", "validation", "negative"]
                ))

                tests.append(TestCase(
                    name="test_login_sql_injection",
                    type="negative",
                    priority="P1",
                    steps=[
                        "Navigate to login page",
                        "Fill username with SQL injection payload",
                        "Fill password with SQL injection payload",
                        "Click submit",
                        "Verify system handles injection safely",
                        "Verify error message or login failure"
                    ],
                    expected_result="SQL injection blocked, error shown",
                    test_data={"username": "' OR '1'='1", "password": "' OR '1'='1"},
                    tags=tags + ["login", "security", "negative"]
                ))

            elif any(k in step_lower for k in ["register", "signup"]):
                tests.append(TestCase(
                    name="test_registration_duplicate_email",
                    type="negative",
                    priority="P1",
                    steps=[
                        "Navigate to registration page",
                        "Fill email with already registered address",
                        "Fill other fields with valid data",
                        "Click register",
                        "Verify duplicate email error is shown"
                    ],
                    expected_result="Error: 'Email already registered'",
                    test_data={"email": "existing@example.com", "password": "Test@123"},
                    tags=tags + ["registration", "negative"]
                ))

                tests.append(TestCase(
                    name="test_registration_password_mismatch",
                    type="negative",
                    priority="P1",
                    steps=[
                        "Navigate to registration page",
                        "Fill password field",
                        "Fill confirm_password with different value",
                        "Click register",
                        "Verify mismatch error is displayed"
                    ],
                    expected_result="Error: 'Passwords do not match'",
                    test_data={"password": "Test@123", "confirm_password": "Different@456"},
                    tags=tags + ["registration", "validation", "negative"]
                ))

                tests.append(TestCase(
                    name="test_registration_invalid_email",
                    type="negative",
                    priority="P1",
                    steps=[
                        "Navigate to registration page",
                        "Fill email with invalid format",
                        "Fill other fields with valid data",
                        "Attempt to register",
                        "Verify email format error"
                    ],
                    expected_result="Error: 'Invalid email format'",
                    test_data={"email": "notanemail", "password": "Test@123"},
                    tags=tags + ["registration", "validation", "negative"]
                ))

            elif "search" in step_lower:
                tests.append(TestCase(
                    name="test_search_no_results",
                    type="negative",
                    priority="P2",
                    steps=[
                        "Navigate to search page",
                        "Enter query that returns no results",
                        "Click search",
                        "Verify 'No results found' message",
                        "Verify suggestions or alternatives shown (if applicable)"
                    ],
                    expected_result="'No results found' message displayed",
                    test_data={"query": "xyzabc123nonexistent"},
                    tags=tags + ["search", "negative"]
                ))

                tests.append(TestCase(
                    name="test_search_empty_query",
                    type="negative",
                    priority="P2",
                    steps=[
                        "Navigate to search page",
                        "Leave search field empty",
                        "Click search button",
                        "Verify validation error or no action"
                    ],
                    expected_result="Validation error or search prevented",
                    test_data={"query": ""},
                    tags=tags + ["search", "validation", "negative"]
                ))

            elif "cart" in step_lower:
                tests.append(TestCase(
                    name="test_add_to_cart_out_of_stock",
                    type="negative",
                    priority="P1",
                    steps=[
                        "Navigate to out-of-stock product",
                        "Verify 'Add to Cart' button is disabled",
                        "Verify 'Out of Stock' message is shown"
                    ],
                    expected_result="Cannot add out-of-stock item to cart",
                    tags=tags + ["cart", "inventory", "negative"]
                ))

        return tests

    # ---- Edge ----

    def _generate_edge_cases(self, suite_name: str, steps: List[str], tags: List[str]) -> List[TestCase]:
        tests: List[TestCase] = []

        for step in steps:
            step_lower = step.lower()

            if "login" in step_lower:
                tests.append(TestCase(
                    name="test_login_special_characters",
                    type="edge",
                    priority="P2",
                    steps=[
                        "Navigate to login page",
                        "Fill username with special characters",
                        "Fill password with special characters",
                        "Click submit",
                        "Verify system handles gracefully"
                    ],
                    expected_result="Special characters handled correctly",
                    test_data={"username": "test@#$%^&*()", "password": "P@$$w0rd!<>?"},
                    tags=tags + ["login", "edge"]
                ))

                tests.append(TestCase(
                    name="test_login_long_credentials",
                    type="edge",
                    priority="P2",
                    steps=[
                        "Navigate to login page",
                        "Fill username with very long string (500+ chars)",
                        "Fill password with very long string (500+ chars)",
                        "Click submit",
                        "Verify system handles or truncates gracefully"
                    ],
                    expected_result="Long credentials handled without crash",
                    test_data={"username": "a" * 500, "password": "b" * 500},
                    tags=tags + ["login", "edge"]
                ))

            elif "search" in step_lower:
                tests.append(TestCase(
                    name="test_search_special_characters",
                    type="edge",
                    priority="P2",
                    steps=[
                        "Navigate to search page",
                        "Enter query with special characters",
                        "Submit search",
                        "Verify results or appropriate handling"
                    ],
                    expected_result="Special characters in search handled",
                    test_data={"query": "@#$%^&*()"},
                    tags=tags + ["search", "edge"]
                ))

                tests.append(TestCase(
                    name="test_search_long_query",
                    type="edge",
                    priority="P2",
                    steps=[
                        "Navigate to search page",
                        "Enter very long query (1000+ chars)",
                        "Submit search",
                        "Verify query is handled or truncated"
                    ],
                    expected_result="Long query handled gracefully",
                    test_data={"query": "long query " * 100},
                    tags=tags + ["search", "edge"]
                ))

            elif "cart" in step_lower:
                tests.append(TestCase(
                    name="test_cart_maximum_quantity",
                    type="edge",
                    priority="P2",
                    steps=[
                        "Navigate to product page",
                        "Set quantity to maximum allowed (e.g., 99)",
                        "Add to cart",
                        "Verify cart accepts max quantity",
                        "Attempt to add more",
                        "Verify limit message or prevention"
                    ],
                    expected_result="Maximum quantity enforced",
                    test_data={"quantity": 99},
                    tags=tags + ["cart", "edge"]
                ))

        return tests

    # ---- Boundary ----

    def _generate_boundary_tests(self, suite_name: str, steps: List[str], tags: List[str]) -> List[TestCase]:
        tests: List[TestCase] = []

        for step in steps:
            step_lower = step.lower()

            if any(kw in step_lower for kw in ["password", "register", "signup"]):
                tests.append(TestCase(
                    name="test_password_minimum_length",
                    type="boundary",
                    priority="P2",
                    steps=[
                        "Navigate to registration/password page",
                        "Enter password with 7 characters (min-1)",
                        "Verify rejection or validation error",
                        "Enter password with 8 characters (min)",
                        "Verify acceptance"
                    ],
                    expected_result="Minimum password length enforced (8 chars)",
                    test_data={"password_short": "Test@12", "password_min": "Test@123"},
                    tags=tags + ["password", "validation", "boundary"]
                ))

                tests.append(TestCase(
                    name="test_password_maximum_length",
                    type="boundary",
                    priority="P2",
                    steps=[
                        "Navigate to registration/password page",
                        "Enter password with 128 characters (max)",
                        "Verify acceptance",
                        "Enter password with 129 characters (max+1)",
                        "Verify rejection or truncation"
                    ],
                    expected_result="Maximum password length enforced (128 chars)",
                    test_data={"password_max": "a" * 128, "password_too_long": "a" * 129},
                    tags=tags + ["password", "validation", "boundary"]
                ))

            elif any(kw in step_lower for kw in ["quantity", "cart", "product"]):
                tests.append(TestCase(
                    name="test_quantity_minimum",
                    type="boundary",
                    priority="P2",
                    steps=[
                        "Navigate to product page",
                        "Try to set quantity to 0 (min-1)",
                        "Verify rejection or default to 1",
                        "Set quantity to 1 (min)",
                        "Verify acceptance"
                    ],
                    expected_result="Minimum quantity enforced (1)",
                    test_data={"quantity_zero": 0, "quantity_min": 1},
                    tags=tags + ["cart", "quantity", "boundary"]
                ))

        return tests

    # ==================== POM ====================

    def _extract_page_objects(self, suite_name: str, steps: List[str]) -> List[PageObjectModel]:
        page_objects: List[PageObjectModel] = []

        if any("login" in str(s).lower() for s in steps):
            page_objects.append(PageObjectModel(
                name="LoginPage",
                url="/login",
                selectors={
                    "usernameInput": "input[name='username'], input[type='email']",
                    "passwordInput": "input[name='password'], input[type='password']",
                    "submitButton": "button[type='submit'], button:has-text('Sign In')",
                    "errorMessage": ".error-message, .alert-danger",
                    "forgotPasswordLink": "a:has-text('Forgot Password')"
                },
                methods=[
                    {"name": "login", "params": ["username", "password"]},
                    {"name": "isLoggedIn", "params": []},
                    {"name": "getErrorMessage", "params": []}
                ]
            ))

        if any(("register" in str(s).lower()) or ("signup" in str(s).lower()) for s in steps):
            page_objects.append(PageObjectModel(
                name="RegistrationPage",
                url="/register",
                selectors={
                    "emailInput": "input[type='email']",
                    "passwordInput": "input[name='password']",
                    "confirmPasswordInput": "input[name='confirmPassword']",
                    "submitButton": "button[type='submit']",
                    "successMessage": ".success-message",
                    "errorMessage": ".error-message"
                },
                methods=[
                    {"name": "register", "params": ["email", "password", "confirmPassword"]},
                    {"name": "getSuccessMessage", "params": []},
                    {"name": "getErrorMessage", "params": []}
                ]
            ))

        if any("cart" in str(s).lower() for s in steps):
            page_objects.append(PageObjectModel(
                name="CartPage",
                url="/cart",
                selectors={
                    "cartItems": ".cart-item",
                    "cartCount": ".cart-count",
                    "checkoutButton": "button:has-text('Checkout')",
                    "emptyCartMessage": ".empty-cart-message",
                    "removeItemButton": ".remove-item"
                },
                methods=[
                    {"name": "getItemCount", "params": []},
                    {"name": "removeItem", "params": ["itemIndex"]},
                    {"name": "proceedToCheckout", "params": []}
                ]
            ))

        return page_objects

    def _generate_page_objects(self, workspace: Path, suites: List[TestSuite]) -> None:
        pages_dir = workspace / "pages"
        all_pos: Dict[str, PageObjectModel] = {}
        for suite in suites:
            for po in suite.page_objects:
                all_pos.setdefault(po.name, po)

        for po_name, po in all_pos.items():
            po_file = pages_dir / f"{po_name}.ts"
            po_file.write_text(self._generate_page_object_code(po), encoding="utf-8")

        logger.info(f"📄 Generated {len(all_pos)} page object files")

    def _generate_page_object_code(self, po: PageObjectModel) -> str:
        def escape_single(s: str) -> str:
            return s.replace("\\", "\\\\").replace("'", "\\'")

        selectors_code = ",\n    ".join([f"{k}: '{escape_single(v)}'" for k, v in po.selectors.items()])

        methods: List[str] = []
        for method in po.methods:
            params = ", ".join(method.get("params", []))
            methods.append(
                f"  async {method['name']}({params}) {{\n"
                f"    // TODO: Implement {method['name']}\n"
                f"  }}\n"
            )
        methods_code = "".join(methods)

        return (
            "import { Page, Locator } from '@playwright/test';\n\n"
            f"export class {po.name} {{\n"
            "  readonly page: Page;\n"
            "  readonly selectors = {\n"
            f"    {selectors_code}\n"
            "  };\n\n"
            f"  constructor(page: Page) {{\n"
            "    this.page = page;\n"
            "  }\n\n"
            f"  async navigate() {{\n"
            f"    await this.page.goto('{po.url}');\n"
            "  }\n"
            f"{methods_code}"
            "  locator(selectorKey: keyof typeof this.selectors): Locator {\n"
            "    return this.page.locator(this.selectors[selectorKey]);\n"
            "  }\n"
            "}\n"
        )

    # ==================== Test Files ====================

    def _write_test_file(self, workspace: Path, suite: TestSuite) -> None:
        suite_name = suite.name.replace(" ", "_").lower()
        test_dir = workspace / "tests"
        test_file = test_dir / f"{suite_name}.spec.ts"

        imports = ["import { test, expect } from '@playwright/test';"]
        if suite.page_objects:
            for po in suite.page_objects:
                imports.append(f"import {{ {po.name} }} from '../pages/{po.name}';")

        code_lines: List[str] = []
        code_lines.extend(imports)
        code_lines.append("")
        code_lines.append(f"test.describe('{suite.name}', () => {{")
        code_lines.append("  test.beforeEach(async ({ page }) => {")
        code_lines.append(f"    await page.goto('{self.base_url}');")
        code_lines.append("  });")
        code_lines.append("")

        for tc in suite.tests:
            code_lines.append(self._generate_test_function(tc))

        code_lines.append("});")
        code = "\n".join(code_lines)

        test_file.write_text(code, encoding="utf-8")
        logger.info(f"📝 Written {len(suite.tests)} tests to {test_file}")

    def _generate_test_function(self, test_case: TestCase) -> str:
        tags = f"@{test_case.type} @{test_case.priority}"
        if test_case.tags:
            tags += " @" + " @".join(test_case.tags)

        # Test options (Playwright test.extend options are not inline; we document retries/timeout)
        options_comment: List[str] = []
        if test_case.timeout_ms is not None:
            options_comment.append(f"timeout={test_case.timeout_ms}ms")
        if test_case.retry_count is not None:
            options_comment.append(f"retries={test_case.retry_count}")
        opt_str = f" // Options: {', '.join(options_comment)}" if options_comment else ""

        lines: List[str] = []
        lines.append(f"  test('{test_case.name} {tags}', async ({'{'} page {'}'}) => {{{opt_str}")
        lines.append(f"    // Test Type: {test_case.type.upper()}")
        lines.append(f"    // Priority: {test_case.priority}")
        lines.append(f"    // Expected: {test_case.expected_result}")
        lines.append("")

        if test_case.preconditions:
            lines.append("    // Preconditions:")
            for pre in test_case.preconditions:
                lines.append(f"    // - {pre}")
            lines.append("")

        for i, step in enumerate(test_case.steps, 1):
            lines.append(f"    // Step {i}: {step}")
            lines.append(self._generate_step_code(step, test_case.test_data))

        lines.append("  });")
        return "\n".join(lines)

    def _generate_step_code(self, step: str, test_data: Optional[Dict[str, Any]]) -> str:
        s = step.lower()

        # Navigation
        if "navigate" in s or "goto" in s:
            if "login" in s:
                return "    await page.goto('/login');"
            if "register" in s or "signup" in s:
                return "    await page.goto('/register');"
            if "cart" in s:
                return "    await page.goto('/cart');"
            if "checkout" in s:
                return "    await page.goto('/checkout');"
            return "    await page.goto('/');"

        # Fill actions
        if "fill" in s:
            td = test_data or {}
            if "username" in s or "email" in s:
                value = td.get("username") or td.get("email") or "test@example.com"
                return "    await page.fill('input[name=\"username\"], input[type=\"email\"]', '" + str(value).replace("'", "\\'") + "');"
            if "password" in s and "confirm" not in s:
                value = td.get("password", "password")
                return "    await page.fill('input[name=\"password\"]', '" + str(value).replace("'", "\\'") + "');"
            if "confirm" in s and "password" in s:
                value = td.get("confirm_password", "password")
                return "    await page.fill('input[name=\"confirmPassword\"]', '" + str(value).replace("'", "\\'") + "');"
            if "search" in s or "query" in s:
                value = td.get("query", "search term")
                return "    await page.fill('input[name=\"search\"], input[type=\"search\"]', '" + str(value).replace("'", "\\'") + "');"
            return "    // TODO: Implement fill action"

        # Click actions
        if "click" in s:
            if "submit" in s or "login" in s:
                return "    await page.click('button[type=\"submit\"]');"
            if "register" in s or "signup" in s:
                return "    await page.click('button[type=\"submit\"], button:has-text(\"Register\")');"
            if "add to cart" in s:
                return "    await page.click('button:has-text(\"Add to Cart\")');"
            if "checkout" in s:
                return "    await page.click('button:has-text(\"Checkout\"), a:has-text(\"Proceed to Checkout\")');"
            if "search" in s:
                return "    await page.click('button[type=\"submit\"], button:has-text(\"Search\")');"
            return "    // TODO: Implement click action"

        # Waits
        if "wait" in s:
            if "navigation" in s:
                return "    await page.waitForNavigation();"
            if "load" in s:
                return "    await page.waitForLoadState('networkidle');"
            return "    await page.waitForTimeout(1000);"

        # Verification
        if any(kw in s for kw in ["verify", "assert", "check"]):
            if "visible" in s or "displayed" in s:
                if "dashboard" in s:
                    return "    await expect(page.locator('text=Dashboard, .dashboard')).toBeVisible();"
                if "error" in s:
                    return "    await expect(page.locator('.error-message, .alert-danger')).toBeVisible();"
                if "success" in s:
                    return "    await expect(page.locator('.success-message, .alert-success')).toBeVisible();"
                if "cart" in s:
                    return "    await expect(page.locator('.cart-count')).toBeVisible();"
                return "    // TODO: Implement visibility check"
            if "error" in s:
                return "    await expect(page.locator('.error-message')).toBeVisible();"
            if "success" in s:
                return "    await expect(page.locator('.success-message')).toBeVisible();"
            if "count" in s or "cart" in s:
                return (
                    "    const count = await page.locator('.cart-count').textContent();\n"
                    "    expect(parseInt(count || '0')).toBeGreaterThan(0);"
                )
            return "    // TODO: Implement verification"

        return f"    // TODO: Implement step: {step}"

    # ==================== Utils & Fixtures ====================

    def _generate_utils(self, workspace: Path) -> None:
        utils_dir = workspace / "utils"

        helpers_ts = (
            "import { Page } from '@playwright/test';\n\n"
            "export class TestHelpers {\n"
            "  static async waitForElement(page: Page, selector: string, timeout = 5000) {\n"
            "    await page.waitForSelector(selector, { timeout });\n"
            "  }\n\n"
            "  static async fillForm(page: Page, data: Record<string, string>) {\n"
            "    for (const [key, value] of Object.entries(data)) {\n"
            "      await page.fill(`[name=\"${key}\"]`, value);\n"
            "    }\n"
            "  }\n\n"
            "  static generateRandomEmail(): string {\n"
            "    return `test${Date.now()}@example.com`;\n"
            "  }\n\n"
            "  static generateRandomString(length = 10): string {\n"
            "    return Math.random().toString(36).substring(2, length + 2);\n"
            "  }\n"
            "}\n"
        )
        (utils_dir / "helpers.ts").write_text(helpers_ts, encoding="utf-8")

        data_gen_ts = (
            "export class DataGenerator {\n"
            "  static validEmail(): string {\n"
            "    return `user${Date.now()}@example.com`;\n"
            "  }\n\n"
            "  static validPassword(): string {\n"
            "    return 'Test@123456';\n"
            "  }\n\n"
            "  static invalidEmail(): string {\n"
            "    return 'notanemail';\n"
            "  }\n\n"
            "  static sqlInjection(): string {\n"
            "    return \"' OR '1'='1\";\n"
            "  }\n\n"
            "  static xssPayload(): string {\n"
            '    return \'<script>alert("XSS")</script>\';\n'
            "  }\n\n"
            "  static longString(length = 1000): string {\n"
            "    return 'a'.repeat(length);\n"
            "  }\n\n"
            "  static specialCharacters(): string {\n"
            "    return '!@#$%^&*()_+-=[]{}|;:,.<>?';\n"
            "  }\n"
            "}\n"
        )
        (utils_dir / "dataGenerator.ts").write_text(data_gen_ts, encoding="utf-8")

        logger.info("🛠️ Generated utility files")

    def _generate_fixtures(self, workspace: Path) -> None:
        fixtures_dir = workspace / "fixtures"

        custom_fixtures_ts = (
            "import { test as base } from '@playwright/test';\n\n"
            "// Example: Authenticated user fixture\n"
            "export const test = base.extend({\n"
            "  authenticatedPage: async ({ page }, use) => {\n"
            "    // Perform login\n"
            "    await page.goto('/login');\n"
            "    await page.fill('[name=\"username\"]', 'testuser@example.com');\n"
            "    await page.fill('[name=\"password\"]', 'Test@123');\n"
            "    await page.click('button[type=\"submit\"]');\n"
            "    await page.waitForNavigation();\n"
            "    await use(page);\n"
            "    await page.goto('/logout');\n"
            "  },\n"
            "});\n\n"
            "export { expect } from '@playwright/test';\n"
        )
        (fixtures_dir / "customFixtures.ts").write_text(custom_fixtures_ts, encoding="utf-8")

        logger.info("🔧 Generated custom fixtures")

    def _generate_test_data(self, workspace: Path, suites: List[TestSuite]) -> None:
        data_dir = workspace / "data"
        all_test_data: Dict[str, Any] = {}
        for suite in suites:
            for test in suite.tests:
                if test.test_data:
                    all_test_data[test.name] = test.test_data

        (data_dir / "testData.json").write_text(json.dumps(all_test_data, indent=2), encoding="utf-8")
        logger.info("📊 Generated test data files")

    # ==================== Config ====================

    def _write_playwright_config(self, workspace: Path) -> None:
        browsers_cfg = self.ui_config.get("browsers", "chromium")
        browsers = [b.strip() for b in browsers_cfg.split(",") if b.strip()]
        if not browsers:
            browsers = ["chromium"]

        # Map: name -> devices alias
        device_alias = {
            "chromium": "Desktop Chrome",
            "firefox": "Desktop Firefox",
            "webkit": "Desktop Safari",
        }

        project_entries: List[str] = []
        for b in browsers:
            alias = device_alias.get(b, f"Desktop {b.capitalize()}")
            project_entries.append(
                f"    {{ name: '{b}', use: {{ ...devices['{alias}'] }} }}"
            )

        cfg = (
            "import { defineConfig, devices } from '@playwright/test';\n\n"
            "export default defineConfig({\n"
            "  testDir: './tests',\n"
            f"  fullyParallel: {str(self.ui_config.get('fully_parallel', False)).lower()},\n"
            f"  forbidOnly: {str(self.ui_config.get('forbid_only', True)).lower()},\n"
            f"  retries: {int(self.ui_config.get('retries', 1))},\n"
            f"  workers: {int(self.ui_config.get('workers', 4))},\n"
            "  reporter: [\n"
            "    ['list'],\n"
            "    ['html', { outputFolder: 'reports/playwright' }],\n"
            "    ['json', { outputFile: 'reports/playwright/report.json' }],\n"
            "    ['junit', { outputFile: 'reports/junit/results.xml' }]\n"
            "  ],\n"
            "  use: {\n"
            f"    baseURL: '{self.base_url}',\n"
            "    trace: 'retain-on-failure',\n"
            "    screenshot: 'only-on-failure',\n"
            "    video: 'retain-on-failure',\n"
            f"    actionTimeout: {int(self.ui_config.get('test_timeout_ms', 45000))},\n"
            "  },\n"
            "  projects: [\n"
            + ",\n".join(project_entries) + "\n"
            "  ],\n"
            "  outputDir: 'reports/test-results',\n"
            "});\n"
        )
        (workspace / "playwright.config.ts").write_text(cfg, encoding="utf-8")

    def _write_package_json(self, workspace: Path) -> None:
        pkg = {
            "name": self.project_name,
            "version": "1.0.0",
            "description": "AI-generated Playwright test framework with full coverage",
            "private": True,
            "scripts": {
                "test": "playwright test",
                "test:smoke": "playwright test --grep @smoke",
                "test:negative": "playwright test --grep @negative",
                "test:edge": "playwright test --grep @edge",
                "test:boundary": "playwright test --grep @boundary",
                "test:p0": "playwright test --grep @P0",
                "test:p1": "playwright test --grep @P1",
                "test:headed": "playwright test --headed",
                "test:debug": "playwright test --debug",
                "report": "playwright show-report reports/playwright"
            },
            "devDependencies": {
                "@playwright/test": "^1.44.0",
                "typescript": "^5.4.0"
            }
        }
        (workspace / "package.json").write_text(json.dumps(pkg, indent=2), encoding="utf-8")

    def _write_tsconfig(self, workspace: Path) -> None:
        tsconfig = {
            "compilerOptions": {
                "target": "ES2020",
                "module": "commonjs",
                "strict": True,
                "esModuleInterop": True,
                "skipLibCheck": True,
                "forceConsistentCasingInFileNames": True
            }
        }
        (workspace / "tsconfig.json").write_text(json.dumps(tsconfig, indent=2), encoding="utf-8")

    def _write_gitignore(self, workspace: Path) -> None:
        gi_lines = [
            "node_modules/",
            "reports/",
            "test-results/",
            "playwright-report/",
            "playwright/.cache/",
            "*.mp4",
            "*.webm",
            "*.png",
            "*.jpg",
            ".env",
            ".env.local",
        ]
        (workspace / ".gitignore").write_text("\n".join(gi_lines) + "\n", encoding="utf-8")

    def _write_editorconfig(self, workspace: Path) -> None:
        content = (
            "root = true\n\n"
            "[*]\n"
            "charset = utf-8\n"
            "end_of_line = lf\n"
            "insert_final_newline = true\n"
            "indent_style = space\n"
            "indent_size = 2\n"
        )
        (workspace / ".editorconfig").write_text(content, encoding="utf-8")

    def _write_nvmrc(self, workspace: Path) -> None:
        # Pin to a modern LTS (Playwright supports active LTS)
        (workspace / ".nvmrc").write_text("18\n", encoding="utf-8")

    def _write_readme(self, workspace: Path) -> None:
        readme = [
            f"# {self.project_name} — Playwright Test Framework",
            "",
            "## Overview",
            "This framework was automatically generated with complete test coverage including:",
            "- ✅ Smoke tests (happy path)",
            "- ✅ Negative tests (error scenarios)",
            "- ✅ Edge case tests",
            "- ✅ Boundary value tests",
            "",
            "## Prerequisites",
            "- Node.js 18+ (`nvm use` if you have `.nvmrc`)",
            "- `npm i -g playwright` (optional; the local devDependency will be used otherwise)",
            "",
            "## Setup",
            "```bash",
            "npm install",
            "npx playwright install --with-deps",
            "```",
            "",
            "## Run Tests",
            "```bash",
            "npm test",
            "npm run test:smoke",
            "npm run test:negative",
            "npm run test:edge",
            "npm run test:boundary",
            "```",
            "",
            "## Reports",
            "- HTML: `reports/playwright` (open with `npm run report`)",
            "- JSON: `reports/playwright/report.json`",
            "- JUnit: `reports/junit/results.xml`",
            "",
            "## Structure",
            "```\n"
            "tests/          # Spec files\n"
            "pages/          # Page Object Model (POM)\n"
            "utils/          # Helpers & generators\n"
            "fixtures/       # Custom fixtures (e.g., authenticatedPage)\n"
            "data/           # Data-driven test JSON\n"
            "reports/        # Test artifacts & reports\n"
            "```\n",
            "",
            "## CI",
            "A GitHub Actions workflow is included under `.github/workflows/ci.yml`.",
            "",
        ]
        (workspace / "README.md").write_text("\n".join(readme), encoding="utf-8")

    # ==================== CI ====================

    def _generate_github_actions(self, workspace: Path) -> None:
        yml = [
            "name: Playwright CI",
            "",
            "on:",
            "  push:",
            "    branches: [ main, master ]",
            "  pull_request:",
            "    branches: [ main, master ]",
            "",
            "jobs:",
            "  test:",
            "    runs-on: ubuntu-latest",
            "    steps:",
            "      - uses: actions/checkout@v4",
            "      - uses: actions/setup-node@v4",
            "        with:",
            "          node-version: 18",
            "      - name: Install dependencies",
            "        run: npm ci",
            "      - name: Install Playwright browsers",
            "        run: npx playwright install --with-deps",
            "      - name: Run tests",
            "        run: npm test",
            "      - name: Upload HTML report",
            "        if: always()",
            "        uses: actions/upload-artifact@v4",
            "        with:",
            "          name: playwright-html-report",
            "          path: reports/playwright",
            "      - name: Upload traces",
            "        if: always()",
            "        uses: actions/upload-artifact@v4",
            "        with:",
            "          name: playwright-traces",
            "          path: reports/traces",
        ]
        (workspace / ".github" / "workflows" / "ci.yml").write_text("\n".join(yml) + "\n", encoding="utf-8")
