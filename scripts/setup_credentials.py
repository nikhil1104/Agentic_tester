#!/usr/bin/env python3
# scripts/setup_credentials.py
"""
Secure Credential Setup CLI v2.1 - Production Grade

Highlights
- Subcommands: add, list, delete, audit, wizard, export, import, test, completion
- Non-interactive mode for CI/CD pipelines
- Strict input validation (alias, service, TTL)
- Secure-by-default: refuses to store without secure backend (vault/keyring)
- Optional explicit override: --allow-insecure (NOT recommended)
- JSON or table output with optional rich formatting
- Clear exit codes for automation
- Bulk import/export (export is metadata-only; passwords never exported)
- Shell completion (bash/zsh)
- Health/test mode for verification
- Graceful handling of interrupts
- Structured logging

Exit Codes:
  0  Success
  1  General error
  2  Invalid input / validation error
  3  Conflict (e.g., alias exists)
  4  Not found
  5  Unsupported operation
  10 Secure backend required (refused)
  130 Interrupted by user (Ctrl+C)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import datetime
from getpass import getpass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure project root on sys.path
sys.path.insert(0, str(Path(__file__).parent.parent))

# NOTE: keep this import aligned with your project
from modules.credential_manager import CredentialManager as SecureCredentialsManager  # type: ignore

# =============== Optional Rich Formatting ===============
try:
    from rich.console import Console  # type: ignore
    from rich.table import Table  # type: ignore
    from rich.prompt import Confirm  # type: ignore
    _HAS_RICH = True
    _console = Console()
except Exception:
    _HAS_RICH = False
    _console = None  # type: ignore

# ===================== Logging =========================
logger = logging.getLogger("scripts.setup_credentials")
if not logger.handlers:
    level = os.getenv("CRED_SETUP_LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, level, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

# ====================== Types ==========================
@dataclass
class CLIResult:
    ok: bool
    code: int = 0
    message: str = ""

# =================== Validation ========================
_ALIAS_RE = re.compile(r"^[a-zA-Z0-9_\-\.]{2,64}$")
_SERVICE_RE = re.compile(r"^[a-zA-Z0-9_\-\.]{2,64}$")

def validate_alias(alias: str) -> Tuple[bool, str]:
    if not alias:
        return False, "Alias is required"
    if not _ALIAS_RE.match(alias):
        return False, "Alias must be 2-64 chars: letters, numbers, _ - ."
    return True, ""

def validate_service(service: str) -> Tuple[bool, str]:
    if not service:
        return False, "Service is required"
    if not _SERVICE_RE.match(service):
        return False, "Service must be 2-64 chars: letters, numbers, _ - ."
    return True, ""

def parse_ttl_hours(value: Optional[str]) -> Optional[int]:
    if value in (None, ""):
        return None
    try:
        v = int(value)
        if v <= 0:
            raise ValueError("TTL must be positive")
        return v
    except Exception as e:
        raise argparse.ArgumentTypeError(f"TTL must be a positive integer (hours): {e}")

# =================== Output Helpers ====================
def _print(msg: str = "", style: Optional[str] = None):
    if _HAS_RICH and style and _console:
        _console.print(msg, style=style)
    else:
        print(msg)

def _plain_table(rows: List[Dict[str, Any]], fields: List[Tuple[str, str]]):
    headers = [h for _, h in fields]
    if not rows:
        print(" | ".join(headers))
        print("-+-".join("-" * len(h) for h in headers))
        return
    widths = []
    for k, h in fields:
        max_cell = max((len(str(r.get(k, ""))) for r in rows), default=0)
        widths.append(max(len(h), max_cell))
    line = " | ".join(h.ljust(w) for h, w in zip(headers, widths))
    sep = "-+-".join("-" * w for w in widths)
    print(line)
    print(sep)
    for r in rows:
        print(" | ".join(str(r.get(k, "")).ljust(w) for (k, _), w in zip(fields, widths)))

def print_table(rows: List[Dict[str, Any]], fields: List[Tuple[str, str]], title: str):
    if _HAS_RICH and _console:
        table = Table(title=title)
        for _, header in fields:
            table.add_column(header)
        for row in rows:
            table.add_row(*[str(row.get(k, "")) for k, _ in fields])
        _console.print(table)
    else:
        _plain_table(rows, fields)

def mask_username(username: str) -> str:
    if not username:
        return ""
    if "@" in username:
        name, domain = username.split("@", 1)
        if len(name) <= 2:
            masked = "*" * len(name)
        else:
            masked = name[0] + "*" * (len(name) - 2) + name[-1]
        return f"{masked}@{domain}"
    if len(username) <= 2:
        return "*" * len(username)
    return username[0] + "*" * (len(username) - 2) + username[-1]

# =========== Manager Health & Security ===========
def manager_health(mgr: SecureCredentialsManager) -> Dict[str, Any]:
    """
    Inspect manager capabilities in a tolerant way.
    Expected keys (if available):
      - vault_enabled: bool
      - keyring_enabled: bool
      - storage_path: str
      - last_error: Optional[str]
    """
    health: Dict[str, Any] = {
        "vault_enabled": False,
        "keyring_enabled": False,
        "storage_path": None,
        "last_error": None,
    }
    # attribute probes
    for key in list(health.keys()):
        if hasattr(mgr, key):
            try:
                health[key] = getattr(mgr, key)
            except Exception:
                pass
    # prefer explicit health() if provided
    if hasattr(mgr, "health") and callable(getattr(mgr, "health")):
        try:
            data = mgr.health()  # type: ignore
            if isinstance(data, dict):
                health.update(data)
        except Exception as e:
            health["last_error"] = str(e)
    return health

def print_health_guidance(h: Dict[str, Any]):
    vault = bool(h.get("vault_enabled"))
    keyring = bool(h.get("keyring_enabled"))
    _print("\nBackend status:", style="bold")
    _print(f"  • Vault:   {'ENABLED' if vault else 'disabled'}")
    _print(f"  • Keyring: {'ENABLED' if keyring else 'disabled'}")
    if h.get("storage_path"):
        _print(f"  • Storage: {h['storage_path']}")
    if not (vault or keyring):
        _print("\n❗ A secure backend is not active. To enable one:", style="yellow")
        _print("   1) Install cryptography for vault:")
        _print("      pip install cryptography")
        _print("   2) OR enable OS keyring:")
        _print("      pip install keyring")
        _print("   3) Re-run this tool.\n")
        _print("   To proceed UNSAFELY for local/dev only, pass --allow-insecure (NOT recommended).", style="red")

def require_secure_backend_or_fail(mgr: SecureCredentialsManager, allow_insecure: bool) -> bool:
    h = manager_health(mgr)
    if h.get("vault_enabled") or h.get("keyring_enabled"):
        return True
    print_health_guidance(h)
    if not allow_insecure:
        _print("\nRefusing to store credentials without a secure backend.", style="red")
        return False
    _print("\n⚠️ Proceeding INSECURELY by explicit request (--allow-insecure).", style="red")
    return True

# ================== Commands: add ====================
def cmd_add(args: argparse.Namespace, mgr: SecureCredentialsManager) -> CLIResult:
    if not require_secure_backend_or_fail(mgr, allow_insecure=getattr(args, "allow_insecure", False)):
        return CLIResult(False, 10, "secure_backend_required")

    alias = args.alias or input("Alias (e.g., 'gmail_test'): ").strip()
    ok, err = validate_alias(alias)
    if not ok:
        return CLIResult(False, 2, err)

    username = args.username or input("Username: ").strip()
    if not username:
        return CLIResult(False, 2, "Username is required")

    password = args.password or getpass("Password (hidden): ")
    if not password:
        return CLIResult(False, 2, "Password is required")

    service = args.service or input("Service (e.g., 'gmail'): ").strip()
    ok, err = validate_service(service)
    if not ok:
        return CLIResult(False, 2, err)

    ttl_hours = args.ttl_hours
    if ttl_hours is None and args.interactive:
        ttl_input = input("Expire after hours (blank=never): ").strip()
        try:
            ttl_hours = parse_ttl_hours(ttl_input)
        except argparse.ArgumentTypeError as e:
            return CLIResult(False, 2, str(e))

    # Prevent accidental overwrite (unless --force)
    try:
        existing = mgr.list_credentials() or {}
    except Exception:
        existing = {}
    if alias in existing and not args.force:
        return CLIResult(False, 3, f"Alias '{alias}' already exists. Use --force to overwrite.")

    try:
        success = mgr.store_credential(
            alias=alias,
            username=username,
            password=password,
            service=service,
            ttl_hours=ttl_hours,
            overwrite=getattr(args, "force", False),  # supported by your manager? else ignored
        )
    except TypeError:
        success = mgr.store_credential(
            alias=alias, username=username, password=password, service=service, ttl_hours=ttl_hours
        )

    if not success:
        last_err = getattr(mgr, "last_error", None)
        if last_err:
            logger.error("Failed to store credential: %s", last_err)
        return CLIResult(False, 1, "Failed to store credential (check logs)")

    _print(f"✅ Credential '{alias}' stored securely", style="green")
    _print(f"💡 Use in commands: Test {service} using {alias}")
    return CLIResult(True, 0, "")

# ================== Commands: list ===================
def cmd_list(args: argparse.Namespace, mgr: SecureCredentialsManager) -> CLIResult:
    try:
        creds = mgr.list_credentials() or {}
    except Exception as e:
        logger.error("List failed: %s", e)
        return CLIResult(False, 1, "list_failed")

    rows: List[Dict[str, Any]] = []
    for alias, info in creds.items():
        username = info.get("username") or ""
        if not args.show_usernames:
            username = mask_username(username)
        rows.append({
            "alias": alias,
            "service": info.get("service", ""),
            "username": username,
            "created_at": info.get("created_at", ""),
            "access_count": info.get("access_count", 0),
            "expires_at": info.get("expires_at") or "-",
        })

    if args.format == "json":
        print(json.dumps(rows, indent=2))
    else:
        print_table(rows, [
            ("alias", "Alias"),
            ("service", "Service"),
            ("username", "Username"),
            ("created_at", "Created"),
            ("access_count", "Accesses"),
            ("expires_at", "Expires"),
        ], title="Stored Credentials")
        _print(f"\nTotal: {len(rows)}", style="dim")

    return CLIResult(True, 0, "")

# ================== Commands: delete =================
def cmd_delete(args: argparse.Namespace, mgr: SecureCredentialsManager) -> CLIResult:
    alias = args.alias or input("Alias to delete: ").strip()
    ok, err = validate_alias(alias)
    if not ok:
        return CLIResult(False, 2, err)

    if not args.yes:
        question = f"Delete '{alias}'? This cannot be undone."
        if _HAS_RICH and _console:
            proceed = Confirm.ask(question)
        else:
            proceed = input(f"{question} [y/N]: ").strip().lower() in {"y", "yes"}
        if not proceed:
            _print("Aborted.", style="yellow")
            return CLIResult(True, 0, "aborted")

    try:
        ok = mgr.delete_credential(alias)
    except Exception as e:
        logger.error("Delete failed: %s", e)
        ok = False

    if ok:
        _print(f"✅ Deleted '{alias}'", style="green")
        return CLIResult(True, 0, "")
    else:
        _print(f"❌ Not found: '{alias}'", style="red")
        return CLIResult(False, 4, "not_found")

# ================== Commands: audit ==================
def cmd_audit(args: argparse.Namespace, mgr: SecureCredentialsManager) -> CLIResult:
    limit = max(1, int(args.limit or 20))
    try:
        entries = mgr.get_audit_log(limit) or []
    except Exception as e:
        logger.error("Audit failed: %s", e)
        entries = []

    if args.format == "json":
        print(json.dumps(entries, indent=2))
        return CLIResult(True, 0, "")

    rows = []
    for e in entries:
        rows.append({
            "timestamp": e.get("timestamp", ""),
            "action": e.get("action", ""),
            "alias": e.get("identifier", "") or e.get("alias", ""),
            "source": e.get("source", ""),
        })
    print_table(rows, [
        ("timestamp", "Timestamp"),
        ("action", "Action"),
        ("alias", "Alias"),
        ("source", "Source"),
    ], title=f"Recent Audit Log (last {limit})")
    return CLIResult(True, 0, "")

# ================== Commands: export =================
def cmd_export(args: argparse.Namespace, mgr: SecureCredentialsManager) -> CLIResult:
    try:
        creds = mgr.list_credentials() or {}
    except Exception as e:
        logger.error("Export list failed: %s", e)
        return CLIResult(False, 1, "export_failed")

    export = {
        "version": "2.1",
        "exported_at": datetime.now().isoformat(),
        "credentials": {
            alias: {
                "service": info.get("service"),
                "username": info.get("username"),
                "created_at": info.get("created_at"),
                "expires_at": info.get("expires_at"),
                # passwords intentionally NOT exported
            }
            for alias, info in creds.items()
        }
    }

    output_path = Path(args.output or "credentials_export.json").expanduser().resolve()
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as f:
            json.dump(export, f, indent=2)
    except Exception as e:
        return CLIResult(False, 1, f"Failed to write export file: {e}")

    _print(f"✅ Exported {len(creds)} credentials to {output_path}", style="green")
    _print("⚠️  Note: Passwords not included (re-enter on import)", style="yellow")
    return CLIResult(True, 0, "")

# ================== Commands: import =================
def cmd_import(args: argparse.Namespace, mgr: SecureCredentialsManager) -> CLIResult:
    # Require secure backend unless explicitly overridden
    if not require_secure_backend_or_fail(mgr, allow_insecure=getattr(args, "allow_insecure", False)):
        return CLIResult(False, 10, "secure_backend_required")

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        return CLIResult(False, 2, f"File not found: {input_path}")

    try:
        data = json.loads(input_path.read_text(encoding="utf-8"))
    except Exception as e:
        return CLIResult(False, 2, f"Invalid JSON file: {e}")

    creds = data.get("credentials", {})
    if not isinstance(creds, dict) or not creds:
        _print("⚠️ No credentials found in file", style="yellow")
        return CLIResult(True, 0, "")

    added = 0
    skipped = 0
    _print(f"\n📦 Importing {len(creds)} credentials from {input_path}\n")

    for alias, info in creds.items():
        ok, err = validate_alias(alias)
        if not ok:
            _print(f"  ⚠️ Skipped '{alias}': {err}", style="yellow")
            skipped += 1
            continue

        service = info.get("service") or "unknown"
        username = info.get("username") or ""

        _print(f"📌 {alias} ({service})")
        password = getpass(f"  Password for {username or alias}: ")
        if not password:
            _print("  ⚠️ Skipped (no password entered)", style="yellow")
            skipped += 1
            continue

        try:
            ok = mgr.store_credential(
                alias=alias,
                username=username,
                password=password,
                service=service,
                ttl_hours=None,
            )
        except Exception as e:
            logger.error("Import store failed for %s: %s", alias, e)
            ok = False

        if ok:
            _print("  ✅ Imported", style="green")
            added += 1
        else:
            _print("  ❌ Failed", style="red")
            skipped += 1

    _print(f"\n✅ Imported {added} credential(s) ({skipped} skipped)", style="green")
    return CLIResult(True, 0, "")

# ================== Commands: test ====================
def cmd_test(_: argparse.Namespace, mgr: SecureCredentialsManager) -> CLIResult:
    _print("\nCredential Manager Health", style="bold")
    h = manager_health(mgr)
    print_health_guidance(h)

    # Optional roundtrip test if secure backend available
    if h.get("vault_enabled") or h.get("keyring_enabled"):
        alias = "__cm_test__"
        try:
            mgr.store_credential(alias=alias, username="probe", password="probe", service="probe", ttl_hours=1, overwrite=True)  # type: ignore
            ok = alias in (mgr.list_credentials() or {})
            mgr.delete_credential(alias)
            if ok:
                _print("✅ Roundtrip test passed", style="green")
                return CLIResult(True, 0, "")
        except Exception as e:
            logger.error("Roundtrip failed: %s", e)
            _print("❌ Roundtrip test failed (see logs)", style="red")
            return CLIResult(False, 1, "roundtrip_failed")

    _print("⚠️ Roundtrip test not performed (no secure backend)", style="yellow")
    return CLIResult(True, 0, "")

# ================= Commands: completion ===============
def cmd_completion(args: argparse.Namespace, _: SecureCredentialsManager) -> CLIResult:
    shell = args.shell
    if shell == "bash":
        script = r"""
# Bash completion for setup_credentials.py
_setup_credentials_complete() {
    local cur prev commands
    cur="${COMP_WORDS[COMP_CWORD]}"
    prev="${COMP_WORDS[COMP_CWORD-1]}"
    commands="add list delete audit wizard export import test completion"

    if [ $COMP_CWORD -eq 1 ]; then
        COMPREPLY=( $(compgen -W "$commands" -- "$cur") )
    fi
}
complete -F _setup_credentials_complete setup_credentials.py
""".strip("\n")
    elif shell == "zsh":
        script = r"""
# Zsh completion for setup_credentials.py
#compdef setup_credentials.py

_setup_credentials() {
    local -a commands
    commands=(
        'add:Add or update credential'
        'list:List stored credentials'
        'delete:Delete credential'
        'audit:Show audit log'
        'wizard:Interactive wizard'
        'export:Export credentials (metadata only)'
        'import:Import credentials from file'
        'test:Test credential manager'
        'completion:Generate shell completion'
    )
    _describe 'commands' commands
}
_setup_credentials "$@"
""".strip("\n")
    else:
        return CLIResult(False, 2, f"Unknown shell: {shell}")

    print(script)
    _print(f"\n# Add to your ~/.{shell}rc:", style="dim")
    _print(f'# eval "$(python scripts/setup_credentials.py completion --shell {shell})"', style="dim")
    return CLIResult(True, 0, "")

# ================= Commands: wizard ===================
def cmd_wizard(_: argparse.Namespace, mgr: SecureCredentialsManager) -> CLIResult:
    _print("\n" + "=" * 70)
    _print("🔒 Secure Credentials Setup Wizard v2.1", style="bold")
    _print("=" * 70)

    print_health_guidance(manager_health(mgr))

    while True:
        _print("\n📋 Options:")
        _print("  1. Add new credential")
        _print("  2. List credentials")
        _print("  3. Delete credential")
        _print("  4. View audit log")
        _print("  5. Export credentials")
        _print("  6. Import credentials")
        _print("  7. Test credential manager")
        _print("  8. Exit")

        try:
            choice = input("\nYour choice [1-8]: ").strip()
        except (EOFError, KeyboardInterrupt):
            _print("\n👋 Goodbye!")
            return CLIResult(True, 0, "")

        if choice == "1":
            res = cmd_add(argparse.Namespace(
                alias=None, username=None, password=None, service=None,
                ttl_hours=None, force=False, interactive=True, allow_insecure=False
            ), mgr)
            if not res.ok:
                _print(f"❌ {res.message}", style="red")

        elif choice == "2":
            cmd_list(argparse.Namespace(format="table", show_usernames=False), mgr)

        elif choice == "3":
            cmd_delete(argparse.Namespace(alias=None, yes=False), mgr)

        elif choice == "4":
            cmd_audit(argparse.Namespace(limit=20, format="table"), mgr)

        elif choice == "5":
            cmd_export(argparse.Namespace(output="credentials_export.json"), mgr)

        elif choice == "6":
            filename = input("Import file path: ").strip()
            if filename:
                cmd_import(argparse.Namespace(input=filename, allow_insecure=False), mgr)

        elif choice == "7":
            cmd_test(argparse.Namespace(), mgr)

        elif choice == "8":
            _print("\n👋 Goodbye!")
            return CLIResult(True, 0, "")

        else:
            _print("Please enter a number 1–8.", style="yellow")

# ================= Argument Parser ====================
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="setup_credentials",
        description="Secure Credential Setup CLI v2.1",
        epilog="For command-specific help: setup_credentials <command> --help"
    )
    sub = p.add_subparsers(dest="command", required=True)

    # add
    ap = sub.add_parser("add", help="Add or update a credential")
    ap.add_argument("--alias", help="Credential alias (unique identifier)")
    ap.add_argument("--service", help="Service name (e.g., gmail)")
    ap.add_argument("--username", help="Username or email")
    ap.add_argument("--password", help="Password (WARNING: avoid in shell history)")
    ap.add_argument("--ttl-hours", type=parse_ttl_hours, help="Expiry in hours (positive integer)")
    ap.add_argument("--force", action="store_true", help="Overwrite if alias exists")
    ap.add_argument("--interactive", action="store_true", help="Prompt for missing values")
    ap.add_argument("--allow-insecure", action="store_true",
                    help="EXPLICITLY allow insecure storage if no secure backend (NOT RECOMMENDED)")
    ap.set_defaults(func=cmd_add)

    # list
    lp = sub.add_parser("list", help="List stored credentials")
    lp.add_argument("--format", choices=("table", "json"), default="table", help="Output format")
    lp.add_argument("--show-usernames", action="store_true", help="Show full usernames")
    lp.set_defaults(func=cmd_list)

    # delete
    dp = sub.add_parser("delete", help="Delete credential by alias")
    dp.add_argument("--alias", help="Alias to delete")
    dp.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")
    dp.set_defaults(func=cmd_delete)

    # audit
    aud = sub.add_parser("audit", help="Show recent audit log entries")
    aud.add_argument("--limit", type=int, default=20, help="Max entries to show")
    aud.add_argument("--format", choices=("table", "json"), default="table", help="Output format")
    aud.set_defaults(func=cmd_audit)

    # export
    ep = sub.add_parser("export", help="Export credentials metadata for backup")
    ep.add_argument("--output", "-o", default="credentials_export.json", help="Output file")
    ep.set_defaults(func=cmd_export)

    # import
    ip = sub.add_parser("import", help="Import credentials from JSON file")
    ip.add_argument("--input", "-i", required=True, help="Input JSON file path")
    ip.add_argument("--allow-insecure", action="store_true",
                    help="EXPLICITLY allow insecure storage if no secure backend (NOT RECOMMENDED)")
    ip.set_defaults(func=cmd_import)

    # test
    tp = sub.add_parser("test", help="Health + optional roundtrip test")
    tp.set_defaults(func=cmd_test)

    # completion
    cp = sub.add_parser("completion", help="Generate shell completion script")
    cp.add_argument("--shell", choices=["bash", "zsh"], required=True, help="Shell type")
    cp.set_defaults(func=cmd_completion)

    # wizard
    wz = sub.add_parser("wizard", help="Interactive setup wizard (menus)")
    wz.set_defaults(func=cmd_wizard)

    return p

# ===================== Main ===========================
def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        mgr = SecureCredentialsManager()
    except Exception as e:
        logger.error("Failed to initialize SecureCredentialsManager: %s", e)
        _print("❌ Failed to initialize credential manager. See logs.", style="red")
        return 1

    try:
        res: CLIResult = args.func(args, mgr)  # type: ignore[attr-defined]
        if not res.ok and res.message:
            logger.warning("Command failed: %s", res.message)
        return res.code
    except KeyboardInterrupt:
        _print("\n⚠️ Aborted by user.", style="yellow")
        return 130
    except EOFError:
        _print("\n⚠️ No input. Exiting.", style="yellow")
        return 1
    except Exception as e:
        logger.exception("Unhandled error: %s", e)
        _print("Unexpected error. See logs for details.", style="red")
        return 1

if __name__ == "__main__":
    sys.exit(main())
