"""
Version Manager (Phase 6.0) — semantic versioning, changelog, release helpers
-------------------------------------------------------------------------------
Responsibilities:
 - Maintain a manifest (versions.json) that records current version, history.
 - Create changelog entries for releases.
 - Bump semantic versions (major/minor/patch/prerelease).
 - Optionally create annotated git tags and (dry-run) shell commands.
 - Keep all operations idempotent and safe for CI usage.

Design considerations / best-practice:
 - Manifest is a small JSON file kept under repo (./.release/versions.json).
 - Changelog appended in Markdown at ./RELEASE_NOTES.md (human + machine readable).
 - All destructive operations (git tag/push) are optional and require `git_push=True`.
 - Functions are synchronous/simple to integrate in CLI or CI steps.
"""

import os
import json
import datetime
import subprocess
from typing import Dict, Optional, Tuple

# Location to persist release artifacts / manifest
RELEASE_DIR = os.getenv("RELEASE_DIR", ".release")
MANIFEST_FILE = os.path.join(RELEASE_DIR, "versions.json")
RELEASE_NOTES_FILE = os.path.join(RELEASE_DIR, "RELEASE_NOTES.md")

# Default manifest structure
DEFAULT_MANIFEST = {
    "current_version": "0.0.0",
    "history": []  # list of { version, timestamp, notes, changes }
}


# -------------------------
# Utilities
# -------------------------
def _ensure_release_dir():
    if not os.path.exists(RELEASE_DIR):
        os.makedirs(RELEASE_DIR, exist_ok=True)


def _now_iso():
    return datetime.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


# -------------------------
# Manifest functions
# -------------------------
def load_manifest() -> Dict:
    """Load or initialize the manifest file."""
    _ensure_release_dir()
    if not os.path.exists(MANIFEST_FILE):
        save_manifest(DEFAULT_MANIFEST)
        return DEFAULT_MANIFEST.copy()
    try:
        with open(MANIFEST_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as e:
        raise RuntimeError(f"Failed to load manifest: {e}")


def save_manifest(manifest: Dict) -> None:
    """Persist the manifest to disk atomically."""
    _ensure_release_dir()
    tmp = MANIFEST_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    os.replace(tmp, MANIFEST_FILE)


# -------------------------
# Version helpers
# -------------------------
def _parse_version(v: str) -> Tuple[int, int, int, Optional[str]]:
    """Return major, minor, patch, prerelease (or None)."""
    if "-" in v:
        core, pre = v.split("-", 1)
    else:
        core, pre = v, None
    parts = core.split(".")
    if len(parts) != 3:
        raise ValueError("Version must be in semver format MAJOR.MINOR.PATCH")
    major, minor, patch = [int(x) for x in parts]
    return major, minor, patch, pre


def _format_version(major: int, minor: int, patch: int, prerelease: Optional[str] = None) -> str:
    v = f"{major}.{minor}.{patch}"
    if prerelease:
        v = f"{v}-{prerelease}"
    return v


def bump_version(current: str, level: str = "patch", prerelease: Optional[str] = None) -> str:
    """
    Bump a semantic version.
    level: 'major' | 'minor' | 'patch'
    prerelease: optional string for '-alpha' '-rc1' etc.
    """
    major, minor, patch, _ = _parse_version(current)
    if level == "major":
        major += 1
        minor = 0
        patch = 0
    elif level == "minor":
        minor += 1
        patch = 0
    elif level == "patch":
        patch += 1
    else:
        raise ValueError("level must be 'major', 'minor' or 'patch'")
    return _format_version(major, minor, patch, prerelease)


# -------------------------
# Changelog / Release Notes
# -------------------------
def _append_release_notes(version: str, notes: str, changes: Optional[list] = None) -> str:
    """Append a markdown entry to RELEASE_NOTES.md and return path."""
    _ensure_release_dir()
    timestamp = _now_iso()
    header = f"## {version} — {timestamp}\n\n"
    body = notes.strip() + "\n\n" if notes else ""
    if changes:
        body += "### Notable changes\n\n"
        for c in changes:
            body += f"- {c}\n"
        body += "\n"
    body += "---\n\n"

    # ensure file exists
    if not os.path.exists(RELEASE_NOTES_FILE):
        with open(RELEASE_NOTES_FILE, "w", encoding="utf-8") as fh:
            fh.write("# Release Notes\n\n")

    with open(RELEASE_NOTES_FILE, "a", encoding="utf-8") as fh:
        fh.write(header)
        fh.write(body)
    return RELEASE_NOTES_FILE


# -------------------------
# High-level release flow
# -------------------------
def create_release(new_version: str, notes: str = "", changes: Optional[list] = None, git_tag: bool = False, git_push: bool = False, dry_run: bool = False) -> Dict:
    """
    Create a release:
      - update manifest
      - append release notes
      - optionally create an annotated git tag and push it

    Returns manifest entry dict.
    """
    manifest = load_manifest()
    current = manifest.get("current_version", "0.0.0")
    if dry_run:
        print(f"[dry-run] Will create release {new_version} (current {current})")

    # create entry
    entry = {
        "version": new_version,
        "timestamp": _now_iso(),
        "notes": notes,
        "changes": changes or []
    }
    manifest.setdefault("history", []).append(entry)
    manifest["current_version"] = new_version

    if not dry_run:
        save_manifest(manifest)
        _append_release_notes(new_version, notes or f"Release {new_version}", changes)

    # Optional git tag
    if git_tag:
        tag_msg = notes or f"Release {new_version}"
        cmd_tag = ["git", "tag", "-a", new_version, "-m", tag_msg]
        print("Running:", " ".join(cmd_tag))
        if not dry_run:
            try:
                subprocess.run(cmd_tag, check=True)
            except Exception as e:
                raise RuntimeError(f"git tag failed: {e}")

        if git_push:
            cmd_push = ["git", "push", "origin", new_version]
            print("Running:", " ".join(cmd_push))
            if not dry_run:
                try:
                    subprocess.run(cmd_push, check=True)
                except Exception as e:
                    raise RuntimeError(f"git push failed: {e}")

    return entry


# -------------------------
# Convenience: bump-and-release
# -------------------------
def bump_and_release(level: str = "patch", notes: str = "", changes: Optional[list] = None, git_tag: bool = False, git_push: bool = False, dry_run: bool = False) -> Dict:
    """
    Convenience helper:
     - load manifest
     - bump version
     - create release entry
    """
    manifest = load_manifest()
    current = manifest.get("current_version", "0.0.0")
    # support prerelease token via level: e.g. 'patch:rc1' or 'minor:alpha'
    prerelease = None
    if ":" in level:
        level, prerelease = level.split(":", 1)
    new_version = bump_version(current, level, prerelease)
    return create_release(new_version, notes=notes, changes=changes, git_tag=git_tag, git_push=git_push, dry_run=dry_run)


# -------------------------
# Small CLI for local use
# -------------------------
def _cli():
    """Minimal CLI for local/manual invocation (keeps dependencies minimal)."""
    import argparse
    parser = argparse.ArgumentParser(description="Version Manager (create releases)")
    parser.add_argument("--bump", choices=["major", "minor", "patch"], help="bump level")
    parser.add_argument("--release", help="create release with explicit version (overrides bump)")
    parser.add_argument("--notes", help="release notes", default="")
    parser.add_argument("--changes", help="comma-separated change list", default="")
    parser.add_argument("--git-tag", action="store_true", help="create annotated git tag")
    parser.add_argument("--git-push", action="store_true", help="git push tag to origin (requires auth)")
    parser.add_argument("--dry-run", action="store_true", help="don't write files or run git commands")
    args = parser.parse_args()

    changes = [c.strip() for c in args.changes.split(",")] if args.changes else []
    if args.release:
        entry = create_release(args.release, notes=args.notes, changes=changes, git_tag=args.git_tag, git_push=args.git_push, dry_run=args.dry_run)
    elif args.bump:
        entry = bump_and_release(args.bump, notes=args.notes, changes=changes, git_tag=args.git_tag, git_push=args.git_push, dry_run=args.dry_run)
    else:
        print("No action specified. Use --bump or --release")
        return
    print("Release entry created:", json.dumps(entry, indent=2))


if __name__ == "__main__":
    _cli()
