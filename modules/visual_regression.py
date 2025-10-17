# modules/visual_regression.py
"""
Visual Regression Testing Engine (Production-Grade)
Integrates with Percy.io and supports custom implementations

Features:
- Percy.io integration
- Screenshot comparison
- Baseline management
- Responsive testing
- Visual diff reports
- Approval workflow
- CI/CD integration
"""

from __future__ import annotations

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

import httpx
from PIL import Image, ImageChops, ImageDraw

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class VisualConfig:
    """Visual regression configuration"""
    percy_token: Optional[str] = field(default_factory=lambda: os.getenv("PERCY_TOKEN"))
    baseline_dir: str = "./data/visual_baselines"
    diff_dir: str = "./data/visual_diffs"
    threshold: float = 0.01  # 1% difference threshold
    widths: List[int] = field(default_factory=lambda: [1920, 1280, 768, 375])
    enable_percy: bool = True
    enable_local: bool = True
    min_height: int = 1024
    
    def __post_init__(self):
        if self.enable_percy and not self.percy_token:
            logger.warning("Percy integration enabled but PERCY_TOKEN not set")
            self.enable_percy = False


@dataclass
class VisualSnapshot:
    """Visual snapshot metadata"""
    name: str
    url: str
    width: int
    height: int
    screenshot_path: str
    baseline_path: Optional[str] = None
    diff_path: Optional[str] = None
    diff_percentage: float = 0.0
    approved: bool = False
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


# ==================== Percy.io Integration ====================

class PercyClient:
    """
    Production-grade Percy.io API client.
    """
    
    def __init__(self, token: str):
        if not token:
            raise ValueError("Percy token required")
        
        self.token = token
        self.base_url = "https://percy.io/api/v1"
        
        self.client = httpx.Client(
            timeout=30.0,
            headers={
                "Authorization": f"Token {token}",
                "Content-Type": "application/json"
            }
        )
        
        logger.info("✅ Percy client initialized")
    
    def create_build(self, project_slug: str, branch: str = "main") -> str:
        """Create a new Percy build"""
        try:
            response = self.client.post(
                f"{self.base_url}/builds",
                json={
                    "data": {
                        "type": "builds",
                        "attributes": {
                            "branch": branch,
                            "target-branch": "main",
                            "commit-sha": "HEAD",
                            "commit-message": "Automated visual regression tests"
                        }
                    }
                }
            )
            response.raise_for_status()
            
            build_id = response.json()["data"]["id"]
            logger.info(f"✅ Percy build created: {build_id}")
            return build_id
        
        except Exception as e:
            logger.error(f"Failed to create Percy build: {e}")
            raise
    
    def upload_snapshot(
        self,
        build_id: str,
        name: str,
        screenshot_path: str,
        widths: List[int],
        min_height: int = 1024
    ) -> Dict[str, Any]:
        """Upload snapshot to Percy"""
        try:
            # Read screenshot
            with open(screenshot_path, "rb") as f:
                screenshot_data = f.read()
            
            # Upload
            response = self.client.post(
                f"{self.base_url}/builds/{build_id}/snapshots",
                json={
                    "data": {
                        "type": "snapshots",
                        "attributes": {
                            "name": name,
                            "widths": widths,
                            "minimum-height": min_height,
                            "enable-javascript": True
                        }
                    }
                },
                files={"snapshot": screenshot_data}
            )
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"✅ Snapshot uploaded: {name}")
            return result
        
        except Exception as e:
            logger.error(f"Snapshot upload failed for {name}: {e}")
            raise
    
    def finalize_build(self, build_id: str) -> None:
        """Finalize Percy build"""
        try:
            response = self.client.post(
                f"{self.base_url}/builds/{build_id}/finalize"
            )
            response.raise_for_status()
            logger.info(f"✅ Percy build finalized: {build_id}")
        
        except Exception as e:
            logger.error(f"Build finalization failed: {e}")
    
    def close(self):
        self.client.close()


# ==================== Local Visual Comparison ====================

class LocalVisualComparator:
    """
    Local screenshot comparison engine.
    Fallback when Percy is unavailable.
    """
    
    def __init__(self, baseline_dir: str, diff_dir: str, threshold: float = 0.01):
        self.baseline_dir = Path(baseline_dir)
        self.diff_dir = Path(diff_dir)
        self.threshold = threshold
        
        self.baseline_dir.mkdir(parents=True, exist_ok=True)
        self.diff_dir.mkdir(parents=True, exist_ok=True)
    
    def compare(
        self,
        name: str,
        screenshot_path: str,
        update_baseline: bool = False
    ) -> Tuple[bool, float, Optional[str]]:
        """
        Compare screenshot with baseline.
        
        Args:
            name: Snapshot name
            screenshot_path: Path to current screenshot
            update_baseline: Update baseline instead of comparing
        
        Returns:
            (passed, diff_percentage, diff_image_path)
        """
        baseline_path = self.baseline_dir / f"{name}.png"
        
        # If no baseline exists, create it
        if not baseline_path.exists() or update_baseline:
            self._save_baseline(screenshot_path, baseline_path)
            logger.info(f"✅ Baseline saved: {name}")
            return True, 0.0, None
        
        # Load images
        try:
            current = Image.open(screenshot_path).convert("RGB")
            baseline = Image.open(baseline_path).convert("RGB")
        except Exception as e:
            logger.error(f"Failed to load images: {e}")
            return False, 1.0, None
        
        # Resize if needed
        if current.size != baseline.size:
            logger.warning(f"Size mismatch: {current.size} vs {baseline.size}")
            baseline = baseline.resize(current.size, Image.Resampling.LANCZOS)
        
        # Calculate difference
        diff_percentage, diff_image = self._calculate_diff(current, baseline)
        
        # Save diff if threshold exceeded
        diff_path = None
        if diff_percentage > self.threshold:
            diff_path = self.diff_dir / f"{name}_diff.png"
            diff_image.save(diff_path)
            logger.warning(f"⚠️ Visual diff detected: {name} ({diff_percentage:.2%})")
        
        passed = diff_percentage <= self.threshold
        return passed, diff_percentage, str(diff_path) if diff_path else None
    
    def _save_baseline(self, screenshot_path: str, baseline_path: Path) -> None:
        """Save screenshot as baseline"""
        img = Image.open(screenshot_path)
        img.save(baseline_path)
    
    def _calculate_diff(self, img1: Image.Image, img2: Image.Image) -> Tuple[float, Image.Image]:
        """
        Calculate pixel difference between images.
        
        Returns:
            (diff_percentage, diff_image)
        """
        # Calculate difference
        diff = ImageChops.difference(img1, img2)
        
        # Convert to grayscale for analysis
        diff_gray = diff.convert('L')
        diff_data = list(diff_gray.getdata())
        
        # Calculate percentage of different pixels
        total_pixels = len(diff_data)
        different_pixels = sum(1 for pixel in diff_data if pixel > 10)  # Tolerance threshold
        diff_percentage = different_pixels / total_pixels if total_pixels > 0 else 0.0
        
        # Create visual diff image (highlight differences in red)
        diff_highlight = Image.new('RGB', img1.size)
        draw = ImageDraw.Draw(diff_highlight)
        
        for y in range(img1.height):
            for x in range(img1.width):
                idx = y * img1.width + x
                if diff_data[idx] > 10:
                    draw.point((x, y), fill=(255, 0, 0))  # Red for differences
                else:
                    draw.point((x, y), fill=img1.getpixel((x, y)))
        
        return diff_percentage, diff_highlight


# ==================== Main Visual Regression Engine ====================

class VisualRegressionEngine:
    """
    Production-grade visual regression testing engine.
    
    Features:
    - Percy.io integration for cloud comparison
    - Local comparison fallback
    - Responsive testing (multiple widths)
    - Baseline management
    - Approval workflow
    """
    
    def __init__(self, config: Optional[VisualConfig] = None):
        self.config = config or VisualConfig()
        
        # Percy client (if enabled)
        self.percy: Optional[PercyClient] = None
        if self.config.enable_percy and self.config.percy_token:
            try:
                self.percy = PercyClient(self.config.percy_token)
            except Exception as e:
                logger.warning(f"Percy initialization failed: {e}")
                self.config.enable_percy = False
        
        # Local comparator
        self.local: Optional[LocalVisualComparator] = None
        if self.config.enable_local:
            self.local = LocalVisualComparator(
                baseline_dir=self.config.baseline_dir,
                diff_dir=self.config.diff_dir,
                threshold=self.config.threshold
            )
        
        # Snapshot registry
        self.snapshots: List[VisualSnapshot] = []
        
        # Percy build ID (for current test run)
        self._percy_build_id: Optional[str] = None
        
        logger.info("✅ Visual regression engine initialized")
    
    # ==================== Snapshot Capture ====================
    
    def snapshot(
        self,
        page,  # Playwright Page object
        name: str,
        widths: Optional[List[int]] = None
    ) -> VisualSnapshot:
        """
        Capture visual snapshot.
        
        Args:
            page: Playwright page object
            name: Snapshot name
            widths: Viewport widths to test
        
        Returns:
            VisualSnapshot metadata
        """
        widths = widths or self.config.widths
        
        # Sanitize name for filename
        safe_name = name.replace(" ", "_").replace("/", "_")
        
        # Capture screenshots for each width
        snapshots = []
        
        for width in widths:
            # Set viewport
            page.set_viewport_size({"width": width, "height": self.config.min_height})
            
            # Take screenshot
            screenshot_path = Path(self.config.baseline_dir) / f"{safe_name}_{width}px.png"
            screenshot_path.parent.mkdir(parents=True, exist_ok=True)
            
            page.screenshot(path=str(screenshot_path), full_page=True)
            
            # Create snapshot metadata
            snapshot = VisualSnapshot(
                name=f"{name} @ {width}px",
                url=page.url,
                width=width,
                height=self.config.min_height,
                screenshot_path=str(screenshot_path)
            )
            
            # Compare with baseline (local)
            if self.local:
                passed, diff_pct, diff_path = self.local.compare(
                    name=f"{safe_name}_{width}px",
                    screenshot_path=str(screenshot_path)
                )
                
                snapshot.diff_percentage = diff_pct
                snapshot.diff_path = diff_path
                snapshot.approved = passed
            
            # Upload to Percy (if enabled)
            if self.percy and self._percy_build_id:
                try:
                    self.percy.upload_snapshot(
                        build_id=self._percy_build_id,
                        name=f"{name} @ {width}px",
                        screenshot_path=str(screenshot_path),
                        widths=[width],
                        min_height=self.config.min_height
                    )
                except Exception as e:
                    logger.error(f"Percy upload failed: {e}")
            
            snapshots.append(snapshot)
            self.snapshots.append(snapshot)
        
        logger.info(f"✅ Captured {len(snapshots)} snapshots for: {name}")
        return snapshots[0]  # Return first snapshot
    
    # ==================== Session Management ====================
    
    def start_session(self, project_slug: str = "ai-qa-tests", branch: str = "main") -> None:
        """Start visual testing session (creates Percy build)"""
        if self.percy:
            try:
                self._percy_build_id = self.percy.create_build(project_slug, branch)
                logger.info(f"✅ Visual testing session started")
            except Exception as e:
                logger.error(f"Failed to start Percy session: {e}")
    
    def end_session(self) -> None:
        """End visual testing session (finalizes Percy build)"""
        if self.percy and self._percy_build_id:
            try:
                self.percy.finalize_build(self._percy_build_id)
                logger.info("✅ Visual testing session ended")
            except Exception as e:
                logger.error(f"Failed to end Percy session: {e}")
        
        self._percy_build_id = None
    
    # ==================== Baseline Management ====================
    
    def update_baselines(self) -> int:
        """Update all baselines with current screenshots"""
        if not self.local:
            logger.warning("Local comparison not enabled")
            return 0
        
        count = 0
        for snapshot in self.snapshots:
            if snapshot.screenshot_path:
                baseline_path = Path(self.config.baseline_dir) / Path(snapshot.screenshot_path).name
                
                try:
                    img = Image.open(snapshot.screenshot_path)
                    img.save(baseline_path)
                    count += 1
                except Exception as e:
                    logger.error(f"Failed to update baseline for {snapshot.name}: {e}")
        
        logger.info(f"✅ Updated {count} baselines")
        return count
    
    def clear_baselines(self) -> int:
        """Clear all baselines (use with caution!)"""
        count = 0
        for baseline_file in Path(self.config.baseline_dir).glob("*.png"):
            baseline_file.unlink()
            count += 1
        
        logger.warning(f"⚠️ Cleared {count} baselines")
        return count
    
    # ==================== Reporting ====================
    
    def get_report(self) -> Dict[str, Any]:
        """Generate visual testing report"""
        total = len(self.snapshots)
        passed = sum(1 for s in self.snapshots if s.approved)
        failed = total - passed
        
        failures = [
            {
                "name": s.name,
                "diff_percentage": s.diff_percentage,
                "diff_path": s.diff_path,
                "screenshot_path": s.screenshot_path
            }
            for s in self.snapshots if not s.approved
        ]
        
        return {
            "total_snapshots": total,
            "passed": passed,
            "failed": failed,
            "pass_rate": passed / total if total > 0 else 0.0,
            "failures": failures,
            "percy_enabled": self.config.enable_percy,
            "percy_build_id": self._percy_build_id,
            "timestamp": datetime.now().isoformat()
        }
    
    def export_report(self, output_file: str) -> None:
        """Export report to JSON"""
        report = self.get_report()
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✅ Report exported to {output_file}")
    
    # ==================== Cleanup ====================
    
    def close(self) -> None:
        """Cleanup resources"""
        if self.percy:
            self.percy.close()


# ==================== Integration with Self-Healing ====================

class VisualAwareSelfHealing:
    """
    Self-healing enhanced with visual regression.
    Captures visual snapshots after healing.
    """
    
    def __init__(self, page, visual_engine: VisualRegressionEngine):
        from modules.self_healing import SelfHealing
        
        self.healer = SelfHealing(page)
        self.visual = visual_engine
        self.page = page
    
    def find_and_click(self, hint: str, test_name: str) -> bool:
        """Click with visual snapshot"""
        success = self.healer.find_and_click(hint)
        
        if success:
            # Capture visual snapshot after action
            self.visual.snapshot(self.page, f"{test_name}_after_{hint}")
        
        return success


# ==================== Usage Example ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize engine
    visual = VisualRegressionEngine()
    
    # Start session
    visual.start_session()
    
    # Example with Playwright
    from playwright.sync_api import sync_playwright
    
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        
        page.goto("https://example.com")
        
        # Capture snapshot
        visual.snapshot(page, "Homepage", widths=[1920, 768])
        
        browser.close()
    
    # End session
    visual.end_session()
    
    # Get report
    report = visual.get_report()
    print(json.dumps(report, indent=2))
    
    visual.close()
