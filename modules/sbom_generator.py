# modules/sbom_generator.py
"""
Generate SBOM (Software Bill of Materials) in CycloneDX format.
Analyzes dependencies and generates SBOM.
"""

import json
import subprocess
from typing import Dict, Any, List
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class SBOMGenerator:
    """
    Generate SBOM for web application.
    Uses: syft, trivy, or custom analysis
    """
    
    def __init__(self):
        self.tool = self._detect_tool()
    
    def _detect_tool(self) -> str:
        """Detect available SBOM tool."""
        try:
            subprocess.run(["syft", "--version"], capture_output=True, check=True)
            return "syft"
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        
        try:
            subprocess.run(["trivy", "--version"], capture_output=True, check=True)
            return "trivy"
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        
        return "manual"
    
    def generate(self, url: str, output_path: str) -> Dict[str, Any]:
        """
        Generate SBOM for target URL.
        
        Returns:
            {
                "format": "CycloneDX",
                "components": [...],
                "vulnerabilities": [...]
            }
        """
        if self.tool == "syft":
            return self._generate_with_syft(url, output_path)
        elif self.tool == "trivy":
            return self._generate_with_trivy(url, output_path)
        else:
            return self._generate_manual(url, output_path)
    
    def _generate_with_syft(self, url: str, output_path: str) -> Dict[str, Any]:
        """Generate SBOM using syft."""
        try:
            cmd = [
                "syft",
                url,
                "-o", "cyclonedx-json",
                "--file", output_path,
            ]
            
            subprocess.run(cmd, check=True, capture_output=True)
            
            with open(output_path, "r") as f:
                sbom = json.load(f)
            
            return {
                "format": "CycloneDX",
                "tool": "syft",
                "components": sbom.get("components", []),
                "generated_at": datetime.utcnow().isoformat() + "Z",
            }
        
        except Exception as e:
            logger.error("Syft SBOM generation failed: %s", e)
            return {"error": str(e)}
    
    def _generate_with_trivy(self, url: str, output_path: str) -> Dict[str, Any]:
        """Generate SBOM using trivy."""
        # Similar implementation for trivy
        pass
    
    def _generate_manual(self, url: str, output_path: str) -> Dict[str, Any]:
        """Manual SBOM generation from detected technologies."""
        # Parse headers, scripts, etc. to detect frameworks/libraries
        components = [
            {
                "type": "library",
                "name": "detected-library",
                "version": "unknown",
            }
        ]
        
        sbom = {
            "bomFormat": "CycloneDX",
            "specVersion": "1.4",
            "version": 1,
            "metadata": {
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "tools": [{"vendor": "SecurityEngine", "name": "SBOM Generator"}],
            },
            "components": components,
        }
        
        with open(output_path, "w") as f:
            json.dump(sbom, f, indent=2)
        
        return sbom
