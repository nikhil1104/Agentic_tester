# modules/threat_intel.py
"""
Real-time threat intelligence integration.
Integrates with: AlienVault OTX, AbuseIPDB, VirusTotal
"""

import asyncio
import httpx
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class ThreatIntelligence:
    """
    Integrate with threat intelligence feeds.
    Requires API keys for: ALIENVAULT_API_KEY, VIRUSTOTAL_API_KEY
    """
    
    def __init__(self):
        import os
        self.otx_api_key = os.getenv("ALIENVAULT_API_KEY")
        self.vt_api_key = os.getenv("VIRUSTOTAL_API_KEY")
        self.enabled = bool(self.otx_api_key or self.vt_api_key)
    
    async def check_url_reputation(self, url: str) -> Dict[str, Any]:
        """Check URL against threat intelligence feeds."""
        if not self.enabled:
            return {"enabled": False}
        
        results = {
            "url": url,
            "malicious": False,
            "sources": [],
        }
        
        # Check AlienVault OTX
        if self.otx_api_key:
            otx_result = await self._check_otx(url)
            if otx_result.get("malicious"):
                results["malicious"] = True
                results["sources"].append("AlienVault OTX")
        
        # Check VirusTotal
        if self.vt_api_key:
            vt_result = await self._check_virustotal(url)
            if vt_result.get("malicious"):
                results["malicious"] = True
                results["sources"].append("VirusTotal")
        
        return results
    
    async def _check_otx(self, url: str) -> Dict[str, Any]:
        """Check AlienVault OTX."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {"X-OTX-API-KEY": self.otx_api_key}
                response = await client.get(
                    f"https://otx.alienvault.com/api/v1/indicators/url/{url}/general",
                    headers=headers,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    pulse_count = data.get("pulse_info", {}).get("count", 0)
                    
                    return {
                        "malicious": pulse_count > 0,
                        "pulse_count": pulse_count,
                    }
        except Exception as e:
            logger.error("OTX check failed: %s", e)
        
        return {"malicious": False}
    
    async def _check_virustotal(self, url: str) -> Dict[str, Any]:
        """Check VirusTotal."""
        try:
            import base64
            url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
            
            async with httpx.AsyncClient(timeout=10.0) as client:
                headers = {"x-apikey": self.vt_api_key}
                response = await client.get(
                    f"https://www.virustotal.com/api/v3/urls/{url_id}",
                    headers=headers,
                )
                
                if response.status_code == 200:
                    data = response.json()
                    stats = data.get("data", {}).get("attributes", {}).get("last_analysis_stats", {})
                    malicious_count = stats.get("malicious", 0)
                    
                    return {
                        "malicious": malicious_count > 0,
                        "malicious_count": malicious_count,
                        "total_scans": sum(stats.values()),
                    }
        except Exception as e:
            logger.error("VirusTotal check failed: %s", e)
        
        return {"malicious": False}
