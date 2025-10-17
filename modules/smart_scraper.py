"""
Smart Scraper - LLM-powered intelligent crawling
"""
import os
import logging
from typing import Dict, List, Any
from openai import OpenAI

logger = logging.getLogger(__name__)


class SmartScraper:
    """LLM-powered intelligent web scraping"""
    
    def __init__(self):
        self.enabled = os.getenv("SMART_CRAWLING_ENABLED", "false").lower() == "true"
        self.model = os.getenv("SMART_CRAWLING_MODEL", "gpt-4o")
        
        if self.enabled:
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            logger.info("🧠 Smart Scraper initialized (LLM-powered)")
    
    def prioritize_urls(self, urls: List[str], goal: str) -> List[Dict[str, Any]]:
        """
        Use LLM to prioritize which URLs to crawl
        
        Args:
            urls: List of discovered URLs
            goal: Testing goal (e.g., "test checkout flow")
        
        Returns:
            Prioritized URLs with scores and reasons
        """
        if not self.enabled or not urls:
            return [{"url": u, "priority": 50, "reason": "default"} for u in urls]
        
        prompt = f"""
Given these URLs from a website, prioritize which ones are most important to crawl for testing purposes.

Goal: {goal}

URLs:
{chr(10).join(f"- {url}" for url in urls[:20])}

Return a JSON array with each URL, priority (0-100), and reason:
[
  {{"url": "...", "priority": 90, "reason": "Login page - critical for auth testing"}},
  ...
]

Focus on: forms, authentication, checkout, APIs, critical user flows.
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            
            import json
            result = json.loads(response.choices[0].message.content)
            return result.get("urls", [])
        
        except Exception as e:
            logger.warning(f"Smart prioritization failed: {e}")
            return [{"url": u, "priority": 50} for u in urls]
    
    def suggest_test_scenarios(self, page_data: Dict[str, Any]) -> List[str]:
        """
        Generate test scenarios based on page content
        
        Args:
            page_data: Scraped page data (HTML, forms, etc.)
        
        Returns:
            List of suggested test scenarios
        """
        if not self.enabled:
            return []
        
        prompt = f"""
Analyze this page and suggest specific test scenarios:

Page URL: {page_data.get('url')}
Page Title: {page_data.get('title')}
Forms: {len(page_data.get('forms', []))}
Links: {len(page_data.get('links', []))}
Input Fields: {[f['name'] for f in page_data.get('forms', [])[0].get('inputs', [])]} if page_data.get('forms') else []

Suggest 5-10 specific test scenarios that should be covered.
Return as JSON array: ["Test scenario 1", "Test scenario 2", ...]
"""
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.7
            )
            
            import json
            scenarios = json.loads(response.choices[0].message.content)
            return scenarios if isinstance(scenarios, list) else []
        
        except Exception as e:
            logger.warning(f"Scenario generation failed: {e}")
            return []
