# modules/serper_integration.py
"""
Serper.dev API Integration (Production-Grade)
Real-time SERP data for competitive analysis and test discovery

Features:
- Google Search API integration
- Rate limiting and retry logic
- Response caching
- Competitor analysis
- Similar site discovery
- Test pattern mining from search results
- Async support for parallel queries
"""

from __future__ import annotations

import os
import json
import time
import hashlib
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class SerperConfig:
    """Serper.dev API configuration"""
    api_key: str = field(default_factory=lambda: os.getenv("SERPER_API_KEY", ""))
    base_url: str = "https://google.serper.dev"
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    enable_cache: bool = True
    cache_dir: str = "./data/serper_cache"
    cache_ttl_hours: int = 24
    results_per_query: int = 10
    country: str = "us"
    language: str = "en"
    
    def __post_init__(self):
        if not self.api_key:
            logger.warning("SERPER_API_KEY not set. Set via environment variable.")


@dataclass
class SearchResult:
    """Search result data model"""
    title: str
    link: str
    snippet: str
    position: int
    domain: str
    date: Optional[str] = None
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any], position: int) -> SearchResult:
        """Create from Serper API response"""
        link = data.get("link", "")
        domain = urlparse(link).netloc if link else ""
        
        return cls(
            title=data.get("title", ""),
            link=link,
            snippet=data.get("snippet", ""),
            position=position,
            domain=domain,
            date=data.get("date")
        )


# ==================== Cache Manager ====================

class SerperCache:
    """Disk-based cache for Serper API responses"""
    
    def __init__(self, cache_dir: str, ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)
    
    def _get_cache_key(self, query: str, **params) -> str:
        """Generate cache key from query and params"""
        combined = f"{query}:{json.dumps(params, sort_keys=True)}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def get(self, query: str, **params) -> Optional[Dict[str, Any]]:
        """Get cached response"""
        key = self._get_cache_key(query, **params)
        cache_file = self.cache_dir / f"{key}.json"
        
        if not cache_file.exists():
            return None
        
        try:
            # Check TTL
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mtime > self.ttl:
                cache_file.unlink()
                return None
            
            # Load cached data
            with open(cache_file, "r", encoding="utf-8") as f:
                return json.load(f)
        
        except Exception as e:
            logger.debug(f"Cache read failed: {e}")
            return None
    
    def set(self, query: str, response: Dict[str, Any], **params) -> None:
        """Cache response"""
        try:
            key = self._get_cache_key(query, **params)
            cache_file = self.cache_dir / f"{key}.json"
            
            with open(cache_file, "w", encoding="utf-8") as f:
                json.dump(response, f, indent=2)
        
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")
    
    def clear(self) -> int:
        """Clear all cached data"""
        count = 0
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
            count += 1
        return count


# ==================== Main Serper Client ====================

class SerperClient:
    """
    Production-grade Serper.dev API client.
    
    Features:
    - Synchronous and asynchronous support
    - Automatic retries with exponential backoff
    - Response caching
    - Rate limiting
    - Error handling
    """
    
    def __init__(self, config: Optional[SerperConfig] = None):
        """
        Initialize Serper client.
        
        Args:
            config: Serper configuration
        """
        self.config = config or SerperConfig()
        
        if not self.config.api_key:
            raise ValueError("Serper API key required. Set SERPER_API_KEY environment variable.")
        
        # HTTP client
        self.client = httpx.Client(
            timeout=self.config.timeout,
            headers={
                "X-API-KEY": self.config.api_key,
                "Content-Type": "application/json"
            }
        )
        
        # Async client (lazy initialized)
        self._async_client: Optional[httpx.AsyncClient] = None
        
        # Cache
        self.cache = SerperCache(
            cache_dir=self.config.cache_dir,
            ttl_hours=self.config.cache_ttl_hours
        ) if self.config.enable_cache else None
        
        # Rate limiting
        self._last_request_time = 0.0
        self._min_request_interval = 0.1  # 10 requests/second max
        
        logger.info("✅ Serper client initialized")
    
    # ==================== Core Search ====================
    
    def search(
        self,
        query: str,
        num_results: Optional[int] = None,
        country: Optional[str] = None,
        language: Optional[str] = None,
        use_cache: bool = True
    ) -> Dict[str, Any]:
        """
        Perform Google search via Serper API.
        
        Args:
            query: Search query
            num_results: Number of results to return
            country: Country code (us, uk, etc.)
            language: Language code (en, es, etc.)
            use_cache: Use cached results if available
        
        Returns:
            Search results
        """
        params = {
            "num": num_results or self.config.results_per_query,
            "gl": country or self.config.country,
            "hl": language or self.config.language
        }
        
        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(query, **params)
            if cached:
                logger.debug(f"Cache hit for query: {query}")
                return cached
        
        # Rate limiting
        self._rate_limit()
        
        # Make request with retries
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.post(
                    f"{self.config.base_url}/search",
                    json={"q": query, **params}
                )
                response.raise_for_status()
                
                result = response.json()
                
                # Cache result
                if self.cache:
                    self.cache.set(query, result, **params)
                
                return result
            
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 429:  # Rate limited
                    wait_time = (2 ** attempt) * self.config.retry_delay
                    logger.warning(f"Rate limited, waiting {wait_time}s...")
                    time.sleep(wait_time)
                elif e.response.status_code >= 500:  # Server error
                    if attempt < self.config.max_retries - 1:
                        time.sleep(self.config.retry_delay)
                    else:
                        raise
                else:
                    raise
            
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    logger.warning(f"Search attempt {attempt + 1} failed: {e}")
                    time.sleep(self.config.retry_delay)
                else:
                    logger.error(f"Search failed after {self.config.max_retries} attempts")
                    raise
        
        return {}
    
    async def search_async(
        self,
        query: str,
        num_results: Optional[int] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """Async version of search"""
        if not self._async_client:
            self._async_client = httpx.AsyncClient(
                timeout=self.config.timeout,
                headers=self.client.headers
            )
        
        params = {
            "num": num_results or self.config.results_per_query,
            "gl": kwargs.get("country", self.config.country),
            "hl": kwargs.get("language", self.config.language)
        }
        
        response = await self._async_client.post(
            f"{self.config.base_url}/search",
            json={"q": query, **params}
        )
        response.raise_for_status()
        
        return response.json()
    
    def _rate_limit(self) -> None:
        """Enforce rate limiting"""
        now = time.time()
        elapsed = now - self._last_request_time
        
        if elapsed < self._min_request_interval:
            time.sleep(self._min_request_interval - elapsed)
        
        self._last_request_time = time.time()
    
    # ==================== Parsed Results ====================
    
    def get_organic_results(self, query: str, **kwargs) -> List[SearchResult]:
        """Get parsed organic search results"""
        response = self.search(query, **kwargs)
        
        results = []
        for i, item in enumerate(response.get("organic", []), 1):
            results.append(SearchResult.from_dict(item, position=i))
        
        return results
    
    def get_related_searches(self, query: str) -> List[str]:
        """Get related search queries"""
        response = self.search(query)
        
        related = []
        for item in response.get("relatedSearches", []):
            if "query" in item:
                related.append(item["query"])
        
        return related
    
    def get_people_also_ask(self, query: str) -> List[Dict[str, str]]:
        """Get People Also Ask questions"""
        response = self.search(query)
        
        questions = []
        for item in response.get("peopleAlsoAsk", []):
            questions.append({
                "question": item.get("question", ""),
                "snippet": item.get("snippet", ""),
                "link": item.get("link", "")
            })
        
        return questions
    
    # ==================== Test Intelligence Features ====================
    
    def find_competitors(self, url: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Find competitor websites for a given URL.
        
        Args:
            url: Target URL
            top_k: Number of competitors to return
        
        Returns:
            List of competitor info
        """
        # Extract domain
        domain = urlparse(url).netloc
        
        # Search for similar sites
        query = f"site similar to {domain}"
        results = self.get_organic_results(query, num_results=top_k * 2)
        
        # Filter out the target domain
        competitors = []
        for result in results:
            if result.domain != domain and result.domain:
                competitors.append({
                    "domain": result.domain,
                    "url": result.link,
                    "title": result.title,
                    "snippet": result.snippet
                })
                
                if len(competitors) >= top_k:
                    break
        
        return competitors
    
    def discover_test_patterns(self, technology: str) -> List[str]:
        """
        Discover common test patterns for a technology.
        
        Args:
            technology: Technology name (e.g., "React", "Django")
        
        Returns:
            List of test pattern descriptions
        """
        query = f"{technology} testing best practices patterns"
        results = self.get_organic_results(query, num_results=10)
        
        patterns = []
        for result in results:
            # Extract test patterns from snippets
            snippet = result.snippet.lower()
            
            if any(kw in snippet for kw in ["test", "testing", "pattern", "practice"]):
                patterns.append(f"{result.title}: {result.snippet}")
        
        return patterns
    
    def find_test_documentation(self, framework: str) -> List[Dict[str, str]]:
        """
        Find official test documentation for a framework.
        
        Args:
            framework: Framework name
        
        Returns:
            List of documentation links
        """
        query = f"{framework} official testing documentation"
        results = self.get_organic_results(query, num_results=5)
        
        docs = []
        for result in results:
            # Prefer official docs
            if any(domain in result.domain for domain in ["github.io", "readthedocs", framework.lower()]):
                docs.append({
                    "title": result.title,
                    "url": result.link,
                    "snippet": result.snippet,
                    "domain": result.domain
                })
        
        return docs
    
    def search_known_bugs(self, error_message: str) -> List[Dict[str, Any]]:
        """
        Search for known bugs/issues related to an error.
        
        Args:
            error_message: Error message to search
        
        Returns:
            List of relevant discussions/issues
        """
        # Clean error message
        clean_error = error_message[:100]  # Limit length
        
        query = f"{clean_error} site:github.com OR site:stackoverflow.com"
        results = self.get_organic_results(query, num_results=10)
        
        issues = []
        for result in results:
            issues.append({
                "title": result.title,
                "url": result.link,
                "snippet": result.snippet,
                "source": "GitHub" if "github" in result.domain else "StackOverflow"
            })
        
        return issues
    
    # ==================== Batch Operations ====================
    
    async def batch_search(self, queries: List[str], **kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Perform multiple searches concurrently.
        
        Args:
            queries: List of search queries
            **kwargs: Additional search parameters
        
        Returns:
            Dictionary mapping queries to results
        """
        import asyncio
        
        if not self._async_client:
            self._async_client = httpx.AsyncClient(
                timeout=self.config.timeout,
                headers=self.client.headers
            )
        
        # Create tasks
        tasks = [self.search_async(q, **kwargs) for q in queries]
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Map results
        output = {}
        for query, result in zip(queries, results):
            if isinstance(result, Exception):
                logger.error(f"Batch search failed for '{query}': {result}")
                output[query] = {}
            else:
                output[query] = result
        
        return output
    
    # ==================== Cleanup ====================
    
    def close(self) -> None:
        """Close HTTP clients"""
        self.client.close()
        if self._async_client:
            import asyncio
            asyncio.run(self._async_client.aclose())
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        self.close()
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, *args):
        if self._async_client:
            await self._async_client.aclose()


# ==================== High-Level Test Intelligence ====================

class TestIntelligence:
    """
    High-level test intelligence using Serper.
    """
    
    def __init__(self, serper_client: Optional[SerperClient] = None):
        self.serper = serper_client or SerperClient()
    
    def analyze_target(self, url: str) -> Dict[str, Any]:
        """
        Comprehensive analysis of a target URL.
        
        Args:
            url: Target URL to analyze
        
        Returns:
            Analysis report
        """
        domain = urlparse(url).netloc
        
        logger.info(f"Analyzing target: {domain}")
        
        analysis = {
            "target": url,
            "domain": domain,
            "analyzed_at": datetime.now().isoformat(),
            "competitors": [],
            "technologies": [],
            "related_queries": []
        }
        
        try:
            # Find competitors
            competitors = self.serper.find_competitors(url, top_k=5)
            analysis["competitors"] = competitors
            
            # Get related searches (for test ideas)
            query = f"{domain} features functionality"
            related = self.serper.get_related_searches(query)
            analysis["related_queries"] = related
            
            # People also ask (for edge cases)
            paa = self.serper.get_people_also_ask(query)
            analysis["edge_case_ideas"] = [q["question"] for q in paa[:5]]
        
        except Exception as e:
            logger.error(f"Target analysis failed: {e}", exc_info=True)
            analysis["error"] = str(e)
        
        return analysis
    
    def discover_test_cases(self, requirement: str, top_k: int = 10) -> List[str]:
        """
        Discover relevant test cases from web.
        
        Args:
            requirement: Test requirement
            top_k: Number of test cases to discover
        
        Returns:
            List of test case descriptions
        """
        query = f"{requirement} test cases examples"
        results = self.serper.get_organic_results(query, num_results=top_k)
        
        test_cases = []
        for result in results:
            test_cases.append(f"{result.title}\n{result.snippet}\nSource: {result.link}")
        
        return test_cases


# ==================== Usage Example ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize client
    client = SerperClient()
    
    # Search
    results = client.search("Playwright testing best practices")
    print(f"Found {len(results.get('organic', []))} results")
    
    # Get organic results
    organic = client.get_organic_results("Python testing patterns", num_results=5)
    for result in organic:
        print(f"{result.position}. {result.title}")
        print(f"   {result.link}")
    
    # Find competitors
    competitors = client.find_competitors("https://playwright.dev", top_k=3)
    print(f"\nCompetitors:")
    for comp in competitors:
        print(f"- {comp['domain']}: {comp['title']}")
    
    # Test intelligence
    intel = TestIntelligence(client)
    analysis = intel.analyze_target("https://example.com")
    print(f"\nAnalysis: {json.dumps(analysis, indent=2)}")
    
    client.close()
