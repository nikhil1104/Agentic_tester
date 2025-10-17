"""
Authenticated Web Scraper - Crawls protected pages after login
"""
from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List
from playwright.async_api import async_playwright, Page, Browser, BrowserContext

logger = logging.getLogger(__name__)


class AuthenticatedScraper:
    """
    Web scraper that handles authentication and crawls protected pages
    """
    
    def __init__(self, credentials: Dict[str, str], login_config: Dict[str, Any]):
        """
        Initialize authenticated scraper
        
        Args:
            credentials: {'username': 'user', 'password': 'pass'}
            login_config: {
                'url': '/login',
                'username_selector': 'input[name="username"]',
                'password_selector': 'input[name="password"]',
                'submit_selector': 'button[type="submit"]',
                'success_indicator': '.dashboard, .home, [data-testid="user-menu"]'
            }
        """
        self.credentials = credentials
        self.login_config = login_config
        self.context: Optional[BrowserContext] = None
        self.authenticated = False
        
    async def login(self, page: Page) -> bool:
        """
        Perform login and verify success
        
        Returns:
            True if login successful, False otherwise
        """
        try:
            login_url = self.login_config.get('url', '/login')
            base_url = self.login_config.get('base_url', 'http://localhost')
            
            full_url = f"{base_url}{login_url}" if not login_url.startswith('http') else login_url
            
            logger.info(f"🔐 Attempting login at {full_url}")
            
            # Navigate to login page
            await page.goto(full_url, wait_until='networkidle')
            
            # Wait for login form
            username_selector = self.login_config.get('username_selector', 'input[name="username"]')
            await page.wait_for_selector(username_selector, timeout=10000)
            
            # Fill credentials
            logger.info("📝 Filling credentials...")
            await page.fill(username_selector, self.credentials['username'])
            
            password_selector = self.login_config.get('password_selector', 'input[name="password"]')
            await page.fill(password_selector, self.credentials['password'])
            
            # Submit form
            submit_selector = self.login_config.get('submit_selector', 'button[type="submit"]')
            logger.info("🚀 Submitting login form...")
            
            # Click and wait for navigation
            async with page.expect_navigation(timeout=30000):
                await page.click(submit_selector)
            
            # Verify login success
            success_indicator = self.login_config.get('success_indicator', '.dashboard')
            
            try:
                await page.wait_for_selector(success_indicator, timeout=5000)
                logger.info("✅ Login successful!")
                self.authenticated = True
                return True
                
            except Exception as e:
                # Check if URL changed (indicates success even without specific element)
                current_url = page.url
                if 'login' not in current_url.lower():
                    logger.info("✅ Login successful (URL changed)")
                    self.authenticated = True
                    return True
                else:
                    logger.error(f"❌ Login failed: {e}")
                    return False
                    
        except Exception as e:
            logger.error(f"❌ Login error: {e}")
            return False
    
    async def crawl_authenticated(
        self,
        start_url: str,
        max_pages: int = 20,
        max_depth: int = 2
    ) -> Dict[str, Any]:
        """
        Crawl website after authentication
        
        Returns:
            {
                'pages': [{url, html, title, links}],
                'protected_pages': [urls],
                'authentication': {'status': 'success', 'method': 'form'},
                'session_info': {cookies, storage}
            }
        """
        result = {
            'pages': [],
            'protected_pages': [],
            'authentication': {'status': 'pending'},
            'session_info': {},
            'api_calls': [],
            'errors': []
        }
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            self.context = await browser.new_context(
                viewport={'width': 1920, 'height': 1080},
                user_agent='Mozilla/5.0 (compatible; AI-QA-Agent/1.0)',
                java_script_enabled=True
            )
            
            # Enable request interception for API monitoring
            await self.context.route("**/*", lambda route: self._intercept_request(route, result))
            
            page = await self.context.new_page()
            
            try:
                # Step 1: Login
                login_success = await self.login(page)
                
                if not login_success:
                    result['authentication']['status'] = 'failed'
                    result['errors'].append('Authentication failed')
                    return result
                
                result['authentication']['status'] = 'success'
                result['authentication']['method'] = 'form-based'
                
                # Step 2: Save session state
                cookies = await self.context.cookies()
                storage_state = await self.context.storage_state()
                
                result['session_info'] = {
                    'cookies': cookies,
                    'storage_state': storage_state
                }
                
                # Step 3: Crawl authenticated pages
                logger.info(f"🔍 Crawling authenticated application (max_pages={max_pages})")
                
                visited = set()
                to_visit = [(start_url, 0)]  # (url, depth)
                
                while to_visit and len(result['pages']) < max_pages:
                    current_url, depth = to_visit.pop(0)
                    
                    if current_url in visited or depth > max_depth:
                        continue
                    
                    visited.add(current_url)
                    
                    try:
                        logger.info(f"📄 Scraping [{len(result['pages'])+1}/{max_pages}] depth={depth}: {current_url}")
                        
                        await page.goto(current_url, wait_until='networkidle', timeout=30000)
                        
                        # Extract page info
                        page_data = {
                            'url': current_url,
                            'title': await page.title(),
                            'html': await page.content(),
                            'depth': depth,
                            'is_protected': True  # All pages crawled here are protected
                        }
                        
                        # Extract links for further crawling
                        links = await page.eval_on_selector_all(
                            'a[href]',
                            '(elements) => elements.map(e => e.href)'
                        )
                        
                        page_data['links'] = links
                        result['pages'].append(page_data)
                        result['protected_pages'].append(current_url)
                        
                        # Add new links to crawl queue
                        base_domain = self._extract_domain(start_url)
                        for link in links:
                            if self._is_same_domain(link, base_domain) and link not in visited:
                                to_visit.append((link, depth + 1))
                        
                    except Exception as e:
                        logger.error(f"Error crawling {current_url}: {e}")
                        result['errors'].append({'url': current_url, 'error': str(e)})
                
                logger.info(f"✅ Crawled {len(result['pages'])} authenticated pages")
                
            finally:
                await browser.close()
        
        return result
    
    async def _intercept_request(self, route, result: Dict[str, Any]):
        """Intercept and log API calls"""
        request = route.request
        
        # Log API calls
        if request.resource_type in ('xhr', 'fetch'):
            api_call = {
                'method': request.method,
                'url': request.url,
                'resource_type': request.resource_type,
                'headers': request.headers
            }
            result['api_calls'].append(api_call)
        
        await route.continue_()
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL"""
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return f"{parsed.scheme}://{parsed.netloc}"
    
    def _is_same_domain(self, url: str, base_domain: str) -> bool:
        """Check if URL belongs to same domain"""
        return url.startswith(base_domain)
