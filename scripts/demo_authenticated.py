"""
Demo: Test authenticated application (login-protected pages)
"""
import asyncio
from modules.authenticated_scraper import AuthenticatedScraper
from modules.test_generator import TestGenerator
from modules.runner import Runner

async def test_authenticated_app():
    """
    Test an application that requires login
    """
    
    # Configuration
    app_config = {
        'base_url': 'https://authentication.liveperson.net/legacyLogin.html?source=accSelLgcy&stId=54128333',
        'credentials': {
            'username': 'sunny',
            'password': 'Liveperson@2025!!'
        },
        'login_config': {
            'url': '/login',
            'base_url': 'https://authentication.liveperson.net/legacyLogin.html?source=accSelLgcy&stId=54128333',
            'username_selector': 'input[name="email"]',
            'password_selector': 'input[name="password"]',
            'submit_selector': 'button[type="submit"]',
            'success_indicator': '.dashboard, [data-testid="user-menu"]'
        }
    }
    
    print("\n" + "="*60)
    print("🔐 AUTHENTICATED APPLICATION TESTING")
    print("="*60)
    
    # Step 1: Scrape with authentication
    print("\n🔍 Step 1: Crawling authenticated application...")
    
    scraper = AuthenticatedScraper(
        credentials=app_config['credentials'],
        login_config=app_config['login_config']
    )
    
    scan_result = await scraper.crawl_authenticated(
        start_url=f"{app_config['base_url']}/dashboard",
        max_pages=30,
        max_depth=3
    )
    
    print(f"✅ Scanned {len(scan_result['pages'])} pages")
    print(f"✅ Found {len(scan_result['api_calls'])} API calls")
    print(f"✅ Protected pages: {len(scan_result['protected_pages'])}")
    
    # Step 2: Generate comprehensive test plan
    print("\n🧩 Step 2: Generating test plan...")
    
    req = {
        'intent': ['ui_testing', 'api_testing'],
        'details': {
            'url': app_config['base_url'],
            'authentication': 'required',
            'credentials': app_config['credentials'],
            'login_config': app_config['login_config']
        }
    }
    
    generator = TestGenerator()
    plan = generator.generate_plan(req, scan_result)
    
    # Add authentication to plan
    plan['authentication'] = app_config['login_config']
    plan['session_state'] = scan_result['session_info'].get('storage_state')
    
    print(f"✅ Generated test plan:")
    for suite_type, suites in plan['suites'].items():
        print(f"   {suite_type}: {len(suites)} suites")
    
    # Step 3: Execute tests
    print("\n🚀 Step 3: Executing tests...")
    
    runner = Runner()
    results = runner.run(plan)
    
    print(f"\n✅ Execution Complete!")
    print(f"   Run ID: {results['run_id']}")
    print(f"   Success: {results['success']}")
    print(f"   Reports: reports/{results['run_id']}.html")
    
    return results


# Run
if __name__ == "__main__":
    asyncio.run(test_authenticated_app())
