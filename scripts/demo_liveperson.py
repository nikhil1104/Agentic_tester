"""
Demo: Test LivePerson bot platform (authenticated)
"""
from modules.auth_manager import AuthManager
from modules.authenticated_scraper import AuthenticatedScraper
from modules.test_generator import TestGenerator
from modules.runner import Runner
import asyncio

async def main():
    print("🔐 LivePerson Bot Platform Testing\n")
    
    # Step 1: Authenticate
    print("Step 1: Authenticating...")
    auth = AuthManager(
        config_path="config/liveperson_auth.json",
        enable_encryption=True,
        auto_renew=True
    )
    
    session_file = auth.login_and_save_session()
    
    if not session_file:
        print("❌ Authentication failed!")
        return
    
    print(f"✅ Authenticated! Session: {session_file}\n")
    
    # Step 2: Scrape authenticated app
    print("Step 2: Crawling authenticated pages...")
    scraper = AuthenticatedScraper(
        credentials={
            'username': auth.auth_config['username'],
            'password': auth.auth_config['password']
        },
        login_config=auth.auth_config
    )
    
    scan_result = await scraper.crawl_authenticated(
        start_url='https://va-a.botplatform.liveperson.net/dashboard',
        max_pages=30,
        max_depth=2
    )
    
    print(f"✅ Crawled {len(scan_result['pages'])} pages\n")
    
    # Step 3: Generate tests
    print("Step 3: Generating test plan...")
    generator = TestGenerator()
    
    req = {
        'intent': ['ui_testing', 'api_testing'],
        'details': {
            'url': 'https://va-a.botplatform.liveperson.net',
            'authentication': 'required'
        }
    }
    
    plan = generator.generate_plan(req, scan_result)
    plan['session_state'] = session_file
    
    print(f"✅ Generated test plan\n")
    
    # Step 4: Execute
    print("Step 4: Executing tests...")
    runner = Runner()
    results = runner.run(plan)
    
    print(f"\n✅ Complete!")
    print(f"   Run ID: {results['run_id']}")
    print(f"   Reports: reports/{results['run_id']}.html")

if __name__ == "__main__":
    asyncio.run(main())
