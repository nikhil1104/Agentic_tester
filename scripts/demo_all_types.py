"""
Demo: Interactive test generation for ALL test types
Enter URL at runtime like demo_run.py
"""
import os
import sys
os.environ["TG_ENABLE_LLM"] = "true"

from modules.conversational_agent import ConversationalAgent
from modules.async_scraper import AsyncScraper, CrawlConfig
from modules.test_generator import TestGenerator
from modules.runner import Runner


def main():
    """Interactive demo for comprehensive testing"""
    
    print("\n" + "="*70)
    print("🤖 AI QA Agent - Comprehensive Test Suite Generator")
    print("="*70)
    print("\nSupported Test Types:")
    print("  ✅ UI Testing (Playwright)")
    print("  ✅ API Testing (REST/GraphQL)")
    print("  ✅ Performance Testing (Lighthouse)")
    print("  ✅ Security Testing (Headers, TLS)")
    print("="*70)
    
    # Get URL from user
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = input("\n🌐 Enter URL to test: ").strip()
    
    if not url:
        print("❌ Error: URL is required")
        sys.exit(1)
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    print(f"\n🎯 Target: {url}")
    
    # Ask for test types
    print("\n📋 Select test types (press Enter for all):")
    print("  1. All (UI + API + Performance + Security)")
    print("  2. UI Only")
    print("  3. API Only")
    print("  4. UI + API")
    print("  5. API + Performance")
    
    choice = input("\nYour choice [1-5] (default: 1): ").strip() or "1"
    
    # Map choices to intents
    intent_map = {
        "1": ['ui_testing', 'api_testing', 'performance_testing', 'security_testing'],
        "2": ['ui_testing'],
        "3": ['api_testing'],
        "4": ['ui_testing', 'api_testing'],
        "5": ['api_testing', 'performance_testing'],
    }
    
    selected_intents = intent_map.get(choice, intent_map["1"])
    
    print(f"\n✅ Selected: {', '.join([i.replace('_testing', '').upper() for i in selected_intents])}")
    
    # Step 1: Parse requirement
    print("\n" + "="*70)
    print("Step 1: Parsing requirement...")
    print("="*70)
    
    agent = ConversationalAgent()
    req = agent.parse_requirement(f"Test {url}")
    req['intent'] = selected_intents
    
    print(f"✅ Parsed intent: {req['intent']}")
    print(f"✅ Confidence: {req.get('confidence', 1.0):.2f}")
    
    # Step 2: Scan website
    print("\n" + "="*70)
    print("Step 2: Scanning website...")
    print("="*70)
    
    config = CrawlConfig(
        max_pages=10,
        max_depth=2,
        timeout_ms=30000,
        concurrency=3
    )
    
    scraper = AsyncScraper(config)
    
    print(f"🔍 Crawling {url}...")
    scan_result = scraper.deep_scan(url)
    
    print(f"✅ Scanned {len(scan_result.get('pages', []))} pages")
    print(f"✅ Found {len(scan_result.get('links', []))} links")
    print(f"✅ Detected {len(scan_result.get('api_calls', []))} API calls")
    
    # Step 3: Generate test plan
    print("\n" + "="*70)
    print("Step 3: Generating test plan...")
    print("="*70)
    
    generator = TestGenerator()
    plan = generator.generate_plan(req, scan_result)
    
    # Fix API base URLs
    for suite in plan['suites'].get('api', []):
        suite['base_url'] = url
        if suite.get('endpoint', '').startswith('http'):
            suite['endpoint'] = suite['endpoint'].replace(url, '')
    
    # Fix Performance tests (use Lighthouse)
    for suite in plan['suites'].get('performance', []):
        suite['tool'] = 'lighthouse'
        suite['url'] = url
        suite.pop('file', None)
        suite.pop('jmeter_output', None)
    
    # Display generated suites
    print("\n📋 Generated Test Suites:")
    print("-"*70)
    
    total_suites = 0
    for suite_type, suites in plan['suites'].items():
        if suites:
            print(f"\n  {suite_type.upper()}: {len(suites)} suite(s)")
            total_suites += len(suites)
            for suite in suites[:3]:  # Show first 3
                print(f"    • {suite['name']}")
                print(f"      Priority: {suite['priority']}, Risk: {suite['risk_score']}")
            if len(suites) > 3:
                print(f"    ... and {len(suites) - 3} more")
    
    print(f"\n  Total: {total_suites} test suites")
    print("-"*70)
    
    # Ask to proceed
    proceed = input("\n▶️  Execute tests? [Y/n]: ").strip().lower()
    
    if proceed in ['n', 'no']:
        print("\n⏹️  Execution cancelled. Test plan saved.")
        sys.exit(0)
    
    # Step 4: Execute tests
    print("\n" + "="*70)
    print("Step 4: Executing tests...")
    print("="*70)
    
    runner = Runner()
    results = runner.run(plan)
    
    # Display results
    print("\n" + "="*70)
    print("✅ Execution Complete!")
    print("="*70)
    
    print(f"\n📊 Summary:")
    print(f"  Run ID: {results['run_id']}")
    print(f"  Success: {'✅ PASS' if results['success'] else '❌ FAIL'}")
    print(f"  Duration: {results['metrics']['total_duration_s']:.2f}s")
    print(f"  Total Stages: {results['metrics']['total_stages']}")
    print(f"  Successful: {results['metrics']['successful']}")
    print(f"  Failed: {results['metrics']['failed']}")
    
    # Stage breakdown
    print(f"\n📈 Stage Breakdown:")
    print("-"*70)
    
    for stage in results['metrics']['stages']:
        status_str = str(stage.get('status', 'UNKNOWN'))
        status_map = {
            'SUCCESS': '✅',
            'StageStatus.SUCCESS': '✅',
            'FAILURE': '❌',
            'StageStatus.FAILURE': '❌',
            'ERROR': '⚠️',
            'StageStatus.ERROR': '⚠️',
            'SKIPPED': '⏭️',
            'StageStatus.SKIPPED': '⏭️',
        }
        
        status_emoji = status_map.get(status_str, '❓')
        duration = stage.get('duration_s', 0)
        
        print(f"  {status_emoji} {stage['name']}: {status_str.replace('StageStatus.', '')} ({duration:.2f}s)")
        
        if stage.get('retries', 0) > 0:
            print(f"     ↻ Retries: {stage['retries']}")
        
        if stage.get('error'):
            error_msg = str(stage['error'])[:100]
            print(f"     ⚠️  {error_msg}...")
    
    print("-"*70)
    
    # Reports
    print(f"\n📄 Reports:")
    print(f"  HTML: reports/{results['run_id']}.html")
    print(f"  JSON: reports/{results['run_id']}.json")
    
    print("\n" + "="*70)
    print("🎉 Done!")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⏹️  Cancelled by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
