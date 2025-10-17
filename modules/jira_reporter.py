"""
Jira Reporter - Automatically create Jira issues for test failures
"""
import os
import logging
from typing import Dict, Any, List, Optional
from jira import JIRA
from datetime import datetime

logger = logging.getLogger(__name__)


class JiraReporter:
    """Create and update Jira issues for test results"""
    
    def __init__(self):
        self.enabled = os.getenv("JIRA_ENABLED", "false").lower() == "true"
        
        if not self.enabled:
            logger.info("📊 Jira reporting disabled")
            return
        
        self.url = os.getenv("JIRA_URL")
        self.email = os.getenv("JIRA_EMAIL")
        self.api_token = os.getenv("JIRA_API_TOKEN")
        self.project_key = os.getenv("JIRA_PROJECT_KEY", "QA")
        self.issue_type = os.getenv("JIRA_ISSUE_TYPE", "Bug")
        self.auto_create = os.getenv("JIRA_AUTO_CREATE_ISSUES", "false").lower() == "true"
        
        # Connect to Jira
        self.jira = JIRA(
            server=self.url,
            basic_auth=(self.email, self.api_token)
        )
        
        logger.info(f"✅ Jira reporter initialized: {self.url}")
    
    def create_issue_for_failure(
        self,
        test_name: str,
        error: str,
        details: Dict[str, Any]
    ) -> Optional[str]:
        """
        Create Jira issue for test failure
        
        Args:
            test_name: Name of failed test
            error: Error message
            details: Additional details (URL, screenshot, etc.)
        
        Returns:
            Jira issue key (e.g., "QA-123")
        """
        if not self.enabled or not self.auto_create:
            return None
        
        summary = f"Test Failure: {test_name}"
        
        description = f"""
*Test Failure Report*

*Test Name:* {test_name}
*Timestamp:* {datetime.now().isoformat()}
*Environment:* {details.get('environment', 'N/A')}

*Error:*
{{code}}
{error}
{{code}}

*Details:*
- URL: {details.get('url', 'N/A')}
- Browser: {details.get('browser', 'N/A')}
- Run ID: {details.get('run_id', 'N/A')}

*Screenshots:*
{details.get('screenshot_url', 'No screenshot available')}

*Logs:*
{details.get('log_url', 'No logs available')}

---
_Automatically created by AI QA Agent_
"""
        
        try:
            issue = self.jira.create_issue(
                project=self.project_key,
                summary=summary,
                description=description,
                issuetype={'name': self.issue_type},
                labels=['automated-test', 'ai-qa-agent']
            )
            
            logger.info(f"✅ Created Jira issue: {issue.key}")
            return issue.key
        
        except Exception as e:
            logger.error(f"Failed to create Jira issue: {e}")
            return None
    
    def create_test_execution_report(
        self,
        results: Dict[str, Any]
    ) -> Optional[str]:
        """
        Create summary Jira issue for test execution
        
        Args:
            results: Test execution results
        
        Returns:
            Jira issue key
        """
        if not self.enabled:
            return None
        
        metrics = results.get('metrics', {})
        
        summary = f"Test Execution Report - {results['run_id']}"
        
        description = f"""
*Test Execution Summary*

*Run ID:* {results['run_id']}
*Date:* {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
*Duration:* {metrics.get('total_duration_s', 0):.2f}s

*Results:*
- Total Stages: {metrics.get('total_stages', 0)}
- Successful: {metrics.get('successful', 0)} ✅
- Failed: {metrics.get('failed', 0)} ❌
- Success Rate: {(metrics.get('successful', 0) / max(metrics.get('total_stages', 1), 1) * 100):.1f}%

*Stages:*
{self._format_stages(metrics.get('stages', []))}

*Reports:*
- HTML Report: reports/{results['run_id']}.html
- JSON Report: reports/{results['run_id']}.json

---
_Automatically created by AI QA Agent_
"""
        
        try:
            issue = self.jira.create_issue(
                project=self.project_key,
                summary=summary,
                description=description,
                issuetype={'name': 'Task'},
                labels=['test-report', 'ai-qa-agent']
            )
            
            logger.info(f"✅ Created test execution report: {issue.key}")
            return issue.key
        
        except Exception as e:
            logger.error(f"Failed to create execution report: {e}")
            return None
    
    def _format_stages(self, stages: List[Dict[str, Any]]) -> str:
        """Format stages for Jira description"""
        lines = []
        for stage in stages:
            status = str(stage.get('status', 'UNKNOWN')).replace('StageStatus.', '')
            emoji = {'SUCCESS': '✅', 'FAILURE': '❌', 'ERROR': '⚠️'}.get(status, '❓')
            lines.append(f"- {emoji} {stage['name']}: {status} ({stage.get('duration_s', 0):.2f}s)")
        return '\n'.join(lines)
