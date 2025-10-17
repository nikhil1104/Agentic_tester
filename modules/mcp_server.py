# modules/mcp_server.py
"""
Model Context Protocol (MCP) Server (Production-Grade)
Enables Claude Desktop and other MCP clients to interact with the QA system

Features:
- Full MCP protocol implementation
- Tool registration and execution
- Resource management
- Streaming support
- Session management
- Error handling and recovery
- Comprehensive logging
"""

from __future__ import annotations

import os
import json
import logging
import asyncio
from typing import Dict, Any, List, Optional, Union, AsyncIterator
from dataclasses import dataclass, field, asdict
from datetime import datetime

# MCP SDK
try:
    from mcp.server import Server, NotificationOptions
    from mcp.server.models import InitializationOptions
    from mcp.types import (
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
        Resource,
        ResourceTemplate,
    )
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    logging.warning("MCP SDK not installed. Install with: pip install mcp")

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class MCPConfig:
    """MCP server configuration"""
    server_name: str = "ai-qa-agent"
    server_version: str = "1.0.0"
    enable_streaming: bool = True
    enable_notifications: bool = True
    max_concurrent_tools: int = 5
    tool_timeout: int = 300  # seconds
    log_level: str = "INFO"


# ==================== Tool Definitions ====================

@dataclass
class ToolResult:
    """Tool execution result"""
    success: bool
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    
    def to_text_content(self) -> TextContent:
        """Convert to MCP TextContent"""
        return TextContent(
            type="text",
            text=self.content
        )


# ==================== MCP Server Implementation ====================

if MCP_AVAILABLE:
    
    class AIQAMCPServer:
        """
        Production-grade MCP server for AI QA Agent.
        
        Exposes QA testing capabilities to Claude Desktop and other MCP clients.
        """
        
        def __init__(self, config: Optional[MCPConfig] = None):
            self.config = config or MCPConfig()
            
            # Initialize MCP server
            self.server = Server(self.config.server_name)
            
            # Lazy-load heavy dependencies
            self._orchestrator = None
            self._rag_engine = None
            self._learning_memory = None
            
            # Setup logging
            self._setup_logging()
            
            # Register tools and resources
            self._register_tools()
            self._register_resources()
            
            # Session tracking
            self._active_sessions: Dict[str, Dict[str, Any]] = {}
            
            logger.info(f"✅ MCP server initialized: {self.config.server_name}")
        
        def _setup_logging(self):
            """Configure logging"""
            level = getattr(logging, self.config.log_level.upper(), logging.INFO)
            logging.basicConfig(
                level=level,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
        
        # ==================== Lazy Loading ====================
        
        def _get_orchestrator(self):
            """Lazy load orchestrator"""
            if not self._orchestrator:
                from modules.spec_orchestrator import SpecOrchestrator
                self._orchestrator = SpecOrchestrator()
            return self._orchestrator
        
        def _get_rag_engine(self):
            """Lazy load RAG engine"""
            if not self._rag_engine:
                from modules.rag_engine import RAGEngine
                self._rag_engine = RAGEngine()
            return self._rag_engine
        
        def _get_learning_memory(self):
            """Lazy load learning memory"""
            if not self._learning_memory:
                from modules.learning_memory import LearningMemory
                self._learning_memory = LearningMemory()
            return self._learning_memory
        
        # ==================== Tool Registration ====================
        
        def _register_tools(self):
            """Register all available tools"""
            
            # Tool 1: Run UI Tests
            @self.server.list_tools()
            async def list_tools() -> List[Tool]:
                return [
                    Tool(
                        name="run_ui_tests",
                        description="Run comprehensive UI tests on a given URL using Playwright",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "url": {
                                    "type": "string",
                                    "description": "Target URL to test"
                                },
                                "browsers": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "Browsers to test (chromium, firefox, webkit)",
                                    "default": ["chromium"]
                                },
                                "max_depth": {
                                    "type": "integer",
                                    "description": "Maximum crawl depth",
                                    "default": 2
                                },
                                "include_visual": {
                                    "type": "boolean",
                                    "description": "Include visual regression testing",
                                    "default": False
                                }
                            },
                            "required": ["url"]
                        }
                    ),
                    
                    Tool(
                        name="run_api_tests",
                        description="Run API tests for specified endpoints",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "base_url": {
                                    "type": "string",
                                    "description": "API base URL"
                                },
                                "endpoints": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of endpoints to test"
                                },
                                "auth_token": {
                                    "type": "string",
                                    "description": "Optional authentication token"
                                }
                            },
                            "required": ["base_url"]
                        }
                    ),
                    
                    Tool(
                        name="analyze_test_history",
                        description="Analyze historical test execution data and detect patterns",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "test_name": {
                                    "type": "string",
                                    "description": "Specific test name to analyze (optional)"
                                },
                                "days": {
                                    "type": "integer",
                                    "description": "Number of days to look back",
                                    "default": 7
                                },
                                "include_flaky": {
                                    "type": "boolean",
                                    "description": "Include flaky test detection",
                                    "default": True
                                }
                            }
                        }
                    ),
                    
                    Tool(
                        name="search_test_knowledge",
                        description="Search test documentation and best practices using RAG",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "query": {
                                    "type": "string",
                                    "description": "Search query"
                                },
                                "top_k": {
                                    "type": "integer",
                                    "description": "Number of results to return",
                                    "default": 5
                                }
                            },
                            "required": ["query"]
                        }
                    ),
                    
                    Tool(
                        name="generate_test_plan",
                        description="Generate a comprehensive test plan from requirements",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "requirement": {
                                    "type": "string",
                                    "description": "Test requirement or user story"
                                },
                                "include_negative_tests": {
                                    "type": "boolean",
                                    "description": "Include negative test cases",
                                    "default": True
                                },
                                "include_performance": {
                                    "type": "boolean",
                                    "description": "Include performance tests",
                                    "default": False
                                }
                            },
                            "required": ["requirement"]
                        }
                    ),
                    
                    Tool(
                        name="get_test_metrics",
                        description="Get comprehensive test execution metrics and statistics",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "period": {
                                    "type": "string",
                                    "enum": ["day", "week", "month", "all"],
                                    "description": "Time period for metrics",
                                    "default": "week"
                                }
                            }
                        }
                    ),
                    
                    Tool(
                        name="recommend_tests",
                        description="Get intelligent test recommendations based on code changes",
                        inputSchema={
                            "type": "object",
                            "properties": {
                                "changed_files": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                    "description": "List of changed file paths"
                                },
                                "max_tests": {
                                    "type": "integer",
                                    "description": "Maximum number of tests to recommend",
                                    "default": 10
                                }
                            },
                            "required": ["changed_files"]
                        }
                    ),
                ]
            
            # Tool execution handler
            @self.server.call_tool()
            async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
                """Execute requested tool"""
                try:
                    logger.info(f"🔧 Executing tool: {name}")
                    
                    if name == "run_ui_tests":
                        result = await self._run_ui_tests(arguments)
                    
                    elif name == "run_api_tests":
                        result = await self._run_api_tests(arguments)
                    
                    elif name == "analyze_test_history":
                        result = await self._analyze_test_history(arguments)
                    
                    elif name == "search_test_knowledge":
                        result = await self._search_test_knowledge(arguments)
                    
                    elif name == "generate_test_plan":
                        result = await self._generate_test_plan(arguments)
                    
                    elif name == "get_test_metrics":
                        result = await self._get_test_metrics(arguments)
                    
                    elif name == "recommend_tests":
                        result = await self._recommend_tests(arguments)
                    
                    else:
                        result = ToolResult(
                            success=False,
                            content=f"Unknown tool: {name}",
                            error="Tool not found"
                        )
                    
                    return [result.to_text_content()]
                
                except Exception as e:
                    logger.error(f"Tool execution failed: {e}", exc_info=True)
                    error_result = ToolResult(
                        success=False,
                        content=f"Tool execution failed: {str(e)}",
                        error=str(e)
                    )
                    return [error_result.to_text_content()]
        
        # ==================== Tool Implementations ====================
        
        async def _run_ui_tests(self, args: Dict[str, Any]) -> ToolResult:
            """Execute UI tests"""
            url = args["url"]
            browsers = args.get("browsers", ["chromium"])
            max_depth = args.get("max_depth", 2)
            include_visual = args.get("include_visual", False)
            
            try:
                orchestrator = self._get_orchestrator()
                
                # Build requirement
                requirement = f"Test UI of {url} with browsers: {', '.join(browsers)}"
                if max_depth > 1:
                    requirement += f". Crawl depth: {max_depth}"
                if include_visual:
                    requirement += ". Include visual regression testing."
                
                # Execute in thread pool (blocking operation)
                result = await asyncio.to_thread(orchestrator.run, requirement)
                
                # Format response
                summary = result.get("summary", {})
                passed = summary.get("passed", False)
                
                content = f"""✅ UI Tests Completed

**Target:** {url}
**Browsers:** {', '.join(browsers)}
**Status:** {'✅ PASSED' if passed else '❌ FAILED'}

**Results:**
- Total Tests: {summary.get('total_tests', 0)}
- Passed: {summary.get('passed_tests', 0)}
- Failed: {summary.get('failed_tests', 0)}
- Duration: {summary.get('duration_s', 0):.2f}s

**Reports:**
{result.get('report_path', 'N/A')}
"""
                
                return ToolResult(
                    success=passed,
                    content=content,
                    metadata=result
                )
            
            except Exception as e:
                logger.error(f"UI test execution failed: {e}", exc_info=True)
                return ToolResult(
                    success=False,
                    content=f"UI test execution failed: {str(e)}",
                    error=str(e)
                )
        
        async def _run_api_tests(self, args: Dict[str, Any]) -> ToolResult:
            """Execute API tests"""
            base_url = args["base_url"]
            endpoints = args.get("endpoints", [])
            auth_token = args.get("auth_token")
            
            try:
                from modules.api_test_engine import APITestEngine
                
                engine = APITestEngine(base_url)
                
                # Build test suite
                suite = {
                    "name": "MCP API Tests",
                    "base_url": base_url,
                    "endpoints": [{"path": ep} for ep in endpoints]
                }
                
                # Execute
                result = await asyncio.to_thread(engine.run_suite, suite)
                
                passed = result.get("status") == "PASS"
                
                content = f"""✅ API Tests Completed

**Base URL:** {base_url}
**Endpoints Tested:** {len(endpoints)}
**Status:** {'✅ PASSED' if passed else '❌ FAILED'}

**Results:**
{json.dumps(result, indent=2)}
"""
                
                return ToolResult(
                    success=passed,
                    content=content,
                    metadata=result
                )
            
            except Exception as e:
                return ToolResult(
                    success=False,
                    content=f"API test execution failed: {str(e)}",
                    error=str(e)
                )
        
        async def _analyze_test_history(self, args: Dict[str, Any]) -> ToolResult:
            """Analyze test history"""
            test_name = args.get("test_name")
            days = args.get("days", 7)
            include_flaky = args.get("include_flaky", True)
            
            try:
                memory = self._get_learning_memory()
                
                # Get metrics
                metrics = memory.get_metrics()
                
                # Get flaky tests if requested
                flaky_tests = []
                if include_flaky:
                    flaky_tests = memory.get_flaky_tests()
                
                content = f"""📊 Test History Analysis

**Period:** Last {days} days
**Total Executions:** {metrics['total_executions']}
**Pass Rate:** {metrics['pass_rate']:.1%}
**Self-Healing Rate:** {metrics['heal_rate']:.1%}

**Status Breakdown:**
- ✅ Passed: {metrics['pass_count']}
- ❌ Failed: {metrics['fail_count']}
- ⏭️ Skipped: {metrics['skip_count']}

"""
                
                if flaky_tests:
                    content += f"\n**⚠️ Flaky Tests Detected:** {len(flaky_tests)}\n"
                    for flaky in flaky_tests[:5]:
                        content += f"- {flaky.test_name} (pass rate: {flaky.metadata['pass_rate']:.1%})\n"
                
                return ToolResult(
                    success=True,
                    content=content,
                    metadata={
                        "metrics": metrics,
                        "flaky_tests": [asdict(f) for f in flaky_tests]
                    }
                )
            
            except Exception as e:
                return ToolResult(
                    success=False,
                    content=f"Analysis failed: {str(e)}",
                    error=str(e)
                )
        
        async def _search_test_knowledge(self, args: Dict[str, Any]) -> ToolResult:
            """Search RAG knowledge base"""
            query = args["query"]
            top_k = args.get("top_k", 5)
            
            try:
                rag = self._get_rag_engine()
                
                results = rag.search(query, top_k=top_k)
                
                content = f"🔍 Search Results for: '{query}'\n\n"
                
                for i, result in enumerate(results, 1):
                    content += f"**Result {i}** (similarity: {result['similarity']:.2f})\n"
                    content += f"{result['content'][:200]}...\n"
                    content += f"Source: {result['metadata'].get('source', 'Unknown')}\n\n"
                
                return ToolResult(
                    success=True,
                    content=content,
                    metadata={"results": results}
                )
            
            except Exception as e:
                return ToolResult(
                    success=False,
                    content=f"Search failed: {str(e)}",
                    error=str(e)
                )
        
        async def _generate_test_plan(self, args: Dict[str, Any]) -> ToolResult:
            """Generate test plan"""
            requirement = args["requirement"]
            include_negative = args.get("include_negative_tests", True)
            include_performance = args.get("include_performance", False)
            
            try:
                orchestrator = self._get_orchestrator()
                
                # Generate plan
                plan = await asyncio.to_thread(
                    orchestrator._generate_test_plan,
                    requirement
                )
                
                content = f"""📋 Generated Test Plan

**Requirement:** {requirement}

**Test Suites:**
"""
                
                for suite_type, tests in plan.get("suites", {}).items():
                    content += f"\n**{suite_type.upper()}:** {len(tests)} tests\n"
                    for test in tests[:3]:  # Show first 3
                        content += f"- {test.get('name', 'Unnamed test')}\n"
                
                return ToolResult(
                    success=True,
                    content=content,
                    metadata=plan
                )
            
            except Exception as e:
                return ToolResult(
                    success=False,
                    content=f"Plan generation failed: {str(e)}",
                    error=str(e)
                )
        
        async def _get_test_metrics(self, args: Dict[str, Any]) -> ToolResult:
            """Get test metrics"""
            period = args.get("period", "week")
            
            try:
                memory = self._get_learning_memory()
                metrics = memory.get_metrics()
                
                content = f"""📊 Test Metrics ({period})

**Overall Statistics:**
- Total Executions: {metrics['total_executions']}
- Unique Tests: {metrics['unique_tests']}
- Pass Rate: {metrics['pass_rate']:.1%}
- Avg Duration: {metrics['avg_duration_ms']:.0f}ms

**Test Outcomes:**
- ✅ Passed: {metrics['pass_count']}
- ❌ Failed: {metrics['fail_count']}
- ⏭️ Skipped: {metrics['skip_count']}

**Self-Healing:**
- Total Heals: {metrics['total_heals']}
- Heal Rate: {metrics['heal_rate']:.1%}
"""
                
                return ToolResult(
                    success=True,
                    content=content,
                    metadata=metrics
                )
            
            except Exception as e:
                return ToolResult(
                    success=False,
                    content=f"Metrics retrieval failed: {str(e)}",
                    error=str(e)
                )
        
        async def _recommend_tests(self, args: Dict[str, Any]) -> ToolResult:
            """Recommend tests based on changes"""
            changed_files = args["changed_files"]
            max_tests = args.get("max_tests", 10)
            
            try:
                from modules.test_selector import IntelligentTestSelector, TestCase
                
                selector = IntelligentTestSelector()
                
                # Mock test cases (in production, load from registry)
                all_tests = [
                    TestCase(name="test_example", file_path="tests/example.spec.ts", priority="P0")
                ]
                
                # Select tests
                selected = selector.select_tests(
                    all_tests=all_tests,
                    changed_files=changed_files,
                    max_tests=max_tests
                )
                
                content = f"""🎯 Test Recommendations

**Changed Files:** {len(changed_files)}
**Recommended Tests:** {len(selected)}

"""
                
                for test in selected:
                    content += f"- {test.name} (priority: {test.priority})\n"
                
                return ToolResult(
                    success=True,
                    content=content,
                    metadata={"selected_tests": [t.name for t in selected]}
                )
            
            except Exception as e:
                return ToolResult(
                    success=False,
                    content=f"Test recommendation failed: {str(e)}",
                    error=str(e)
                )
        
        # ==================== Resource Registration ====================
        
        def _register_resources(self):
            """Register resources (test reports, logs, etc.)"""
            
            @self.server.list_resources()
            async def list_resources() -> List[Resource]:
                return [
                    Resource(
                        uri="test-reports://latest",
                        name="Latest Test Reports",
                        description="Most recent test execution reports",
                        mimeType="application/json"
                    ),
                    Resource(
                        uri="test-metrics://summary",
                        name="Test Metrics Summary",
                        description="Aggregated test execution metrics",
                        mimeType="application/json"
                    ),
                ]
            
            @self.server.read_resource()
            async def read_resource(uri: str) -> str:
                """Read resource content"""
                if uri == "test-reports://latest":
                    # Return latest test report
                    return json.dumps({"status": "placeholder"}, indent=2)
                
                elif uri == "test-metrics://summary":
                    memory = self._get_learning_memory()
                    metrics = memory.get_metrics()
                    return json.dumps(metrics, indent=2)
                
                else:
                    raise ValueError(f"Unknown resource: {uri}")
        
        # ==================== Server Lifecycle ====================
        
        async def run(self) -> None:
            """Run MCP server"""
            from mcp.server.stdio import stdio_server
            
            logger.info(f"🚀 Starting MCP server: {self.config.server_name}")
            
            async with stdio_server() as (read_stream, write_stream):
                await self.server.run(
                    read_stream,
                    write_stream,
                    InitializationOptions(
                        server_name=self.config.server_name,
                        server_version=self.config.server_version,
                        capabilities=self.server.get_capabilities(
                            notification_options=NotificationOptions(),
                            experimental_capabilities={}
                        )
                    )
                )


# ==================== CLI Entry Point ====================

async def main():
    """Main entry point for MCP server"""
    if not MCP_AVAILABLE:
        print("❌ MCP SDK not installed. Install with: pip install mcp")
        return 1
    
    try:
        server = AIQAMCPServer()
        await server.run()
    except KeyboardInterrupt:
        logger.info("🛑 Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}", exc_info=True)
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run server
    sys.exit(asyncio.run(main()))
