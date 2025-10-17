# modules/test_selector.py
"""
Intelligent Test Selector (Production-Grade)
Uses ML and RAG to intelligently select tests to run

Features:
- Changed file analysis
- Test impact prediction
- Risk-based prioritization
- Flaky test filtering
- Execution time optimization
- Coverage-guided selection
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class SelectorConfig:
    """Test selector configuration"""
    enable_ml_ranking: bool = True
    enable_flaky_filter: bool = True
    max_execution_time_s: Optional[int] = None
    priority_weights: Dict[str, float] = field(default_factory=lambda: {
        "P0": 1.0,
        "P1": 0.8,
        "P2": 0.5,
        "P3": 0.3
    })
    flaky_threshold: float = 0.3
    risk_threshold: float = 0.6


@dataclass
class TestCase:
    """Test case metadata"""
    name: str
    file_path: str
    priority: str = "P1"
    tags: List[str] = field(default_factory=list)
    avg_duration_ms: float = 0.0
    pass_rate: float = 1.0
    last_failed: Optional[str] = None
    dependencies: List[str] = field(default_factory=list)
    
    @property
    def is_flaky(self) -> bool:
        """Check if test is flaky"""
        return 0.3 < self.pass_rate < 0.7
    
    @property
    def risk_score(self) -> float:
        """Calculate risk score"""
        priority_score = {"P0": 1.0, "P1": 0.8, "P2": 0.5, "P3": 0.3}.get(self.priority, 0.5)
        flaky_penalty = 0.3 if self.is_flaky else 0.0
        failure_bonus = 0.2 if self.last_failed else 0.0
        
        return min(1.0, priority_score + failure_bonus - flaky_penalty)


# ==================== Test Impact Analyzer ====================

class TestImpactAnalyzer:
    """
    Analyze test impact based on code changes.
    """
    
    def __init__(self):
        # File to test mapping (can be learned over time)
        self.file_test_map: Dict[str, Set[str]] = defaultdict(set)
    
    def analyze_changes(
        self,
        changed_files: List[str],
        all_tests: List[TestCase]
    ) -> List[Tuple[TestCase, float]]:
        """
        Analyze which tests are impacted by changed files.
        
        Args:
            changed_files: List of changed file paths
            all_tests: All available tests
        
        Returns:
            List of (test, impact_score) tuples
        """
        impacted = []
        
        for test in all_tests:
            impact_score = self._calculate_impact(test, changed_files)
            
            if impact_score > 0:
                impacted.append((test, impact_score))
        
        # Sort by impact score
        impacted.sort(key=lambda x: x[1], reverse=True)
        
        return impacted
    
    def _calculate_impact(self, test: TestCase, changed_files: List[str]) -> float:
        """Calculate impact score for a test"""
        score = 0.0
        
        # Direct file match
        for changed_file in changed_files:
            changed_path = Path(changed_file)
            test_path = Path(test.file_path)
            
            # Same file
            if changed_path == test_path:
                score += 1.0
            
            # Same directory
            elif changed_path.parent == test_path.parent:
                score += 0.5
            
            # Check learned associations
            if changed_file in self.file_test_map:
                if test.name in self.file_test_map[changed_file]:
                    score += 0.7
        
        return min(1.0, score)
    
    def learn_association(self, file_path: str, test_name: str) -> None:
        """Learn file-to-test association"""
        self.file_test_map[file_path].add(test_name)


# ==================== ML-Based Test Ranker ====================

class MLTestRanker:
    """
    Machine learning-based test prioritization.
    Uses simple heuristics (can be replaced with actual ML model).
    """
    
    def __init__(self):
        self.feature_weights = {
            "priority": 0.3,
            "pass_rate": 0.2,
            "last_failed": 0.25,
            "duration": 0.15,
            "frequency": 0.1
        }
    
    def rank_tests(
        self,
        tests: List[TestCase],
        context: Optional[Dict[str, Any]] = None
    ) -> List[Tuple[TestCase, float]]:
        """
        Rank tests by predicted importance.
        
        Args:
            tests: List of test cases
            context: Additional context (changed files, etc.)
        
        Returns:
            List of (test, score) tuples, sorted by score
        """
        ranked = []
        
        for test in tests:
            score = self._calculate_score(test, context)
            ranked.append((test, score))
        
        # Sort by score (descending)
        ranked.sort(key=lambda x: x[1], reverse=True)
        
        return ranked
    
    def _calculate_score(self, test: TestCase, context: Optional[Dict]) -> float:
        """Calculate test importance score"""
        score = 0.0
        
        # Priority weight
        priority_map = {"P0": 1.0, "P1": 0.8, "P2": 0.5, "P3": 0.3}
        score += priority_map.get(test.priority, 0.5) * self.feature_weights["priority"]
        
        # Pass rate (lower pass rate = higher priority)
        score += (1.0 - test.pass_rate) * self.feature_weights["pass_rate"]
        
        # Recently failed
        if test.last_failed:
            score += self.feature_weights["last_failed"]
        
        # Duration (faster tests = slightly higher priority)
        if test.avg_duration_ms > 0:
            duration_score = max(0, 1.0 - (test.avg_duration_ms / 60000))  # Normalize to 1 min
            score += duration_score * self.feature_weights["duration"]
        
        return min(1.0, score)


# ==================== Main Test Selector ====================

class IntelligentTestSelector:
    """
    Production-grade intelligent test selector.
    
    Features:
    - Impact analysis based on code changes
    - ML-based test ranking
    - Flaky test filtering
    - Execution time optimization
    - Risk-based prioritization
    """
    
    def __init__(
        self,
        config: Optional[SelectorConfig] = None,
        learning_memory = None,
        rag_engine = None
    ):
        """
        Initialize test selector.
        
        Args:
            config: Selector configuration
            learning_memory: LearningMemory instance
            rag_engine: RAGEngine instance
        """
        self.config = config or SelectorConfig()
        self.memory = learning_memory
        self.rag = rag_engine
        
        # Components
        self.impact_analyzer = TestImpactAnalyzer()
        self.ml_ranker = MLTestRanker()
        
        logger.info("✅ Intelligent test selector initialized")
    
    # ==================== Test Selection ====================
    
    def select_tests(
        self,
        all_tests: List[TestCase],
        changed_files: Optional[List[str]] = None,
        max_tests: Optional[int] = None,
        max_duration_s: Optional[int] = None
    ) -> List[TestCase]:
        """
        Select optimal tests to run.
        
        Args:
            all_tests: All available test cases
            changed_files: List of changed files (for impact analysis)
            max_tests: Maximum number of tests to select
            max_duration_s: Maximum total execution time
        
        Returns:
            Optimally selected test cases
        """
        logger.info(f"Selecting tests from {len(all_tests)} available tests")
        
        # 1. Filter flaky tests if enabled
        if self.config.enable_flaky_filter:
            all_tests = self._filter_flaky(all_tests)
            logger.info(f"After flaky filter: {len(all_tests)} tests")
        
        # 2. Analyze impact if changed files provided
        if changed_files:
            impacted = self.impact_analyzer.analyze_changes(changed_files, all_tests)
            
            # Boost scores for impacted tests
            impacted_tests = set(t.name for t, _ in impacted)
        else:
            impacted_tests = set()
        
        # 3. Rank tests
        if self.config.enable_ml_ranking:
            ranked = self.ml_ranker.rank_tests(
                all_tests,
                context={"changed_files": changed_files, "impacted": impacted_tests}
            )
        else:
            # Simple priority-based ranking
            ranked = [(t, t.risk_score) for t in all_tests]
            ranked.sort(key=lambda x: x[1], reverse=True)
        
        # 4. Apply constraints
        selected = self._apply_constraints(
            ranked,
            max_tests=max_tests,
            max_duration_s=max_duration_s or self.config.max_execution_time_s
        )
        
        logger.info(f"✅ Selected {len(selected)} tests")
        
        return selected
    
    def _filter_flaky(self, tests: List[TestCase]) -> List[TestCase]:
        """Filter out flaky tests"""
        filtered = []
        
        for test in tests:
            if not test.is_flaky:
                filtered.append(test)
            else:
                logger.debug(f"Filtered flaky test: {test.name} (pass_rate={test.pass_rate:.2f})")
        
        return filtered
    
    def _apply_constraints(
        self,
        ranked_tests: List[Tuple[TestCase, float]],
        max_tests: Optional[int] = None,
        max_duration_s: Optional[int] = None
    ) -> List[TestCase]:
        """Apply selection constraints"""
        selected = []
        total_duration = 0.0
        
        for test, score in ranked_tests:
            # Check max tests constraint
            if max_tests and len(selected) >= max_tests:
                break
            
            # Check max duration constraint
            if max_duration_s:
                test_duration_s = test.avg_duration_ms / 1000.0
                if total_duration + test_duration_s > max_duration_s:
                    logger.debug(f"Duration limit reached, skipping: {test.name}")
                    continue
                total_duration += test_duration_s
            
            selected.append(test)
        
        return selected
    
    # ==================== Smart Recommendations ====================
    
    def recommend_for_requirement(
        self,
        requirement: str,
        all_tests: List[TestCase],
        top_k: int = 10
    ) -> List[TestCase]:
        """
        Recommend tests for a requirement using RAG.
        
        Args:
            requirement: Test requirement description
            all_tests: All available tests
            top_k: Number of tests to recommend
        
        Returns:
            Recommended test cases
        """
        if not self.rag:
            logger.warning("RAG engine not available, using fallback")
            return all_tests[:top_k]
        
        # Get RAG recommendations
        rag_results = self.rag.recommend_test_cases(requirement, top_k=top_k * 2)
        
        # Match with actual tests
        recommended_names = set(r['test_case'][:100] for r in rag_results)  # Partial match
        
        matched = []
        for test in all_tests:
            if any(name in test.name for name in recommended_names):
                matched.append(test)
        
        # Fill remaining with high-priority tests
        remaining = [t for t in all_tests if t not in matched]
        remaining.sort(key=lambda x: x.risk_score, reverse=True)
        
        matched.extend(remaining[:top_k - len(matched)])
        
        return matched[:top_k]
    
    # ==================== Statistics ====================
    
    def get_selection_stats(self, selected: List[TestCase]) -> Dict[str, Any]:
        """Get statistics for selected tests"""
        total_duration = sum(t.avg_duration_ms for t in selected)
        
        priority_counts = defaultdict(int)
        for test in selected:
            priority_counts[test.priority] += 1
        
        return {
            "total_tests": len(selected),
            "total_duration_s": total_duration / 1000.0,
            "priority_breakdown": dict(priority_counts),
            "avg_pass_rate": sum(t.pass_rate for t in selected) / len(selected) if selected else 0.0,
            "flaky_count": sum(1 for t in selected if t.is_flaky),
        }


# ==================== Usage Example ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Example tests
    tests = [
        TestCase(name="test_login", file_path="tests/auth.spec.ts", priority="P0", avg_duration_ms=1200, pass_rate=1.0),
        TestCase(name="test_logout", file_path="tests/auth.spec.ts", priority="P1", avg_duration_ms=800, pass_rate=0.98),
        TestCase(name="test_flaky", file_path="tests/flaky.spec.ts", priority="P2", avg_duration_ms=2000, pass_rate=0.5),
    ]
    
    # Initialize selector
    selector = IntelligentTestSelector()
    
    # Select tests
    selected = selector.select_tests(
        all_tests=tests,
        changed_files=["src/auth.ts"],
        max_tests=2
    )
    
    print(f"Selected {len(selected)} tests:")
    for test in selected:
        print(f"- {test.name} (priority={test.priority}, risk={test.risk_score:.2f})")
    
    # Get stats
    stats = selector.get_selection_stats(selected)
    print(f"\nStats: {stats}")
