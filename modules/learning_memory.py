# modules/learning_memory.py
"""
LearningMemory Module (Production-Grade, v2)

AI-powered test history and pattern learning system

Features:
- Persistent storage of test executions (JSONL)
- Semantic search using embeddings (ChromaDB + SentenceTransformers)
- Flaky test detection
- Healed locator recommendations
- Test pattern analysis
- Failure prediction (hooks for future)
- Graceful degradation if optional deps unavailable
"""

from __future__ import annotations

import json
import logging
import math
import os
from dataclasses import dataclass, asdict, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

logger = logging.getLogger(__name__)

# ---------- Optional dependencies (defensive import) ----------
_HAS_CHROMA = False
_HAS_ST = False
try:
    from chromadb import Client, Collection  # type: ignore
    from chromadb.config import Settings as ChromaSettings  # type: ignore
    _HAS_CHROMA = True
except Exception as e:
    logger.debug(f"ChromaDB not available: {e}")
    Client = object  # type: ignore
    Collection = object  # type: ignore
    ChromaSettings = object  # type: ignore

try:
    from sentence_transformers import SentenceTransformer  # type: ignore
    _HAS_ST = True
except Exception as e:
    logger.debug(f"sentence_transformers not available: {e}")
    SentenceTransformer = object  # type: ignore


# ==================== Data Models ====================

@dataclass
class TestExecution:
    """Single test execution record"""
    test_name: str
    status: str  # PASS, FAIL, SKIP, ERROR
    duration_ms: float
    timestamp: str
    locators_used: List[str]
    healed_locators: List[Dict[str, Any]]  # [{hint, original, healed, method}]
    failure_reason: Optional[str] = None
    dom_snapshot: Optional[str] = None
    screenshot_path: Optional[str] = None
    retries: int = 0
    browser: Optional[str] = None
    url: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary with defensive truncation for large blobs."""
        d = asdict(self)
        # Truncate very large fields to keep history JSONL manageable
        if d.get("failure_reason"):
            d["failure_reason"] = str(d["failure_reason"])[:2000]
        if d.get("dom_snapshot"):
            d["dom_snapshot"] = str(d["dom_snapshot"])[:50000]
        return d


@dataclass
class TestPattern:
    """Detected test pattern"""
    pattern_type: str  # flaky, slow, healed, consistent_failure
    test_name: str
    occurrences: int
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealedLocator:
    """Healed locator recommendation"""
    hint: str
    original_locator: str
    healed_locator: str
    method: str  # get_by_role, css, etc.
    success_count: int
    last_used: str
    confidence: float


# ==================== Fallback Embedder ====================

class _FallbackEmbedder:
    """
    Deterministic lightweight embedder used when sentence_transformers isn't available.
    Produces a fixed-size vector by hashing characters. Not semantically rich,
    but preserves API shape & allows the system to run without heavy deps.
    """
    def __init__(self, dim: int = 384):
        self.dim = dim

    def encode(self, text: str) -> List[float]:
        text = (text or "").strip()
        vec = [0] * self.dim
        if not text:
            return [0.0] * self.dim
        # Simple rolling hash -> bucket into dim slots
        h = 2166136261
        for ch in text:
            h ^= ord(ch)
            h *= 16777619
            idx = abs(h) % self.dim
            vec[idx] += 1
        # L2 normalize
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


# ==================== Main Learning Memory ====================

class LearningMemory:
    """
    Production-grade learning memory system with graceful fallbacks.

    - If LEARNING_MEMORY_DISABLE=1, the module becomes a no-op (safe).
    - If embeddings stack fails, falls back to a lightweight embedder and/or disables vector search.
    """

    def __init__(
        self,
        persist_dir: str = "./data/learning_memory",
        collection_name: str = "test_executions",
        embedding_model: Optional[str] = None
    ):
        """
        Initialize learning memory.

        Args:
            persist_dir: Directory for persistent storage
            collection_name: ChromaDB collection name
            embedding_model: Sentence transformer model (env override supported)
        """
        self.disabled = os.getenv("LEARNING_MEMORY_DISABLE", "0") == "1"
        if self.disabled:
            logger.warning("LearningMemory is DISABLED via LEARNING_MEMORY_DISABLE=1")

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)

        self.history_file = self.persist_dir / "history.jsonl"
        self.stats_file = self.persist_dir / "stats.json"
        self.patterns_file = self.persist_dir / "patterns.json"

        # ---- ChromaDB init (optional) ----
        self.collection: Optional[Collection] = None
        self.vectors_enabled = False
        if not self.disabled and _HAS_CHROMA:
            try:
                logger.info("Initializing ChromaDB client...")
                self.client = Client(ChromaSettings(
                    persist_directory=str(self.persist_dir / "chroma"),
                    anonymized_telemetry=False
                ))
                self.collection = self.client.get_or_create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                self.vectors_enabled = True
                logger.info(f"✅ ChromaDB collection ready: {collection_name}")
            except Exception as e:
                self.collection = None
                self.vectors_enabled = False
                logger.warning(f"ChromaDB unavailable, semantic indexing disabled: {e}")

        # ---- Embedding model (optional) ----
        self.embedder = None
        self.embed_enabled = False
        if not self.disabled:
            if os.getenv("LEARNING_MEMORY_DISABLE_EMBEDDINGS", "0") == "1":
                logger.warning("Embeddings disabled via LEARNING_MEMORY_DISABLE_EMBEDDINGS=1")
            else:
                model_name = embedding_model or os.getenv("LEARNING_MEMORY_EMBED_MODEL", "all-MiniLM-L6-v2")
                if _HAS_ST:
                    try:
                        logger.info(f"Loading embedding model: {model_name}")
                        self.embedder = SentenceTransformer(model_name)  # type: ignore
                        self.embed_enabled = True
                        logger.info("✅ Embedding model loaded")
                    except Exception as e:
                        logger.warning(f"Failed to load embedding model '{model_name}': {e}. Falling back.")
                if self.embedder is None:
                    self.embedder = _FallbackEmbedder()
                    self.embed_enabled = True
                    logger.info("🔁 Using fallback embedder (lightweight)")

        # In-memory caches
        self._healed_cache: Dict[str, List[HealedLocator]] = {}
        self._flaky_cache: Optional[List[TestPattern]] = None
        self._stats_cache: Optional[Dict[str, Any]] = None

        # Load existing data (best-effort)
        self._load_existing_data()

    # ==================== Core Operations ====================

    def store_execution(self, execution: TestExecution) -> None:
        """
        Store test execution (durable JSONL + optional vector index).
        Defensive checks prevent crashes from malformed input.
        """
        try:
            # ---- Defensive check (as requested) ----
            if not execution or not getattr(execution, "test_name", None):
                logger.warning("Skipping invalid execution (missing test_name)")
                return

            if self.disabled:
                logger.info("LearningMemory disabled; execution accepted but not persisted.")
                return

            exec_dict = execution.to_dict()
            # Default/fallback normalization
            exec_dict["status"] = str(exec_dict.get("status", "UNKNOWN")).upper()
            try:
                exec_dict["duration_ms"] = float(exec_dict.get("duration_ms", 0.0))
            except Exception:
                exec_dict["duration_ms"] = 0.0

            # Normalize timestamp to ISO8601 if not provided
            ts = exec_dict.get("timestamp")
            if not ts:
                exec_dict["timestamp"] = datetime.now().isoformat()

            # 1) Append to JSONL file (durable storage)
            with open(self.history_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(exec_dict, ensure_ascii=False) + "\n")

            # 2) Index in vector database (best-effort)
            if self.vectors_enabled and self.embed_enabled:
                self._index_execution(exec_dict)
            else:
                logger.debug("Vector indexing skipped (disabled or unavailable)")

            # 3) Update healed locator cache
            if exec_dict.get("healed_locators"):
                self._update_healed_cache_from_dict(exec_dict)

            # 4) Invalidate caches
            self._flaky_cache = None
            self._stats_cache = None

            logger.debug(f"Stored execution: {execution.test_name} ({exec_dict['status']})")

        except Exception as e:
            logger.error(f"Failed to store execution: {e}", exc_info=True)

    def _index_execution(self, exec_data: Dict[str, Any]) -> None:
        """Index execution in ChromaDB (best-effort, fully guarded)."""
        try:
            if not (self.collection and self.embedder and self.vectors_enabled and self.embed_enabled):
                return

            test_name = str(exec_data.get("test_name", "")).strip()
            if not test_name:
                return

            # Create embedding from test name + failure reason
            text = f"{test_name} {exec_data.get('failure_reason', '')}".strip()
            embedding = self.embedder.encode(text)
            # Ensure embedding is a list of floats
            if not isinstance(embedding, list):
                embedding = list(embedding)  # type: ignore

            # Generate unique ID
            doc_id = f"{test_name}_{exec_data.get('timestamp', '')}"

            # Prepare metadata (basic types only)
            metadata = {
                "status": exec_data.get("status", "UNKNOWN"),
                "duration_ms": float(exec_data.get("duration_ms", 0.0)),
                "timestamp": exec_data.get("timestamp", ""),
                "failure_reason": str(exec_data.get("failure_reason", ""))[:500],
                "healed_count": len(exec_data.get("healed_locators", [])),
                "retries": int(exec_data.get("retries", 0) or 0),
                "browser": exec_data.get("browser", "unknown"),
            }

            self.collection.add(
                embeddings=[embedding],  # type: ignore[arg-type]
                documents=[test_name],
                metadatas=[metadata],
                ids=[doc_id]
            )
        except Exception as e:
            # Don't fail the run—just log.
            logger.debug(f"Indexing skipped (error): {e}")

    # ==================== Semantic Search ====================

    def find_similar_failures(
        self,
        test_name: str,
        top_k: int = 5,
        status_filter: Optional[str] = "FAIL"
    ) -> List[Dict[str, Any]]:
        """
        Find similar past failures using semantic search.
        Returns [] if embeddings or vector DB are disabled/unavailable.
        """
        try:
            if not test_name:
                return []

            if not (self.collection and self.embedder and self.vectors_enabled and self.embed_enabled):
                logger.debug("Semantic search unavailable (no vectors/embeddings).")
                return []

            # Generate embedding
            embedding = self.embedder.encode(test_name)
            if not isinstance(embedding, list):
                embedding = list(embedding)  # type: ignore

            where_clause = {"status": status_filter} if status_filter else None

            results = self.collection.query(  # type: ignore
                query_embeddings=[embedding],
                n_results=max(1, int(top_k)),
                where=where_clause
            )
            # Chroma returns dict-of-lists; guard against empty results
            if not results or not results.get("ids") or not results["ids"][0]:
                return []

            similar = []
            ids = results.get("ids", [[]])[0]
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            dists = results.get("distances", [[]])[0] if "distances" in results else [None] * len(ids)

            for i in range(len(ids)):
                dist = dists[i]
                sim = 1.0 - dist if isinstance(dist, (int, float)) else None
                similar.append({
                    "test_name": docs[i] if i < len(docs) else "",
                    "metadata": metas[i] if i < len(metas) else {},
                    "distance": dist,
                    "similarity": sim
                })

            return similar

        except Exception as e:
            logger.debug(f"Semantic search failed: {e}")
            return []

    # ==================== Healed Locators ====================

    def _update_healed_cache_from_dict(self, exec_data: Dict[str, Any]) -> None:
        healed_list = exec_data.get("healed_locators") or []
        timestamp = exec_data.get("timestamp", datetime.now().isoformat())

        for healed in healed_list:
            hint = str(healed.get("hint", "") or "").strip()
            if not hint:
                continue

            if hint not in self._healed_cache:
                self._healed_cache[hint] = []

            found = False
            for existing in self._healed_cache[hint]:
                if existing.healed_locator == healed.get("healed_locator"):
                    existing.success_count += 1
                    existing.last_used = timestamp
                    found = True
                    break

            if not found:
                self._healed_cache[hint].append(HealedLocator(
                    hint=hint,
                    original_locator=healed.get("original_locator", "") or "",
                    healed_locator=healed.get("healed_locator", "") or "",
                    method=healed.get("method", "") or "",
                    success_count=1,
                    last_used=timestamp,
                    confidence=0.8
                ))

    def _update_healed_cache(self, execution: TestExecution) -> None:
        self._update_healed_cache_from_dict(execution.to_dict())

    def get_healed_locators(
        self,
        hint: str,
        min_confidence: float = 0.5
    ) -> List[HealedLocator]:
        """
        Get recommended healed locators for a hint.
        """
        hint = (hint or "").strip()
        if not hint:
            return []

        # Check cache first
        if hint in self._healed_cache:
            return [loc for loc in self._healed_cache[hint] if loc.confidence >= min_confidence]

        # Load from history file
        healed_locators = []
        if self.history_file.exists():
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        exec_data = json.loads(line)
                        for healed in exec_data.get("healed_locators", []):
                            if (healed or {}).get("hint") == hint:
                                healed_locators.append(healed)
                    except Exception:
                        continue

        # Build recommendations
        locator_counts = defaultdict(int)
        for healed in healed_locators:
            key = (healed or {}).get("healed_locator", "")
            if key:
                locator_counts[key] += 1

        recommendations: List[HealedLocator] = []
        total = sum(locator_counts.values())
        for locator, count in locator_counts.items():
            confidence = (count / total) if total > 0 else 0.0
            if confidence >= min_confidence:
                recommendations.append(HealedLocator(
                    hint=hint,
                    original_locator="",
                    healed_locator=locator,
                    method="",
                    success_count=count,
                    last_used=datetime.now().isoformat(),
                    confidence=confidence
                ))

        recommendations.sort(key=lambda x: x.confidence, reverse=True)
        return recommendations

    # ==================== Flaky Test Detection ====================

    def get_flaky_tests(
        self,
        threshold: float = 0.3,
        min_runs: int = 3
    ) -> List[TestPattern]:
        """
        Detect flaky tests (inconsistent pass/fail).
        Flaky if pass_rate between (threshold, 1-threshold), default 0.3 => 30–70%.
        """
        if self._flaky_cache is not None:
            return self._flaky_cache

        test_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"PASS": 0, "FAIL": 0, "total": 0})

        if self.history_file.exists():
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        exec_data = json.loads(line)
                        name = exec_data.get("test_name")
                        status = str(exec_data.get("status", "")).upper()
                        if not name or status not in ("PASS", "FAIL"):
                            continue
                        test_stats[name][status] += 1
                        test_stats[name]["total"] += 1
                    except Exception:
                        continue

        flaky_tests: List[TestPattern] = []
        for name, stats in test_stats.items():
            total = stats["total"]
            if total < min_runs:
                continue
            pass_rate = stats["PASS"] / total if total else 0.0
            if threshold < pass_rate < (1 - threshold):
                # Confidence scaled around 50%: closer to 0.5 => more flaky
                confidence = 1.0 - abs(pass_rate - 0.5) * 2
                flaky_tests.append(TestPattern(
                    pattern_type="flaky",
                    test_name=name,
                    occurrences=total,
                    confidence=confidence,
                    metadata={
                        "pass_rate": pass_rate,
                        "pass_count": stats["PASS"],
                        "fail_count": stats["FAIL"],
                        "total_runs": total
                    }
                ))

        flaky_tests.sort(key=lambda x: x.confidence, reverse=True)
        self._flaky_cache = flaky_tests
        return flaky_tests

    # ==================== Slow Test Detection ====================

    def get_slow_tests(
        self,
        percentile: float = 0.9,
        min_runs: int = 5
    ) -> List[TestPattern]:
        """
        Detect slow tests (top (1 - percentile) cohort by average duration).
        Default percentile=0.9 -> return ~top 10% slowest.
        """
        test_durations: Dict[str, List[float]] = defaultdict(list)

        if self.history_file.exists():
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        exec_data = json.loads(line)
                        name = exec_data.get("test_name")
                        dur = exec_data.get("duration_ms")
                        if name is None or dur is None:
                            continue
                        test_durations[name].append(float(dur))
                    except Exception:
                        continue

        averages: List[Tuple[str, float, int]] = []
        for name, durs in test_durations.items():
            if len(durs) >= min_runs:
                avg = sum(durs) / len(durs)
                averages.append((name, avg, len(durs)))

        averages.sort(key=lambda x: x[1], reverse=True)
        n = len(averages)
        if n == 0:
            return []

        top_fraction = max(0.0, min(1.0, 1.0 - float(percentile)))  # e.g., 0.1 for 90th percentile
        k = max(1, math.ceil(n * top_fraction))  # at least one
        slow_head = averages[:k]

        patterns: List[TestPattern] = []
        for name, avg_ms, run_count in slow_head:
            patterns.append(TestPattern(
                pattern_type="slow",
                test_name=name,
                occurrences=run_count,
                confidence=0.9,
                metadata={
                    "avg_duration_ms": avg_ms,
                    "run_count": run_count
                }
            ))
        return patterns

    # ==================== Statistics ====================

    def get_metrics(self) -> Dict[str, Any]:
        """Get overall learning metrics (cached)."""
        if self._stats_cache is not None:
            return self._stats_cache

        total_executions = 0
        total_heals = 0
        status_counts = {"PASS": 0, "FAIL": 0, "SKIP": 0, "ERROR": 0}
        total_duration = 0.0

        if self.history_file.exists():
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        exec_data = json.loads(line)
                        total_executions += 1
                        total_heals += len(exec_data.get("healed_locators", []) or [])
                        status = str(exec_data.get("status", "")).upper()
                        if status in status_counts:
                            status_counts[status] += 1
                        total_duration += float(exec_data.get("duration_ms", 0.0) or 0.0)
                    except Exception:
                        continue

        metrics = {
            "total_executions": total_executions,
            "total_heals": total_heals,
            "pass_count": status_counts["PASS"],
            "fail_count": status_counts["FAIL"],
            "skip_count": status_counts["SKIP"],
            "error_count": status_counts["ERROR"],
            "pass_rate": (status_counts["PASS"] / total_executions) if total_executions > 0 else 0.0,
            "heal_rate": (total_heals / total_executions) if total_executions > 0 else 0.0,
            "avg_duration_ms": (total_duration / total_executions) if total_executions > 0 else 0.0,
            "unique_tests": len(set(self._get_all_test_names())),
            "vectors_enabled": self.vectors_enabled,
            "embeddings_enabled": self.embed_enabled,
            "disabled": self.disabled,
        }

        self._stats_cache = metrics
        return metrics

    def _get_all_test_names(self) -> List[str]:
        """Get all unique test names (best-effort)."""
        names = set()
        if self.history_file.exists():
            with open(self.history_file, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        exec_data = json.loads(line)
                        name = exec_data.get("test_name")
                        if name:
                            names.add(name)
                    except Exception:
                        continue
        return list(names)

    # ==================== Data Management ====================

    def _load_existing_data(self) -> None:
        """Load and (best-effort) reindex existing history."""
        if not self.history_file.exists():
            logger.info("No existing learning history found.")
            return

        reindexed = 0
        if not (self.collection and self.embedder and self.vectors_enabled and self.embed_enabled):
            logger.info("Skipping reindex (vectors/embeddings unavailable).")
            return

        logger.info("Loading existing test history for reindex...")
        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    exec_data = json.loads(line)
                    doc_id = f"{exec_data.get('test_name', '')}_{exec_data.get('timestamp', '')}"
                    if not doc_id.strip():
                        continue

                    # Check presence (get returns empty lists if not found)
                    present = False
                    try:
                        existing = self.collection.get(ids=[doc_id])  # type: ignore
                        present = bool(existing and existing.get("ids") and existing["ids"])
                    except Exception:
                        present = False

                    if not present:
                        self._index_execution(exec_data)
                        reindexed += 1
                except Exception as e:
                    logger.debug(f"Skipping bad history line: {e}")
        logger.info(f"✅ Reindex complete (added {reindexed})")

    def export_patterns(self) -> None:
        """Export detected patterns to file (best-effort)."""
        try:
            patterns = {
                "flaky_tests": [asdict(p) for p in self.get_flaky_tests()],
                "slow_tests": [asdict(p) for p in self.get_slow_tests()],
                "metrics": self.get_metrics(),
                "exported_at": datetime.now().isoformat()
            }
            with open(self.patterns_file, "w", encoding="utf-8") as f:
                json.dump(patterns, f, indent=2, ensure_ascii=False)
            logger.info(f"✅ Exported patterns to {self.patterns_file}")
        except Exception as e:
            logger.error(f"Failed to export patterns: {e}")

    def cleanup_old_data(self, days: int = 30) -> int:
        """
        Remove test executions older than specified days from JSONL and reindex.
        Returns number of entries removed.
        """
        if not self.history_file.exists():
            return 0

        cutoff_iso = (datetime.now() - timedelta(days=days)).isoformat()
        kept: List[Dict[str, Any]] = []
        total_before = 0

        with open(self.history_file, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    total_before += 1
                    exec_data = json.loads(line)
                    if str(exec_data.get("timestamp", "")) >= cutoff_iso:
                        kept.append(exec_data)
                except Exception:
                    # Skip unreadable lines
                    continue

        # Rewrite file with kept entries
        with open(self.history_file, "w", encoding="utf-8") as f:
            for entry in kept:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        removed = max(0, total_before - len(kept))

        # Rebuild vector index (best-effort)
        if self.collection and self.vectors_enabled:
            try:
                name = getattr(self.collection, "name", None)
                # ✅ Your suggested defensive fix (avoid crash if client missing)
                try:
                    if hasattr(self, "client") and getattr(self, "client", None):
                        self.client.delete_collection(name)  # type: ignore[attr-defined]
                    else:
                        raise AttributeError("No client")
                except Exception:
                    try:
                        if self.collection:
                            self.collection.delete(where={})  # type: ignore
                    except Exception:
                        pass

                # Recreate collection
                if hasattr(self, "client") and getattr(self, "client", None):
                    self.collection = self.client.get_or_create_collection(  # type: ignore
                        name=name,
                        metadata={"hnsw:space": "cosine"}
                    )
                # Reindex kept entries
                if self.collection:
                    for entry in kept:
                        self._index_execution(entry)
            except Exception as e:
                logger.debug(f"Vector reindex during cleanup skipped: {e}")

        logger.info(f"✅ Cleaned up {removed} old entries (kept last {days} days)")
        return removed


# ==================== Usage Example ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Initialize learning memory (honors env flags)
    memory = LearningMemory()

    # Store a test execution (defensive checks handle missing fields)
    execution = TestExecution(
        test_name="test_login_successful",
        status="PASS",
        duration_ms=1250.5,
        timestamp=datetime.now().isoformat(),
        locators_used=["input[name='username']", "button[type='submit']"],
        healed_locators=[{
            "hint": "username",
            "original_locator": "input#user",
            "healed_locator": "input[name='username']",
            "method": "css"
        }]
    )
    memory.store_execution(execution)

    # Get metrics
    metrics = memory.get_metrics()
    print(json.dumps(metrics, indent=2))

    # Detect flaky tests
    flaky = memory.get_flaky_tests()
    print(f"Flaky tests: {len(flaky)}")

    # Export patterns
    memory.export_patterns()
