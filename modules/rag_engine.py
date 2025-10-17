# modules/rag_engine.py
"""
RAG Engine (Production-Grade Implementation)
Retrieval-Augmented Generation for test intelligence

Features:
- ChromaDB vector store with persistence
- Semantic document search
- Test case recommendations
- Company standards ingestion
- Multi-source document loading
- Incremental indexing
- Query expansion
- Relevance scoring
"""

from __future__ import annotations

import os
import json
import logging
import hashlib
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime

# Vector database
from chromadb import Client, Collection
from chromadb.config import Settings as ChromaSettings

# Embeddings
from sentence_transformers import SentenceTransformer

# Document processing
from bs4 import BeautifulSoup
import markdown

logger = logging.getLogger(__name__)


# ==================== Configuration ====================

@dataclass
class RAGConfig:
    """RAG engine configuration"""
    persist_dir: str = "./data/vector_store"
    collection_name: str = "test_knowledge"
    embedding_model: str = "all-MiniLM-L6-v2"
    chunk_size: int = 512  # Characters per chunk
    chunk_overlap: int = 50
    top_k: int = 5
    similarity_threshold: float = 0.7
    enable_reranking: bool = False


@dataclass
class Document:
    """Document representation"""
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc_id,
            "content": self.content,
            "metadata": self.metadata
        }


# ==================== Document Processors ====================

class DocumentProcessor:
    """Process documents from various formats"""
    
    @staticmethod
    def chunk_text(
        text: str,
        chunk_size: int = 512,
        overlap: int = 50
    ) -> List[str]:
        """
        Split text into overlapping chunks.
        
        Args:
            text: Text to chunk
            chunk_size: Size of each chunk
            overlap: Overlap between chunks
        
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = min(start + chunk_size, text_len)
            chunk = text[start:end]
            
            # Try to break at sentence boundary
            if end < text_len:
                last_period = chunk.rfind('.')
                last_newline = chunk.rfind('\n')
                break_point = max(last_period, last_newline)
                
                if break_point > chunk_size // 2:
                    chunk = chunk[:break_point + 1]
                    end = start + break_point + 1
            
            chunks.append(chunk.strip())
            start = end - overlap
        
        return [c for c in chunks if len(c) > 20]  # Filter very short chunks
    
    @staticmethod
    def process_html(html: str) -> str:
        """Extract text from HTML"""
        try:
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text
            text = soup.get_text(separator='\n', strip=True)
            
            # Clean up whitespace
            lines = [line.strip() for line in text.split('\n')]
            text = '\n'.join(line for line in lines if line)
            
            return text
        except Exception as e:
            logger.error(f"HTML processing failed: {e}")
            return ""
    
    @staticmethod
    def process_markdown(md_text: str) -> str:
        """Convert markdown to plain text"""
        try:
            html = markdown.markdown(md_text)
            return DocumentProcessor.process_html(html)
        except Exception as e:
            logger.error(f"Markdown processing failed: {e}")
            return md_text
    
    @staticmethod
    def generate_doc_id(content: str, source: str = "") -> str:
        """Generate unique document ID"""
        combined = f"{source}:{content[:100]}"
        return hashlib.sha256(combined.encode()).hexdigest()[:16]


# ==================== Main RAG Engine ====================

class RAGEngine:
    """
    Production-grade RAG engine for test intelligence.
    
    Features:
    - Persistent vector storage
    - Semantic search
    - Document ingestion from multiple sources
    - Test case recommendations
    - Company standards integration
    """
    
    def __init__(self, config: Optional[RAGConfig] = None):
        """
        Initialize RAG engine.
        
        Args:
            config: RAG configuration
        """
        self.config = config or RAGConfig()
        
        # Setup directories
        self.persist_dir = Path(self.config.persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.docs_file = self.persist_dir / "documents.jsonl"
        self.index_file = self.persist_dir / "index_metadata.json"
        
        # Initialize ChromaDB
        logger.info(f"Initializing RAG with persist_dir: {self.persist_dir}")
        
        try:
            self.client = Client(ChromaSettings(
                persist_directory=str(self.persist_dir / "chroma"),
                anonymized_telemetry=False
            ))
            
            self.collection: Collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            logger.info(f"✅ ChromaDB collection: {self.config.collection_name}")
        except Exception as e:
            logger.error(f"ChromaDB initialization failed: {e}")
            raise
        
        # Initialize embedding model
        logger.info(f"Loading embedding model: {self.config.embedding_model}")
        
        try:
            self.embedder = SentenceTransformer(self.config.embedding_model)
            logger.info("✅ Embedding model loaded")
        except Exception as e:
            logger.error(f"Embedding model load failed: {e}")
            raise
        
        # Document processor
        self.processor = DocumentProcessor()
        
        # Statistics
        self._stats = {
            "total_documents": 0,
            "total_chunks": 0,
            "last_indexed": None
        }
        
        # Load existing index
        self._load_index_metadata()
    
    # ==================== Document Ingestion ====================
    
    def ingest_document(
        self,
        content: str,
        source: str,
        doc_type: str = "text",
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Ingest a single document.
        
        Args:
            content: Document content
            source: Document source/path
            doc_type: Document type (text, html, markdown)
            metadata: Additional metadata
        
        Returns:
            Document ID
        """
        try:
            # Process based on type
            if doc_type == "html":
                text = self.processor.process_html(content)
            elif doc_type == "markdown":
                text = self.processor.process_markdown(content)
            else:
                text = content
            
            if not text or len(text) < 10:
                logger.warning(f"Document too short or empty: {source}")
                return ""
            
            # Generate document ID
            doc_id = self.processor.generate_doc_id(text, source)
            
            # Check if already indexed
            try:
                existing = self.collection.get(ids=[doc_id])
                if existing and len(existing['ids']) > 0:
                    logger.debug(f"Document already indexed: {doc_id}")
                    return doc_id
            except Exception:
                pass
            
            # Chunk document
            chunks = self.processor.chunk_text(
                text,
                chunk_size=self.config.chunk_size,
                overlap=self.config.chunk_overlap
            )
            
            if not chunks:
                logger.warning(f"No chunks generated for: {source}")
                return ""
            
            # Generate embeddings
            embeddings = self.embedder.encode(chunks).tolist()
            
            # Prepare metadata
            base_metadata = {
                "source": source,
                "doc_type": doc_type,
                "indexed_at": datetime.now().isoformat(),
                "chunk_count": len(chunks)
            }
            
            if metadata:
                base_metadata.update(metadata)
            
            # Add to collection
            chunk_ids = [f"{doc_id}_chunk_{i}" for i in range(len(chunks))]
            metadatas = [
                {**base_metadata, "chunk_index": i, "chunk_total": len(chunks)}
                for i in range(len(chunks))
            ]
            
            self.collection.add(
                ids=chunk_ids,
                embeddings=embeddings,
                documents=chunks,
                metadatas=metadatas
            )
            
            # Save document record
            doc_record = {
                "doc_id": doc_id,
                "source": source,
                "doc_type": doc_type,
                "chunk_count": len(chunks),
                "indexed_at": base_metadata["indexed_at"],
                "metadata": metadata or {}
            }
            
            with open(self.docs_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(doc_record) + "\n")
            
            # Update stats
            self._stats["total_documents"] += 1
            self._stats["total_chunks"] += len(chunks)
            self._stats["last_indexed"] = datetime.now().isoformat()
            self._save_index_metadata()
            
            logger.info(f"✅ Indexed document: {source} ({len(chunks)} chunks)")
            return doc_id
        
        except Exception as e:
            logger.error(f"Document ingestion failed for {source}: {e}", exc_info=True)
            return ""
    
    def ingest_directory(
        self,
        directory: str,
        extensions: Optional[List[str]] = None,
        recursive: bool = True
    ) -> int:
        """
        Ingest all documents from a directory.
        
        Args:
            directory: Directory path
            extensions: File extensions to include (default: .txt, .md, .html)
            recursive: Search subdirectories
        
        Returns:
            Number of documents ingested
        """
        if extensions is None:
            extensions = [".txt", ".md", ".html", ".json"]
        
        dir_path = Path(directory)
        if not dir_path.exists():
            logger.error(f"Directory not found: {directory}")
            return 0
        
        count = 0
        pattern = "**/*" if recursive else "*"
        
        for file_path in dir_path.glob(pattern):
            if not file_path.is_file():
                continue
            
            if file_path.suffix.lower() not in extensions:
                continue
            
            try:
                content = file_path.read_text(encoding="utf-8")
                
                # Determine doc type
                if file_path.suffix == ".html":
                    doc_type = "html"
                elif file_path.suffix == ".md":
                    doc_type = "markdown"
                else:
                    doc_type = "text"
                
                doc_id = self.ingest_document(
                    content=content,
                    source=str(file_path),
                    doc_type=doc_type,
                    metadata={"filename": file_path.name}
                )
                
                if doc_id:
                    count += 1
            
            except Exception as e:
                logger.error(f"Failed to ingest {file_path}: {e}")
        
        logger.info(f"✅ Ingested {count} documents from {directory}")
        return count
    
    def ingest_scraped_docs(self, html_dir: str = "data/scraped_docs") -> int:
        """
        Ingest documents from web scraper output.
        
        Args:
            html_dir: Directory containing scraped HTML files
        
        Returns:
            Number of documents ingested
        """
        return self.ingest_directory(
            directory=html_dir,
            extensions=[".html"],
            recursive=False
        )
    
    # ==================== Search & Retrieval ====================
    
    def search(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Semantic search over indexed documents.
        
        Args:
            query: Search query
            top_k: Number of results to return
            filter_metadata: Filter by metadata fields
        
        Returns:
            List of relevant document chunks with metadata
        """
        try:
            top_k = top_k or self.config.top_k
            
            # Generate query embedding
            query_embedding = self.embedder.encode(query).tolist()
            
            # Query collection
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=filter_metadata
            )
            
            # Format results
            search_results = []
            
            for i in range(len(results['ids'][0])):
                result = {
                    "chunk_id": results['ids'][0][i],
                    "content": results['documents'][0][i],
                    "metadata": results['metadatas'][0][i],
                    "distance": results['distances'][0][i] if 'distances' in results else None,
                    "similarity": 1.0 - results['distances'][0][i] if 'distances' in results else None
                }
                
                # Apply similarity threshold
                if result['similarity'] and result['similarity'] >= self.config.similarity_threshold:
                    search_results.append(result)
            
            logger.debug(f"Search '{query}': {len(search_results)} results")
            return search_results
        
        except Exception as e:
            logger.error(f"Search failed: {e}", exc_info=True)
            return []
    
    def get_context(
        self,
        query: str,
        max_tokens: int = 2000
    ) -> str:
        """
        Get relevant context for a query (for LLM prompts).
        
        Args:
            query: Query string
            max_tokens: Maximum tokens in context (approximate)
        
        Returns:
            Concatenated relevant context
        """
        results = self.search(query, top_k=10)
        
        context_parts = []
        total_chars = 0
        max_chars = max_tokens * 4  # Rough estimate: 1 token ≈ 4 chars
        
        for result in results:
            content = result['content']
            source = result['metadata'].get('source', 'Unknown')
            
            part = f"[Source: {source}]\n{content}\n"
            
            if total_chars + len(part) > max_chars:
                break
            
            context_parts.append(part)
            total_chars += len(part)
        
        return "\n---\n".join(context_parts)
    
    # ==================== Test Intelligence ====================
    
    def recommend_test_cases(
        self,
        requirement: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Recommend test cases based on requirement.
        
        Args:
            requirement: Test requirement description
            top_k: Number of recommendations
        
        Returns:
            List of recommended test cases
        """
        # Expand query with test-specific terms
        expanded_query = f"{requirement} test case scenario validation check"
        
        results = self.search(expanded_query, top_k=top_k * 2)
        
        # Filter and rank
        recommendations = []
        
        for result in results:
            content = result['content']
            
            # Check if content looks like a test case
            test_indicators = [
                "test", "verify", "assert", "should", "expect",
                "given", "when", "then", "scenario"
            ]
            
            indicator_count = sum(1 for ind in test_indicators if ind in content.lower())
            
            if indicator_count >= 2:
                recommendations.append({
                    "test_case": content,
                    "source": result['metadata'].get('source', 'Unknown'),
                    "relevance": result['similarity'],
                    "confidence": min(1.0, indicator_count / 5.0)
                })
        
        # Sort by relevance * confidence
        recommendations.sort(
            key=lambda x: x['relevance'] * x['confidence'],
            reverse=True
        )
        
        return recommendations[:top_k]
    
    def get_company_standards(self, domain: str = "testing") -> List[str]:
        """
        Retrieve company testing standards.
        
        Args:
            domain: Domain to search (testing, security, etc.)
        
        Returns:
            List of standard guidelines
        """
        query = f"{domain} standards guidelines best practices"
        
        results = self.search(
            query,
            top_k=10,
            filter_metadata={"doc_type": "standard"}  # If standards are tagged
        )
        
        standards = []
        for result in results:
            standards.append(result['content'])
        
        return standards
    
    # ==================== Statistics & Management ====================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get RAG engine statistics"""
        # Update from collection
        try:
            count_result = self.collection.count()
            self._stats["total_chunks"] = count_result
        except Exception:
            pass
        
        return {
            **self._stats,
            "collection_name": self.config.collection_name,
            "persist_dir": str(self.persist_dir)
        }
    
    def _load_index_metadata(self) -> None:
        """Load index metadata from file"""
        if self.index_file.exists():
            try:
                with open(self.index_file, "r", encoding="utf-8") as f:
                    self._stats = json.load(f)
                logger.info(f"Loaded index metadata: {self._stats['total_documents']} documents")
            except Exception as e:
                logger.warning(f"Failed to load index metadata: {e}")
    
    def _save_index_metadata(self) -> None:
        """Save index metadata to file"""
        try:
            with open(self.index_file, "w", encoding="utf-8") as f:
                json.dump(self._stats, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save index metadata: {e}")
    
    def clear_index(self) -> None:
        """Clear all indexed documents (destructive!)"""
        logger.warning("Clearing RAG index...")
        
        try:
            # Delete collection
            self.client.delete_collection(self.config.collection_name)
            
            # Recreate collection
            self.collection = self.client.get_or_create_collection(
                name=self.config.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            
            # Clear files
            if self.docs_file.exists():
                self.docs_file.unlink()
            if self.index_file.exists():
                self.index_file.unlink()
            
            # Reset stats
            self._stats = {
                "total_documents": 0,
                "total_chunks": 0,
                "last_indexed": None
            }
            
            logger.info("✅ RAG index cleared")
        
        except Exception as e:
            logger.error(f"Failed to clear index: {e}", exc_info=True)
    
    def export_index(self, output_file: str) -> None:
        """
        Export indexed documents to JSON file.
        
        Args:
            output_file: Output file path
        """
        try:
            # Get all documents
            all_docs = []
            
            if self.docs_file.exists():
                with open(self.docs_file, "r", encoding="utf-8") as f:
                    for line in f:
                        all_docs.append(json.loads(line))
            
            # Write to output
            with open(output_file, "w", encoding="utf-8") as f:
                json.dump({
                    "stats": self._stats,
                    "documents": all_docs,
                    "exported_at": datetime.now().isoformat()
                }, f, indent=2)
            
            logger.info(f"✅ Exported index to {output_file}")
        
        except Exception as e:
            logger.error(f"Export failed: {e}", exc_info=True)


# ==================== Integration with Test Generation ====================

class RAGEnhancedTestGenerator:
    """
    Test generator enhanced with RAG for intelligent suggestions.
    """
    
    def __init__(self, rag_engine: RAGEngine):
        self.rag = rag_engine
    
    def generate_with_context(
        self,
        requirement: str,
        base_plan: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance test plan with RAG context.
        
        Args:
            requirement: Test requirement
            base_plan: Base test plan
        
        Returns:
            Enhanced test plan
        """
        # Get relevant context
        context = self.rag.get_context(requirement, max_tokens=1500)
        
        # Get test case recommendations
        recommendations = self.rag.recommend_test_cases(requirement, top_k=5)
        
        # Add to plan metadata
        base_plan.setdefault("rag_context", {})
        base_plan["rag_context"]["context"] = context
        base_plan["rag_context"]["recommendations"] = recommendations
        base_plan["rag_context"]["retrieved_at"] = datetime.now().isoformat()
        
        return base_plan


# ==================== Usage Example ====================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Initialize RAG engine
    rag = RAGEngine()
    
    # Ingest documents
    rag.ingest_document(
        content="Test cases should validate user authentication with valid and invalid credentials.",
        source="test_standards.txt",
        doc_type="text",
        metadata={"category": "authentication", "doc_type": "standard"}
    )
    
    # Search
    results = rag.search("login test cases")
    print(f"Found {len(results)} results")
    
    for result in results:
        print(f"- {result['content'][:100]}... (similarity: {result['similarity']:.2f})")
    
    # Get stats
    stats = rag.get_stats()
    print(f"\nStats: {json.dumps(stats, indent=2)}")
