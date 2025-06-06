"""
Storage layer for paper insights with vector embeddings and local persistence.
Uses ChromaDB for vector search and JSON for full data storage.
"""

import json
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import logging

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .insight_schema import PaperInsights, UserContext, ExtractionMetadata

logger = logging.getLogger(__name__)


class InsightStorage:
    """
    Manages storage and retrieval of paper insights.
    
    Storage structure:
    - storage/papers/: Raw paper data
    - storage/insights/: Extracted insights
    - storage/embeddings/chroma/: Vector database
    - storage/metadata.db: SQLite for quick lookups
    """
    
    def __init__(self, storage_root: str = "storage"):
        """Initialize storage with directory structure."""
        self.storage_root = Path(storage_root)
        self._setup_directories()
        
        # Initialize components
        self._init_vector_db()
        self._init_embedder()
        
        # SQLite connection will be created per-thread
        self._connections = {}
        self._setup_database()
        
        logger.info(f"Initialized storage at {self.storage_root}")
    
    def _setup_directories(self):
        """Create storage directory structure."""
        directories = [
            self.storage_root / "papers",
            self.storage_root / "insights", 
            self.storage_root / "embeddings",
            self.storage_root / "checkpoints"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def _init_vector_db(self):
        """Initialize ChromaDB for vector similarity search."""
        self.chroma_client = chromadb.PersistentClient(
            path=str(self.storage_root / "embeddings" / "chroma"),
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        try:
            self.insights_collection = self.chroma_client.create_collection(
                name="paper_insights",
                metadata={"description": "GenAI paper insights with embeddings"}
            )
        except:
            self.insights_collection = self.chroma_client.get_collection("paper_insights")
    
    @property
    def metadata_conn(self):
        """Get or create thread-local SQLite connection."""
        import threading
        
        thread_id = threading.get_ident()
        if not hasattr(self, '_connections'):
            self._connections = {}
        
        if thread_id not in self._connections:
            db_path = self.storage_root / "metadata.db"
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.row_factory = sqlite3.Row
            self._connections[thread_id] = conn
            
            # Ensure tables exist
            self._create_tables(conn)
        
        return self._connections[thread_id]
    
    def _create_tables(self, conn):
        """Create tables in the given connection."""
        # Drop and recreate insights table to ensure schema is correct
        conn.executescript("""
            DROP TABLE IF EXISTS extraction_metadata;
            DROP TABLE IF EXISTS insights;
            
            CREATE TABLE IF NOT EXISTS papers (
                paper_id TEXT PRIMARY KEY,
                title TEXT NOT NULL,
                authors TEXT,
                published_date TEXT,
                arxiv_categories TEXT,
                stored_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
            
            CREATE TABLE IF NOT EXISTS insights (
                paper_id TEXT PRIMARY KEY,
                study_type TEXT,
                complexity TEXT,
                techniques TEXT,
                quality_score REAL,
                evidence_strength REAL,
                practical_applicability REAL,
                extraction_confidence REAL,
                has_code BOOLEAN,
                has_dataset BOOLEAN,
                key_findings_count INTEGER,
                extraction_timestamp TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            );
            
            CREATE TABLE IF NOT EXISTS extraction_metadata (
                extraction_id TEXT PRIMARY KEY,
                paper_id TEXT,
                extraction_time_seconds REAL,
                api_calls_made INTEGER,
                estimated_cost_usd REAL,
                extractor_version TEXT,
                llm_model TEXT,
                extraction_timestamp TIMESTAMP,
                FOREIGN KEY (paper_id) REFERENCES papers(paper_id)
            );
            
            CREATE INDEX IF NOT EXISTS idx_insights_quality ON insights(quality_score DESC);
            CREATE INDEX IF NOT EXISTS idx_insights_evidence ON insights(evidence_strength DESC);
            CREATE INDEX IF NOT EXISTS idx_insights_applicability ON insights(practical_applicability DESC);
            CREATE INDEX IF NOT EXISTS idx_insights_complexity ON insights(complexity);
            CREATE INDEX IF NOT EXISTS idx_insights_study_type ON insights(study_type);
            CREATE INDEX IF NOT EXISTS idx_papers_published ON papers(published_date DESC);
        """)
        conn.commit()
    
    def _init_embedder(self):
        """Initialize sentence transformer for embeddings."""
        self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
        logger.info("Initialized sentence embedder")
    
    def _setup_database(self):
        """Initialize SQLite database structure."""
        # Just ensure the database exists with correct tables
        # Actual connections will be created per-thread
        db_path = self.storage_root / "metadata.db"
        conn = sqlite3.connect(str(db_path))
        self._create_tables(conn)
        conn.close()
    
    def store_paper(self, paper_data: Dict) -> str:
        """
        Store raw paper data.
        
        Args:
            paper_data: Paper metadata from arXiv
            
        Returns:
            Paper ID
        """
        paper_id = paper_data.get('id', '').split('/')[-1]
        if not paper_id:
            paper_id = f"paper_{datetime.utcnow().timestamp()}"
        
        # Sanitize Unicode in paper data
        sanitized_paper = self._sanitize_unicode(paper_data)
        
        # Store JSON file
        paper_path = self.storage_root / "papers" / f"{paper_id}.json"
        try:
            with open(paper_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_paper, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Failed to store paper {paper_id}: {e}")
            # Try again with more aggressive sanitization
            sanitized_paper = self._sanitize_unicode(paper_data, aggressive=True)
            with open(paper_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_paper, f, indent=2, ensure_ascii=True)
        
        # Store metadata in SQLite
        self.metadata_conn.execute("""
            INSERT OR REPLACE INTO papers 
            (paper_id, title, authors, published_date, arxiv_categories)
            VALUES (?, ?, ?, ?, ?)
        """, (
            paper_id,
            sanitized_paper.get('title', ''),
            json.dumps(sanitized_paper.get('authors', [])),
            sanitized_paper.get('published', ''),
            json.dumps(sanitized_paper.get('categories', []))
        ))
        self.metadata_conn.commit()
        
        logger.info(f"Stored paper {paper_id}: {sanitized_paper.get('title', '')[:50]}...")
        return paper_id
    
    def _sanitize_unicode(self, data, aggressive=False):
        """
        Recursively sanitize Unicode strings in a data structure.
        Removes or replaces invalid Unicode characters like surrogates.
        
        Args:
            data: The data to sanitize
            aggressive: If True, use more aggressive sanitization (ASCII only)
        """
        if isinstance(data, str):
            if aggressive:
                # Remove all non-ASCII characters
                return ''.join(char if ord(char) < 128 else '?' for char in data)
            else:
                # Remove surrogate pairs and other problematic Unicode
                try:
                    # First try: encode to UTF-8 and back
                    return data.encode('utf-8', errors='replace').decode('utf-8', errors='replace')
                except Exception:
                    # Second try: character-by-character cleaning
                    cleaned = []
                    for char in data:
                        try:
                            char.encode('utf-8')
                            cleaned.append(char)
                        except UnicodeEncodeError:
                            cleaned.append('?')  # Replace with placeholder
                    return ''.join(cleaned)
        elif isinstance(data, dict):
            return {key: self._sanitize_unicode(value, aggressive) for key, value in data.items()}
        elif isinstance(data, list):
            return [self._sanitize_unicode(item, aggressive) for item in data]
        else:
            return data
    
    def store_insights(self, paper_id: str, insights: PaperInsights, 
                      extraction_metadata: Optional[ExtractionMetadata] = None):
        """
        Store extracted insights with embeddings.
        
        Args:
            paper_id: Paper identifier
            insights: Extracted insights
            extraction_metadata: Optional extraction metadata
        """
        # Sanitize insights data before storing
        insights_dict = insights.dict()
        sanitized_insights = self._sanitize_unicode(insights_dict)
        
        # Store insights JSON
        insights_path = self.storage_root / "insights" / f"{paper_id}_insights.json"
        try:
            with open(insights_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_insights, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Failed to store insights for {paper_id}: {e}")
            # Try with aggressive sanitization
            sanitized_insights = self._sanitize_unicode(insights_dict, aggressive=True)
            with open(insights_path, 'w', encoding='utf-8') as f:
                json.dump(sanitized_insights, f, indent=2, default=str, ensure_ascii=True)
        
        # Generate embedding from abstract and key findings for enhanced RAG
        paper_data = self.load_paper(paper_id)
        abstract = paper_data.get('summary', '') if paper_data else ''
        key_findings_text = " ".join(insights.key_findings)
        
        # Combine abstract and key findings for comprehensive searchable text
        searchable_text = f"Abstract: {abstract} Key Findings: {key_findings_text}"
        
        # Also include the full searchable text from insights
        full_searchable = insights.to_searchable_text()
        combined_text = f"{searchable_text} {full_searchable}"
        
        # Sanitize the text before embedding
        combined_text = self._sanitize_unicode(combined_text)
        
        embedding = self.embedder.encode([combined_text])[0]
        
        # Store in vector DB with enhanced metadata
        self.insights_collection.add(
            embeddings=[embedding.tolist()],
            documents=[combined_text],
            metadatas=[{
                "paper_id": paper_id,
                "study_type": insights.study_type.value,
                "complexity": insights.implementation_complexity.value,
                "techniques": ", ".join(t.value for t in insights.techniques_used),
                "quality_score": insights.get_quality_score(),
                "evidence_strength": insights.evidence_strength,
                "practical_applicability": insights.practical_applicability,
                "key_findings_count": len(insights.key_findings),
                "has_code": insights.has_code_available,
                "has_dataset": insights.has_dataset_available,
                "published_year": self._extract_year(paper_data) if paper_data else 2020
            }],
            ids=[paper_id]
        )
        
        # Store in SQLite with enhanced fields
        self.metadata_conn.execute("""
            INSERT OR REPLACE INTO insights 
            (paper_id, study_type, complexity, techniques, 
             quality_score, evidence_strength, practical_applicability, extraction_confidence, 
             has_code, has_dataset, key_findings_count, extraction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            paper_id,
            insights.study_type.value,
            insights.implementation_complexity.value,
            json.dumps([t.value for t in insights.techniques_used]),
            insights.get_quality_score(),
            insights.evidence_strength,
            insights.practical_applicability,
            insights.extraction_confidence,
            insights.has_code_available,
            insights.has_dataset_available,
            len(insights.key_findings),
            insights.extraction_timestamp.isoformat()
        ))
        
        # Store extraction metadata if provided
        if extraction_metadata:
            self.metadata_conn.execute("""
                INSERT OR REPLACE INTO extraction_metadata
                (extraction_id, paper_id, extraction_time_seconds, api_calls_made,
                 estimated_cost_usd, extractor_version, llm_model, extraction_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                extraction_metadata.extraction_id,
                paper_id,
                extraction_metadata.extraction_time_seconds,
                extraction_metadata.api_calls_made,
                extraction_metadata.estimated_cost_usd,
                extraction_metadata.extractor_version,
                extraction_metadata.llm_model,
                extraction_metadata.extraction_timestamp.isoformat()
            ))
        
        self.metadata_conn.commit()
        logger.info(f"Stored insights for paper {paper_id} with {len(insights.key_findings)} key findings")
    
    def _extract_year(self, paper_data: Dict) -> int:
        """Extract publication year from paper data."""
        if not paper_data or not paper_data.get('published'):
            return 2020
        try:
            return int(paper_data['published'][:4])
        except:
            return 2020
    
    def find_similar_papers(self, user_context: UserContext, 
                           n_results: int = 20) -> List[Dict]:
        """
        Find papers matching user context using enhanced vector similarity on abstracts and key findings.
        
        Args:
            user_context: User requirements and constraints
            n_results: Number of results to return
            
        Returns:
            List of paper insights with similarity scores, prioritized by recency, quality, evidence, and applicability
        """
        # Generate query embedding from user context
        query_text = user_context.to_search_query()
        query_embedding = self.embedder.encode([query_text])[0]
        
        # Build filters for vector search
        filters = []

        # Complexity filter based on budget
        if user_context.budget_constraint == "low":
            filters.append({"complexity": {"$in": ["low", "medium"]}})
        elif user_context.budget_constraint == "medium":
            filters.append({"complexity": {"$in": ["low", "medium", "high"]}})

        # Combine filters
        where_filters = None
        if len(filters) == 1:
            where_filters = filters[0]
        elif len(filters) > 1:
            where_filters = {"$and": filters}

        # Search with enhanced retrieval (get more for re-ranking)
        results = self.insights_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results * 3,  # Get extra for sophisticated re-ranking
            where=where_filters
        )

        # Post-process and re-rank results by multiple factors
        similar_papers = []
        for i, paper_id in enumerate(results['ids'][0]):
            # Load full insights
            insights = self.load_insights(paper_id)
            if not insights:
                continue
            
            # Additional filtering based on user context
            if not self._matches_user_constraints(insights, user_context):
                continue
            
            metadata = results['metadatas'][0][i]
            similarity_score = 1 - results['distances'][0][i]  # Convert distance to similarity
            
            # Calculate enhanced ranking score prioritizing recency, quality, evidence, applicability
            ranking_score = self._calculate_ranking_score(
                similarity_score, 
                insights, 
                metadata, 
                user_context
            )
            
            similar_papers.append({
                'paper_id': paper_id,
                'insights': insights,
                'similarity_score': similarity_score,
                'ranking_score': ranking_score,
                'metadata': metadata
            })
        
        # Sort by enhanced ranking score (combines multiple factors)
        similar_papers.sort(key=lambda x: x['ranking_score'], reverse=True)
        
        # Return top N results
        return similar_papers[:n_results]
    
    def _calculate_ranking_score(self, similarity_score: float, insights: PaperInsights, 
                               metadata: Dict, user_context: UserContext) -> float:
        """
        Calculate enhanced ranking score prioritizing recency, quality, evidence strength, and practical applicability.
        """
        # Get publication year for recency scoring
        pub_year = metadata.get('published_year', 2020)
        current_year = datetime.now().year
        
        # Recency score (more recent = higher score)
        recency_score = max(0, 1.0 - (current_year - pub_year) * 0.1)  # 10% decay per year
        
        # Quality components
        quality_score = insights.get_quality_score()
        evidence_score = insights.evidence_strength
        applicability_score = insights.practical_applicability
        
        # Key findings richness (more detailed findings = higher score)
        findings_score = min(1.0, len(insights.key_findings) / 8.0)  # Normalize to max 8 findings
        
        # Technique relevance (bonus for preferred techniques)
        technique_bonus = 0.0
        if user_context.preferred_techniques:
            user_techniques = set(t.value for t in user_context.preferred_techniques)
            paper_techniques = set(t.value for t in insights.techniques_used)
            if user_techniques.intersection(paper_techniques):
                technique_bonus = 0.1
        
        # Risk tolerance alignment
        risk_bonus = 0.0
        if user_context.risk_tolerance == "conservative" and evidence_score > 0.7:
            risk_bonus = 0.1
        elif user_context.risk_tolerance == "aggressive" and insights.implementation_complexity.value == "low":
            risk_bonus = 0.1
        
        # Weighted combination emphasizing the key factors
        final_score = (
            similarity_score * 0.25 +        # Vector similarity
            recency_score * 0.25 +           # Recency priority
            quality_score * 0.20 +           # Overall quality
            evidence_score * 0.15 +          # Evidence strength
            applicability_score * 0.10 +     # Practical applicability
            findings_score * 0.05 +          # Key findings richness
            technique_bonus +                # Technique preference bonus
            risk_bonus                       # Risk alignment bonus
        )
        
        return min(1.0, final_score)  # Cap at 1.0
    
    def _matches_user_constraints(self, insights: PaperInsights, 
                                 user_context: UserContext) -> bool:
        """Check if paper insights match user constraints."""
        # Avoided techniques
        if user_context.avoided_techniques:
            for tech in insights.techniques_used:
                if tech in user_context.avoided_techniques:
                    return False
        
        # Risk tolerance constraints
        if user_context.risk_tolerance == "conservative":
            if insights.evidence_strength < 0.6:  # Slightly relaxed for more results
                return False
        
        # Budget constraint via complexity
        if user_context.budget_constraint == "low":
            if insights.implementation_complexity.value in ["high", "very_high"]:
                return False
        
        return True
    
    def load_insights(self, paper_id: str) -> Optional[PaperInsights]:
        """Load insights for a specific paper."""
        insights_path = self.storage_root / "insights" / f"{paper_id}_insights.json"
        if not insights_path.exists():
            return None
        
        with open(insights_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return PaperInsights(**data)
    
    def load_paper(self, paper_id: str) -> Optional[Dict]:
        """Load raw paper data."""
        paper_path = self.storage_root / "papers" / f"{paper_id}.json"
        if not paper_path.exists():
            return None
        
        with open(paper_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_statistics(self) -> Dict:
        """Get enhanced storage statistics."""
        stats = {
            'total_papers': 0,
            'total_insights': 0,
            'papers_with_code': 0,
            'complexity_distribution': {},
            'study_type_distribution': {},
            'average_quality_score': 0.0,
            'average_evidence_strength': 0.0,
            'average_practical_applicability': 0.0,
            'average_key_findings_count': 0.0,
            'total_extraction_cost': 0.0,
            'recent_papers_count': 0  # Papers from last 2 years
        }
        
        # Count files
        stats['total_papers'] = len(list((self.storage_root / "papers").glob("*.json")))
        stats['total_insights'] = len(list((self.storage_root / "insights").glob("*.json")))
        
        # Get distributions and enhanced metrics from SQLite
        cursor = self.metadata_conn.cursor()
        
        # Complexity distribution
        cursor.execute("""
            SELECT complexity, COUNT(*) as count 
            FROM insights 
            GROUP BY complexity
        """)
        stats['complexity_distribution'] = dict(cursor.fetchall())
        
        # Study type distribution
        cursor.execute("""
            SELECT study_type, COUNT(*) as count 
            FROM insights 
            GROUP BY study_type
        """)
        stats['study_type_distribution'] = dict(cursor.fetchall())
        
        # Enhanced quality metrics
        cursor.execute("""
            SELECT 
                AVG(quality_score) as avg_quality,
                AVG(evidence_strength) as avg_evidence,
                AVG(practical_applicability) as avg_applicability,
                AVG(key_findings_count) as avg_findings,
                SUM(CASE WHEN has_code THEN 1 ELSE 0 END) as with_code
            FROM insights
        """)
        result = cursor.fetchone()
        if result:
            stats['average_quality_score'] = result['avg_quality'] or 0.0
            stats['average_evidence_strength'] = result['avg_evidence'] or 0.0
            stats['average_practical_applicability'] = result['avg_applicability'] or 0.0
            stats['average_key_findings_count'] = result['avg_findings'] or 0.0
            stats['papers_with_code'] = result['with_code'] or 0
        
        # Recent papers count (last 2 years)
        current_year = datetime.now().year
        cursor.execute("""
            SELECT COUNT(*) as recent_count
            FROM papers 
            WHERE published_date >= ?
        """, (f"{current_year - 2}-01-01",))
        result = cursor.fetchone()
        if result:
            stats['recent_papers_count'] = result['recent_count'] or 0
        
        # Total cost
        cursor.execute("""
            SELECT SUM(estimated_cost_usd) as total_cost
            FROM extraction_metadata
        """)
        result = cursor.fetchone()
        if result:
            stats['total_extraction_cost'] = result['total_cost'] or 0.0
        
        return stats
    
    def clear_all(self):
        """Clear all stored data (for debugging)."""
        # Clear ChromaDB
        self.chroma_client.delete_collection("paper_insights")
        self._init_vector_db()
        
        # Clear SQLite
        self.metadata_conn.executescript("""
            DELETE FROM extraction_metadata;
            DELETE FROM insights;
            DELETE FROM papers;
        """)
        self.metadata_conn.commit()
        
        # Clear files
        for directory in ["papers", "insights", "checkpoints"]:
            for file in (self.storage_root / directory).glob("*.json"):
                file.unlink()
        
        logger.info("Cleared all storage")
    
    def __del__(self):
        """Cleanup connections on deletion."""
        if hasattr(self, '_connections'):
            for conn in self._connections.values():
                conn.close()