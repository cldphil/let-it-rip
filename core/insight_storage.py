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
        conn.executescript("""
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
                industries TEXT,
                techniques TEXT,
                quality_score REAL,
                extraction_confidence REAL,
                has_code BOOLEAN,
                has_dataset BOOLEAN,
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
            CREATE INDEX IF NOT EXISTS idx_insights_complexity ON insights(complexity);
            CREATE INDEX IF NOT EXISTS idx_insights_study_type ON insights(study_type);
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
        paper_id = paper_data.get('id', '').split('/')[-1]  # Extract arXiv ID
        if not paper_id:
            paper_id = f"paper_{datetime.utcnow().timestamp()}"
        
        # Store JSON file
        paper_path = self.storage_root / "papers" / f"{paper_id}.json"
        with open(paper_path, 'w', encoding='utf-8') as f:
            json.dump(paper_data, f, indent=2, ensure_ascii=False)
        
        # Store metadata in SQLite
        self.metadata_conn.execute("""
            INSERT OR REPLACE INTO papers 
            (paper_id, title, authors, published_date, arxiv_categories)
            VALUES (?, ?, ?, ?, ?)
        """, (
            paper_id,
            paper_data.get('title', ''),
            json.dumps(paper_data.get('authors', [])),
            paper_data.get('published', ''),
            json.dumps(paper_data.get('categories', []))
        ))
        self.metadata_conn.commit()
        
        logger.info(f"Stored paper {paper_id}: {paper_data.get('title', '')[:50]}...")
        return paper_id
    
    def store_insights(self, paper_id: str, insights: PaperInsights, 
                      extraction_metadata: Optional[ExtractionMetadata] = None):
        """
        Store extracted insights with embeddings.
        
        Args:
            paper_id: Paper identifier
            insights: Extracted insights
            extraction_metadata: Optional extraction metadata
        """
        # Store insights JSON
        insights_path = self.storage_root / "insights" / f"{paper_id}_insights.json"
        with open(insights_path, 'w', encoding='utf-8') as f:
            json.dump(insights.dict(), f, indent=2, default=str)
        
        # Generate embedding
        searchable_text = insights.to_searchable_text()
        embedding = self.embedder.encode([searchable_text])[0]
        
        # Store in vector DB
        self.insights_collection.add(
            embeddings=[embedding.tolist()],
            documents=[searchable_text],
            metadatas=[{
                "paper_id": paper_id,
                "study_type": insights.study_type.value,
                "complexity": insights.implementation_complexity.value,
                "industries": ", ".join(i.value for i in insights.industry_applications),
                "techniques": ", ".join(t.value for t in insights.techniques_used),
                "quality_score": insights.get_quality_score(),
                "has_code": insights.has_code_available,
                "has_dataset": insights.has_dataset_available
            }],
            ids=[paper_id]
        )
        
        # Store in SQLite
        self.metadata_conn.execute("""
            INSERT OR REPLACE INTO insights 
            (paper_id, study_type, complexity, industries, techniques, 
             quality_score, extraction_confidence, has_code, has_dataset, extraction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            paper_id,
            insights.study_type.value,
            insights.implementation_complexity.value,
            json.dumps([i.value for i in insights.industry_applications]),
            json.dumps([t.value for t in insights.techniques_used]),
            insights.get_quality_score(),
            insights.extraction_confidence,
            insights.has_code_available,
            insights.has_dataset_available,
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
        logger.info(f"Stored insights for paper {paper_id}")
    
    def find_similar_papers(self, user_context: UserContext, 
                           n_results: int = 20) -> List[Dict]:
        """
        Find papers matching user context using vector similarity.
        
        Args:
            user_context: User requirements and constraints
            n_results: Number of results to return
            
        Returns:
            List of paper insights with similarity scores
        """
        # Generate query embedding
        query_text = user_context.to_search_query()
        query_embedding = self.embedder.encode([query_text])[0]
        
        # Build where clause for filtering
        filters = []

        # Complexity filter
        if user_context.budget_constraint == "low":
            filters.append({"complexity": {"$in": ["low", "medium"]}})
        elif user_context.budget_constraint == "medium":
            filters.append({"complexity": {"$in": ["low", "medium", "high"]}})

        # Industry filter if specific
        if user_context.industry != "general":
            filters.append({"industries": {"$in": [user_context.industry.value]}})

        # Combine filters with $and if more than one
        where_filters = None
        if len(filters) == 1:
            where_filters = filters[0]
        elif len(filters) > 1:
            where_filters = {"$and": filters}

        # Search with filters
        results = self.insights_collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=n_results * 2,  # Get extra for filtering
            where=where_filters
        )

        
        # Post-process results
        similar_papers = []
        for i, paper_id in enumerate(results['ids'][0]):
            # Load full insights
            insights = self.load_insights(paper_id)
            if not insights:
                continue
            
            # Additional filtering based on user context
            if not self._matches_user_constraints(insights, user_context):
                continue
            
            similar_papers.append({
                'paper_id': paper_id,
                'insights': insights,
                'similarity_score': 1 - results['distances'][0][i],  # Convert distance to similarity
                'metadata': results['metadatas'][0][i]
            })
            
            if len(similar_papers) >= n_results:
                break
        
        # Sort by combined score (similarity + quality)
        similar_papers.sort(
            key=lambda x: (
                x['similarity_score'] * 0.7 + 
                x['insights'].get_quality_score() * 0.3
            ),
            reverse=True
        )
        
        return similar_papers
    
    def _matches_user_constraints(self, insights: PaperInsights, 
                                 user_context: UserContext) -> bool:
        """Check if paper insights match user constraints."""
        # Timeline constraint
        if user_context.timeline_weeks:
            paper_weeks = insights.resource_requirements.estimated_time_weeks
            if paper_weeks and paper_weeks > user_context.timeline_weeks:
                return False
        
        # Avoided techniques
        if user_context.avoided_techniques:
            for tech in insights.techniques_used:
                if tech in user_context.avoided_techniques:
                    return False
        
        # Risk tolerance
        if user_context.risk_tolerance == "conservative":
            if insights.evidence_strength < 0.7 or not insights.industry_validation:
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
        """Get storage statistics."""
        stats = {
            'total_papers': 0,
            'total_insights': 0,
            'papers_with_code': 0,
            'complexity_distribution': {},
            'study_type_distribution': {},
            'average_quality_score': 0.0,
            'total_extraction_cost': 0.0
        }
        
        # Count files
        stats['total_papers'] = len(list((self.storage_root / "papers").glob("*.json")))
        stats['total_insights'] = len(list((self.storage_root / "insights").glob("*.json")))
        
        # Get distributions from SQLite
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
        
        # Quality metrics
        cursor.execute("""
            SELECT 
                AVG(quality_score) as avg_quality,
                SUM(CASE WHEN has_code THEN 1 ELSE 0 END) as with_code
            FROM insights
        """)
        result = cursor.fetchone()
        if result:
            stats['average_quality_score'] = result['avg_quality'] or 0.0
            stats['papers_with_code'] = result['with_code'] or 0
        
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