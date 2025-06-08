"""
Storage layer for paper insights with vector embeddings and local persistence.
Uses ChromaDB for vector search and JSON for full data storage.
Enhanced with Supabase cloud storage support.
"""

import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import logging

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

from .insight_schema import PaperInsights, UserContext, ExtractionMetadata
from config import Config

logger = logging.getLogger(__name__)


class InsightStorage:
    """
    Manages storage and retrieval of paper insights.
    Supports both local and cloud (Supabase) storage.
    
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
        
        # Initialize Supabase if cloud storage is enabled
        self.supabase = None
        if Config.USE_CLOUD_STORAGE:
            self._init_supabase()
        
        logger.info(f"Initialized storage at {self.storage_root} (Cloud: {Config.USE_CLOUD_STORAGE})")
    
    def _init_supabase(self):
        """Initialize Supabase client for cloud storage."""
        try:
            from supabase import create_client, Client
            
            url = Config.SUPABASE_URL
            key = Config.SUPABASE_ANON_KEY or Config.SUPABASE_SERVICE_ROLE_KEY
            
            if not url or not key:
                logger.warning("Supabase credentials not found, falling back to local storage")
                return
            
            self.supabase: Client = create_client(url, key)
            logger.info("Initialized Supabase client for cloud storage")
            
        except ImportError:
            logger.warning("Supabase library not installed, falling back to local storage")
        except Exception as e:
            logger.error(f"Failed to initialize Supabase: {e}")
    
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
        # Updated schema without deprecated fields
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
                reputation_score REAL,
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
            
            CREATE INDEX IF NOT EXISTS idx_insights_reputation ON insights(reputation_score DESC);
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
        Store raw paper data in both local and cloud storage.
        
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
        
        # Store in Supabase if enabled
        if self.supabase and Config.USE_CLOUD_STORAGE:
            try:
                # Prepare data for Supabase
                supabase_data = {
                    'id': paper_id,
                    'title': sanitized_paper.get('title', ''),
                    'authors': sanitized_paper.get('authors', []),
                    'summary': sanitized_paper.get('summary', ''),
                    'published_date': sanitized_paper.get('published', ''),
                    'categories': sanitized_paper.get('categories', []),
                    'pdf_url': sanitized_paper.get('pdf_url', ''),
                    'full_text': sanitized_paper.get('full_text', ''),
                    'comments': sanitized_paper.get('comments', ''),
                    'metadata': sanitized_paper  # Store full data as JSON
                }
                
                # Upsert to Supabase
                result = self.supabase.table('papers').upsert(supabase_data).execute()
                logger.info(f"Stored paper {paper_id} in Supabase")
                
            except Exception as e:
                logger.error(f"Failed to store paper in Supabase: {e}")
                # Continue with local storage
        
        # Always store locally (for backup or if cloud storage fails)
        if Config.ENABLE_LOCAL_BACKUP or not self.supabase:
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
        Store extracted insights with embeddings in both local and cloud storage.
        
        Args:
            paper_id: Paper identifier
            insights: Extracted insights
            extraction_metadata: Optional extraction metadata
        """
        # Sanitize insights data before storing
        insights_dict = insights.dict()
        sanitized_insights = self._sanitize_unicode(insights_dict)
        
        # Store in Supabase if enabled
        if self.supabase and Config.USE_CLOUD_STORAGE:
            try:
                # Prepare insights data for Supabase
                supabase_insights = {
                    'id': paper_id,
                    'paper_id': paper_id,
                    'study_type': insights.study_type.value,
                    'techniques_used': [t.value for t in insights.techniques_used],
                    'implementation_complexity': insights.implementation_complexity.value,
                    'reputation_score': insights.get_reputation_score(),
                    'extraction_confidence': insights.extraction_confidence,
                    'has_code': insights.has_code_available,
                    'has_dataset': insights.has_dataset_available,
                    'key_findings_count': len(insights.key_findings),
                    'extraction_timestamp': insights.extraction_timestamp.isoformat(),
                    'total_author_hindex': insights.total_author_hindex,
                    'has_conference_mention': insights.has_conference_mention,
                    'key_findings': insights.key_findings,
                    'limitations': insights.limitations,
                    'problem_addressed': insights.problem_addressed,
                    'prerequisites': insights.prerequisites,
                    'real_world_applications': insights.real_world_applications,
                    'full_insights': sanitized_insights  # Store complete insights as JSON
                }
                
                # Upsert to Supabase
                result = self.supabase.table('insights').upsert(supabase_insights).execute()
                logger.info(f"Stored insights for {paper_id} in Supabase")
                
                # Store extraction metadata if provided
                if extraction_metadata:
                    metadata_data = {
                        'id': extraction_metadata.extraction_id,
                        'paper_id': paper_id,
                        'extraction_time_seconds': extraction_metadata.extraction_time_seconds,
                        'api_calls_made': extraction_metadata.api_calls_made,
                        'estimated_cost_usd': extraction_metadata.estimated_cost_usd,
                        'extractor_version': extraction_metadata.extractor_version,
                        'llm_model': extraction_metadata.llm_model,
                        'extraction_timestamp': extraction_metadata.extraction_timestamp.isoformat()
                    }
                    
                    self.supabase.table('extraction_metadata').upsert(metadata_data).execute()
                    logger.info(f"Stored extraction metadata for {paper_id} in Supabase")
                
            except Exception as e:
                logger.error(f"Failed to store insights in Supabase: {e}")
                # Continue with local storage
        
        # Always store locally for backup
        if Config.ENABLE_LOCAL_BACKUP or not self.supabase:
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
        
        # Store in vector DB with updated metadata (removed deprecated fields)
        self.insights_collection.add(
            embeddings=[embedding.tolist()],
            documents=[combined_text],
            metadatas=[{
                "paper_id": paper_id,
                "study_type": insights.study_type.value,
                "complexity": insights.implementation_complexity.value,
                "techniques": ", ".join(t.value for t in insights.techniques_used),
                "reputation_score": insights.get_reputation_score(),
                "key_findings_count": len(insights.key_findings),
                "has_code": insights.has_code_available,
                "has_dataset": insights.has_dataset_available,
                "published_year": self._extract_year(paper_data) if paper_data else 2020
            }],
            ids=[paper_id]
        )
        
        # Store in SQLite with updated schema (removed deprecated fields)
        self.metadata_conn.execute("""
            INSERT OR REPLACE INTO insights 
            (paper_id, study_type, complexity, techniques, 
             reputation_score, extraction_confidence, 
             has_code, has_dataset, key_findings_count, extraction_timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            paper_id,
            insights.study_type.value,
            insights.implementation_complexity.value,
            json.dumps([t.value for t in insights.techniques_used]),
            insights.get_reputation_score(),
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
        Find papers matching user context using vector similarity.
        
        Args:
            user_context: User requirements and constraints
            n_results: Number of results to return
            
        Returns:
            List of paper insights with similarity scores, prioritized by reputation, recency, and relevance
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
            
            # Calculate enhanced ranking score with updated algorithm
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
        Calculate enhanced ranking score with updated algorithm.
        
        New formula emphasizes:
        - Vector similarity (30%)
        - Reputation score (25%) 
        - Recency (20%)
        - Key findings richness (15%)
        - Technique relevance and risk alignment (10%)
        """
        # Get publication year for recency scoring
        pub_year = metadata.get('published_year', 2020)
        current_year = datetime.now().year
        
        # Recency score (more recent = higher score)
        recency_score = max(0, 1.0 - (current_year - pub_year) * 0.1)  # 10% decay per year
        
        # Reputation score from insights
        reputation_score = insights.get_reputation_score()
        
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
        if user_context.risk_tolerance == "conservative":
            # Prioritize case studies for conservative users
            if insights.study_type.value == "case_study":
                risk_bonus = 0.1
            elif reputation_score > 0.7:  # High reputation papers
                risk_bonus = 0.05
        elif user_context.risk_tolerance == "aggressive":
            # Prioritize low complexity for aggressive/fast implementation
            if insights.implementation_complexity.value == "low":
                risk_bonus = 0.1
        
        # Industry validation bonus (especially for case studies)
        validation_bonus = 0.0
        if insights.study_type.value == "case_study":
            validation_bonus = 0.1
        
        # Updated weighted combination
        final_score = (
            similarity_score * 0.30 +        # Vector similarity (increased weight)
            reputation_score * 0.25 +           # Reputation score (maintained)
            recency_score * 0.20 +           # Recency (maintained)
            findings_score * 0.15 +          # Key findings richness (increased)
            technique_bonus +                # Technique preference bonus
            risk_bonus +                     # Risk alignment bonus
            validation_bonus                 # Industry validation bonus
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
            # For conservative users, prefer validated studies or high reputation
            if insights.get_reputation_score() < 0.5:
                return False
        
        # Budget constraint via complexity
        if user_context.budget_constraint == "low":
            if insights.implementation_complexity.value in ["high", "very_high"]:
                return False
        
        return True
    
    def load_insights(self, paper_id: str) -> Optional[PaperInsights]:
        """Load insights for a specific paper from cloud or local storage."""
        # Try Supabase first if enabled
        if self.supabase and Config.USE_CLOUD_STORAGE:
            try:
                result = self.supabase.table('insights').select('*').eq('id', paper_id).execute()
                
                if result.data and len(result.data) > 0:
                    # Convert from Supabase format back to PaperInsights
                    insights_data = result.data[0]
                    
                    # Use full_insights if available, otherwise reconstruct
                    if 'full_insights' in insights_data and insights_data['full_insights']:
                        return PaperInsights(**insights_data['full_insights'])
                    else:
                        # Fallback: reconstruct from individual fields
                        logger.warning(f"Reconstructing insights for {paper_id} from individual fields")
                        # This would need proper reconstruction logic
                        return None
                        
            except Exception as e:
                logger.error(f"Failed to load insights from Supabase: {e}")
        
        # Fallback to local storage
        insights_path = self.storage_root / "insights" / f"{paper_id}_insights.json"
        if not insights_path.exists():
            return None
        
        with open(insights_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return PaperInsights(**data)
    
    def load_paper(self, paper_id: str) -> Optional[Dict]:
        """Load raw paper data from cloud or local storage."""
        # Try Supabase first if enabled
        if self.supabase and Config.USE_CLOUD_STORAGE:
            try:
                result = self.supabase.table('papers').select('*').eq('id', paper_id).execute()
                
                if result.data and len(result.data) > 0:
                    paper_data = result.data[0]
                    
                    # Use metadata field if available, otherwise use individual fields
                    if 'metadata' in paper_data and paper_data['metadata']:
                        return paper_data['metadata']
                    else:
                        # Reconstruct paper data from individual fields
                        return {
                            'id': paper_data.get('id'),
                            'title': paper_data.get('title'),
                            'authors': paper_data.get('authors', []),
                            'summary': paper_data.get('summary', ''),
                            'published': paper_data.get('published_date', ''),
                            'categories': paper_data.get('categories', []),
                            'pdf_url': paper_data.get('pdf_url', ''),
                            'full_text': paper_data.get('full_text', ''),
                            'comments': paper_data.get('comments', '')
                        }
                        
            except Exception as e:
                logger.error(f"Failed to load paper from Supabase: {e}")
        
        # Fallback to local storage
        paper_path = self.storage_root / "papers" / f"{paper_id}.json"
        if not paper_path.exists():
            return None
        
        with open(paper_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def get_statistics(self) -> Dict:
        """
        Get enhanced storage statistics from cloud or local storage.
        """
        stats = {
            'total_papers': 0,
            'total_insights': 0,
            'papers_with_code': 0,
            'complexity_distribution': {},
            'study_type_distribution': {},
            'average_reputation_score': 0.0,
            'average_key_findings_count': 0.0,
            'recent_papers_count': 0,  # Papers from last 2 years
            'case_studies_count': 0  # Count of case studies
        }
        
        # Try Supabase first if enabled
        if self.supabase and Config.USE_CLOUD_STORAGE:
            try:
                # Get counts from Supabase
                papers_result = self.supabase.table('papers').select('id', count='exact').execute()
                stats['total_papers'] = papers_result.count
                
                insights_result = self.supabase.table('insights').select('*').execute()
                insights_data = insights_result.data
                stats['total_insights'] = len(insights_data)
                
                if insights_data:
                    # Process insights for statistics
                    total_reputation = 0.0
                    total_findings = 0
                    complexity_counts = {}
                    study_type_counts = {}
                    current_year = datetime.now().year
                    
                    for insight in insights_data:
                        # Reputation score
                        reputation = insight.get('reputation_score', 0)
                        total_reputation += reputation
                        
                        # Key findings count
                        findings_count = insight.get('key_findings_count', 0)
                        total_findings += findings_count
                        
                        # Code availability
                        if insight.get('has_code'):
                            stats['papers_with_code'] += 1
                        
                        # Case studies
                        if insight.get('study_type') == 'case_study':
                            stats['case_studies_count'] += 1
                        
                        # Complexity distribution
                        complexity = insight.get('implementation_complexity', 'unknown')
                        complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
                        
                        # Study type distribution
                        study_type = insight.get('study_type', 'unknown')
                        study_type_counts[study_type] = study_type_counts.get(study_type, 0) + 1
                    
                    # Calculate averages
                    if stats['total_insights'] > 0:
                        stats['average_reputation_score'] = round(total_reputation / stats['total_insights'], 2)
                        stats['average_key_findings_count'] = round(total_findings / stats['total_insights'], 1)
                    
                    stats['complexity_distribution'] = complexity_counts
                    stats['study_type_distribution'] = study_type_counts
                
                # Get recent papers count (would need to query papers table with date filter)
                # For now, this is a simplified version
                
                return stats
                
            except Exception as e:
                logger.error(f"Failed to get statistics from Supabase: {e}")
                # Fall through to local storage
        
        # Fallback to local storage statistics
        # Count paper files
        paper_files = list((self.storage_root / "papers").glob("*.json"))
        stats['total_papers'] = len(paper_files)
        
        # Process insights files directly
        insights_files = list((self.storage_root / "insights").glob("*_insights.json"))
        stats['total_insights'] = len(insights_files)
        
        if stats['total_insights'] == 0:
            return stats
        
        # Aggregate metrics from JSON files
        total_reputation = 0.0
        total_findings = 0
        
        complexity_counts = {}
        study_type_counts = {}
        
        current_year = datetime.now().year
        
        for insight_file in insights_files:
            try:
                # Load insights
                with open(insight_file, 'r', encoding='utf-8') as f:
                    insight_data = json.load(f)
                
                # Create PaperInsights object for reputation score calculation
                insights = PaperInsights(**insight_data)
                
                # Update metrics
                reputation_score = insights.get_reputation_score()
                total_reputation += reputation_score
                total_findings += len(insights.key_findings)
                
                # Count code availability
                if insights.has_code_available:
                    stats['papers_with_code'] += 1

                # Count case studies
                if insights.study_type.value == "case_study":
                    stats['case_studies_count'] += 1
                
                # Update distributions
                complexity = insights.implementation_complexity.value
                complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
                
                study_type = insights.study_type.value
                study_type_counts[study_type] = study_type_counts.get(study_type, 0) + 1
                
                # Check if paper is recent
                paper_id = insight_file.stem.replace("_insights", "")
                paper_data = self.load_paper(paper_id)
                if paper_data and paper_data.get('published'):
                    try:
                        pub_year = int(paper_data['published'][:4])
                        if pub_year >= current_year - 2:
                            stats['recent_papers_count'] += 1
                    except:
                        pass
                        
            except Exception as e:
                logger.warning(f"Error processing insights file {insight_file}: {e}")
                continue
        
        # Calculate averages
        if stats['total_insights'] > 0:
            stats['average_reputation_score'] = round(total_reputation / stats['total_insights'], 2)
            stats['average_key_findings_count'] = round(total_findings / stats['total_insights'], 1)
        
        # Set distributions
        stats['complexity_distribution'] = complexity_counts
        stats['study_type_distribution'] = study_type_counts
        
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
        
        # Clear Supabase if enabled
        if self.supabase and Config.USE_CLOUD_STORAGE:
            try:
                # Delete all records from Supabase tables
                # Note: This is a destructive operation - use with caution
                self.supabase.table('extraction_metadata').delete().neq('id', '').execute()
                self.supabase.table('insights').delete().neq('id', '').execute()
                self.supabase.table('papers').delete().neq('id', '').execute()
                logger.info("Cleared all data from Supabase")
            except Exception as e:
                logger.error(f"Failed to clear Supabase data: {e}")
        
        logger.info("Cleared all storage")
    
    def __del__(self):
        """Cleanup connections on deletion."""
        if hasattr(self, '_connections'):
            for conn in self._connections.values():
                conn.close()