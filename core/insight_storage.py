"""
Cloud-only storage layer for paper insights using Supabase.
Handles papers, insights, and vector embeddings entirely through Supabase.
"""

import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import logging

from sentence_transformers import SentenceTransformer
from supabase import create_client, Client

from .insight_schema import (
    PaperInsights, UserContext, ExtractionMetadata,
    generate_paper_uuid, sanitize_for_cloud_storage
)
from config import Config

logger = logging.getLogger(__name__)


class InsightStorage:
    """
    Cloud-only storage for paper insights using Supabase.
    
    Handles:
    - Paper metadata and full text storage
    - Insight extraction results with vector embeddings
    - Vector similarity search using Supabase functions
    - Processing statistics and analytics
    """
    
    def __init__(self):
        """Initialize cloud-only storage with Supabase."""
        self._init_supabase()
        self._init_embedder()
        
        logger.info("Initialized cloud-only storage with Supabase")
    
    def _init_supabase(self):
        """Initialize Supabase client."""
        url = Config.SUPABASE_URL
        key = Config.SUPABASE_ANON_KEY or Config.SUPABASE_SERVICE_ROLE_KEY
        
        if not url or not key:
            raise ValueError("Supabase credentials are required for cloud-only operation")
        
        try:
            self.supabase: Client = create_client(url, key)
            logger.info(f"Connected to Supabase at {url[:30]}...")
            
            # Test connection
            self._test_connection()
            
        except Exception as e:
            logger.error(f"Failed to initialize Supabase: {e}")
            raise
    
    def _test_connection(self):
        """Test Supabase connection and required tables."""
        try:
            # Test basic connectivity by querying papers table
            result = self.supabase.table(Config.TABLE_PAPERS).select('id').limit(1).execute()
            logger.info("✅ Supabase connection verified")
            
        except Exception as e:
            logger.error(f"❌ Supabase connection test failed: {e}")
            raise ConnectionError(f"Cannot connect to Supabase: {e}")
    
    def _init_embedder(self):
        """Initialize sentence transformer for embeddings."""
        try:
            self.embedder = SentenceTransformer(Config.VECTOR_EMBEDDING_MODEL)
            logger.info(f"Initialized embedder: {Config.VECTOR_EMBEDDING_MODEL}")
        except Exception as e:
            logger.error(f"Failed to initialize embedder: {e}")
            raise
    
    def store_paper(self, paper_data: Dict) -> str:
        """
        Store raw paper data in Supabase.
        
        Args:
            paper_data: Paper metadata from arXiv
            
        Returns:
            Paper ID (arxiv ID)
        """
        paper_id = paper_data.get('id', '').split('/')[-1]
        if not paper_id:
            paper_id = f"paper_{datetime.utcnow().timestamp()}"
        
        # Sanitize paper data for cloud storage
        sanitized_paper = sanitize_for_cloud_storage(paper_data)
        
        try:
            # Generate consistent UUID for Supabase
            supabase_uuid = generate_paper_uuid(paper_id)
            
            # Prepare data for Supabase with proper validation
            supabase_data = {
                'id': supabase_uuid,
                'paper_id': paper_id,
                'title': sanitized_paper.get('title', '')[:Config.MAX_TITLE_LENGTH],
                'authors': sanitized_paper.get('authors', [])[:Config.MAX_AUTHORS_DISPLAY],
                'summary': sanitized_paper.get('summary', '')[:Config.MAX_ABSTRACT_LENGTH],
                'published_date': sanitized_paper.get('published', ''),
                'arxiv_categories': sanitized_paper.get('categories', []),
                'pdf_url': sanitized_paper.get('pdf_url', ''),
                'comments': sanitized_paper.get('comments', ''),
            }
            
            # Upsert to Supabase (insert or update if exists)
            result = self.supabase.table(Config.TABLE_PAPERS).upsert(supabase_data).execute()
            
            if result.data:
                logger.info(f"Stored paper {paper_id}: {sanitized_paper.get('title', '')[:50]}...")
            else:
                logger.warning(f"No data returned when storing paper {paper_id}")
            
            return paper_id
            
        except Exception as e:
            logger.error(f"Failed to store paper {paper_id}: {e}")
            raise
    
    def store_insights(self, paper_id: str, insights: PaperInsights, 
                      extraction_metadata: Optional[ExtractionMetadata] = None):
        """
        Store extracted insights with embeddings in Supabase.
        
        Args:
            paper_id: Paper identifier
            insights: Extracted insights
            extraction_metadata: Optional extraction metadata
        """
        try:
            # Generate UUID for Supabase
            supabase_uuid = generate_paper_uuid(paper_id)
            
            # Generate embedding for vector search
            searchable_text = insights.to_searchable_text()
            embedding = self.embedder.encode([searchable_text])[0]
            
            # Prepare insights data for Supabase
            insights_data = {
                'id': supabase_uuid,
                'paper_id': paper_id,
                'study_type': insights.study_type.value,
                'techniques_used': [t.value for t in insights.techniques_used],
                'implementation_complexity': insights.implementation_complexity.value,
                'reputation_score': insights.get_reputation_score(),
                'extraction_confidence': insights.extraction_confidence,
                'has_code': insights.has_code_available,
                'has_dataset': insights.has_dataset_available,
                'key_findings_count': len(insights.key_findings),
                'total_author_hindex': insights.total_author_hindex,
                'has_conference_mention': insights.has_conference_mention,
                'industry_validation': insights.industry_validation,
                'key_findings': insights.key_findings,
                'limitations': insights.limitations,
                'problem_addressed': insights.problem_addressed,
                'prerequisites': insights.prerequisites,
                'real_world_applications': insights.real_world_applications,
                'extraction_timestamp': insights.extraction_timestamp.isoformat(),
                'embedding': embedding.tolist(),  # Store embedding for vector search
                'searchable_text': searchable_text,
                'full_insights': insights.to_supabase_dict()  # Store complete insights
            }
            
            # Store insights
            result = self.supabase.table(Config.TABLE_INSIGHTS).upsert(insights_data).execute()
            
            if result.data:
                logger.info(f"Stored insights for {paper_id} with {len(insights.key_findings)} key findings")
            
            # Store extraction metadata if provided
            if extraction_metadata:
                self._store_extraction_metadata(paper_id, supabase_uuid, extraction_metadata)
            
        except Exception as e:
            logger.error(f"Failed to store insights for {paper_id}: {e}")
            raise
    
    def _store_extraction_metadata(self, paper_id: str, paper_uuid: str, 
                                  extraction_metadata: ExtractionMetadata):
        """Store extraction metadata in Supabase."""
        try:
            metadata_data = {
                'id': str(uuid.uuid4()),
                'extraction_id': extraction_metadata.extraction_id,
                'paper_id': paper_id,
                'paper_uuid': paper_uuid,
                'extraction_time_seconds': extraction_metadata.extraction_time_seconds,
                'api_calls_made': extraction_metadata.api_calls_made,
                'estimated_cost_usd': extraction_metadata.estimated_cost_usd,
                'extractor_version': extraction_metadata.extractor_version,
                'llm_model': extraction_metadata.llm_model,
                'extraction_timestamp': extraction_metadata.extraction_timestamp.isoformat(),
                'sections_found': extraction_metadata.sections_found,
                'section_lengths': extraction_metadata.section_lengths,
                'extraction_errors': extraction_metadata.extraction_errors,
                'success': extraction_metadata.success
            }
            
            self.supabase.table(Config.TABLE_EXTRACTION_METADATA).upsert(metadata_data).execute()
            logger.info(f"Stored extraction metadata for {paper_id}")
            
        except Exception as e:
            logger.warning(f"Failed to store extraction metadata for {paper_id}: {e}")
    
    def find_similar_papers(self, user_context: UserContext, 
                           n_results: int = 20) -> List[Dict]:
        """
        Find papers matching user context using Supabase vector similarity.
        
        Args:
            user_context: User requirements and constraints
            n_results: Number of results to return
            
        Returns:
            List of paper insights with similarity scores
        """
        try:
            # Generate query embedding from user context
            query_text = user_context.to_search_query()
            query_embedding = self.embedder.encode([query_text])[0]
            
            # Use Supabase RPC function for vector similarity search
            results = self.supabase.rpc(
                Config.FUNCTION_MATCH_INSIGHTS,
                {
                    'query_embedding': query_embedding.tolist(),
                    'match_threshold': Config.VECTOR_SIMILARITY_THRESHOLD,
                    'match_count': min(n_results * 2, Config.MAX_VECTOR_SEARCH_RESULTS)  # Get extra for filtering
                }
            ).execute()
            
            if not results.data:
                logger.info("No similar papers found")
                return []
            
            # Process and filter results
            similar_papers = []
            for result in results.data:
                try:
                    # Load full insights from stored data
                    if 'full_insights' in result and result['full_insights']:
                        insights = PaperInsights.from_supabase_dict(result['full_insights'])
                    else:
                        # Fallback: reconstruct from individual fields
                        insights = self._reconstruct_insights_from_result(result)
                    
                    # Apply user constraints
                    if not self._matches_user_constraints(insights, user_context):
                        continue
                    
                    # Calculate enhanced ranking score
                    similarity_score = result.get('similarity', 0)
                    ranking_score = self._calculate_ranking_score(
                        similarity_score, insights, result, user_context
                    )
                    
                    similar_papers.append({
                        'paper_id': result['paper_id'],
                        'insights': insights,
                        'similarity_score': similarity_score,
                        'ranking_score': ranking_score,
                        'metadata': {
                            'reputation_score': result.get('reputation_score', 0),
                            'study_type': result.get('study_type', 'unknown'),
                            'complexity': result.get('implementation_complexity', 'unknown')
                        }
                    })
                    
                except Exception as e:
                    logger.warning(f"Error processing search result: {e}")
                    continue
            
            # Sort by ranking score and return top results
            similar_papers.sort(key=lambda x: x['ranking_score'], reverse=True)
            
            logger.info(f"Found {len(similar_papers)} relevant papers for user context")
            return similar_papers[:n_results]
            
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            return []
    
    def _reconstruct_insights_from_result(self, result: Dict) -> PaperInsights:
        """Reconstruct PaperInsights from Supabase result when full_insights not available."""
        from .insight_schema import StudyType, TechniqueCategory, ComplexityLevel
        
        try:
            # Map basic fields
            insights_data = {
                'paper_id': result.get('paper_id', ''),
                'key_findings': result.get('key_findings', []),
                'limitations': result.get('limitations', []),
                'study_type': StudyType(result.get('study_type', 'unknown')),
                'techniques_used': [TechniqueCategory(t) for t in result.get('techniques_used', [])],
                'implementation_complexity': ComplexityLevel(result.get('implementation_complexity', 'unknown')),
                'problem_addressed': result.get('problem_addressed', ''),
                'prerequisites': result.get('prerequisites', []),
                'real_world_applications': result.get('real_world_applications', []),
                'total_author_hindex': result.get('total_author_hindex', 0),
                'has_conference_mention': result.get('has_conference_mention', False),
                'industry_validation': result.get('industry_validation', False),
                'extraction_confidence': result.get('extraction_confidence', 0.5),
                'has_code_available': result.get('has_code', False),
                'has_dataset_available': result.get('has_dataset', False)
            }
            
            # Parse timestamp if available
            if result.get('extraction_timestamp'):
                try:
                    insights_data['extraction_timestamp'] = datetime.fromisoformat(
                        result['extraction_timestamp'].replace('Z', '+00:00')
                    )
                except:
                    pass
            
            return PaperInsights(**insights_data)
            
        except Exception as e:
            logger.error(f"Failed to reconstruct insights: {e}")
            # Return minimal insights
            return PaperInsights(
                paper_id=result.get('paper_id', ''),
                key_findings=['Reconstruction failed - limited data available']
            )
    
    def _calculate_ranking_score(self, similarity_score: float, insights: PaperInsights, 
                               result: Dict, user_context: UserContext) -> float:
        """Calculate enhanced ranking score for search results."""
        # Get publication year for recency scoring
        pub_year = result.get('published_year', 2020)
        current_year = datetime.now().year
        
        # Recency score
        recency_score = max(0, 1.0 - (current_year - pub_year) * Config.RECENCY_DECAY_RATE)
        
        # Reputation score
        reputation_score = insights.get_reputation_score()
        
        # Key findings richness
        findings_score = min(1.0, len(insights.key_findings) / 8.0)
        
        # Technique relevance bonus
        technique_bonus = 0.0
        if user_context.preferred_techniques:
            user_techniques = set(t.value for t in user_context.preferred_techniques)
            paper_techniques = set(t.value for t in insights.techniques_used)
            if user_techniques.intersection(paper_techniques):
                technique_bonus = 0.1
        
        # Risk tolerance alignment
        risk_bonus = 0.0
        if user_context.risk_tolerance == "conservative":
            if insights.study_type.value == "case_study":
                risk_bonus = 0.1
            elif reputation_score > 0.7:
                risk_bonus = 0.05
        elif user_context.risk_tolerance == "aggressive":
            if insights.implementation_complexity.value == "low":
                risk_bonus = 0.1
        
        # Industry validation bonus
        validation_bonus = 0.1 if insights.industry_validation else 0.0
        
        # Calculate final score using configured weights
        weights = Config.RANKING_WEIGHTS
        final_score = (
            similarity_score * weights['similarity'] +
            reputation_score * weights['reputation'] +
            recency_score * weights['recency'] +
            findings_score * weights['findings_richness'] +
            technique_bonus + risk_bonus + validation_bonus
        )
        
        return min(1.0, final_score)
    
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
            if insights.get_reputation_score() < Config.CONSERVATIVE_REPUTATION_THRESHOLD:
                return False
        
        # Budget constraint via complexity
        if user_context.budget_constraint == "low":
            if insights.implementation_complexity.value in ["high", "very_high"]:
                return False
        
        return True
    
    def load_insights(self, paper_id: str) -> Optional[PaperInsights]:
        """
        Load insights for a specific paper from Supabase.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            PaperInsights object or None if not found
        """
        try:
            result = self.supabase.table(Config.TABLE_INSIGHTS).select('*').eq(
                'paper_id', paper_id
            ).execute()
            
            if not result.data:
                return None
            
            insights_data = result.data[0]
            
            # Use full_insights if available
            if 'full_insights' in insights_data and insights_data['full_insights']:
                return PaperInsights.from_supabase_dict(insights_data['full_insights'])
            else:
                # Fallback: reconstruct from individual fields
                return self._reconstruct_insights_from_result(insights_data)
                
        except Exception as e:
            logger.error(f"Failed to load insights for {paper_id}: {e}")
            return None
    
    def load_paper(self, paper_id: str) -> Optional[Dict]:
        """
        Load raw paper data from Supabase.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            Paper data dictionary or None if not found
        """
        try:
            result = self.supabase.table(Config.TABLE_PAPERS).select('*').eq(
                'paper_id', paper_id
            ).execute()
            
            if not result.data:
                return None
            
            paper_data = result.data[0]
            
            # Use metadata field if available
            if 'metadata' in paper_data and paper_data['metadata']:
                return paper_data['metadata']
            else:
                # Reconstruct from individual fields
                return {
                    'id': paper_data.get('paper_id'),
                    'title': paper_data.get('title'),
                    'authors': paper_data.get('authors', []),
                    'summary': paper_data.get('summary', ''),
                    'published': paper_data.get('published_date', ''),
                    'categories': paper_data.get('arxiv_categories', []),
                    'pdf_url': paper_data.get('pdf_url', ''),
                    'comments': paper_data.get('comments', '')
                }
                
        except Exception as e:
            logger.error(f"Failed to load paper {paper_id}: {e}")
            return None
    
    def get_statistics(self) -> Dict:
        """
        Get comprehensive storage statistics from Supabase.
        
        Returns:
            Dictionary with storage statistics
        """
        try:
            stats = {
                'total_papers': 0,
                'total_insights': 0,
                'papers_with_code': 0,
                'complexity_distribution': {},
                'study_type_distribution': {},
                'average_reputation_score': 0.0,
                'average_key_findings_count': 0.0,
                'recent_papers_count': 0,
                'case_studies_count': 0,
                'top_techniques': {},
                'average_extraction_confidence': 0.0
            }
            
            # Get paper count
            papers_result = self.supabase.table(Config.TABLE_PAPERS).select(
                'id', count='exact'
            ).execute()
            stats['total_papers'] = papers_result.count or 0
            
            # Get insights with aggregated data
            insights_result = self.supabase.table(Config.TABLE_INSIGHTS).select(
                'study_type, implementation_complexity, reputation_score, '
                'key_findings_count, has_code, extraction_confidence, techniques_used'
            ).execute()
            
            insights_data = insights_result.data or []
            stats['total_insights'] = len(insights_data)
            
            if insights_data:
                # Calculate aggregated statistics
                total_reputation = 0.0
                total_findings = 0
                total_confidence = 0.0
                complexity_counts = {}
                study_type_counts = {}
                technique_counts = {}
                
                for insight in insights_data:
                    # Reputation and findings
                    reputation = insight.get('reputation_score', 0) or 0
                    total_reputation += reputation
                    
                    findings_count = insight.get('key_findings_count', 0) or 0
                    total_findings += findings_count
                    
                    confidence = insight.get('extraction_confidence', 0) or 0
                    total_confidence += confidence
                    
                    # Code availability
                    if insight.get('has_code'):
                        stats['papers_with_code'] += 1
                    
                    # Case studies
                    if insight.get('study_type') == 'case_study':
                        stats['case_studies_count'] += 1
                    
                    # Distributions
                    complexity = insight.get('implementation_complexity', 'unknown')
                    complexity_counts[complexity] = complexity_counts.get(complexity, 0) + 1
                    
                    study_type = insight.get('study_type', 'unknown')
                    study_type_counts[study_type] = study_type_counts.get(study_type, 0) + 1
                    
                    # Technique popularity
                    techniques = insight.get('techniques_used', []) or []
                    for technique in techniques:
                        technique_counts[technique] = technique_counts.get(technique, 0) + 1
                
                # Calculate averages
                if stats['total_insights'] > 0:
                    stats['average_reputation_score'] = round(
                        total_reputation / stats['total_insights'], 2
                    )
                    stats['average_key_findings_count'] = round(
                        total_findings / stats['total_insights'], 1
                    )
                    stats['average_extraction_confidence'] = round(
                        total_confidence / stats['total_insights'], 2
                    )
                
                # Set distributions
                stats['complexity_distribution'] = complexity_counts
                stats['study_type_distribution'] = study_type_counts
                
                # Top 10 techniques
                sorted_techniques = sorted(
                    technique_counts.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )
                stats['top_techniques'] = dict(sorted_techniques[:10])
            
            # Get recent papers count (last 2 years)
            current_year = datetime.now().year
            recent_cutoff = f"{current_year - 2}-01-01"
            
            recent_result = self.supabase.table(Config.TABLE_PAPERS).select(
                'id', count='exact'
            ).gte('published_date', recent_cutoff).execute()
            
            stats['recent_papers_count'] = recent_result.count or 0
            
            logger.info(f"Retrieved statistics: {stats['total_papers']} papers, {stats['total_insights']} insights")
            return stats
            
        except Exception as e:
            logger.error(f"Failed to get statistics: {e}")
            return {
                'error': str(e),
                'total_papers': 0,
                'total_insights': 0
            }
    
    def get_processing_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent processing history from Supabase.
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of processing history records
        """
        try:
            result = self.supabase.table(Config.TABLE_PROCESSING_LOGS).select(
                '*'
            ).order('created_at', desc=True).limit(limit).execute()
            
            return result.data or []
            
        except Exception as e:
            logger.error(f"Failed to get processing history: {e}")
            return []
    
    def store_processing_log(self, log_data: Dict) -> bool:
        """
        Store processing log entry in Supabase.
        
        Args:
            log_data: Processing log data
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Ensure required fields
            log_entry = {
                'id': str(uuid.uuid4()),
                'batch_name': log_data.get('batch_name', 'unknown'),
                'papers_processed': log_data.get('papers_processed', 0),
                'successful_extractions': log_data.get('successful_extractions', 0),
                'failed_extractions': log_data.get('failed_extractions', 0),
                'total_cost_usd': log_data.get('total_cost_usd', 0.0),
                'processing_time_seconds': log_data.get('processing_time_seconds', 0.0),
                'success_rate': log_data.get('success_rate', 0.0),
                'date_range': log_data.get('date_range', {}),
                'created_at': datetime.utcnow().isoformat()
            }
            
            self.supabase.table(Config.TABLE_PROCESSING_LOGS).insert(log_entry).execute()
            logger.info(f"Stored processing log: {log_entry['batch_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to store processing log: {e}")
            return False
    
    def clear_all(self):
        """
        Clear all stored data from Supabase.
        WARNING: This is destructive and will delete all data.
        """
        try:
            logger.warning("🚨 Clearing ALL data from Supabase - this is destructive!")
            
            # Delete in reverse dependency order
            tables_to_clear = [
                Config.TABLE_EXTRACTION_METADATA,
                Config.TABLE_PROCESSING_LOGS,
                Config.TABLE_INSIGHTS,
                Config.TABLE_PAPERS
            ]
            
            for table in tables_to_clear:
                try:
                    # Delete all records (be very careful with this!)
                    result = self.supabase.table(table).delete().neq('id', '').execute()
                    logger.info(f"Cleared table: {table}")
                except Exception as e:
                    logger.error(f"Failed to clear table {table}: {e}")
            
            logger.info("✅ All data cleared from Supabase")
            
        except Exception as e:
            logger.error(f"Failed to clear all data: {e}")
            raise
    
    def health_check(self) -> Dict:
        """
        Perform health check on Supabase connection and tables.
        
        Returns:
            Health status dictionary
        """
        health_status = {
            'healthy': True,
            'timestamp': datetime.utcnow().isoformat(),
            'checks': {}
        }
        
        # Test basic connectivity
        try:
            result = self.supabase.table(Config.TABLE_PAPERS).select('id').limit(1).execute()
            health_status['checks']['connectivity'] = {'status': 'ok', 'message': 'Connected to Supabase'}
        except Exception as e:
            health_status['healthy'] = False
            health_status['checks']['connectivity'] = {'status': 'error', 'message': str(e)}
        
        # Test vector search function
        try:
            dummy_embedding = [0.1] * 384  # 384-dimensional dummy vector
            result = self.supabase.rpc(Config.FUNCTION_MATCH_INSIGHTS, {
                'query_embedding': dummy_embedding,
                'match_threshold': 0.5,
                'match_count': 1
            }).execute()
            health_status['checks']['vector_search'] = {'status': 'ok', 'message': 'Vector search function working'}
        except Exception as e:
            health_status['healthy'] = False
            health_status['checks']['vector_search'] = {'status': 'error', 'message': str(e)}
        
        # Test embedder
        try:
            test_embedding = self.embedder.encode(['test text'])
            health_status['checks']['embedder'] = {'status': 'ok', 'message': f'Embedder working, dimension: {len(test_embedding[0])}'}
        except Exception as e:
            health_status['healthy'] = False
            health_status['checks']['embedder'] = {'status': 'error', 'message': str(e)}
        
        return health_status