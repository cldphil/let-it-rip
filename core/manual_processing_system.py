"""
Manual processing system with date range selection capabilities.
Allows users to control when and what papers to process.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple
import logging

# Updated imports to use available modules
from core import SyncBatchProcessor, InsightStorage
from config import Config

logger = logging.getLogger(__name__)

# Enhanced ArxivGenAIFetcher with date range support
class DateRangeArxivFetcher:
    """Extended arXiv fetcher with date range capabilities."""
    
    def __init__(self):
        """Initialize with arXiv API base URL."""
        self.base_url = "http://export.arxiv.org/api/query"
        self.current_year = datetime.now().year
    
    def fetch_papers_date_range(self, start_date: datetime, end_date: datetime,
                               max_results: int = 100, include_full_text: bool = True) -> List[Dict]:
        """
        Fetch papers within a specific date range.
        
        Args:
            start_date: Start date for papers
            end_date: End date for papers
            max_results: Maximum papers to fetch
            include_full_text: Whether to extract full text
            
        Returns:
            List of paper dictionaries
        """
        # Format dates for arXiv API
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        # Build date-specific query
        query = self._build_date_range_query(start_str, end_str, max_results)
        url = self.base_url + query
        
        logger.info(f"Fetching papers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        return self._fetch_and_process(url, include_full_text)
    
    def _build_date_range_query(self, start_date: str, end_date: str, max_results: int) -> str:
        """Build arXiv query for specific date range."""
        # Search terms for generative AI research
        search_terms = [
            "generative artificial intelligence",
            "generative AI", 
            "large language model",
            "LLM",
            "GPT",
            "diffusion model",
            "generative model",
            "text generation",
            "image generation"
        ]
        
        # Build OR query for search terms
        query_parts = []
        for term in search_terms:
            query_parts.append(f'all:"{term}"')
        
        search_query = " OR ".join(query_parts)
        
        # Add date filter
        date_filter = f"submittedDate:[{start_date}* TO {end_date}*]"
        
        full_query = f"({search_query}) AND {date_filter}"
        
        return f"?search_query={self._url_encode(full_query)}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    
    def _url_encode(self, text: str) -> str:
        """URL encode the query string."""
        import urllib.parse
        return urllib.parse.quote(text)
    
    def _fetch_and_process(self, url: str, include_full_text: bool) -> List[Dict]:
        """Fetch and process papers from arXiv API."""
        try:
            import requests
            import xml.etree.ElementTree as ET
            
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
            
            papers = []
            for i, entry in enumerate(entries, 1):
                logger.info(f"Processing paper {i}/{len(entries)}...")
                paper = self._parse_arxiv_entry(entry)
                
                # Extract full text if requested and PDF URL is available
                if include_full_text and paper.get('pdf_url'):
                    paper['full_text'] = self._extract_pdf_text(paper['pdf_url'])
                    paper['text_length'] = len(paper.get('full_text', ''))
                
                papers.append(paper)
                
                # Rate limiting
                if include_full_text and i < len(entries):
                    import time
                    time.sleep(1)
            
            logger.info(f"Successfully processed {len(papers)} papers")
            return papers
            
        except Exception as e:
            logger.error(f"Error fetching papers: {e}")
            return []
    
    def _parse_arxiv_entry(self, entry) -> Dict:
        """Parse a single arXiv entry from XML response."""
        # Import the original fetcher to reuse its parsing logic
        from services.arxiv_fetcher import ArxivGenAIFetcher
        original_fetcher = ArxivGenAIFetcher()
        return original_fetcher.parse_arxiv_entry(entry)
    
    def _extract_pdf_text(self, pdf_url: str) -> str:
        """Extract text from PDF URL."""
        # Import the original fetcher to reuse its PDF extraction logic
        from services.arxiv_fetcher import ArxivGenAIFetcher
        original_fetcher = ArxivGenAIFetcher()
        return original_fetcher.extract_pdf_text(pdf_url)

class ManualProcessingController:
    """
    Controls manual processing with date range selection and progress tracking.
    """
    
    def __init__(self):
        """Initialize manual processing controller."""
        # Use the factory function from core
        self.storage = InsightStorage()
        
        # Use SyncBatchProcessor with cloud-optimized settings
        if Config.USE_CLOUD_STORAGE:
            batch_size = Config.FREE_TIER_BATCH_SIZE
        else:
            batch_size = Config.BATCH_SIZE
            
        self.processor = SyncBatchProcessor(storage=self.storage, batch_size=batch_size)
        
        # Initialize fetcher with date range support
        self.fetcher = DateRangeArxivFetcher()
        
        # Processing statistics
        self.last_processing_stats = {}
        
        logger.info("Initialized manual processing controller")
    
    def get_available_date_ranges(self) -> Dict:
        """
        Get suggested date ranges for processing.
        
        Returns:
            Dict with suggested date ranges and their paper counts (estimated)
        """
        today = datetime.now()
        
        suggested_ranges = {
            'last_24_hours': {
                'start_date': today - timedelta(days=1),
                'end_date': today,
                'description': 'Last 24 hours',
                'estimated_papers': '5-15'
            },
            'last_week': {
                'start_date': today - timedelta(days=7),
                'end_date': today,
                'description': 'Last 7 days',
                'estimated_papers': '20-50'
            },
            'last_month': {
                'start_date': today - timedelta(days=30),
                'end_date': today,
                'description': 'Last 30 days',
                'estimated_papers': '100-200'
            },
            'this_year': {
                'start_date': datetime(today.year, 1, 1),
                'end_date': today,
                'description': f'Year {today.year}',
                'estimated_papers': '1000+'
            }
        }
        
        return suggested_ranges
    
    def validate_date_range(self, start_date: datetime, end_date: datetime) -> Tuple[bool, str]:
        """
        Validate a date range for processing.
        
        Args:
            start_date: Start date for processing
            end_date: End date for processing
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        if start_date >= end_date:
            return False, "Start date must be before end date"
        
        if end_date > datetime.now():
            return False, "End date cannot be in the future"
        
        # Check maximum range
        days_diff = (end_date - start_date).days
        if days_diff > Config.MAX_PROCESSING_DAYS:
            return False, f"Date range too large. Maximum {Config.MAX_PROCESSING_DAYS} days allowed"
        
        # Check minimum date (arXiv started in 1991)
        min_date = datetime(1991, 1, 1)
        if start_date < min_date:
            return False, "Start date cannot be before 1991"
        
        return True, ""
    
    def estimate_processing_cost(self, start_date: datetime, end_date: datetime, 
                                max_papers: Optional[int] = None) -> Dict:
        """
        Estimate processing cost and time for a date range.
        
        Args:
            start_date: Start date
            end_date: End date
            max_papers: Maximum papers to process (None for no limit)
            
        Returns:
            Dict with cost and time estimates
        """
        days_diff = (end_date - start_date).days
        
        # Rough estimates based on typical arXiv GenAI paper volumes
        estimated_papers_per_day = {
            'recent': 15,  # Last 30 days
            'moderate': 10,  # 30-365 days ago
            'older': 5     # More than 1 year ago
        }
        
        # Determine paper density based on recency
        days_ago = (datetime.now() - end_date).days
        if days_ago <= 30:
            papers_per_day = estimated_papers_per_day['recent']
        elif days_ago <= 365:
            papers_per_day = estimated_papers_per_day['moderate']
        else:
            papers_per_day = estimated_papers_per_day['older']
        
        estimated_total_papers = days_diff * papers_per_day
        
        # Apply max_papers limit if specified
        if max_papers:
            estimated_total_papers = min(estimated_total_papers, max_papers)
        
        # Cost estimates
        cost_per_paper = 0.005  # $0.005 per paper (API costs)
        estimated_cost = estimated_total_papers * cost_per_paper
        
        # Time estimates (based on 2 papers per minute processing)
        estimated_time_minutes = estimated_total_papers * 0.5
        
        # Reputation filtering impact
        min_reputation = Config.MINIMUM_REPUTATION_SCORE
        if min_reputation > 0:
            # Estimate filtering impact (rough approximation)
            if min_reputation >= 0.5:
                filtering_reduction = 0.7  # 70% filtered out
            elif min_reputation >= 0.3:
                filtering_reduction = 0.5  # 50% filtered out
            elif min_reputation >= 0.1:
                filtering_reduction = 0.3  # 30% filtered out
            else:
                filtering_reduction = 0.1  # 10% filtered out
            
            estimated_total_papers = int(estimated_total_papers * (1 - filtering_reduction))
            estimated_cost *= (1 - filtering_reduction)
            estimated_time_minutes *= (1 - filtering_reduction)
        
        return {
            'estimated_papers': estimated_total_papers,
            'estimated_cost_usd': round(estimated_cost, 2),
            'estimated_time_minutes': round(estimated_time_minutes, 1),
            'estimated_time_hours': round(estimated_time_minutes / 60, 1),
            'days_in_range': days_diff,
            'papers_per_day_avg': round(estimated_total_papers / max(days_diff, 1), 1),
            'reputation_filter_active': min_reputation > 0,
            'min_reputation_score': min_reputation
        }
    
    def check_existing_papers(self, start_date: datetime, end_date: datetime) -> Dict:
        """
        Check how many papers from this date range are already processed.
        
        Args:
            start_date: Start date
            end_date: End date
            
        Returns:
            Dict with existing paper counts and recommendations
        """
        try:
            # Get papers in date range from storage
            if hasattr(self.storage, 'supabase'):
                # Cloud storage query
                result = self.storage.supabase.table('papers').select('id, published_date').gte(
                    'published_date', start_date.strftime('%Y-%m-%d')
                ).lte(
                    'published_date', end_date.strftime('%Y-%m-%d')
                ).execute()
                
                existing_papers = len(result.data)
            else:
                # Local storage fallback
                existing_papers = 0  # Would need to implement local date range query
            
            return {
                'existing_papers': existing_papers,
                'recommendation': 'skip_duplicates' if existing_papers > 0 else 'proceed',
                'message': f"Found {existing_papers} papers already processed in this date range"
            }
            
        except Exception as e:
            logger.warning(f"Could not check existing papers: {e}")
            return {
                'existing_papers': 0,
                'recommendation': 'proceed',
                'message': "Could not check for existing papers"
            }
    
    def process_date_range(self, start_date: datetime, end_date: datetime, 
                          max_papers: Optional[int] = None,
                          skip_existing: bool = True,
                          progress_callback=None) -> Dict:
        """
        Process papers from a specific date range.
        
        Args:
            start_date: Start date for papers
            end_date: End date for papers
            max_papers: Maximum papers to process
            skip_existing: Whether to skip already processed papers
            progress_callback: Function to call with progress updates
            
        Returns:
            Dict with processing results and statistics
        """
        # Validate date range
        is_valid, error_msg = self.validate_date_range(start_date, end_date)
        if not is_valid:
            return {'error': error_msg, 'success': False}
        
        if progress_callback:
            progress_callback("Validating date range...", 0)
        
        # Get cost estimate
        estimate = self.estimate_processing_cost(start_date, end_date, max_papers)
        
        if progress_callback:
            progress_callback(f"Fetching papers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...", 10)
        
        try:
            # Fetch papers with date range
            papers = self.fetcher.fetch_papers_date_range(
                start_date=start_date,
                end_date=end_date,
                max_results=max_papers or 1000,  # Default max
                include_full_text=True
            )
            
            if not papers:
                return {
                    'error': 'No papers found in the specified date range',
                    'success': False,
                    'estimate': estimate
                }
            
            if progress_callback:
                progress_callback(f"Found {len(papers)} papers. Starting processing...", 20)
            
            # Filter existing papers if requested
            if skip_existing:
                papers = self._filter_existing_papers(papers)
                if progress_callback:
                    progress_callback(f"Processing {len(papers)} new papers...", 30)
            
            # Process papers
            checkpoint_name = f"manual_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            
            # Create progress wrapper for batch processor
            def batch_progress(current, total):
                if progress_callback:
                    progress = 30 + (current / total) * 60  # 30-90% range
                    progress_callback(f"Processing paper {current}/{total}...", progress)
            
            stats = self.processor.process_papers(
                papers,
                checkpoint_name=checkpoint_name,
                progress_callback=batch_progress
            )
            
            if progress_callback:
                progress_callback("Processing complete!", 100)
            
            # Enhanced results
            results = {
                'success': True,
                'papers_found': len(papers),
                'papers_processed': stats.get('successful', 0),
                'papers_failed': stats.get('failed', 0),
                'processing_time_seconds': stats.get('total_time', 0),
                'processing_cost_usd': stats.get('total_cost', 0),
                'date_range': {
                    'start': start_date.strftime('%Y-%m-%d'),
                    'end': end_date.strftime('%Y-%m-%d'),
                    'days': (end_date - start_date).days
                },
                'estimate_vs_actual': {
                    'estimated_papers': estimate['estimated_papers'],
                    'actual_papers': len(papers),
                    'estimated_cost': estimate['estimated_cost_usd'],
                    'actual_cost': stats.get('total_cost', 0)
                },
                'reputation_filtering': {
                    'active': Config.MINIMUM_REPUTATION_SCORE > 0,
                    'threshold': Config.MINIMUM_REPUTATION_SCORE,
                    'papers_stored': stats.get('successful', 0)
                }
            }
            
            # Store results for future reference
            self.last_processing_stats = results
            
            return results
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(error_msg)
            
            if progress_callback:
                progress_callback(f"Error: {error_msg}", 0)
            
            return {
                'error': error_msg,
                'success': False,
                'estimate': estimate
            }
    
    def _filter_existing_papers(self, papers: List[Dict]) -> List[Dict]:
        """Filter out papers that are already in storage."""
        new_papers = []
        
        for paper in papers:
            paper_id = paper.get('id', '').split('/')[-1]
            if not paper_id:
                continue
            
            # Check if paper already exists
            existing_paper = self.storage.load_paper(paper_id)
            if not existing_paper:
                new_papers.append(paper)
        
        logger.info(f"Filtered {len(papers) - len(new_papers)} existing papers")
        return new_papers
    
    def get_processing_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent processing history.
        
        Args:
            limit: Maximum number of recent batches to return
            
        Returns:
            List of processing history records
        """
        try:
            if hasattr(self.storage, 'supabase'):
                result = self.storage.supabase.table('processing_logs').select(
                    '*'
                ).order('created_at', desc=True).limit(limit).execute()
                
                return result.data
            else:
                # Local storage fallback
                return []
                
        except Exception as e:
            logger.error(f"Failed to get processing history: {e}")
            return []
    
    def get_storage_usage(self) -> Dict:
        """Get current storage usage statistics."""
        return self.storage.get_statistics()