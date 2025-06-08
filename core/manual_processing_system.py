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

from services.semantic_scholar_hidx import SemanticScholarAPI
from services.arxiv_fetcher import ArxivFetcher

logger = logging.getLogger(__name__)

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
        self.fetcher = ArxivFetcher()
        
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
                
                # Ensure all numeric fields are properly typed
                history_data = result.data
                for record in history_data:
                    # Convert cost fields to float if they exist
                    if 'total_cost' in record:
                        try:
                            # Remove $ sign if present and convert to float
                            cost_str = str(record['total_cost']).replace('$', '').strip()
                            record['total_cost'] = float(cost_str) if cost_str else 0.0
                        except (ValueError, TypeError):
                            record['total_cost'] = 0.0
                    
                    if 'processing_cost_usd' in record:
                        try:
                            cost_str = str(record['processing_cost_usd']).replace('$', '').strip()
                            record['processing_cost_usd'] = float(cost_str) if cost_str else 0.0
                        except (ValueError, TypeError):
                            record['processing_cost_usd'] = 0.0
                    
                    # Ensure other numeric fields are properly typed
                    if 'papers_processed' in record:
                        try:
                            record['papers_processed'] = int(record['papers_processed'])
                        except (ValueError, TypeError):
                            record['papers_processed'] = 0
                    
                    if 'success_rate' in record:
                        try:
                            record['success_rate'] = float(record['success_rate'])
                        except (ValueError, TypeError):
                            record['success_rate'] = 0.0
                
                return history_data
            else:
                # Local storage fallback
                return []
                
        except Exception as e:
            logger.error(f"Failed to get processing history: {e}")
            return []
        
    def process_date_range_enhanced(self, start_date: datetime, end_date: datetime, 
                               max_papers: Optional[int] = None,
                               skip_existing: bool = True,
                               progress_callback=None) -> Dict:
        """
        Enhanced process papers from a specific date range with detailed progress tracking.
        
        Args:
            start_date: Start date for papers
            end_date: End date for papers
            max_papers: Maximum papers to process
            skip_existing: Whether to skip already processed papers
            progress_callback: Function to call with detailed progress updates
            
        Returns:
            Dict with processing results and statistics
        """
        # Validate date range
        is_valid, error_msg = self.validate_date_range(start_date, end_date)
        if not is_valid:
            return {'error': error_msg, 'success': False}
        
        if progress_callback:
            progress_callback("Validating date range and preparing...", 0, phase='initializing')
        
        # Get cost estimate
        estimate = self.estimate_processing_cost(start_date, end_date, max_papers)
        
        if progress_callback:
            progress_callback(
                f"Fetching papers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}...", 
                5, 
                phase='fetching',
                progress_papers_found=0
            )
        
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
                progress_callback(
                    f"Found {len(papers)} papers. Preparing for processing...", 
                    10, 
                    phase='processing',
                    progress_papers_found=len(papers)
                )
            
            # Filter existing papers if requested
            original_count = len(papers)
            if skip_existing:
                papers = self._filter_existing_papers(papers)
                if progress_callback:
                    filtered_count = original_count - len(papers)
                    progress_callback(
                        f"Filtered {filtered_count} existing papers. Processing {len(papers)} new papers...", 
                        15,
                        progress_papers_found=len(papers)
                    )
            
            if not papers:
                return {
                    'error': 'All papers in the date range have already been processed',
                    'success': True,
                    'papers_found': original_count,
                    'papers_processed': 0,
                    'estimate': estimate
                }
            
            # Process papers with enhanced progress tracking
            checkpoint_name = f"manual_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}"
            
            # Enhanced processing with detailed callbacks
            stats = self._process_papers_with_detailed_progress(
                papers,
                checkpoint_name,
                progress_callback
            )
            
            if progress_callback:
                progress_callback("Processing complete! Calculating final statistics...", 95)
            
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
                },
                'detailed_stats': stats
            }
            
            # Store results for future reference
            self.last_processing_stats = results
            
            if progress_callback:
                progress_callback("Processing complete!", 100, phase='completed')
            
            return results
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            logger.error(error_msg)
            
            if progress_callback:
                progress_callback(f"Error: {error_msg}", 0, phase='error', error_details=error_msg)
            
            return {
                'error': error_msg,
                'success': False,
                'estimate': estimate
            }

    def _process_papers_with_detailed_progress(self, papers: List[Dict], 
                                            checkpoint_name: str,
                                            progress_callback=None) -> Dict:
        """
        Process papers with detailed progress tracking for enhanced UI feedback.
        
        Args:
            papers: List of paper dictionaries to process
            checkpoint_name: Name for the processing checkpoint
            progress_callback: Callback function for progress updates
            
        Returns:
            Processing statistics dictionary
        """
        from datetime import datetime
        import time
        
        start_time = datetime.utcnow()
        
        # Initialize statistics
        stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_time': 0.0,
            'total_cost': 0.0,
            'errors': []
        }
        
        total_papers = len(papers)
        processed_papers = 0
        insights_generated = 0
        
        logger.info(f"Starting enhanced processing of {total_papers} papers")
        
        # Process papers one by one for detailed progress tracking
        for i, paper in enumerate(papers, 1):
            paper_start_time = time.time()
            
            # Get paper title for progress display
            paper_title = paper.get('title', f"Paper {i}")
            
            if progress_callback:
                base_progress = 15 + (i - 1) / total_papers * 80  # 15-95% range for processing
                progress_callback(
                    f"Processing paper {i}/{total_papers}",
                    base_progress,
                    current_paper_title=paper_title,
                    progress_papers_processed=processed_papers,
                    progress_insights_generated=insights_generated
                )
            
            try:
                # Store raw paper data
                paper_id = paper.get('id', '').split('/')[-1]
                if not paper_id:
                    paper_id = f"unknown_{datetime.utcnow().timestamp()}"
                
                stored_id = self.storage.store_paper(paper)
                
                # Extract insights
                insights, metadata = self.processor.extractor.extract_insights(paper)
                
                # Store insights
                self.storage.store_insights(stored_id, insights, metadata)
                
                # Update statistics
                stats['successful'] += 1
                insights_generated += 1
                processed_papers += 1
                
                if metadata.estimated_cost_usd:
                    stats['total_cost'] += metadata.estimated_cost_usd
                
                # Progress update after successful processing
                if progress_callback:
                    current_progress = 15 + i / total_papers * 80
                    progress_callback(
                        f"Successfully processed: {paper_title[:60]}...",
                        current_progress,
                        progress_papers_processed=processed_papers,
                        progress_insights_generated=insights_generated
                    )
                
                logger.info(f"Successfully processed paper {i}/{total_papers}: {paper_title[:50]}...")
                
            except Exception as e:
                stats['failed'] += 1
                processed_papers += 1
                error_msg = f"Failed to process {paper_title[:50]}: {str(e)}"
                stats['errors'].append(error_msg)
                
                logger.error(error_msg)
                
                # Progress update after failed processing
                if progress_callback:
                    current_progress = 15 + i / total_papers * 80
                    progress_callback(
                        f"Failed to process: {paper_title[:60]}...",
                        current_progress,
                        progress_papers_processed=processed_papers,
                        progress_insights_generated=insights_generated,
                        error_details=error_msg
                    )
            
            # Small delay between papers to avoid overwhelming the system
            paper_processing_time = time.time() - paper_start_time
            if paper_processing_time < 1.0:  # Minimum 1 second per paper for UI responsiveness
                time.sleep(1.0 - paper_processing_time)
        
        # Calculate final statistics
        stats['total_processed'] = len(papers)
        stats['total_time'] = (datetime.utcnow() - start_time).total_seconds()
        
        logger.info(f"Enhanced processing complete: {stats}")
        return stats

    # Also add this helper method to provide real-time status
    def get_processing_status(self) -> Dict:
        """
        Get current processing status for real-time monitoring.
        
        Returns:
            Dict with current processing status
        """
        try:
            # Get current storage statistics
            storage_stats = self.storage.get_statistics()
            
            # Get latest processing batch info if available
            latest_batch = None
            if hasattr(self, 'last_processing_stats') and self.last_processing_stats:
                latest_batch = self.last_processing_stats
            
            status = {
                'storage_stats': storage_stats,
                'latest_batch': latest_batch,
                'timestamp': datetime.now().isoformat(),
                'system_healthy': True
            }
            
            # Check if any processing is currently running
            # This would need to be implemented based on your processing architecture
            status['processing_active'] = False
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get processing status: {e}")
            return {
                'error': str(e),
                'system_healthy': False,
                'timestamp': datetime.now().isoformat()
            }
    
    def get_storage_usage(self) -> Dict:
        """Get current storage usage statistics."""
        return self.storage.get_statistics()