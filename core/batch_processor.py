"""
Cloud-only batch processing system for handling large paper volumes.
Uses in-memory state tracking and Supabase for persistence.
"""

import asyncio
from datetime import datetime
from typing import List, Dict, Optional, Set
import logging
import traceback

from .insight_extractor import InsightExtractor
from .insight_storage import InsightStorage

logger = logging.getLogger(__name__)


class BatchProcessor:
    """
    Cloud-only batch processor for papers with in-memory state tracking.
    
    Features:
    - Parallel processing within batches
    - Automatic retry on failures
    - Progress tracking and cost estimation
    - Cloud-native design without local checkpointing
    """
    
    def __init__(self, 
                 extractor: Optional[InsightExtractor] = None,
                 storage: Optional[InsightStorage] = None,
                 batch_size: int = 10,
                 max_workers: int = 3):
        """
        Initialize batch processor.
        
        Args:
            extractor: Insight extractor (will create if not provided)
            storage: Storage system (will create if not provided)
            batch_size: Number of papers per batch
            max_workers: Maximum concurrent extractions
        """
        self.extractor = extractor or InsightExtractor()
        self.storage = storage or InsightStorage()
        self.batch_size = batch_size
        self.max_workers = max_workers
        
        # In-memory processing state
        self.current_session = {
            'session_id': f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'start_time': None,
            'processed_papers': set(),
            'failed_papers': set(),
            'current_batch': 0,
            'total_batches': 0
        }
        
        # Processing statistics
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_time': 0.0,
            'total_cost': 0.0,
            'errors': []
        }
        
        logger.info(f"Initialized cloud-only batch processor (batch_size={batch_size})")
    
    async def process_papers(self, 
                           papers: List[Dict], 
                           session_name: Optional[str] = None,
                           force_reprocess: bool = False,
                           progress_callback=None) -> Dict:
        """
        Process a list of papers with progress tracking.
        
        Args:
            papers: List of paper dictionaries
            session_name: Name for this processing session
            force_reprocess: Whether to reprocess already processed papers
            progress_callback: Function for progress updates
            
        Returns:
            Processing statistics
        """
        start_time = datetime.utcnow()
        self.current_session['start_time'] = start_time
        
        if session_name:
            self.current_session['session_id'] = session_name
        
        # Filter papers to process
        papers_to_process = []
        for paper in papers:
            paper_id = paper.get('id', '').split('/')[-1]
            if not paper_id:
                logger.warning(f"Skipping paper without ID: {paper.get('title', 'Unknown')}")
                self.stats['skipped'] += 1
                continue
            
            # Check if already processed (unless force reprocess)
            if not force_reprocess and self._is_already_processed(paper_id):
                logger.info(f"Skipping already processed paper: {paper_id}")
                self.stats['skipped'] += 1
                continue
            
            papers_to_process.append(paper)
        
        if not papers_to_process:
            logger.warning("No papers to process after filtering")
            return self.stats
        
        # Calculate batch information
        self.current_session['total_batches'] = (len(papers_to_process) + self.batch_size - 1) // self.batch_size
        
        logger.info(f"Processing {len(papers_to_process)} papers in {self.current_session['total_batches']} batches")
        
        # Process in batches
        for i in range(0, len(papers_to_process), self.batch_size):
            batch = papers_to_process[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            self.current_session['current_batch'] = batch_num
            
            logger.info(f"Processing batch {batch_num}/{self.current_session['total_batches']}")
            
            # Update progress
            if progress_callback:
                progress = (batch_num - 1) / self.current_session['total_batches'] * 100
                progress_callback(
                    f"Processing batch {batch_num}/{self.current_session['total_batches']}", 
                    progress,
                    current_batch=batch_num,
                    total_batches=self.current_session['total_batches']
                )
            
            # Process batch
            batch_results = await self._process_batch(batch, progress_callback)
            
            # Update statistics
            for paper_id, success in batch_results.items():
                if success:
                    self.current_session['processed_papers'].add(paper_id)
                    self.stats['successful'] += 1
                else:
                    self.current_session['failed_papers'].add(paper_id)
                    self.stats['failed'] += 1
            
            # Rate limiting between batches
            if i + self.batch_size < len(papers_to_process):
                await asyncio.sleep(2)
        
        # Calculate final statistics
        self.stats['total_processed'] = len(papers_to_process)
        self.stats['total_time'] = (datetime.utcnow() - start_time).total_seconds()
        
        # Store processing log in cloud storage
        await self._store_processing_log()
        
        logger.info(f"Processing complete: {self.stats}")
        
        # Final progress update
        if progress_callback:
            progress_callback(
                f"Processing complete! Processed {self.stats['successful']} papers successfully",
                100,
                phase='completed'
            )
        
        return self.stats
    
    async def _process_batch(self, batch: List[Dict], progress_callback=None) -> Dict[str, bool]:
        """
        Process a single batch of papers concurrently.
        
        Returns:
            Dict mapping paper_id to success status
        """
        results = {}
        
        # Create tasks for concurrent processing
        tasks = []
        for paper in batch:
            task = asyncio.create_task(self._process_single_paper(paper))
            paper_id = paper.get('id', '').split('/')[-1]
            tasks.append((paper_id, task))
        
        # Wait for all tasks with timeout
        for i, (paper_id, task) in enumerate(tasks):
            try:
                success = await asyncio.wait_for(task, timeout=300)  # 5 minute timeout per paper
                results[paper_id] = success
                
                # Update progress within batch
                if progress_callback:
                    paper_title = next((p.get('title', '') for p in batch if p.get('id', '').split('/')[-1] == paper_id), '')
                    progress_callback(
                        f"Completed paper {i+1}/{len(batch)}: {paper_title[:50]}...",
                        current_paper_in_batch=i+1,
                        total_papers_in_batch=len(batch)
                    )
                    
            except asyncio.TimeoutError:
                logger.error(f"Timeout processing paper {paper_id}")
                results[paper_id] = False
                self.stats['errors'].append({
                    'paper_id': paper_id,
                    'error': 'Processing timeout (5 minutes)',
                    'timestamp': datetime.utcnow().isoformat()
                })
            except Exception as e:
                logger.error(f"Error processing paper {paper_id}: {str(e)}")
                results[paper_id] = False
                self.stats['errors'].append({
                    'paper_id': paper_id,
                    'error': str(e),
                    'timestamp': datetime.utcnow().isoformat()
                })
        
        return results
    
    async def _process_single_paper(self, paper: Dict) -> bool:
        """
        Process a single paper with retry logic.
        
        Returns:
            True if successful, False otherwise
        """
        paper_id = paper.get('id', '').split('/')[-1]
        if not paper_id:
            paper_id = f"unknown_{datetime.utcnow().timestamp()}"
        
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                # Store raw paper data
                stored_id = self.storage.store_paper(paper)
                
                # Extract insights synchronously (InsightExtractor is not async)
                # Run in executor to avoid blocking
                loop = asyncio.get_event_loop()
                insights, metadata = await loop.run_in_executor(
                    None, 
                    self.extractor.extract_insights, 
                    paper
                )
                
                # Store insights
                self.storage.store_insights(stored_id, insights, metadata)
                
                # Update cost tracking
                if metadata.estimated_cost_usd:
                    self.stats['total_cost'] += metadata.estimated_cost_usd
                
                logger.info(f"Successfully processed paper {paper_id}")
                return True
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"Failed to process paper {paper_id} after {max_retries} attempts: {str(e)}")
                    logger.error(f"Traceback: {traceback.format_exc()}")
                    return False
                else:
                    logger.warning(f"Attempt {attempt + 1} failed for paper {paper_id}, retrying...")
                    await asyncio.sleep(2 ** attempt)  # Exponential backoff
        
        return False
    
    def _is_already_processed(self, paper_id: str) -> bool:
        """
        Check if a paper has already been processed by querying cloud storage.
        
        Args:
            paper_id: Paper identifier
            
        Returns:
            True if already processed, False otherwise
        """
        try:
            # Check if insights exist for this paper
            insights = self.storage.load_insights(paper_id)
            return insights is not None
        except Exception as e:
            logger.warning(f"Error checking if paper {paper_id} is processed: {e}")
            return False
    
    async def _store_processing_log(self):
        """Store processing log in cloud storage."""
        try:
            log_data = {
                'batch_name': self.current_session['session_id'],
                'papers_processed': self.stats['total_processed'],
                'successful_extractions': self.stats['successful'],
                'failed_extractions': self.stats['failed'],
                'total_cost_usd': self.stats['total_cost'],
                'processing_time_seconds': self.stats['total_time'],
                'success_rate': (
                    self.stats['successful'] / max(1, self.stats['total_processed'])
                ),
                'date_range': {
                    'start': self.current_session['start_time'].isoformat(),
                    'end': datetime.utcnow().isoformat()
                },
                'error_count': len(self.stats['errors'])
            }
            
            self.storage.store_processing_log(log_data)
            logger.info(f"Stored processing log for session: {self.current_session['session_id']}")
            
        except Exception as e:
            logger.warning(f"Failed to store processing log: {e}")
    
    def get_current_session_status(self) -> Dict:
        """Get current session status for monitoring."""
        return {
            'session_id': self.current_session['session_id'],
            'current_batch': self.current_session['current_batch'],
            'total_batches': self.current_session['total_batches'],
            'papers_processed': len(self.current_session['processed_papers']),
            'papers_failed': len(self.current_session['failed_papers']),
            'current_stats': self.stats.copy(),
            'elapsed_time': (
                (datetime.utcnow() - self.current_session['start_time']).total_seconds()
                if self.current_session['start_time'] else 0
            )
        }
    
    def reset_session(self):
        """Reset current session state."""
        self.current_session = {
            'session_id': f"batch_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
            'start_time': None,
            'processed_papers': set(),
            'failed_papers': set(),
            'current_batch': 0,
            'total_batches': 0
        }
        
        self.stats = {
            'total_processed': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_time': 0.0,
            'total_cost': 0.0,
            'errors': []
        }
        
        logger.info("Reset batch processor session")


class SyncBatchProcessor:
    """Synchronous wrapper for BatchProcessor."""
    
    def __init__(self, *args, **kwargs):
        """Initialize with async processor."""
        self.async_processor = BatchProcessor(*args, **kwargs)
    
    def process_papers(self, papers: List[Dict], **kwargs) -> Dict:
        """Synchronous wrapper for process_papers."""
        # Create new event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.async_processor.process_papers(papers, **kwargs)
            )
        finally:
            loop.close()
    
    def get_current_session_status(self) -> Dict:
        """Get current session status."""
        return self.async_processor.get_current_session_status()
    
    def reset_session(self):
        """Reset current session."""
        self.async_processor.reset_session()
    
    @property
    def stats(self) -> Dict:
        """Get processing statistics."""
        return self.async_processor.stats
    
    @property
    def extractor(self):
        """Get the insight extractor instance."""
        return self.async_processor.extractor
    
    @property
    def storage(self):
        """Get the storage instance."""
        return self.async_processor.storage