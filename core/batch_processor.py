"""
Batch processing system with checkpointing for handling large paper volumes.
Supports incremental processing and failure recovery.
"""

import asyncio
import json
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Set
import logging
from concurrent.futures import ThreadPoolExecutor
import traceback

from .insight_extractor import InsightExtractor
from .insight_storage import InsightStorage
from .insight_schema import PaperInsights, ExtractionMetadata

logger = logging.getLogger(__name__)

class BatchProcessor:
    """
    Processes papers in batches with checkpointing and error recovery.
    
    Features:
    - Incremental processing with checkpoints
    - Parallel processing within batches
    - Automatic retry on failures
    - Progress tracking and cost estimation
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
        self.checkpoint_dir = Path("storage/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    async def process_papers(self, 
                           papers: List[Dict], 
                           checkpoint_name: Optional[str] = None,
                           force_reprocess: bool = False) -> Dict:
        """
        Process a list of papers with checkpointing.
        
        Args:
            papers: List of paper dictionaries
            checkpoint_name: Name for checkpoint file (auto-generated if not provided)
            force_reprocess: Whether to reprocess already processed papers
            
        Returns:
            Processing statistics
        """
        start_time = datetime.utcnow()
        
        # Setup checkpoint
        if not checkpoint_name:
            checkpoint_name = f"batch_{start_time.strftime('%Y%m%d_%H%M%S')}"
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.json"
        
        # Load checkpoint if exists
        processed_ids = self._load_checkpoint(checkpoint_file)
        
        # Filter papers to process
        papers_to_process = []
        for paper in papers:
            paper_id = paper.get('id', '').split('/')[-1]
            if not paper_id:
                logger.warning(f"Skipping paper without ID: {paper.get('title', 'Unknown')}")
                self.stats['skipped'] += 1
                continue
            
            if paper_id in processed_ids and not force_reprocess:
                logger.info(f"Skipping already processed paper: {paper_id}")
                self.stats['skipped'] += 1
                continue
            
            papers_to_process.append(paper)
        
        logger.info(f"Processing {len(papers_to_process)} papers in batches of {self.batch_size}")
        
        # Process in batches
        for i in range(0, len(papers_to_process), self.batch_size):
            batch = papers_to_process[i:i + self.batch_size]
            batch_num = i // self.batch_size + 1
            total_batches = (len(papers_to_process) + self.batch_size - 1) // self.batch_size
            
            logger.info(f"Processing batch {batch_num}/{total_batches}")
            
            # Process batch
            batch_results = await self._process_batch(batch)
            
            # Update checkpoint
            for paper_id, success in batch_results.items():
                if success:
                    processed_ids.add(paper_id)
                    self.stats['successful'] += 1
                else:
                    self.stats['failed'] += 1
            
            self._save_checkpoint(checkpoint_file, processed_ids)
            
            # Rate limiting between batches
            if i + self.batch_size < len(papers_to_process):
                await asyncio.sleep(2)
        
        # Calculate final statistics
        self.stats['total_processed'] = len(papers_to_process)
        self.stats['total_time'] = (datetime.utcnow() - start_time).total_seconds()
        
        # Save final statistics
        self._save_statistics(checkpoint_name)
        
        logger.info(f"Processing complete: {self.stats}")
        return self.stats
    
    async def _process_batch(self, batch: List[Dict]) -> Dict[str, bool]:
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
        for paper_id, task in tasks:
            try:
                success = await asyncio.wait_for(task, timeout=60)  # 60 second timeout
                results[paper_id] = success
            except asyncio.TimeoutError:
                logger.error(f"Timeout processing paper {paper_id}")
                results[paper_id] = False
                self.stats['errors'].append({
                    'paper_id': paper_id,
                    'error': 'Timeout',
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
                
                # Extract insights
                insights, metadata = await self.extractor.extract_insights(paper)
                
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
    
    def _load_checkpoint(self, checkpoint_file: Path) -> Set[str]:
        """Load processed paper IDs from checkpoint."""
        if not checkpoint_file.exists():
            return set()
        
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
                return set(data.get('processed_ids', []))
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}")
            return set()
    
    def _save_checkpoint(self, checkpoint_file: Path, processed_ids: Set[str]):
        """Save checkpoint with processed paper IDs."""
        checkpoint_data = {
            'processed_ids': list(processed_ids),
            'timestamp': datetime.utcnow().isoformat(),
            'stats': self.stats
        }
        
        try:
            with open(checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
    
    def _save_statistics(self, checkpoint_name: str):
        """Save processing statistics."""
        stats_file = self.checkpoint_dir / f"{checkpoint_name}_stats.json"
        
        stats_data = {
            'checkpoint_name': checkpoint_name,
            'timestamp': datetime.utcnow().isoformat(),
            'statistics': self.stats,
            'storage_stats': self.storage.get_statistics()
        }
        
        try:
            with open(stats_file, 'w') as f:
                json.dump(stats_data, f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save statistics: {e}")
    
    def get_checkpoint_status(self, checkpoint_name: str) -> Optional[Dict]:
        """Get status of a checkpoint."""
        checkpoint_file = self.checkpoint_dir / f"{checkpoint_name}.json"
        if not checkpoint_file.exists():
            return None
        
        try:
            with open(checkpoint_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Failed to load checkpoint status: {e}")
            return None
    
    def list_checkpoints(self) -> List[Dict]:
        """List all available checkpoints."""
        checkpoints = []
        
        for checkpoint_file in self.checkpoint_dir.glob("*.json"):
            if checkpoint_file.name.endswith("_stats.json"):
                continue
            
            try:
                with open(checkpoint_file, 'r') as f:
                    data = json.load(f)
                    checkpoints.append({
                        'name': checkpoint_file.stem,
                        'timestamp': data.get('timestamp'),
                        'processed_count': len(data.get('processed_ids', [])),
                        'stats': data.get('stats', {})
                    })
            except Exception as e:
                logger.warning(f"Failed to load checkpoint {checkpoint_file}: {e}")
                continue
        
        return sorted(checkpoints, key=lambda x: x.get('timestamp', ''), reverse=True)


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
    
    def get_checkpoint_status(self, checkpoint_name: str) -> Optional[Dict]:
        """Get checkpoint status."""
        return self.async_processor.get_checkpoint_status(checkpoint_name)
    
    def list_checkpoints(self) -> List[Dict]:
        """List all checkpoints."""
        return self.async_processor.list_checkpoints()
    
    @property
    def stats(self) -> Dict:
        """Get processing statistics."""
        return self.async_processor.stats