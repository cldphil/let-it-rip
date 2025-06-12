"""
Refactored manual processing system with redundancies removed.
Now serves purely as a workflow orchestrator.
"""

import os
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Union
import logging

from core import SyncBatchProcessor, InsightStorage
from config import Config
from services.arxiv_fetcher import ArxivFetcher  # Use the existing fetcher

logger = logging.getLogger(__name__)


class ManualProcessingController:
    """
    Orchestrates manual processing workflows with date range selection.
    
    This controller coordinates between:
    - ArxivFetcher: For fetching papers by date
    - BatchProcessor: For processing papers
    - InsightStorage: For storing results and tracking history
    
    Responsibilities:
    - Date range validation and estimation
    - Cost and time estimation
    - Processing orchestration with progress tracking
    - Processing history management
    """
    
    def __init__(self):
        """Initialize manual processing controller."""
        self.storage = InsightStorage()
        
        self.processor = SyncBatchProcessor(storage=self.storage, batch_size=Config.BATCH_SIZE)
        
        # Use the existing ArxivFetcher instead of creating a redundant one
        self.fetcher = ArxivFetcher()
        
        # Processing statistics
        self.last_processing_stats = {}
        
        logger.info("Initialized manual processing controller")
    
    
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
            if hasattr(self.storage, 'supabase') and self.storage.supabase:
                # Cloud storage query
                result = self.storage.supabase.table('papers').select('paper_id, published_date').gte(
                    'published_date', start_date.strftime('%Y-%m-%d')
                ).lte(
                    'published_date', end_date.strftime('%Y-%m-%d')
                ).execute()
                
                existing_papers = len(result.data)
            else:
                # Local storage - would need to implement date range query
                # For now, return 0 as we don't have an efficient way to query by date locally
                existing_papers = 0
            
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
        
        This is a simplified version that delegates to process_date_range_enhanced.
        
        Args:
            start_date: Start date for papers
            end_date: End date for papers
            max_papers: Maximum papers to process
            skip_existing: Whether to skip already processed papers
            progress_callback: Function to call with progress updates
            
        Returns:
            Dict with processing results and statistics
        """
        return self.process_date_range_enhanced(
            start_date, end_date, max_papers, skip_existing, progress_callback
        )
    
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
            # Fetch papers using the existing ArxivFetcher
            papers = self.fetcher.fetch_papers_date_range(
                start_date=start_date,
                end_date=end_date,
                max_results=max_papers or 1000,
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
            
            # Process using batch processor with progress tracking
            stats = self._process_papers_with_detailed_progress(
                papers,
                checkpoint_name,
                progress_callback
            )
            
            if progress_callback:
                progress_callback("Processing complete! Calculating final statistics...", 95)
            
            # Store processing history if using cloud storage
            if hasattr(self.storage, 'supabase') and self.storage.supabase:
                try:
                    # REPLACE THE history_entry DICTIONARY WITH THIS:
                    history_entry = {
                        'batch_name': checkpoint_name,
                        'papers_processed': stats.get('successful', 0),
                        'successful_extractions': stats.get('successful', 0),
                        'failed_extractions': stats.get('failed', 0),
                        'total_cost_usd': stats.get('total_cost', 0),
                        'processing_time_seconds': stats.get('total_time', 0),
                        'date_range': {  # Store as JSON object, not separate fields
                            'start': start_date.isoformat(),
                            'end': end_date.isoformat()
                        },
                        'success_rate': stats.get('successful', 0) / len(papers) if papers else 0
                    }
                    
                    self.storage.supabase.table('processing_logs').insert(history_entry).execute()
                except Exception as e:
                    logger.warning(f"Failed to store processing history: {e}")
            
            # Enhanced results
            results = {
                'success': True,
                'papers_found': len(papers),
                'papers_processed': stats.get('successful', 0),
                'papers_failed': stats.get('failed', 0),
                'processing_time_seconds': stats.get('total_time', 0),
                'total_cost': stats.get('total_cost', 0),  # Use consistent field name
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
    
    def _process_papers_with_detailed_progress(self, papers: List[Dict], 
                                             checkpoint_name: str,
                                             progress_callback=None) -> Dict:
        """
        Process papers with detailed progress tracking for enhanced UI feedback.
        
        This method provides granular progress updates during processing.
        
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
    
    def get_processing_history(self, limit: int = 10) -> List[Dict]:
        """
        Get recent processing history with enhanced column mapping and error handling.
        
        Args:
            limit: Maximum number of recent batches to return
            
        Returns:
            List of processing history records with consistent field names
        """
        try:
            if hasattr(self.storage, 'supabase') and self.storage.supabase:
                # Query with error handling
                try:
                    result = self.storage.supabase.table('processing_logs').select(
                        '*'
                    ).order('created_at', desc=True).limit(limit).execute()
                    
                    if not result.data:
                        logger.info("No processing history found")
                        return []
                    
                    logger.info(f"Retrieved {len(result.data)} processing history records")
                    
                except Exception as e:
                    logger.error(f"Failed to query processing history: {e}")
                    return []
                
                # Enhanced data mapping and validation
                history_data = []
                
                for i, record in enumerate(result.data):
                    try:
                        mapped_record = self._process_history_record(record)
                        if mapped_record:  # Only include valid records
                            history_data.append(mapped_record)
                            
                    except Exception as e:
                        logger.warning(f"Failed to process history record {i}: {e}")
                        continue  # Skip problematic records
                
                logger.info(f"Successfully processed {len(history_data)} history records")
                return history_data
            
            else:
                # Local storage fallback
                logger.info("Using local storage for processing history")
                return self._get_local_processing_history(limit)
                
        except Exception as e:
            logger.error(f"Failed to get processing history: {e}")
            return []

    def _process_history_record(self, record: Dict) -> Optional[Dict]:
        """
        Process a single history record with enhanced mapping and validation.
        
        Args:
            record: Raw database record
            
        Returns:
            Processed record with consistent field names, or None if invalid
        """
        try:
            # Start with the raw record
            processed_record = record.copy()
            
            # Apply column mapping from database to application field names
            column_mappings = {
                # Database column → Application field
                'failed_extractions': 'papers_failed',
                'successful_extractions': 'papers_successful', 
                'total_cost_usd': 'total_cost'
            }
            
            for db_column, app_field in column_mappings.items():
                if db_column in processed_record:
                    processed_record[app_field] = processed_record[db_column]
                    # Keep original for compatibility, but prioritize mapped version
            
            # Ensure all numeric fields are properly typed and validated
            numeric_fields = {
                'papers_processed': {'type': int, 'default': 0, 'min': 0},
                'papers_failed': {'type': int, 'default': 0, 'min': 0},
                'papers_successful': {'type': int, 'default': 0, 'min': 0},
                'failed_extractions': {'type': int, 'default': 0, 'min': 0},
                'successful_extractions': {'type': int, 'default': 0, 'min': 0},
                'total_cost': {'type': float, 'default': 0.0, 'min': 0.0},
                'total_cost_usd': {'type': float, 'default': 0.0, 'min': 0.0},
                'processing_time_seconds': {'type': float, 'default': 0.0, 'min': 0.0},
                'success_rate': {'type': float, 'default': 0.0, 'min': 0.0, 'max': 1.0}
            }
            
            for field, config in numeric_fields.items():
                if field in processed_record:
                    processed_record[field] = self._validate_numeric_field(
                        processed_record[field], 
                        config
                    )
            
            # Calculate derived fields if missing
            processed_record = self._calculate_derived_fields(processed_record)
            
            # Validate required fields exist
            required_fields = ['batch_name', 'created_at']
            for field in required_fields:
                if not processed_record.get(field):
                    logger.warning(f"Record missing required field: {field}")
                    return None
            
            # Format timestamps for consistency
            if 'created_at' in processed_record:
                processed_record['created_at'] = self._format_timestamp(processed_record['created_at'])
            
            return processed_record
            
        except Exception as e:
            logger.error(f"Error processing history record: {e}")
            return None

    def _validate_numeric_field(self, value, config: Dict) -> Union[int, float]:
        """
        Validate and convert a numeric field with error handling.
        
        Args:
            value: Raw field value
            config: Validation configuration
            
        Returns:
            Validated numeric value
        """
        try:
            # Handle string representations
            if isinstance(value, str):
                # Remove currency symbols and whitespace
                cleaned_value = value.replace('$', '').replace(',', '').strip()
                if not cleaned_value:
                    return config['default']
                value = cleaned_value
            
            # Convert to expected type
            expected_type = config['type']
            converted_value = expected_type(value)
            
            # Apply range validation
            if 'min' in config and converted_value < config['min']:
                logger.warning(f"Value {converted_value} below minimum {config['min']}, using minimum")
                converted_value = config['min']
            
            if 'max' in config and converted_value > config['max']:
                logger.warning(f"Value {converted_value} above maximum {config['max']}, using maximum")
                converted_value = config['max']
            
            return converted_value
            
        except (ValueError, TypeError) as e:
            logger.warning(f"Failed to convert field value '{value}': {e}, using default")
            return config['default']

    def _calculate_derived_fields(self, record: Dict) -> Dict:
        """
        Calculate derived fields for consistency and completeness.
        
        Args:
            record: Processing history record
            
        Returns:
            Record with calculated derived fields
        """
        try:
            # Calculate total processed if missing
            if 'papers_processed' not in record or record['papers_processed'] == 0:
                papers_failed = record.get('papers_failed', record.get('failed_extractions', 0))
                papers_successful = record.get('papers_successful', record.get('successful_extractions', 0))
                record['papers_processed'] = papers_failed + papers_successful
            
            # Calculate success rate if missing or invalid
            papers_processed = record.get('papers_processed', 0)
            if papers_processed > 0:
                papers_successful = record.get('papers_successful', record.get('successful_extractions', 0))
                calculated_success_rate = papers_successful / papers_processed
                
                # Only update if missing or clearly wrong
                current_success_rate = record.get('success_rate', 0)
                if current_success_rate == 0 or current_success_rate > 1:
                    record['success_rate'] = round(calculated_success_rate, 3)
            
            # Ensure both column name versions exist for compatibility
            if 'papers_failed' in record and 'failed_extractions' not in record:
                record['failed_extractions'] = record['papers_failed']
            
            if 'papers_successful' in record and 'successful_extractions' not in record:
                record['successful_extractions'] = record['papers_successful']
            
            if 'total_cost' in record and 'total_cost_usd' not in record:
                record['total_cost_usd'] = record['total_cost']
            
            return record
            
        except Exception as e:
            logger.warning(f"Error calculating derived fields: {e}")
            return record

    def _format_timestamp(self, timestamp_value) -> str:
        """
        Format timestamp for consistent display.
        
        Args:
            timestamp_value: Raw timestamp value
            
        Returns:
            Formatted timestamp string
        """
        try:
            if isinstance(timestamp_value, str):
                # Parse ISO format and reformat for display
                from datetime import datetime
                if 'T' in timestamp_value:
                    dt = datetime.fromisoformat(timestamp_value.replace('Z', '+00:00'))
                    return dt.strftime('%Y-%m-%d %H:%M:%S')
                else:
                    return timestamp_value
            else:
                return str(timestamp_value)
                
        except Exception as e:
            logger.warning(f"Error formatting timestamp: {e}")
            return str(timestamp_value)

    def _get_local_processing_history(self, limit: int) -> List[Dict]:
        """
        Fallback method for local storage processing history.
        
        Args:
            limit: Maximum records to return
            
        Returns:
            List of processing history records
        """
        try:
            # This would implement local file-based history if needed
            # For now, return empty list
            logger.info("Local processing history not implemented")
            return []
            
        except Exception as e:
            logger.error(f"Error getting local processing history: {e}")
            return []

    # ADDITIONAL ENHANCEMENT: Add processing history statistics method

    def get_processing_statistics(self, days: int = 30) -> Dict:
        """
        Get processing statistics for the last N days.
        
        Args:
            days: Number of days to analyze
            
        Returns:
            Dict with processing statistics
        """
        try:
            # Get recent history
            recent_history = self.get_processing_history(limit=100)  # Get more for stats
            
            if not recent_history:
                return {
                    'total_batches': 0,
                    'total_papers_processed': 0,
                    'total_successful': 0,
                    'total_failed': 0,
                    'average_success_rate': 0.0,
                    'total_cost': 0.0,
                    'average_processing_time': 0.0
                }
            
            # Filter by date range
            from datetime import datetime, timedelta
            cutoff_date = datetime.now() - timedelta(days=days)
            
            filtered_history = []
            for record in recent_history:
                try:
                    record_date = datetime.fromisoformat(record['created_at'].replace('Z', '+00:00'))
                    if record_date >= cutoff_date:
                        filtered_history.append(record)
                except:
                    # Include records with unparseable dates
                    filtered_history.append(record)
            
            # Calculate statistics
            stats = {
                'total_batches': len(filtered_history),
                'total_papers_processed': sum(r.get('papers_processed', 0) for r in filtered_history),
                'total_successful': sum(r.get('papers_successful', r.get('successful_extractions', 0)) for r in filtered_history),
                'total_failed': sum(r.get('papers_failed', r.get('failed_extractions', 0)) for r in filtered_history),
                'total_cost': sum(r.get('total_cost', r.get('total_cost_usd', 0)) for r in filtered_history),
                'total_processing_time': sum(r.get('processing_time_seconds', 0) for r in filtered_history)
            }
            
            # Calculate averages
            if stats['total_papers_processed'] > 0:
                stats['average_success_rate'] = stats['total_successful'] / stats['total_papers_processed']
            else:
                stats['average_success_rate'] = 0.0
            
            if stats['total_batches'] > 0:
                stats['average_processing_time'] = stats['total_processing_time'] / stats['total_batches']
            else:
                stats['average_processing_time'] = 0.0
            
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating processing statistics: {e}")
            return {}
    
    def get_storage_usage(self) -> Dict:
        """Get current storage usage statistics."""
        return self.storage.get_statistics()
    
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
                'system_healthy': True,
                'processing_active': False  # Would need to track active processing
            }
            
            return status
            
        except Exception as e:
            logger.error(f"Failed to get processing status: {e}")
            return {
                'error': str(e),
                'system_healthy': False,
                'timestamp': datetime.now().isoformat()
            }