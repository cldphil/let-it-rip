"""
Metadata extractor that integrates with the new hierarchical extraction system.
This module acts as a bridge between the old system and new core modules.
"""

import re
import json
from typing import Dict, List, Optional
import logging

from core import (
    SyncHierarchicalExtractor, 
    PaperInsights,
    ExtractionMetadata
)
from config import Config

logger = logging.getLogger(__name__)


class MetadataExtractor:
    """
    Wrapper class to maintain compatibility with existing code while using new extraction system.
    """
    
    def __init__(self):
        """Initialize the metadata extractor."""
        self.config = Config
        self.hierarchical_extractor = SyncHierarchicalExtractor()
        logger.info("Initialized MetadataExtractor with hierarchical extraction system")
    
    def process_paper(self, paper_metadata: Dict, full_text: str = "") -> Dict:
        """
        Process a paper and extract insights using the new system.
        
        Args:
            paper_metadata: Basic paper metadata from arXiv
            full_text: Full paper text (optional)
            
        Returns:
            Enhanced paper metadata with business tags and quality score
        """
        # Add full text to paper metadata if provided
        if full_text:
            paper_metadata['full_text'] = full_text
        
        try:
            # Extract insights using hierarchical extractor
            insights, extraction_metadata = self.hierarchical_extractor.extract_insights(paper_metadata)
            
            # Convert to old format for compatibility
            enhanced_metadata = self._convert_to_legacy_format(
                paper_metadata, 
                insights, 
                extraction_metadata
            )
            
            return enhanced_metadata
            
        except Exception as e:
            logger.error(f"Error processing paper: {e}")
            # Return minimal metadata on error
            return self._create_minimal_metadata(paper_metadata)
    
    def _convert_to_legacy_format(self, 
                                 paper_metadata: Dict, 
                                 insights: PaperInsights,
                                 extraction_metadata: ExtractionMetadata) -> Dict:
        """Convert new insights format to legacy format for compatibility."""
        enhanced_metadata = paper_metadata.copy()
        
        # Extract business tags in old format
        business_tags = {
            'methodology_type': insights.study_type.value,
            'industry': [ind.value for ind in insights.industry_applications],
            'implementation_complexity': insights.implementation_complexity.value,
            'team_size_required': insights.resource_requirements.team_size.value,
            'success_metrics': [
                metric.metric_name for metric in insights.success_metrics
            ],
            'technical_requirements': [
                tech.value for tech in insights.techniques_used
            ],
            'confidence_score': insights.extraction_confidence
        }
        
        enhanced_metadata.update({
            'business_tags': business_tags,
            'quality_score': insights.get_quality_score(),
            'sections_extracted': extraction_metadata.section_lengths,
            
            # New fields from insights
            'key_findings': insights.key_findings,
            'main_contribution': insights.main_contribution,
            'limitations': insights.limitations,
            'prerequisites': insights.prerequisites,
            'evidence_strength': insights.evidence_strength,
            'practical_applicability': insights.practical_applicability,
            'has_code': insights.has_code_available,
            'has_dataset': insights.has_dataset_available,
            
            # Extraction metadata
            'extraction_time': extraction_metadata.extraction_time_seconds,
            'extraction_cost': extraction_metadata.estimated_cost_usd
        })
        
        return enhanced_metadata
    
    def _create_minimal_metadata(self, paper_metadata: Dict) -> Dict:
        """Create minimal metadata when extraction fails."""
        enhanced_metadata = paper_metadata.copy()
        
        enhanced_metadata.update({
            'business_tags': {
                'methodology_type': 'unknown',
                'industry': ['general'],
                'implementation_complexity': 'unknown',
                'team_size_required': 'not_specified',
                'success_metrics': [],
                'technical_requirements': [],
                'confidence_score': 0.1
            },
            'quality_score': 0.3,
            'sections_extracted': {},
            'extraction_error': True
        })
        
        return enhanced_metadata
    
    def extract_paper_sections(self, full_text: str, abstract: str = "") -> Dict[str, str]:
        """
        Legacy method for section extraction - now handled by hierarchical extractor.
        """
        # This is handled internally by the hierarchical extractor
        # Keeping for compatibility
        return {
            'abstract': abstract[:500],
            'introduction': '',
            'methodology': '',
            'results': '',
            'conclusion': ''
        }
    
    def calculate_quality_score(self, paper_metadata: Dict, business_tags: Dict) -> float:
        """
        Legacy method for quality score calculation.
        Now handled by PaperInsights.get_quality_score()
        """
        # For compatibility, return a simple score
        return paper_metadata.get('quality_score', 0.5)