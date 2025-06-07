"""
Core modules for the GenAI Research Implementation Platform.
"""

from .insight_schema import (
    PaperInsights,
    UserContext,
    StudyType,
    TechniqueCategory,
    ComplexityLevel,
    ExtractionMetadata
)

from .insight_extractor import InsightExtractor

from .insight_storage import InsightStorage

from .batch_processor import (
    BatchProcessor,
    SyncBatchProcessor
)

from .synthesis_engine import SynthesisEngine

__all__ = [
    # Schema
    'PaperInsights',
    'UserContext',
    'StudyType',
    'TechniqueCategory',
    'ComplexityLevel',
    'ExtractionMetadata',
    
    # Extractors
    'InsightExtractor',
    
    # Storage
    'InsightStorage',
    
    # Processing
    'BatchProcessor',
    'SyncBatchProcessor',
    
    # Synthesis
    'SynthesisEngine'
]