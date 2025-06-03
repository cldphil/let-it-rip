"""
Core modules for the GenAI Research Implementation Platform.
"""

from .insight_schema import (
    PaperInsights,
    UserContext,
    StudyType,
    TechniqueCategory,
    ComplexityLevel,
    Industry,
    TeamSize,
    ResourceRequirements,
    SuccessMetric,
    ExtractionMetadata
)

from .hierarchical_extractor import (
    HierarchicalInsightExtractor,
    SyncHierarchicalExtractor
)

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
    'Industry',
    'TeamSize',
    'ResourceRequirements',
    'SuccessMetric',
    'ExtractionMetadata',
    
    # Extractors
    'HierarchicalInsightExtractor',
    'SyncHierarchicalExtractor',
    
    # Storage
    'InsightStorage',
    
    # Processing
    'BatchProcessor',
    'SyncBatchProcessor',
    
    # Synthesis
    'SynthesisEngine'
]