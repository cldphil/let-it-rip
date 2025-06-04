"""
Structured insight schema for research papers.
Defines the data models for extracted insights and metadata.
"""

from typing import List, Dict, Optional, Any
from enum import Enum
from pydantic import BaseModel, Field, validator
from datetime import datetime


class StudyType(str, Enum):
    """Types of research studies."""
    EMPIRICAL = "empirical"
    CASE_STUDY = "case_study"
    THEORETICAL = "theoretical"
    PILOT = "pilot"
    SURVEY = "survey"
    META_ANALYSIS = "meta_analysis"
    REVIEW = "review"
    UNKNOWN = "unknown"


class TechniqueCategory(str, Enum):
    """Categories of AI/ML techniques used."""
    FINE_TUNING = "fine_tuning"
    RAG = "retrieval_augmented_generation"
    PROMPT_ENGINEERING = "prompt_engineering"
    MULTI_AGENT = "multi_agent"
    CHAIN_OF_THOUGHT = "chain_of_thought"
    REINFORCEMENT_LEARNING = "reinforcement_learning"
    FEW_SHOT_LEARNING = "few_shot_learning"
    ZERO_SHOT_LEARNING = "zero_shot_learning"
    TRANSFER_LEARNING = "transfer_learning"
    ENSEMBLE_METHODS = "ensemble_methods"
    OTHER = "other"


class ComplexityLevel(str, Enum):
    """Implementation complexity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"
    UNKNOWN = "unknown"


class ResourceRequirements(BaseModel):
    """Detailed resource requirements for implementation."""
    compute_requirements: Optional[str] = None  # e.g., "4 GPUs", "TPU cluster"
    data_requirements: Optional[str] = None  # e.g., "1M labeled examples"
    budget_tier: Optional[str] = None  # e.g., "low", "medium", "high", "enterprise"
    special_hardware: Optional[List[str]] = Field(default_factory=list)
    cloud_services: Optional[List[str]] = Field(default_factory=list)


class SuccessMetric(BaseModel):
    """Individual success metric with details."""
    metric_name: str
    improvement_value: Optional[float] = None
    improvement_unit: Optional[str] = None  # e.g., "percentage", "time_seconds"
    baseline_comparison: Optional[str] = None
    statistical_significance: Optional[bool] = None


class PaperInsights(BaseModel):
    """Comprehensive insights extracted from a research paper."""
    
    # Identification
    paper_id: str  # arXiv ID or unique identifier
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    extraction_version: str = "1.0"
    
    # Core insights - Enhanced key findings
    key_findings: List[str] = Field(
        default_factory=list,
        description="Detailed key findings from the paper (up to 10, can be multiple sentences each)"
    )
    limitations: List[str] = Field(
        default_factory=list,
        description="Acknowledged limitations or constraints"
    )
    future_work: List[str] = Field(
        default_factory=list,
        description="Suggested future research directions"
    )
    
    # Categorization
    study_type: StudyType = StudyType.UNKNOWN
    techniques_used: List[TechniqueCategory] = Field(default_factory=list)
    
    # Implementation details
    implementation_complexity: ComplexityLevel = ComplexityLevel.UNKNOWN
    resource_requirements: ResourceRequirements = Field(default_factory=ResourceRequirements)
    success_metrics: List[SuccessMetric] = Field(default_factory=list)
    
    # Context and synthesis helpers
    problem_addressed: str = Field(
        default="",
        description="What specific problem does this research solve?"
    )
    prerequisites: List[str] = Field(
        default_factory=list,
        description="Technical prerequisites for implementation"
    )
    comparable_approaches: List[str] = Field(
        default_factory=list,
        description="Alternative approaches mentioned or compared"
    )
    real_world_applications: List[str] = Field(
        default_factory=list,
        description="Specific real-world use cases mentioned"
    )
    
    # Quality indicators
    evidence_strength: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Strength of empirical evidence (0-1)"
    )
    practical_applicability: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="How applicable to real-world scenarios (0-1)"
    )
    extraction_confidence: float = Field(
        default=0.5,
        ge=0.0,
        le=1.0,
        description="Confidence in extraction accuracy (0-1)"
    )
    
    # Additional metadata
    has_code_available: bool = False
    has_dataset_available: bool = False
    reproducibility_score: Optional[float] = Field(None, ge=0.0, le=1.0)
    industry_validation: bool = Field(
        default=False,
        description="Whether results were validated in industry setting"
    )
    
    @validator('key_findings')
    def limit_key_findings(cls, v):
        """Ensure we don't have too many key findings."""
        return v[:10] if len(v) > 10 else v
    
    def to_searchable_text(self) -> str:
        """Convert insights to searchable text for embeddings."""
        parts = [
            f"Problem: {self.problem_addressed}",
            f"Study type: {self.study_type.value}",
            f"Techniques: {', '.join(t.value for t in self.techniques_used)}",
            f"Complexity: {self.implementation_complexity.value}",
            "Key findings: " + " ".join(self.key_findings),
            "Prerequisites: " + " ".join(self.prerequisites),
            "Applications: " + " ".join(self.real_world_applications)
        ]
        
        return " ".join(filter(None, parts))
    
    def get_quality_score(self) -> float:
        """Calculate overall quality score."""
        # Weighted average of different factors
        weights = {
            'evidence': 0.3,
            'applicability': 0.3,
            'confidence': 0.2,
            'reproducibility': 0.1,
            'validation': 0.1
        }
        
        score = (
            weights['evidence'] * self.evidence_strength +
            weights['applicability'] * self.practical_applicability +
            weights['confidence'] * self.extraction_confidence
        )
        
        # Bonus for reproducibility
        if self.reproducibility_score is not None:
            score += weights['reproducibility'] * self.reproducibility_score
        else:
            score += weights['reproducibility'] * 0.3  # Default low score
        
        # Bonus for industry validation
        if self.industry_validation:
            score += weights['validation'] * 1.0
        else:
            score += weights['validation'] * 0.0
        
        return min(1.0, score)  # Cap at 1.0


class UserContext(BaseModel):
    """User context for personalized recommendations."""
    
    # Organization profile
    company_size: str = "medium"  # startup, small, medium, large, enterprise
    maturity_level: str = "pilot_ready"  # greenfield, pilot_ready, scaling, optimizing
    
    # Technical capabilities
    team_skills: List[str] = Field(default_factory=list)
    existing_infrastructure: List[str] = Field(default_factory=list)
    preferred_cloud: Optional[str] = None
    
    # Constraints
    budget_constraint: Optional[str] = None  # low, medium, high, unlimited
    risk_tolerance: str = "moderate"  # conservative, moderate, aggressive
    
    # Preferences
    preferred_techniques: List[TechniqueCategory] = Field(default_factory=list)
    avoided_techniques: List[TechniqueCategory] = Field(default_factory=list)
    success_criteria: List[str] = Field(default_factory=list)
    
    # Use case specifics
    use_case_description: str = ""
    specific_problems: List[str] = Field(default_factory=list)
    expected_outcomes: List[str] = Field(default_factory=list)
    
    def to_search_query(self) -> str:
        """Convert user context to search query."""
        parts = [
            f"Company size: {self.company_size}",
            f"Use case: {self.use_case_description}",
            f"Problems: {' '.join(self.specific_problems)}",
            f"Skills: {' '.join(self.team_skills)}"
        ]
        
        return " ".join(filter(None, parts))


class ExtractionMetadata(BaseModel):
    """Metadata about the extraction process."""
    
    extraction_id: str
    paper_id: str
    extraction_timestamp: datetime = Field(default_factory=datetime.utcnow)
    
    # Extraction details
    extractor_version: str
    llm_model: str
    llm_temperature: float
    
    # Performance metrics
    extraction_time_seconds: float
    tokens_used: Optional[int] = None
    api_calls_made: int = 1
    
    # Quality metrics
    sections_found: Dict[str, bool] = Field(default_factory=dict)
    section_lengths: Dict[str, int] = Field(default_factory=dict)
    extraction_errors: List[str] = Field(default_factory=list)
    
    # Cost tracking
    estimated_cost_usd: Optional[float] = None
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }