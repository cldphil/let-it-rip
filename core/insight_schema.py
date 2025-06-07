"""
Structured insight schema for research papers.
Defines the data models for extracted insights and metadata.
"""

from typing import List, Dict, Optional
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
    """Expanded categories of AI/ML techniques used in modern GenAI research."""
    
    # Core GenAI Techniques
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
    
    # Modern Generative Models
    DIFFUSION_MODELS = "diffusion_models"
    FLOW_MATCHING = "flow_matching"
    AUTOREGRESSIVE_MODELING = "autoregressive_modeling"
    VARIATIONAL_INFERENCE = "variational_inference"
    ENERGY_BASED_MODELING = "energy_based_modeling"
    
    # Multimodal & Vision-Language
    MULTIMODAL_LEARNING = "multimodal_learning"
    VISION_LANGUAGE_MODELING = "vision_language_modeling"
    TEXT_TO_3D_SYNTHESIS = "text_to_3d_synthesis"
    SEMANTIC_SEGMENTATION = "semantic_segmentation"
    OBJECT_DETECTION = "object_detection"
    GAUSSIAN_SPLATTING = "gaussian_splatting"
    
    # Advanced Training & Optimization
    CONTRASTIVE_LEARNING = "contrastive_learning"
    SELF_SUPERVISED_LEARNING = "self_supervised_learning"
    MULTI_TASK_LEARNING = "multi_task_learning"
    PREFERENCE_OPTIMIZATION = "preference_optimization"
    REWARD_MODELING = "reward_modeling"
    LOW_RANK_ADAPTATION = "low_rank_adaptation"  # LoRA
    
    # Architecture & Attention Innovations
    ATTENTION_MECHANISMS = "attention_mechanisms"
    CROSS_ATTENTION = "cross_attention"
    TRANSFORMER_ARCHITECTURE = "transformer_architecture"
    MEMORY_AUGMENTATION = "memory_augmentation"
    
    # Inference & Optimization
    INFERENCE_OPTIMIZATION = "inference_optimization"
    KV_CACHE_OPTIMIZATION = "kv_cache_optimization"
    APPROXIMATE_ATTENTION = "approximate_attention"
    DYNAMIC_SPARSIFICATION = "dynamic_sparsification"
    MODEL_PARALLELIZATION = "model_parallelization"
    
    # Reasoning & Decision Making
    SELF_REASONING = "self_reasoning"
    ADAPTIVE_DECISION_MAKING = "adaptive_decision_making"
    CAUSAL_DECODING = "causal_decoding"
    SELF_CORRECTION = "self_correction"
    IN_CONTEXT_LEARNING = "in_context_learning"
    
    # Safety & Alignment
    SAFETY_ALIGNMENT = "safety_alignment"
    CONSTITUTIONAL_AI = "constitutional_ai"
    HUMAN_PREFERENCE_EVALUATION = "human_preference_evaluation"
    
    # Data & Preprocessing
    DATA_SYNTHESIS = "data_synthesis"
    SYNTHETIC_DATA_GENERATION = "synthetic_data_generation"
    WEB_SCRAPING = "web_scraping"
    DOCUMENT_PARSING = "document_parsing"
    
    # Specialized Applications
    LEGAL_AI = "legal_ai"
    MEDICAL_AI = "medical_ai"
    SCIENTIFIC_REASONING = "scientific_reasoning"
    
    # Other/Uncategorized
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
    
    # New quality indicators
    total_author_hindex: int = Field(
        default=0,
        description="Sum of h-indices for all authors"
    )
    has_conference_mention: bool = Field(
        default=False,
        description="Whether paper mentions conference/workshop acceptance"
    )
    author_hindices: Dict[str, int] = Field(
        default_factory=dict,
        description="Individual h-indices for each author"
    )
    
    # Confidence in extraction
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
        """
        Calculate quality score based on author h-index and conference mention.
        
        Formula: quality_score = (total_author_hindex * conference_multiplier) / 100
        where conference_multiplier is 1.5 if conference mentioned, 1.0 otherwise.
        
        Normalized to 0-1 range by dividing by 100 (assuming max reasonable total h-index of ~100).
        """
        conference_multiplier = 1.5 if self.has_conference_mention else 1.0
        raw_score = self.total_author_hindex * conference_multiplier
        
        # Normalize to 0-1 range (cap at 1.0)
        quality_score = min(1.0, raw_score / 100.0)
        
        return quality_score


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
    timeline_weeks: Optional[int] = None
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