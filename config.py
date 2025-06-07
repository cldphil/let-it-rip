"""
Configuration settings for the Research Implementation Platform.
Updated to remove deprecated evidence and applicability weights.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration settings."""
    
    # API Configuration
    LLM_API_KEY = os.getenv('LLM_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    
    # LLM Settings
    LLM_MODEL = "claude-sonnet-4-20250514"  # Anthropic Claude Sonnet 4
    LLM_TEMPERATURE = 0.1  # Low temperature for consistent structured output
    LLM_MAX_TOKENS = 2000  # Increased for longer key findings
    LLM_TIMEOUT = 30  # Timeout in seconds
    
    # arXiv API Settings
    ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
    ARXIV_MAX_RESULTS_DEFAULT = 20
    ARXIV_REQUEST_DELAY = 1.0  # Seconds between requests (be respectful)
    ARXIV_TIMEOUT = 30
    
    # Text Extraction Settings
    ABSTRACT_MAX_CHARS = 500
    INTRODUCTION_MAX_CHARS = 1000
    METHODOLOGY_MAX_CHARS = 800
    RESULTS_MAX_CHARS = 600
    CONCLUSION_MAX_CHARS = 600
    
    # Updated Quality Scoring Weights (removed deprecated evidence and applicability weights)
    RECENCY_WEIGHT = 0.25  # Prioritize recent research
    QUALITY_WEIGHT = 0.35  # Increased weight for objective quality score (author h-index + conference)
    VALIDATION_WEIGHT = 0.20  # Industry validation importance
    CASE_STUDY_WEIGHT = 0.20  # Real-world implementation evidence
    RECENCY_DECAY_RATE = 0.1  # Quality decay per year (10%)
    
    # Quality Score Configuration
    CONFERENCE_MULTIPLIER = 1.5  # Multiplier for papers with conference validation
    QUALITY_SCORE_NORMALIZATION = 100.0  # Divisor for normalizing h-index to 0-1 range
    MAX_QUALITY_SCORE = 1.0  # Maximum quality score cap
    
    # Author H-Index Configuration
    SEMANTIC_SCHOLAR_CACHE_DAYS = 30  # Days to cache author h-index data
    AUTHOR_HINDEX_TIMEOUT = 10  # Timeout for author lookup requests
    MIN_HINDEX_FOR_QUALITY = 5  # Minimum h-index to consider for quality bonus
    
    # Processing Settings
    BATCH_SIZE = 10  # Number of papers to process in each batch
    MAX_RETRIES = 3  # Retry attempts for failed API calls
    ENABLE_FULL_TEXT = True  # Enable full text extraction for enhanced key findings
    ENABLE_LLM_TAGGING = True  # Whether to use LLM for tagging (vs heuristics)
    ENABLE_AUTHOR_LOOKUP = True  # Whether to fetch author h-indices
    
    # Output Settings
    OUTPUT_DIR = "output"
    METADATA_FILENAME = "papers_metadata.json"
    FULL_TEXT_FILENAME = "papers_fulltext.json"
    
    # Search Terms for GenAI Research
    GENAI_SEARCH_TERMS = [
        "generative artificial intelligence",
        "generative AI",
        "large language model",
        "LLM",
        "GPT",
        "diffusion model",
        "generative model",
        "text generation",
        "image generation",
        "prompt engineering",
        "fine-tuning",
        "RLHF",
        "transformer"
    ]
    
    # Business Tag Categories (streamlined for objective analysis)
    METHODOLOGY_TYPES = [
        "case_study", 
        "theoretical", 
        "empirical", 
        "pilot", 
        "survey",
        "meta_analysis",
        "review",
        "unknown"
    ]
    
    COMPLEXITY_LEVELS = [
        "low",
        "medium", 
        "high",
        "very_high",
        "unknown"
    ]
    
    # Updated success metrics focused on measurable outcomes
    SUCCESS_METRICS = [
        "roi_mentioned",
        "kpis_listed", 
        "user_metrics",
        "performance_gains",
        "cost_reduction",
        "time_savings",
        "accuracy_improvement",
        "scalability_metrics",
        "deployment_stats",
        "user_satisfaction_scores"
    ]
    
    TECHNICAL_REQUIREMENTS = [
        "fine_tuning",
        "retrieval_augmented_generation",
        "prompt_engineering",
        "multi_agent",
        "chain_of_thought",
        "reinforcement_learning",
        "few_shot_learning",
        "zero_shot_learning",
        "transfer_learning",
        "ensemble_methods"
    ]
    
    # Enhanced Key Findings Configuration
    MAX_KEY_FINDINGS = 10  # Allow up to 10 detailed findings per paper
    MIN_FINDING_LENGTH = 50  # Minimum characters per finding
    FOCUS_AREAS = [
        "methods",
        "results", 
        "implications",
        "uniqueness",
        "practical_applications",
        "conclusions",
        "limitations",
        "validation_evidence",
        "industry_deployment"
    ]
    
    # Conference Detection Configuration
    MAJOR_AI_CONFERENCES = [
        'neurips', 'nips', 'icml', 'iclr', 'aaai', 'ijcai', 'cvpr', 'iccv', 'eccv',
        'acl', 'emnlp', 'naacl', 'coling', 'sigir', 'kdd', 'www', 'icra', 'iros',
        'uai', 'aistats', 'colt', 'interspeech', 'asru', 'icassp'
    ]
    
    CONFERENCE_INDICATORS = [
        'workshop', 'symposium', 'tutorial', 'accepted at', 'accepted to', 
        'to appear in', 'published in', 'presented at', 'submission to', 
        'camera ready', 'conference paper'
    ]
    
    # Case Study Detection Configuration
    CASE_STUDY_STRONG_INDICATORS = [
        'case study', 'case-study', 'field study', 'industrial case study',
        'case study at', 'deployment case study', 'real-world case study'
    ]
    
    CASE_STUDY_WEAK_INDICATORS = [
        'deployment', 'production', 'real-world', 'industrial application',
        'company', 'organization', 'enterprise', 'pilot study'
    ]
    
    IMPLEMENTATION_EVIDENCE = [
        'deployed at', 'implemented at', 'used by', 'adopted by',
        'operational at', 'in production at', 'live system',
        'fortune 500', 'fortune 1000', 'employees use',
        'customers use', 'users daily', 'requests per day',
        'deployed system', 'production deployment'
    ]
    
    # Ranking Algorithm Configuration (updated weights)
    RANKING_WEIGHTS = {
        'similarity': 0.30,      # Vector similarity
        'quality': 0.25,         # Author h-index + conference validation
        'recency': 0.20,         # Time-based relevance
        'findings_richness': 0.15,  # Number and depth of key findings
        'validation_bonus': 0.10    # Industry validation and technique match bonuses
    }
    
    # Synthesis Engine Configuration
    MAX_PAPERS_FOR_SYNTHESIS = 25  # Maximum papers to include in synthesis
    CASE_STUDY_BONUS_THRESHOLD = 0.7  # Minimum case study score for bonus
    VALIDATION_BONUS_THRESHOLD = 0.8  # Minimum validation score for bonus
    CONSERVATIVE_QUALITY_THRESHOLD = 0.5  # Quality threshold for conservative users
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present."""
        if cls.ENABLE_LLM_TAGGING and not any([
            cls.LLM_API_KEY, 
            cls.OPENAI_API_KEY, 
            cls.ANTHROPIC_API_KEY,
            cls.GOOGLE_API_KEY
        ]):
            print("Warning: LLM tagging enabled but no API key found. Will fall back to heuristics.")
        
        if not os.path.exists(cls.OUTPUT_DIR):
            os.makedirs(cls.OUTPUT_DIR)
            print(f"Created output directory: {cls.OUTPUT_DIR}")
        
        # Validate ranking weights sum to reasonable total
        total_ranking_weight = sum(cls.RANKING_WEIGHTS.values())
        if abs(total_ranking_weight - 1.0) > 0.01:
            print(f"Warning: Ranking weights sum to {total_ranking_weight:.2f}, expected ~1.0")
    
    @classmethod
    def get_active_api_key(cls):
        """Get the first available API key."""
        for key in [cls.LLM_API_KEY, cls.OPENAI_API_KEY, cls.ANTHROPIC_API_KEY, cls.GOOGLE_API_KEY]:
            if key:
                return key
        return None
    
    @classmethod
    def get_quality_score_config(cls):
        """Get quality score calculation configuration."""
        return {
            'conference_multiplier': cls.CONFERENCE_MULTIPLIER,
            'normalization_factor': cls.QUALITY_SCORE_NORMALIZATION,
            'max_score': cls.MAX_QUALITY_SCORE,
            'min_hindex_threshold': cls.MIN_HINDEX_FOR_QUALITY
        }
    
    @classmethod
    def get_ranking_config(cls):
        """Get ranking algorithm configuration."""
        return {
            'weights': cls.RANKING_WEIGHTS,
            'recency_decay_rate': cls.RECENCY_DECAY_RATE,
            'case_study_bonus_threshold': cls.CASE_STUDY_BONUS_THRESHOLD,
            'validation_bonus_threshold': cls.VALIDATION_BONUS_THRESHOLD,
            'conservative_quality_threshold': cls.CONSERVATIVE_QUALITY_THRESHOLD
        }
    
    @classmethod
    def get_detection_config(cls):
        """Get detection configuration for conferences and case studies."""
        return {
            'conferences': cls.MAJOR_AI_CONFERENCES,
            'conference_indicators': cls.CONFERENCE_INDICATORS,
            'case_study_strong': cls.CASE_STUDY_STRONG_INDICATORS,
            'case_study_weak': cls.CASE_STUDY_WEAK_INDICATORS,
            'implementation_evidence': cls.IMPLEMENTATION_EVIDENCE
        }

# Initialize configuration
Config.validate_config()