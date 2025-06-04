"""
Configuration settings for the Research Implementation Platform.
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
    
    # Quality Scoring Weights (updated to prioritize key factors)
    RECENCY_WEIGHT = 0.25  # Prioritize recent research
    QUALITY_WEIGHT = 0.25  # Overall quality score
    EVIDENCE_WEIGHT = 0.25  # Evidence strength
    APPLICABILITY_WEIGHT = 0.25  # Practical applicability
    RECENCY_DECAY_RATE = 0.1  # Quality decay per year (10%)
    
    # Processing Settings
    BATCH_SIZE = 10  # Number of papers to process in each batch
    MAX_RETRIES = 3  # Retry attempts for failed API calls
    ENABLE_FULL_TEXT = True  # Enable full text extraction for enhanced key findings
    ENABLE_LLM_TAGGING = True  # Whether to use LLM for tagging (vs heuristics)
    
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
    
    # Business Tag Categories (streamlined - removed industry, team_size, timeline)
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
    
    SUCCESS_METRICS = [
        "roi_mentioned",
        "kpis_listed", 
        "user_metrics",
        "performance_gains",
        "cost_reduction",
        "time_savings",
        "accuracy_improvement"
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
        "limitations"
    ]
    
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
    
    @classmethod
    def get_active_api_key(cls):
        """Get the first available API key."""
        for key in [cls.LLM_API_KEY, cls.OPENAI_API_KEY, cls.ANTHROPIC_API_KEY, cls.GOOGLE_API_KEY]:
            if key:
                return key
        return None

# Initialize configuration
Config.validate_config()