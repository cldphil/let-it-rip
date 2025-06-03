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
    LLM_MODEL = "claude-3-sonnet-20240229"  # Anthropic Claude 3 Sonnet
    LLM_TEMPERATURE = 0.1  # Low temperature for consistent structured output
    LLM_MAX_TOKENS = 1000  # Max tokens for metadata extraction
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
    
    # Quality Scoring Weights
    ACADEMIC_WEIGHT = 0.4
    PRACTICAL_WEIGHT = 0.4
    RECENCY_WEIGHT = 0.2
    RECENCY_DECAY_RATE = 0.2  # Quality decay per year
    
    # Processing Settings
    BATCH_SIZE = 10  # Number of papers to process in each batch
    MAX_RETRIES = 3  # Retry attempts for failed API calls
    ENABLE_FULL_TEXT = False  # Whether to extract full text by default
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
    
    # Business Tag Categories
    METHODOLOGY_TYPES = [
        "case_study", 
        "theoretical", 
        "empirical", 
        "pilot", 
        "survey",
        "not_specified"
    ]
    
    INDUSTRIES = [
        "healthcare",
        "finance", 
        "retail",
        "manufacturing",
        "education",
        "government",
        "general",
        "other"
    ]
    
    COMPLEXITY_LEVELS = [
        "low",
        "medium", 
        "high",
        "not_specified"
    ]
    
    TEAM_SIZES = [
        "solo",
        "small_team",
        "large_team", 
        "enterprise",
        "not_specified"
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
        "machine_learning",
        "deep_learning",
        "nlp",
        "computer_vision", 
        "cloud_computing",
        "apis",
        "data_engineering",
        "mlops",
        "gpu_required"
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