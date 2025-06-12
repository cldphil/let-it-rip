"""
Configuration settings for the Research Implementation Platform.
Cloud-only configuration optimized for Supabase storage.
"""

import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration settings - Cloud-only with Supabase."""
    
    # API Configuration
    LLM_API_KEY = os.getenv('LLM_API_KEY')
    OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
    ANTHROPIC_API_KEY = os.getenv('ANTHROPIC_API_KEY')
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    
    # Supabase Configuration (Required)
    SUPABASE_URL = os.getenv('SUPABASE_URL')
    SUPABASE_ANON_KEY = os.getenv('SUPABASE_ANON_KEY')
    SUPABASE_SERVICE_ROLE_KEY = os.getenv('SUPABASE_SERVICE_ROLE_KEY')
    
    # Quality Filtering
    MINIMUM_REPUTATION_SCORE = float(os.getenv('MINIMUM_REPUTATION_SCORE', '0.0'))
    
    # LLM Settings
    LLM_MODEL = "claude-sonnet-4-20250514"  # Anthropic Claude Sonnet 4
    LLM_TEMPERATURE = 0.1  # Low temperature for consistent structured output
    LLM_MAX_TOKENS = 2000  # Increased for longer key findings
    LLM_TIMEOUT = 30  # Timeout in seconds
    
    # arXiv API Settings
    ARXIV_BASE_URL = "http://export.arxiv.org/api/query"
    ARXIV_MAX_RESULTS_DEFAULT = 20
    ARXIV_REQUEST_DELAY = 1.0  # Seconds between requests
    ARXIV_TIMEOUT = 30
    
    # Semantic Scholar API Settings (Rate limited)
    SEMANTIC_SCHOLAR_MIN_INTERVAL = 0.2  # Minimum seconds between API calls
    SEMANTIC_SCHOLAR_MAX_RETRIES = 2  # Reduced retries for faster processing
    SEMANTIC_SCHOLAR_TIMEOUT = 10  # Timeout for author lookups
    MAX_AUTHORS_PER_PAPER = 3  # Limit authors to reduce API calls
    CONTINUE_ON_API_ERRORS = True  # Continue processing if author lookup fails
    
    # Processing Settings
    BATCH_SIZE = 5  # Optimized batch size for cloud processing
    MAX_RETRIES = 3  # Retry attempts for failed API calls
    ENABLE_FULL_TEXT = True  # Enable full text extraction
    ENABLE_LLM_TAGGING = True  # Use LLM for extraction
    ENABLE_AUTHOR_LOOKUP = True  # Fetch author h-indices
    
    # Text Extraction Limits
    ABSTRACT_MAX_CHARS = 500
    INTRODUCTION_MAX_CHARS = 1000
    METHODOLOGY_MAX_CHARS = 1500
    RESULTS_MAX_CHARS = 2000
    CONCLUSION_MAX_CHARS = 2000
    
    # Vector Search Configuration
    VECTOR_EMBEDDING_MODEL = 'all-MiniLM-L6-v2'  # SentenceTransformer model
    VECTOR_SIMILARITY_THRESHOLD = 0.6  # Minimum similarity for matches
    MAX_VECTOR_SEARCH_RESULTS = 50  # Maximum results before filtering
    
    # Reputation Scoring Configuration
    RECENCY_WEIGHT = 0.25  # Prioritize recent research
    REPUTATION_WEIGHT = 0.35  # Author h-index + conference validation
    VALIDATION_WEIGHT = 0.20  # Industry validation importance
    CASE_STUDY_WEIGHT = 0.20  # Real-world implementation evidence
    RECENCY_DECAY_RATE = 0.1  # Reputation decay per year (10%)
    
    # Reputation Score Calculation
    CONFERENCE_MULTIPLIER = 1.5  # Multiplier for papers with conference validation
    REPUTATION_SCORE_NORMALIZATION = 100.0  # Divisor for normalizing h-index
    MAX_REPUTATION_SCORE = 100  # Maximum reputation score cap
    MIN_HINDEX_FOR_REPUTATION = 1  # Minimum h-index for reputation bonus
    
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
    
    # Study Type Categories
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
    
    # Success Metrics Categories
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
    
    # Technical Requirements
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
    MIN_FINDING_LENGTH = 100  # Minimum characters per finding
    
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
    
    # Ranking Algorithm Configuration
    RANKING_WEIGHTS = {
        'similarity': 0.30,      # Vector similarity
        'reputation': 0.10,      # Author h-index + conference validation
        'recency': 0.20,         # Time-based relevance
        'findings_richness': 0.30,  # Number and depth of key findings
        'validation_bonus': 0.10    # Industry validation and technique match bonuses
    }
    
    # Synthesis Engine Configuration
    MAX_PAPERS_FOR_SYNTHESIS = 25  # Maximum papers to include in synthesis
    CASE_STUDY_BONUS_THRESHOLD = 0.7  # Minimum case study score for bonus
    VALIDATION_BONUS_THRESHOLD = 0.8  # Minimum validation score for bonus
    CONSERVATIVE_REPUTATION_THRESHOLD = 0.0  # Reputation threshold for conservative users
    
    # Manual Processing Configuration
    DEFAULT_PROCESSING_DAYS = 7  # Default to last 7 days
    MAX_PROCESSING_DAYS = 365  # Maximum days in a single batch
    
    # Rate Limiting and Performance
    SUPABASE_MAX_BATCH_SIZE = 1000  # Maximum records per batch insert
    SUPABASE_REQUEST_TIMEOUT = 30  # Timeout for Supabase requests
    VECTOR_SEARCH_TIMEOUT = 10  # Timeout for vector similarity searches
    
    # Database Table Names (for consistency)
    TABLE_PAPERS = 'papers'
    TABLE_INSIGHTS = 'insights'
    TABLE_EXTRACTION_METADATA = 'extraction_metadata'
    TABLE_PROCESSING_LOGS = 'processing_logs'
    TABLE_USER_CONTEXTS = 'user_contexts'
    
    # Supabase Function Names
    FUNCTION_MATCH_INSIGHTS = 'match_insights'  # Vector similarity search function
    FUNCTION_GET_PAPER_STATS = 'get_paper_statistics'  # Statistics function
    
    # Data Validation Rules
    MAX_TITLE_LENGTH = 500
    MAX_ABSTRACT_LENGTH = 5000
    MAX_FINDING_LENGTH = 1000
    MAX_AUTHORS_DISPLAY = 10
    
    # Processing Limits
    MAX_PAPERS_PER_REQUEST = 1000
    MAX_CONCURRENT_EXTRACTIONS = 3
    MAX_EXTRACTION_TIME_MINUTES = 10
    
    @classmethod
    def validate_config(cls):
        """Validate that required configuration is present."""
        errors = []
        warnings = []
        
        # Check required Supabase configuration
        if not cls.SUPABASE_URL:
            errors.append("SUPABASE_URL is required for cloud-only operation")
        
        if not (cls.SUPABASE_ANON_KEY or cls.SUPABASE_SERVICE_ROLE_KEY):
            errors.append("SUPABASE_ANON_KEY or SUPABASE_SERVICE_ROLE_KEY is required")
        
        # Check LLM API keys
        if cls.ENABLE_LLM_TAGGING and not any([
            cls.LLM_API_KEY, 
            cls.OPENAI_API_KEY, 
            cls.ANTHROPIC_API_KEY,
            cls.GOOGLE_API_KEY
        ]):
            warnings.append("LLM tagging enabled but no API key found. Will fall back to heuristics.")
        
        # Validate reputation score configuration
        if cls.MINIMUM_REPUTATION_SCORE < 0 or cls.MINIMUM_REPUTATION_SCORE > 1:
            warnings.append(f"MINIMUM_REPUTATION_SCORE should be between 0 and 1, got {cls.MINIMUM_REPUTATION_SCORE}")
        
        # Validate date range configuration
        if cls.DEFAULT_PROCESSING_DAYS > cls.MAX_PROCESSING_DAYS:
            warnings.append(f"DEFAULT_PROCESSING_DAYS ({cls.DEFAULT_PROCESSING_DAYS}) exceeds MAX_PROCESSING_DAYS ({cls.MAX_PROCESSING_DAYS})")
        
        # Validate ranking weights
        total_ranking_weight = sum(cls.RANKING_WEIGHTS.values())
        if abs(total_ranking_weight - 1.0) > 0.01:
            warnings.append(f"Ranking weights sum to {total_ranking_weight:.2f}, expected ~1.0")
        
        # Validate batch sizes
        if cls.BATCH_SIZE > cls.SUPABASE_MAX_BATCH_SIZE:
            warnings.append(f"BATCH_SIZE ({cls.BATCH_SIZE}) exceeds SUPABASE_MAX_BATCH_SIZE ({cls.SUPABASE_MAX_BATCH_SIZE})")
        
        # Print validation results
        if errors:
            print("❌ Configuration Errors:")
            for error in errors:
                print(f"   - {error}")
            raise ValueError("Configuration validation failed")
        
        if warnings:
            print("⚠️  Configuration Warnings:")
            for warning in warnings:
                print(f"   - {warning}")
        
        print("✅ Cloud-only configuration validated successfully")
    
    @classmethod
    def get_active_api_key(cls):
        """Get the first available API key."""
        for key in [cls.LLM_API_KEY, cls.ANTHROPIC_API_KEY, cls.OPENAI_API_KEY, cls.GOOGLE_API_KEY]:
            if key:
                return key
        return None
    
    @classmethod
    def get_supabase_config(cls):
        """Get Supabase configuration."""
        return {
            'url': cls.SUPABASE_URL,
            'anon_key': cls.SUPABASE_ANON_KEY,
            'service_role_key': cls.SUPABASE_SERVICE_ROLE_KEY,
            'timeout': cls.SUPABASE_REQUEST_TIMEOUT,
            'max_batch_size': cls.SUPABASE_MAX_BATCH_SIZE,
            'vector_search_timeout': cls.VECTOR_SEARCH_TIMEOUT
        }
    
    @classmethod
    def get_semantic_scholar_config(cls):
        """Get Semantic Scholar API configuration."""
        return {
            'timeout': cls.SEMANTIC_SCHOLAR_TIMEOUT,
            'max_retries': cls.SEMANTIC_SCHOLAR_MAX_RETRIES,
            'min_interval': cls.SEMANTIC_SCHOLAR_MIN_INTERVAL,
            'max_authors': cls.MAX_AUTHORS_PER_PAPER,
            'continue_on_errors': cls.CONTINUE_ON_API_ERRORS
        }
    
    @classmethod
    def get_reputation_score_config(cls):
        """Get reputation score calculation configuration."""
        return {
            'conference_multiplier': cls.CONFERENCE_MULTIPLIER,
            'normalization_factor': cls.REPUTATION_SCORE_NORMALIZATION,
            'max_score': cls.MAX_REPUTATION_SCORE,
            'min_hindex_threshold': cls.MIN_HINDEX_FOR_REPUTATION,
            'minimum_reputation_score': cls.MINIMUM_REPUTATION_SCORE
        }
    
    @classmethod
    def get_ranking_config(cls):
        """Get ranking algorithm configuration."""
        return {
            'weights': cls.RANKING_WEIGHTS,
            'recency_decay_rate': cls.RECENCY_DECAY_RATE,
            'case_study_bonus_threshold': cls.CASE_STUDY_BONUS_THRESHOLD,
            'validation_bonus_threshold': cls.VALIDATION_BONUS_THRESHOLD,
            'conservative_reputation_threshold': cls.CONSERVATIVE_REPUTATION_THRESHOLD
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
    
    @classmethod
    def get_vector_config(cls):
        """Get vector search configuration."""
        return {
            'model': cls.VECTOR_EMBEDDING_MODEL,
            'similarity_threshold': cls.VECTOR_SIMILARITY_THRESHOLD,
            'max_search_results': cls.MAX_VECTOR_SEARCH_RESULTS,
            'timeout': cls.VECTOR_SEARCH_TIMEOUT
        }
    
    @classmethod
    def get_processing_limits(cls):
        """Get processing limits and constraints."""
        return {
            'max_papers_per_request': cls.MAX_PAPERS_PER_REQUEST,
            'max_concurrent_extractions': cls.MAX_CONCURRENT_EXTRACTIONS,
            'max_extraction_time_minutes': cls.MAX_EXTRACTION_TIME_MINUTES,
            'batch_size': cls.BATCH_SIZE,
            'max_retries': cls.MAX_RETRIES
        }
    
    @classmethod
    def get_table_names(cls):
        """Get all table names for database operations."""
        return {
            'papers': cls.TABLE_PAPERS,
            'insights': cls.TABLE_INSIGHTS,
            'extraction_metadata': cls.TABLE_EXTRACTION_METADATA,
            'processing_logs': cls.TABLE_PROCESSING_LOGS,
            'user_contexts': cls.TABLE_USER_CONTEXTS
        }
    
    @classmethod
    def get_supabase_functions(cls):
        """Get Supabase function names."""
        return {
            'match_insights': cls.FUNCTION_MATCH_INSIGHTS,
            'get_paper_stats': cls.FUNCTION_GET_PAPER_STATS
        }
    
    @classmethod
    def print_current_config(cls):
        """Print current configuration summary."""
        print("\n" + "=" * 60)
        print("Cloud-Only Configuration Summary")
        print("=" * 60)
        
        print(f"☁️  Storage: Supabase Cloud")
        print(f"🔍 Min Reputation Score: {cls.MINIMUM_REPUTATION_SCORE}")
        print(f"🤖 LLM Model: {cls.LLM_MODEL}")
        print(f"📈 Author Lookup: {'✅' if cls.ENABLE_AUTHOR_LOOKUP else '❌'}")
        print(f"👥 Max Authors per Paper: {cls.MAX_AUTHORS_PER_PAPER}")
        print(f"⏰ API Request Interval: {cls.SEMANTIC_SCHOLAR_MIN_INTERVAL}s")
        print(f"🔄 Max API Retries: {cls.SEMANTIC_SCHOLAR_MAX_RETRIES}")
        print(f"📦 Batch Size: {cls.BATCH_SIZE}")
        print(f"🛡️  Continue on API Errors: {'✅' if cls.CONTINUE_ON_API_ERRORS else '❌'}")
        print(f"🔧 Vector Model: {cls.VECTOR_EMBEDDING_MODEL}")
        print(f"📊 Max Vector Results: {cls.MAX_VECTOR_SEARCH_RESULTS}")
        
        # Show Supabase connection status
        supabase_configured = bool(cls.SUPABASE_URL and (cls.SUPABASE_ANON_KEY or cls.SUPABASE_SERVICE_ROLE_KEY))
        print(f"🗄️  Supabase Configured: {'✅' if supabase_configured else '❌'}")
        
        print("=" * 60)
    
    @classmethod
    def get_data_validation_rules(cls):
        """Get data validation rules for input sanitization."""
        return {
            'max_title_length': cls.MAX_TITLE_LENGTH,
            'max_abstract_length': cls.MAX_ABSTRACT_LENGTH,
            'max_finding_length': cls.MAX_FINDING_LENGTH,
            'max_authors_display': cls.MAX_AUTHORS_DISPLAY,
            'max_key_findings': cls.MAX_KEY_FINDINGS,
            'min_finding_length': cls.MIN_FINDING_LENGTH
        }
    
    @classmethod
    def is_cloud_ready(cls) -> bool:
        """Check if configuration is ready for cloud operations."""
        return bool(
            cls.SUPABASE_URL and 
            (cls.SUPABASE_ANON_KEY or cls.SUPABASE_SERVICE_ROLE_KEY) and
            cls.get_active_api_key()
        )

# Initialize and validate configuration on import
Config.validate_config()