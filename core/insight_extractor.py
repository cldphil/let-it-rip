"""
Enhanced insight extractor with intelligent section detection.
Focuses on extracting actionable insights from relevant paper sections.
"""

import re
import json
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import anthropic
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'research_tools'))
from services.semantic_scholar_hidx import SemanticScholarAPI

from .insight_schema import (
    PaperInsights, StudyType, TechniqueCategory, 
    ComplexityLevel, ExtractionMetadata
)
from config import Config

logger = logging.getLogger(__name__)

# Enhanced technique mapping for the extractor
TECHNIQUE_MAPPING = {
    # Core GenAI (existing)
    'rag': TechniqueCategory.RAG,
    'retrieval_augmented_generation': TechniqueCategory.RAG,
    'retrieval-augmented': TechniqueCategory.RAG,
    'fine_tuning': TechniqueCategory.FINE_TUNING,
    'fine-tuning': TechniqueCategory.FINE_TUNING,
    'finetuning': TechniqueCategory.FINE_TUNING,
    'prompt_engineering': TechniqueCategory.PROMPT_ENGINEERING,
    'prompt-engineering': TechniqueCategory.PROMPT_ENGINEERING,
    'prompting': TechniqueCategory.PROMPT_ENGINEERING,
    'multi_agent': TechniqueCategory.MULTI_AGENT,
    'multi-agent': TechniqueCategory.MULTI_AGENT,
    'multiagent': TechniqueCategory.MULTI_AGENT,
    'chain_of_thought': TechniqueCategory.CHAIN_OF_THOUGHT,
    'chain-of-thought': TechniqueCategory.CHAIN_OF_THOUGHT,
    'cot': TechniqueCategory.CHAIN_OF_THOUGHT,
    'reinforcement_learning': TechniqueCategory.REINFORCEMENT_LEARNING,
    'reinforcement-learning': TechniqueCategory.REINFORCEMENT_LEARNING,
    'rlhf': TechniqueCategory.REINFORCEMENT_LEARNING,
    'few_shot_learning': TechniqueCategory.FEW_SHOT_LEARNING,
    'few-shot': TechniqueCategory.FEW_SHOT_LEARNING,
    'few shot': TechniqueCategory.FEW_SHOT_LEARNING,
    'zero_shot_learning': TechniqueCategory.ZERO_SHOT_LEARNING,
    'zero-shot': TechniqueCategory.ZERO_SHOT_LEARNING,
    'zero shot': TechniqueCategory.ZERO_SHOT_LEARNING,
    'transfer_learning': TechniqueCategory.TRANSFER_LEARNING,
    'transfer-learning': TechniqueCategory.TRANSFER_LEARNING,
    'ensemble_methods': TechniqueCategory.ENSEMBLE_METHODS,
    'ensemble': TechniqueCategory.ENSEMBLE_METHODS,
    
    # Modern Generative Models
    'diffusion_models': TechniqueCategory.DIFFUSION_MODELS,
    'diffusion': TechniqueCategory.DIFFUSION_MODELS,
    'flow_matching': TechniqueCategory.FLOW_MATCHING,
    'autoregressive_modeling': TechniqueCategory.AUTOREGRESSIVE_MODELING,
    'autoregressive': TechniqueCategory.AUTOREGRESSIVE_MODELING,
    'variational_inference': TechniqueCategory.VARIATIONAL_INFERENCE,
    'energy_based_modeling': TechniqueCategory.ENERGY_BASED_MODELING,
    'score_matching': TechniqueCategory.ENERGY_BASED_MODELING,
    'denoising_objectives': TechniqueCategory.DIFFUSION_MODELS,
    
    # Multimodal & Vision-Language
    'multimodal_learning': TechniqueCategory.MULTIMODAL_LEARNING,
    'multimodal': TechniqueCategory.MULTIMODAL_LEARNING,
    'vision_language_modeling': TechniqueCategory.VISION_LANGUAGE_MODELING,
    'text_to_3d_synthesis': TechniqueCategory.TEXT_TO_3D_SYNTHESIS,
    'text-to-3d': TechniqueCategory.TEXT_TO_3D_SYNTHESIS,
    'semantic_segmentation': TechniqueCategory.SEMANTIC_SEGMENTATION,
    'object_detection': TechniqueCategory.OBJECT_DETECTION,
    'caption_generation': TechniqueCategory.VISION_LANGUAGE_MODELING,
    'gaussian_splatting': TechniqueCategory.GAUSSIAN_SPLATTING,
    '3d_gaussian_splatting': TechniqueCategory.GAUSSIAN_SPLATTING,
    '3d_reconstruction': TechniqueCategory.GAUSSIAN_SPLATTING,
    
    # Advanced Training & Optimization
    'contrastive_learning': TechniqueCategory.CONTRASTIVE_LEARNING,
    'self_supervised_learning': TechniqueCategory.SELF_SUPERVISED_LEARNING,
    'multi_task_learning': TechniqueCategory.MULTI_TASK_LEARNING,
    'preference_optimization': TechniqueCategory.PREFERENCE_OPTIMIZATION,
    'reward_modeling': TechniqueCategory.REWARD_MODELING,
    'low_rank_adaptation': TechniqueCategory.LOW_RANK_ADAPTATION,
    'lora': TechniqueCategory.LOW_RANK_ADAPTATION,
    'multi_stage_training': TechniqueCategory.MULTI_TASK_LEARNING,
    'two_stage_training': TechniqueCategory.MULTI_TASK_LEARNING,
    
    # Architecture & Attention
    'attention_mechanisms': TechniqueCategory.ATTENTION_MECHANISMS,
    'attention_analysis': TechniqueCategory.ATTENTION_MECHANISMS,
    'cross_attention': TechniqueCategory.CROSS_ATTENTION,
    'transformer_architecture': TechniqueCategory.TRANSFORMER_ARCHITECTURE,
    'memory_augmentation': TechniqueCategory.MEMORY_AUGMENTATION,
    'attention_replacement': TechniqueCategory.ATTENTION_MECHANISMS,
    'architectural_modification': TechniqueCategory.TRANSFORMER_ARCHITECTURE,
    
    # Inference & Optimization
    'inference_optimization': TechniqueCategory.INFERENCE_OPTIMIZATION,
    'kv_cache_optimization': TechniqueCategory.KV_CACHE_OPTIMIZATION,
    'cache_compression': TechniqueCategory.KV_CACHE_OPTIMIZATION,
    'approximate_attention': TechniqueCategory.APPROXIMATE_ATTENTION,
    'dynamic_sparsification': TechniqueCategory.DYNAMIC_SPARSIFICATION,
    'sparsity_exploitation': TechniqueCategory.DYNAMIC_SPARSIFICATION,
    'model_parallelization': TechniqueCategory.MODEL_PARALLELIZATION,
    'context_compression': TechniqueCategory.INFERENCE_OPTIMIZATION,
    'dynamic_thresholding': TechniqueCategory.INFERENCE_OPTIMIZATION,
    
    # Reasoning & Decision Making
    'self_reasoning': TechniqueCategory.SELF_REASONING,
    'self_guided_reasoning': TechniqueCategory.SELF_REASONING,
    'adaptive_decision_making': TechniqueCategory.ADAPTIVE_DECISION_MAKING,
    'causal_decoding': TechniqueCategory.CAUSAL_DECODING,
    'self_correction': TechniqueCategory.SELF_CORRECTION,
    'self_consistency': TechniqueCategory.SELF_CORRECTION,
    'self_reflection': TechniqueCategory.SELF_REASONING,
    'in_context_learning': TechniqueCategory.IN_CONTEXT_LEARNING,
    'multi_stage_reasoning': TechniqueCategory.SELF_REASONING,
    'verifier_feedback': TechniqueCategory.SELF_CORRECTION,
    'best_of_n_sampling': TechniqueCategory.INFERENCE_OPTIMIZATION,
    
    # Safety & Alignment
    'safety_alignment': TechniqueCategory.SAFETY_ALIGNMENT,
    'human_preference_evaluation': TechniqueCategory.HUMAN_PREFERENCE_EVALUATION,
    'constitutional_interpretation': TechniqueCategory.CONSTITUTIONAL_AI,
    
    # Data & Preprocessing
    'data_synthesis': TechniqueCategory.DATA_SYNTHESIS,
    'synthetic_data_generation': TechniqueCategory.SYNTHETIC_DATA_GENERATION,
    'web_scraping': TechniqueCategory.WEB_SCRAPING,
    'web_search_integration': TechniqueCategory.WEB_SCRAPING,
    'document_parsing': TechniqueCategory.DOCUMENT_PARSING,
    'text_parsing': TechniqueCategory.DOCUMENT_PARSING,
    'multi_modal_processing': TechniqueCategory.MULTIMODAL_LEARNING,
    'preprocessing_pipelines': TechniqueCategory.DOCUMENT_PARSING,
    'data_preprocessing': TechniqueCategory.DOCUMENT_PARSING,
    'data_selection': TechniqueCategory.DATA_SYNTHESIS,
    'adaptive_masking': TechniqueCategory.SELF_SUPERVISED_LEARNING,
    
    # All the specific techniques from your error log
    'representation_similarity_analysis': TechniqueCategory.CONTRASTIVE_LEARNING,
    'spatial_reasoning': TechniqueCategory.SELF_REASONING,
    'numerical_layout_generation': TechniqueCategory.TEXT_TO_3D_SYNTHESIS,
    'spatio_temporal_grounding': TechniqueCategory.MULTIMODAL_LEARNING,
    'sequential_decomposition': TechniqueCategory.SELF_REASONING,
    'sparse_sampling': TechniqueCategory.DYNAMIC_SPARSIFICATION,
    'grafting': TechniqueCategory.TRANSFER_LEARNING,
    'token_interleaving': TechniqueCategory.TRANSFORMER_ARCHITECTURE,
    'foundation_models': TechniqueCategory.TRANSFER_LEARNING,
    'rollout_replay': TechniqueCategory.REINFORCEMENT_LEARNING,
    'constrained_optimization': TechniqueCategory.PREFERENCE_OPTIMIZATION,
    'primal_dual_algorithms': TechniqueCategory.PREFERENCE_OPTIMIZATION,
    'logit_margin_flattening': TechniqueCategory.INFERENCE_OPTIMIZATION,
    'power_law_modeling': TechniqueCategory.TRANSFORMER_ARCHITECTURE,
    'benchmark_development': TechniqueCategory.OTHER,
    'human_annotation': TechniqueCategory.HUMAN_PREFERENCE_EVALUATION,
    'temporal_evaluation_metrics': TechniqueCategory.OTHER,
    'hierarchical_action_space': TechniqueCategory.REINFORCEMENT_LEARNING,
    'knowledge_decomposition': TechniqueCategory.SELF_REASONING,
    'confidence_weighting': TechniqueCategory.PREFERENCE_OPTIMIZATION,
    'classifier_guidance': TechniqueCategory.DIFFUSION_MODELS,
    'latent_space_optimization': TechniqueCategory.VARIATIONAL_INFERENCE,
    'natural_language_inference': TechniqueCategory.SELF_REASONING,
    'neuron_encryption': TechniqueCategory.SAFETY_ALIGNMENT,
    'selective_decryption': TechniqueCategory.SAFETY_ALIGNMENT,
    'task_specific_neuron_extraction': TechniqueCategory.TRANSFER_LEARNING,
    'ai_augmented_optimization': TechniqueCategory.REINFORCEMENT_LEARNING,
    'team_simulation': TechniqueCategory.MULTI_AGENT,
    'behavioral_modeling': TechniqueCategory.MULTI_AGENT,
    'mcmc': TechniqueCategory.VARIATIONAL_INFERENCE,
    'parallel_tempering': TechniqueCategory.VARIATIONAL_INFERENCE,
    'neural_samplers': TechniqueCategory.VARIATIONAL_INFERENCE,
    'structured_decomposition': TechniqueCategory.SELF_REASONING,
    'legal_precedent_analysis': TechniqueCategory.LEGAL_AI,
    'policy_analysis': TechniqueCategory.LEGAL_AI,
    'dual_state_modeling': TechniqueCategory.GAUSSIAN_SPLATTING,
    'photometric_consistency': TechniqueCategory.GAUSSIAN_SPLATTING,
    'search_algorithms': TechniqueCategory.ADAPTIVE_DECISION_MAKING,
    'universal_approximation': TechniqueCategory.TRANSFORMER_ARCHITECTURE,
    'cross_modal_evaluation': TechniqueCategory.MULTIMODAL_LEARNING,
    'memorization_quantification': TechniqueCategory.OTHER,
    'tree_sampling': TechniqueCategory.ADAPTIVE_DECISION_MAKING,
    'reward_optimization': TechniqueCategory.REWARD_MODELING,
    'model_merging': TechniqueCategory.ENSEMBLE_METHODS,
    'dynamical_systems_theory': TechniqueCategory.OTHER,
    'morse_smale_analysis': TechniqueCategory.OTHER,
    'gradient_flow_methods': TechniqueCategory.OTHER,
    'neural_networks': TechniqueCategory.TRANSFORMER_ARCHITECTURE,
    'stochastic_differential_equations': TechniqueCategory.OTHER,
    'gumbel_softmax': TechniqueCategory.VARIATIONAL_INFERENCE,
    'evidentiality_assessment': TechniqueCategory.SAFETY_ALIGNMENT,
    'constrained_decoding': TechniqueCategory.INFERENCE_OPTIMIZATION,
    'comparative_evaluation': TechniqueCategory.OTHER,
    'multimodal_analysis': TechniqueCategory.MULTIMODAL_LEARNING,
    'controlled_error_injection': TechniqueCategory.OTHER,
    'mechanistic_interpretability': TechniqueCategory.OTHER,
    'systematic_ablation': TechniqueCategory.OTHER,
    'edge_analysis': TechniqueCategory.OTHER,
    'prefix_tokens': TechniqueCategory.PROMPT_ENGINEERING,
    
    # Fallback
    'other': TechniqueCategory.OTHER
}

class InsightExtractor:
    """
    Simplified but intelligent insight extractor that identifies and extracts
    from the most relevant sections of research papers.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize extractor with API client."""
        self.api_key = api_key or Config.get_active_api_key()
        if not self.api_key:
            raise ValueError("No API key found. Please set ANTHROPIC_API_KEY in .env")
        
        self.client = anthropic.Anthropic(api_key=self.api_key)
        
        # Initialize Semantic Scholar API for author metrics and conference detection
        self.semantic_scholar = SemanticScholarAPI()
        
        # Section patterns for intelligent detection
        self.section_patterns = {
            'problem_context': {
                'keywords': ['abstract', 'introduction', 'motivation', 'background', 
                            'problem statement', 'overview', 'summary'],
                'priority': 1,
                'max_length': 1200,
                'required': True
            },
            'methodology': {
                'keywords': ['method', 'methodology', 'approach', 'architecture', 'design', 
                            'implementation', 'system', 'framework', 'model', 'algorithm',
                            'technique', 'solution', 'our approach', 'proposed method'],
                'priority': 2,
                'max_length': 1500,
                'required': True
            },
            'results': {
                'keywords': ['results', 'evaluation', 'experiments', 'findings', 'performance',
                            'outcomes', 'analysis', 'comparison', 'benchmarks', 'metrics',
                            'assessment', 'validation'],
                'priority': 3,
                'max_length': 1200,
                'required': False
            },
            'case_study': {
                'keywords': ['case study', 'case-study', 'deployment', 'real-world', 
                            'application', 'use case', 'pilot', 'production', 'industrial',
                            'commercial', 'field study', 'practical', 'in practice',
                            'company', 'organization', 'enterprise'],
                'priority': 4,
                'max_length': 1500,
                'required': False
            },
            'limitations': {
                'keywords': ['limitation', 'limitations', 'challenge', 'constraint', 
                            'future work', 'discussion', 'threats to validity', 
                            'lessons learned', 'drawback', 'weakness', 'open questions'],
                'priority': 5,
                'max_length': 800,
                'required': False
            }
        }
    
    def _fetch_author_metrics_safe(self, paper: Dict, insights: PaperInsights, 
                               metadata: ExtractionMetadata):
        """
        Safely fetch author metrics with proper error handling.
        
        Args:
            paper: Paper data
            insights: Insights object to update
            metadata: Metadata object for error tracking
        """
        try:
            # Limit authors to process
            authors_to_process = paper.get('authors', [])[:Config.MAX_AUTHORS_PER_PAPER]
            
            if len(paper.get('authors', [])) > Config.MAX_AUTHORS_PER_PAPER:
                logger.info(f"Limiting author lookup to first {Config.MAX_AUTHORS_PER_PAPER} of {len(paper.get('authors', []))} authors")
            
            # Get author h-indices with error handling
            try:
                total_hindex, individual_hindices = self.semantic_scholar.get_paper_total_hindex(authors_to_process)
                insights.total_author_hindex = total_hindex
                insights.author_hindices = individual_hindices
                logger.info(f"Total author h-index: {total_hindex}")
            except Exception as e:
                logger.warning(f"Author h-index lookup failed: {e}")
                insights.total_author_hindex = 0
                insights.author_hindices = {}
                metadata.extraction_errors.append(f"Author lookup failed: {str(e)}")
                
                # Continue on error if configured
                if not Config.CONTINUE_ON_API_ERRORS:
                    raise
            
            # Detect conference/workshop with error handling
            try:
                is_conference = self.semantic_scholar.detect_conference_mention(paper)
                insights.has_conference_mention = is_conference
                logger.info(f"Conference detected: {is_conference}")
            except Exception as e:
                logger.warning(f"Conference detection failed: {e}")
                insights.has_conference_mention = False
                metadata.extraction_errors.append(f"Conference detection failed: {str(e)}")
                
                # Continue on error if configured
                if not Config.CONTINUE_ON_API_ERRORS:
                    raise
                    
        except Exception as e:
            logger.error(f"Error in author metrics fetch: {e}")
            # Set defaults
            insights.total_author_hindex = 0
            insights.has_conference_mention = False
            insights.author_hindices = {}
            
            # Only raise if we're not configured to continue on errors
            if not Config.CONTINUE_ON_API_ERRORS:
                raise

    def extract_insights(self, paper: Dict) -> Tuple[PaperInsights, ExtractionMetadata]:
        """
        Extract insights using intelligent section detection.
        
        Args:
            paper: Paper data with title, abstract, and optionally full_text
            
        Returns:
            Tuple of (PaperInsights, ExtractionMetadata)
        """
        start_time = datetime.utcnow()
        metadata = ExtractionMetadata(
            extraction_id=f"{paper.get('id', 'unknown')}_{start_time.timestamp()}",
            paper_id=paper.get('id', 'unknown'),
            extractor_version="2.1",  # Updated version
            llm_model=Config.LLM_MODEL,
            llm_temperature=Config.LLM_TEMPERATURE,
            extraction_time_seconds=0.0,
            api_calls_made=0
        )
        
        try:
            # Determine if we have full text or just abstract
            has_full_text = bool(paper.get('full_text'))
            
            if has_full_text:
                # Extract relevant sections intelligently
                logger.info(f"Extracting sections from full text for: {paper.get('title', '')[:50]}...")
                sections = self._extract_relevant_sections(paper.get('full_text', ''))
                
                # Update metadata with sections found
                metadata.sections_found = {
                    section: len(content) > 0 
                    for section, content in sections.items()
                }
                metadata.section_lengths = {
                    section: len(content) 
                    for section, content in sections.items()
                }
            else:
                # Use abstract and title only
                logger.info(f"Using abstract-only extraction for: {paper.get('title', '')[:50]}...")
                sections = {
                    'problem_context': f"Title: {paper.get('title', '')}\n\nAbstract: {paper.get('summary', '')}",
                }
            
            # Detect if this is a case study
            is_case_study = self._detect_case_study(paper, sections)
            
            # Extract insights using LLM
            insights_data = self._extract_insights_from_sections(
                paper, 
                sections, 
                is_case_study
            )
            metadata.api_calls_made += 1
            
            # Create insights object
            insights = self._create_insights_object(insights_data, paper.get('id', ''))
            
            # Fetch author h-indices and detect conference with improved error handling
            logger.info("Fetching author metrics and conference detection...")
            try:
                # Check if API has the status method (backward compatibility)
                if hasattr(self.semantic_scholar, 'get_api_status'):
                    api_status = self.semantic_scholar.get_api_status()
                    
                    if api_status['consecutive_failures'] >= api_status['max_consecutive_failures']:
                        logger.warning("Skipping author lookup due to API rate limit issues")
                        insights.total_author_hindex = 0
                        insights.has_conference_mention = False
                        
                        # Add note about skipped lookup
                        metadata.extraction_errors.append("Author h-index lookup skipped due to API rate limits")
                    else:
                        # Proceed with author lookup
                        self._fetch_author_metrics_safe(paper, insights, metadata)
                else:
                    # Old API version without status tracking - just try to fetch
                    self._fetch_author_metrics_safe(paper, insights, metadata)
                
                # Log the final reputation score
                reputation_score = insights.get_reputation_score()
                logger.info(f"Calculated reputation score: {reputation_score:.3f}")
                
            except Exception as e:
                logger.error(f"Failed to fetch author metrics or conference detection: {e}")
                # Set defaults and continue
                insights.total_author_hindex = 0
                insights.has_conference_mention = False
                insights.author_hindices = {}
                metadata.extraction_errors.append(f"Author metrics extraction failed: {str(e)}")
            
            # Set extraction confidence based on sections found
            if has_full_text:
                required_sections_found = sum(1 for name in ['problem_context', 'methodology'] 
                                            if sections.get(name))
                total_sections_found = len([s for s in sections.values() if s])
                insights.extraction_confidence = min(0.9, 0.4 + (total_sections_found * 0.15))
            else:
                insights.extraction_confidence = 0.5  # Lower confidence for abstract-only
            
            # Calculate extraction time and cost
            extraction_time = (datetime.utcnow() - start_time).total_seconds()
            metadata.extraction_time_seconds = extraction_time
            metadata.estimated_cost_usd = metadata.api_calls_made * 0.008  # ~$0.008 per call
            
            logger.info(f"Extraction complete in {extraction_time:.2f}s")
            
            return insights, metadata
            
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            metadata.extraction_errors.append(str(e))
            return self._create_minimal_insights(paper), metadata
    
    def _extract_relevant_sections(self, full_text: str) -> Dict[str, str]:
        """
        Intelligently identify and extract relevant sections using semantic matching.
        """
        sections = {}
        text_lower = full_text.lower()
        
        # Clean and normalize text
        text = re.sub(r'\s+', ' ', full_text)
        
        # Try to identify section boundaries
        section_matches = []
        
        # Look for numbered sections (e.g., "1. Introduction", "2. Methods")
        numbered_pattern = r'(?:^|\n)\s*(\d+\.?\s*[A-Z][a-z]+.*?)(?=\n)'
        numbered_sections = re.finditer(numbered_pattern, text, re.MULTILINE)
        
        # Look for bold/emphasized sections (e.g., "**Methods**", "### Results")
        emphasized_pattern = r'(?:^|\n)\s*(?:\*\*|###?)\s*([A-Z][a-z]+.*?)(?:\*\*|###?)?\s*(?=\n)'
        emphasized_sections = re.finditer(emphasized_pattern, text, re.MULTILINE)
        
        # Combine all section markers
        for match in numbered_sections:
            section_matches.append((match.start(), match.group(1).strip()))
        for match in emphasized_sections:
            section_matches.append((match.start(), match.group(1).strip()))
        
        # Sort by position
        section_matches.sort(key=lambda x: x[0])
        
        # Extract content for each section type
        for section_type, config in self.section_patterns.items():
            found_content = ""
            
            # First, try to find sections by title matching
            for i, (pos, section_title) in enumerate(section_matches):
                title_lower = section_title.lower()
                
                # Check if this section title matches our keywords
                if any(keyword in title_lower for keyword in config['keywords']):
                    # Extract content until next section or end
                    start_pos = pos
                    end_pos = section_matches[i + 1][0] if i + 1 < len(section_matches) else len(text)
                    
                    content = text[start_pos:end_pos].strip()
                    
                    # Remove the section title from content
                    content = re.sub(r'^.*?\n', '', content, count=1).strip()
                    
                    # Limit length
                    if len(content) > config['max_length']:
                        content = content[:config['max_length']] + "..."
                    
                    found_content = content
                    break
            
            # If not found by section title, try context-based extraction
            if not found_content and config.get('required', False):
                found_content = self._extract_by_context(text, text_lower, config['keywords'], config['max_length'])
            
            if found_content:
                sections[section_type] = found_content
        
        # Special handling for case studies - might be embedded in other sections
        if 'case_study' not in sections:
            case_study_content = self._find_case_study_content(text, text_lower)
            if case_study_content:
                sections['case_study'] = case_study_content
        
        return sections
    
    def _extract_by_context(self, text: str, text_lower: str, keywords: List[str], max_length: int) -> str:
        """
        Extract content based on contextual clues when section headers aren't found.
        """
        best_match = ""
        best_score = 0
        
        # Split into paragraphs
        paragraphs = text.split('\n\n')
        
        for i, para in enumerate(paragraphs):
            if len(para) < 50:  # Skip very short paragraphs
                continue
            
            para_lower = para.lower()
            
            # Score based on keyword density
            score = sum(1 for keyword in keywords if keyword in para_lower)
            
            # Bonus for certain indicator phrases
            if any(phrase in para_lower for phrase in ['we present', 'we propose', 'our approach', 'this paper']):
                score += 2
            
            if score > best_score:
                best_score = score
                # Include some context (previous and next paragraph if available)
                context_start = max(0, i - 1)
                context_end = min(len(paragraphs), i + 2)
                best_match = '\n\n'.join(paragraphs[context_start:context_end])
        
        if best_match and len(best_match) > max_length:
            best_match = best_match[:max_length] + "..."
        
        return best_match
    
    def _find_case_study_content(self, text: str, text_lower: str) -> str:
        """
        Special logic to find case study content that might be embedded.
        """
        case_indicators = [
            r'(?:case study|field study|pilot study|real-world deployment)',
            r'(?:company|organization|enterprise|industry partner)',
            r'(?:deployed at|implemented at|used by|adopted by)',
            r'(?:production system|live system|operational)'
        ]
        
        best_excerpt = ""
        
        for indicator in case_indicators:
            matches = list(re.finditer(indicator, text_lower))
            for match in matches:
                # Extract surrounding context
                start = max(0, match.start() - 200)
                end = min(len(text), match.end() + 800)
                excerpt = text[start:end].strip()
                
                # Look for reputation indicators
                if any(term in excerpt.lower() for term in ['results', 'performance', 'improved', 'reduced']):
                    if len(excerpt) > len(best_excerpt):
                        best_excerpt = excerpt
        
        if best_excerpt and len(best_excerpt) > 1500:
            best_excerpt = best_excerpt[:1500] + "..."
        
        return best_excerpt
    
    def _detect_case_study(self, paper: Dict, sections: Dict) -> bool:
        """
        Detect if this paper contains a case study with more stringent criteria.
        A paper is only marked as a case study if it describes an actual implementation
        at a specific organization with concrete results.
        """
        # Check title and abstract
        title_abstract = f"{paper.get('title', '')} {paper.get('summary', '')}".lower()
        
        # Strong case study indicators - must have explicit case study mention
        strong_indicators = [
            'case study', 'case-study', 'field study', 'industrial case study',
            'case study at', 'deployment case study', 'real-world case study'
        ]
        
        # Weak indicators that need additional evidence
        weak_indicators = [
            'deployment', 'production', 'real-world', 'industrial application',
            'company', 'organization', 'enterprise', 'pilot study'
        ]
        
        # Implementation evidence - needed to confirm it's an actual case study
        implementation_evidence = [
            'deployed at', 'implemented at', 'used by', 'adopted by',
            'operational at', 'in production at', 'live system',
            'fortune 500', 'fortune 1000', 'employees use',
            'customers use', 'users daily', 'requests per day',
            'deployed system', 'production deployment'
        ]
        
        # Check for strong indicators in title
        title = paper.get('title', '').lower()
        has_strong_in_title = any(indicator in title for indicator in strong_indicators)
        
        # Check for case study section
        has_case_study_section = bool(sections.get('case_study'))
        
        # Count evidence across all text
        all_text = title_abstract
        for section_content in sections.values():
            all_text += " " + section_content.lower()
        
        # Count weak indicators
        weak_indicator_count = sum(1 for indicator in weak_indicators if indicator in all_text)
        
        # Count implementation evidence
        implementation_count = sum(1 for evidence in implementation_evidence if evidence in all_text)
        
        # Decision logic - much more stringent
        if has_strong_in_title:
            # If "case study" is in the title, likely a real case study
            return True
        
        if has_case_study_section and implementation_count >= 2:
            # Has dedicated case study section with implementation evidence
            return True
        
        # Need strong evidence of actual deployment
        if weak_indicator_count >= 3 and implementation_count >= 3:
            # Multiple weak indicators AND strong implementation evidence
            return True
        
        # Check for specific patterns that indicate real deployment
        deployment_patterns = [
            r'deployed (?:at|in|to) \w+ (?:company|organization|corporation)',
            r'implemented at \w+',
            r'case study (?:at|of|from) \w+',
            r'\d+[,\d]* (?:users|employees|customers)',
            r'\d+[,\d]* (?:requests|queries|transactions) per',
            r'fortune \d+',
            r'production (?:system|deployment|environment) at'
        ]
        
        import re
        pattern_matches = sum(1 for pattern in deployment_patterns 
                            if re.search(pattern, all_text, re.IGNORECASE))
        
        if pattern_matches >= 2:
            return True
        
        return False
    
    def _extract_insights_from_sections(self, paper: Dict, sections: Dict, is_case_study: bool) -> Dict:
        """
        Extract insights from identified sections using LLM.
        """
        # Prepare sections text
        sections_text = ""
        for section_name, content in sections.items():
            if content:
                section_title = section_name.replace('_', ' ').title()
                sections_text += f"\n[{section_title}]\n{content}\n"
        
        # Build prompt with case study awareness
        prompt = f"""Analyze this research paper and extract actionable insights.

Title: {paper.get('title', '')}
Paper Type: {'Case Study' if is_case_study else 'Research Paper'}

{sections_text}

Extract comprehensive insights and return ONLY a JSON object with this exact format (no markdown, no extra text):
{{
    "key_findings": [
        "Detailed finding 1 - Be specific about methods, results, and implications. Include quantitative improvements where mentioned. At least 25 words",
        "Detailed finding 2 - Focus on practical applications and real-world impact. Mention specific techniques or innovations. At least 25 words.",
        "Detailed finding 3 - Highlight unique contributions or novel approaches. Include performance metrics if available. At least 50 words.",
        "Detailed finding 4 - Discuss implementation details, requirements, or deployment considerations. At least 75 words.",
        "Detailed finding 5 - Cover broader implications, lessons learned, or future directions. At least 25 words."
    ],
    "limitations": ["Key limitation 1", "Key limitation 2"],
    "study_type": "{self._infer_study_type(is_case_study, sections)}",
    "techniques_used": ["rag", "fine_tuning", "prompt_engineering"],
    "implementation_complexity": "low",
    "problem_addressed": "What specific problem does this solve?",
    "prerequisites": ["Technical requirements"],
    "real_world_applications": ["Specific use cases mentioned"],
    "has_code_available": false,
    "has_dataset_available": false,
}}

{self._get_case_study_instructions() if is_case_study else ''}

IMPORTANT: 
1. Focus on extracting PRACTICAL, IMPLEMENTABLE insights that practitioners can use.
2. Return ONLY the JSON object, no markdown formatting, no explanations.
3. Ensure all JSON values are properly formatted (strings in quotes, numbers without quotes).
4. For implementation_complexity use one of: "low", "medium", "high", "very_high".
5. Write 25-100 words for each Key Finding, ensuring that every pracitcal, applicable detail is summarized from the text.
"""

        try:
            response = self.client.messages.create(
                model=Config.LLM_MODEL,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            # Extract JSON from response, handling potential markdown formatting
            response_text = response.content[0].text
            
            # Try to extract JSON from markdown code blocks if present
            import re
            json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response_text, re.DOTALL)
            if json_match:
                json_text = json_match.group(1)
            else:
                # Try to find JSON directly
                json_match = re.search(r'(\{.*\})', response_text, re.DOTALL)
                if json_match:
                    json_text = json_match.group(1)
                else:
                    json_text = response_text
            
            result = json.loads(json_text)
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing failed: {e}")
            logger.error(f"Response text: {response_text[:500]}...")  # Log first 500 chars
            return self._create_fallback_insights(paper, sections, is_case_study)
        except Exception as e:
            logger.error(f"LLM extraction failed: {e}")
            return self._create_fallback_insights(paper, sections, is_case_study)
    
    def _get_case_study_instructions(self) -> str:
        """Additional instructions for case study papers."""
        return """
            CASE STUDY SPECIFIC: Please pay special attention to:
            - Organization/company involved (if mentioned)
            - Scale of deployment (users, requests, data volume)
            - Specific business outcomes achieved
            - Implementation timeline and phases
            - Challenges faced during deployment
            - Lessons learned from real-world usage"""
    
    def _infer_study_type(self, is_case_study: bool, sections: Dict) -> str:
        """Infer study type based on content with stricter case study criteria."""
        if is_case_study:
            return "case_study"
        
        # Check for other study types based on section content
        all_text = ' '.join(sections.values()).lower()
        
        # Strong empirical indicators
        empirical_indicators = [
            'experiment', 'benchmark', 'evaluation', 'measured', 'tested',
            'results show', 'performance evaluation', 'empirical study',
            'quantitative results', 'statistical analysis', 'hypothesis test'
        ]
        empirical_count = sum(1 for indicator in empirical_indicators if indicator in all_text)
        
        # Pilot study indicators (not a full case study)
        pilot_indicators = [
            'pilot', 'trial', 'prototype', 'proof of concept', 'poc',
            'preliminary implementation', 'initial deployment', 'early results'
        ]
        pilot_count = sum(1 for indicator in pilot_indicators if indicator in all_text)
        
        # Survey/review indicators
        survey_indicators = [
            'survey', 'review', 'analysis of', 'systematic review',
            'literature review', 'meta-analysis', 'comparison of approaches',
            'overview of methods', 'state of the art'
        ]
        survey_count = sum(1 for indicator in survey_indicators if indicator in all_text)
        
        # Theoretical indicators
        theoretical_indicators = [
            'framework', 'theoretical', 'model', 'formalization',
            'mathematical proof', 'theorem', 'conceptual', 'abstract model',
            'formal analysis', 'theoretical contribution'
        ]
        theoretical_count = sum(1 for indicator in theoretical_indicators if indicator in all_text)
        
        # Determine study type based on strongest evidence
        max_count = max(empirical_count, pilot_count, survey_count, theoretical_count)
        
        if max_count == 0:
            return "unknown"
        
        if empirical_count == max_count:
            return "empirical"
        elif pilot_count == max_count:
            return "pilot"
        elif survey_count == max_count:
            # Distinguish between survey and review
            if 'meta-analysis' in all_text or 'meta analysis' in all_text:
                return "meta_analysis"
            elif 'survey' in all_text and 'participants' in all_text:
                return "survey"
            else:
                return "review"
        elif theoretical_count == max_count:
            return "theoretical"
        
        return "unknown"
    
    def _create_fallback_insights(self, paper: Dict, sections: Dict, is_case_study: bool) -> Dict:
        """Create basic insights when LLM extraction fails."""
        # Extract key findings from abstract or sections
        key_findings = []
        
        abstract = paper.get('summary', '')
        if abstract:
            sentences = abstract.split('. ')
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in 
                      ['achieve', 'improve', 'result', 'show', 'demonstrate', 'find']):
                    key_findings.append(sentence.strip() + '.')
        
        # Add findings from sections if available
        if sections.get('results'):
            results_sentences = sections['results'].split('. ')[:2]
            key_findings.extend([s.strip() + '.' for s in results_sentences if s.strip()])
        
        if not key_findings:
            key_findings = ["Analysis of " + paper.get('title', 'research paper')]
        
        # Ensure we have at least 10 findings (pad with generic ones if needed)
        while len(key_findings) < 10:
            key_findings.append(f"Additional analysis needed for finding {len(key_findings) + 1}")
        
        return {
            "key_findings": key_findings[:10],
            "limitations": ["Full analysis not available due to extraction error"],
            "study_type": "case_study" if is_case_study else "unknown",
            "techniques_used": self._infer_techniques_from_text(paper, sections),
            "implementation_complexity": "unknown",
            "problem_addressed": paper.get('title', 'Unknown problem'),
            "prerequisites": [],
            "real_world_applications": [],
            "has_code_available": False,
            "has_dataset_available": False,
        }
    
    def _infer_techniques_from_text(self, paper: Dict, sections: Dict) -> List[str]:
        """Infer techniques from paper text when LLM fails using enhanced mapping."""
        techniques = []
        
        # Combine all text for analysis
        all_text = f"{paper.get('title', '')} {paper.get('summary', '')} "
        for section_text in sections.values():
            all_text += section_text + " "
        
        all_text_lower = all_text.lower()
        
        # Check against our enhanced technique mapping
        for keyword, technique_category in TECHNIQUE_MAPPING.items():
            if keyword in all_text_lower:
                techniques.append(technique_category.value)
        
        # Remove duplicates and return unique techniques
        return list(set(techniques)) if techniques else []
    
    def _create_insights_object(self, raw_insights: Dict, paper_id: str) -> PaperInsights:
        """Create validated PaperInsights object from raw extraction."""        
        # Map study type to valid enum value
        study_type_mapping = {
            'experimental': 'empirical',
            'experiment': 'empirical',
            'case-study': 'case_study',
            'case study': 'case_study',
            'field study': 'case_study',
            'theoretical': 'theoretical',
            'theory': 'theoretical',
            'framework': 'theoretical',
            'pilot': 'pilot',
            'prototype': 'pilot',
            'survey': 'survey',
            'review': 'review',
            'meta-analysis': 'meta_analysis',
            'meta analysis': 'meta_analysis',
            'empirical': 'empirical',
            'case_study': 'case_study',
            'meta_analysis': 'meta_analysis',
            'unknown': 'unknown'
        }
        
        raw_study_type = raw_insights.get('study_type', 'unknown').lower()
        mapped_study_type = study_type_mapping.get(raw_study_type, 'unknown')
        
        # Log if mapping was needed
        if raw_study_type != mapped_study_type:
            logger.info(f"Mapped study type '{raw_study_type}' to '{mapped_study_type}'")
        
        # Map techniques to valid enum values
        valid_techniques = []
        for tech in raw_insights.get('techniques_used', []):
            tech_lower = tech.lower().replace(' ', '_').replace('-', '_')
            
            if tech_lower in TECHNIQUE_MAPPING:
                valid_techniques.append(TECHNIQUE_MAPPING[tech_lower])
            else:
                # Try to find partial matches
                found = False
                for key, value in TECHNIQUE_MAPPING.items():
                    if key in tech_lower or tech_lower in key:
                        valid_techniques.append(value)
                        found = True
                        break
                if not found:
                    # No more warnings - just map to OTHER
                    valid_techniques.append(TechniqueCategory.OTHER)
        
        # Map complexity level
        complexity_mapping = {
            'low': ComplexityLevel.LOW,
            'medium': ComplexityLevel.MEDIUM,
            'high': ComplexityLevel.HIGH,
            'very_high': ComplexityLevel.VERY_HIGH,
            'very high': ComplexityLevel.VERY_HIGH,
            'extreme': ComplexityLevel.VERY_HIGH,
            'unknown': ComplexityLevel.UNKNOWN
        }
        
        raw_complexity = raw_insights.get('implementation_complexity', 'unknown').lower()
        mapped_complexity = complexity_mapping.get(raw_complexity, ComplexityLevel.UNKNOWN)
        
        # Create insights object with validated values
        insights = PaperInsights(
            paper_id=paper_id,
            key_findings=raw_insights.get('key_findings', [])[:10],
            limitations=raw_insights.get('limitations', []),
            
            study_type=StudyType(mapped_study_type),
            techniques_used=list(set(valid_techniques)),  # Remove duplicates
            
            implementation_complexity=mapped_complexity,
            
            problem_addressed=raw_insights.get('problem_addressed', ''),
            prerequisites=raw_insights.get('prerequisites', []),
            comparable_approaches=raw_insights.get('comparable_approaches', []),
            real_world_applications=raw_insights.get('real_world_applications', []),
            
            has_code_available=bool(raw_insights.get('has_code_available', False)),
            has_dataset_available=bool(raw_insights.get('has_dataset_available', False)),
        )
        
        return insights
    
    def _create_minimal_insights(self, paper: Dict) -> PaperInsights:
        """Create minimal insights when extraction fails."""
        insights = PaperInsights(
            paper_id=paper.get('id', 'unknown'),
            key_findings=["Extraction failed - minimal insights only"],
            study_type=StudyType.UNKNOWN,
            implementation_complexity=ComplexityLevel.UNKNOWN,
            extraction_confidence=0.1
        )
        
        # Even for minimal insights, try to get author metrics and conference detection
        try:
            total_hindex, individual_hindices = self.semantic_scholar.get_paper_total_hindex(paper.get('authors', []))
            is_conference = self.semantic_scholar.detect_conference_mention(paper)
            insights.total_author_hindex = total_hindex
            insights.has_conference_mention = is_conference
            logger.info(f"Minimal insights with reputation score: {insights.get_reputation_score()}")
        except Exception as e:
            logger.warning(f"Failed to fetch author metrics for minimal insights: {e}")
        
        return insights