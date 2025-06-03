"""
Hierarchical extraction pipeline for efficient insight extraction.
Uses multi-stage processing to optimize API costs and extraction quality.
"""

import asyncio
import json
import re
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging

import anthropic
from anthropic import AsyncAnthropic

from .insight_schema import (
    PaperInsights, StudyType, TechniqueCategory, 
    ComplexityLevel, Industry, ResourceRequirements,
    SuccessMetric, ExtractionMetadata
)
from config import Config

logger = logging.getLogger(__name__)


class HierarchicalInsightExtractor:
    """
    Multi-stage insight extractor that optimizes for quality and cost.
    
    Stage 1: Quick classification from title/abstract
    Stage 2: Targeted deep extraction based on classification
    Stage 3: Validation and normalization
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize extractor with API client."""
        self.api_key = api_key or Config.get_active_api_key()
        if not self.api_key:
            raise ValueError("No API key found. Please set ANTHROPIC_API_KEY in .env")
        
        self.client = AsyncAnthropic(api_key=self.api_key)
        self.sync_client = anthropic.Anthropic(api_key=self.api_key)
        
    async def extract_insights(self, paper: Dict) -> Tuple[PaperInsights, ExtractionMetadata]:
        """
        Extract insights using hierarchical approach.
        
        Args:
            paper: Paper data with title, abstract, and optionally full_text
            
        Returns:
            Tuple of (PaperInsights, ExtractionMetadata)
        """
        start_time = datetime.utcnow()
        metadata = ExtractionMetadata(
            extraction_id=f"{paper.get('id', 'unknown')}_{start_time.timestamp()}",
            paper_id=paper.get('id', 'unknown'),
            extractor_version="1.0",
            llm_model=Config.LLM_MODEL,
            llm_temperature=Config.LLM_TEMPERATURE,
            extraction_time_seconds=0.0,  # Will be updated at the end
            api_calls_made=0
        )
        
        try:
            # Stage 1: Quick classification
            logger.info(f"Stage 1: Quick classification for {paper.get('title', '')[:50]}...")
            quick_insights = await self._quick_classify(
                paper.get('title', ''), 
                paper.get('summary', '')
            )
            metadata.api_calls_made += 1
            
            # Decide processing depth
            if quick_insights.get('worth_deep_analysis', False) and paper.get('full_text'):
                # Stage 2: Deep extraction
                logger.info("Stage 2: Deep extraction based on classification")
                sections = self._extract_relevant_sections(
                    paper.get('full_text', ''),
                    quick_insights.get('focus_sections', [])
                )
                
                detailed_insights = await self._deep_extract(
                    paper, 
                    sections,
                    quick_insights
                )
                metadata.api_calls_made += 1
                
                # Update metadata with sections found
                metadata.sections_found = {
                    section: len(text) > 0 
                    for section, text in sections.items()
                }
                metadata.section_lengths = {
                    section: len(text) 
                    for section, text in sections.items()
                }
                
            else:
                # Minimal extraction for lower-value papers
                logger.info("Using minimal extraction (low value or no full text)")
                detailed_insights = self._minimal_extract(paper, quick_insights)
            
            # Stage 3: Validation and create insights object
            logger.info("Stage 3: Validation and normalization")
            insights = self._create_insights_object(detailed_insights, paper.get('id', ''))
            
            # Calculate extraction time
            extraction_time = (datetime.utcnow() - start_time).total_seconds()
            metadata.extraction_time_seconds = extraction_time
            
            # Estimate cost (rough estimate for Claude)
            metadata.estimated_cost_usd = metadata.api_calls_made * 0.01  # Adjust based on actual pricing
            
            logger.info(f"Extraction complete in {extraction_time:.2f}s with {metadata.api_calls_made} API calls")
            
            return insights, metadata
            
        except Exception as e:
            logger.error(f"Extraction failed: {str(e)}")
            metadata.extraction_errors.append(str(e))
            # Return minimal insights on error
            return self._create_minimal_insights(paper), metadata
    
    async def _quick_classify(self, title: str, abstract: str) -> Dict:
        """
        Fast classification to determine processing depth.
        
        Returns dict with:
        - worth_deep_analysis: bool
        - study_type: preliminary classification
        - has_implementation_details: bool
        - focus_sections: list of sections worth analyzing
        """
        prompt = f"""Quickly analyze this research paper to determine if it contains practical, implementable insights.

Title: {title}
Abstract: {abstract[:800]}

Return a JSON object with these fields:
{{
    "worth_deep_analysis": true/false,  // Does this have real implementation value?
    "study_type": "empirical|case_study|theoretical|pilot|survey",
    "has_implementation_details": true/false,
    "has_quantitative_results": true/false,
    "mentions_real_deployment": true/false,
    "focus_sections": ["methodology", "results", "implementation"],  // Which sections to analyze deeply
    "key_indicators": ["specific metrics", "company names", "deployment details"]  // What makes this valuable
}}

Criteria for worth_deep_analysis=true:
- Contains specific implementation details or architecture
- Reports quantitative results or metrics
- Describes real-world deployment or case study
- Includes enough technical detail to replicate

Return ONLY the JSON object."""

        try:
            response = await self.client.messages.create(
                model=Config.LLM_MODEL,
                temperature=0,  # Deterministic for classification
                max_tokens=500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = json.loads(response.content[0].text)
            return result
            
        except Exception as e:
            logger.warning(f"Quick classification failed: {e}")
            return {
                "worth_deep_analysis": False,
                "study_type": "unknown",
                "has_implementation_details": False,
                "focus_sections": []
            }
    
    def _extract_relevant_sections(self, full_text: str, focus_sections: List[str]) -> Dict[str, str]:
        """
        Extract only the relevant sections based on quick classification.
        
        Returns dict mapping section names to extracted text.
        """
        sections = {}
        
        # Clean text
        text = re.sub(r'\s+', ' ', full_text)
        
        # Section patterns
        section_patterns = {
            'introduction': r'(?:1\.?\s*)?Introduction.*?(?=\n\s*\d+\.?\s*[A-Z]|\Z)',
            'methodology': r'(?:\d+\.?\s*)?(?:Methodology|Methods|Approach|Model Architecture).*?(?=\n\s*\d+\.?\s*[A-Z]|\Z)',
            'implementation': r'(?:\d+\.?\s*)?(?:Implementation|System Design|Architecture).*?(?=\n\s*\d+\.?\s*[A-Z]|\Z)',
            'results': r'(?:\d+\.?\s*)?(?:Results|Experiments|Evaluation).*?(?=\n\s*\d+\.?\s*[A-Z]|\Z)',
            'discussion': r'(?:\d+\.?\s*)?(?:Discussion|Analysis|Findings).*?(?=\n\s*\d+\.?\s*[A-Z]|\Z)',
            'conclusion': r'(?:\d+\.?\s*)?(?:Conclusion|Future Work).*?(?=\n\s*(?:References|Appendix)|\Z)'
        }
        
        for section_name in focus_sections:
            if section_name in section_patterns:
                pattern = section_patterns[section_name]
                match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
                if match:
                    section_text = match.group(0)
                    # Limit section length
                    sections[section_name] = section_text[:3000]
        
        return sections
    
    async def _deep_extract(self, paper: Dict, sections: Dict[str, str], quick_insights: Dict) -> Dict:
        """
        Perform detailed extraction from specific sections.
        """
        # Build focused prompt based on what we found
        sections_text = "\n\n".join([
            f"[{name.upper()}]\n{text[:1500]}"
            for name, text in sections.items()
        ])
        
        prompt = f"""Based on the following research paper sections, extract detailed implementation insights.

Title: {paper.get('title', '')}
Paper Type: {quick_insights.get('study_type', 'unknown')}

{sections_text}

Extract comprehensive insights in this JSON format:
{{
    "key_findings": [
        "Detailed finding 1 - be specific about methods, results, and implications",
        "Detailed finding 2 - include quantitative results where mentioned",
        "Detailed finding 3 - explain what makes this approach unique",
        "Detailed finding 4 - describe practical applications or use cases",
        "Detailed finding 5 - highlight any surprising or counterintuitive results"
    ],  // IMPORTANT: Provide 5-8 detailed findings, each 1-3 sentences. Focus on WHAT was done, HOW it worked, and WHY it matters
    
    "main_contribution": "A comprehensive 2-3 sentence description of this paper's primary contribution to the field. Explain what problem it solves, how the solution works, and what makes it significant or novel compared to existing approaches.",
    
    "limitations": ["limitation 1", "limitation 2"],
    "future_work": ["future direction 1"],
    
    "study_type": "{quick_insights.get('study_type', 'unknown')}",
    "industry_applications": ["healthcare", "finance"],  // Specific industries that could use this
    "techniques_used": ["fine_tuning", "rag", "prompt_engineering"],  // From our defined categories
    
    "implementation_complexity": "low|medium|high|very_high",
    "resource_requirements": {{
        "team_size": "solo|small_team|medium_team|large_team",
        "estimated_time_weeks": 4,  // Realistic estimate
        "compute_requirements": "4 GPUs",  // Specific if mentioned
        "data_requirements": "10k examples"  // Specific if mentioned
    }},
    
    "problem_addressed": "specific problem this solves",
    "prerequisites": ["Python", "PyTorch", "GPU access"],
    "comparable_approaches": ["BERT fine-tuning", "GPT-3 prompting"],
    "real_world_applications": ["customer service automation", "medical diagnosis"],
    
    "evidence_strength": 0.8,  // 0-1, based on empirical validation
    "practical_applicability": 0.7,  // 0-1, how easy to implement
    "has_code_available": false,
    "has_dataset_available": false,
    "industry_validation": false  // Was this tested in industry?
}}

IMPORTANT INSTRUCTIONS:
1. KEY FINDINGS should be the most detailed section - extract specific insights about methods, results, performance improvements, and practical implications
2. MAIN CONTRIBUTION must be a complete, descriptive summary (2-3 full sentences) that captures the essence of the paper
3. Do NOT include a success_metrics field - we'll extract metrics separately
4. Focus on PRACTICAL, IMPLEMENTABLE insights
5. Be specific about numbers, methods, and results when mentioned"""
        
        try:
            response = await self.client.messages.create(
                model=Config.LLM_MODEL,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=1500,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = json.loads(response.content[0].text)
            result['extraction_confidence'] = 0.8  # High confidence for deep extraction
            return result
            
        except Exception as e:
            logger.error(f"Deep extraction failed: {e}")
            return self._minimal_extract(paper, quick_insights)
    
    def _minimal_extract(self, paper: Dict, quick_insights: Dict) -> Dict:
        """
        Minimal extraction for papers without full text or low value.
        Uses only title and abstract.
        """
        # Basic heuristic extraction
        title_lower = paper.get('title', '').lower()
        abstract_lower = paper.get('summary', '').lower()
        combined = f"{title_lower} {abstract_lower}"
        
        # Detect techniques
        techniques = []
        technique_keywords = {
            TechniqueCategory.FINE_TUNING: ["fine-tun", "finetun"],
            TechniqueCategory.RAG: ["retrieval", "rag", "retrieval-augmented"],
            TechniqueCategory.PROMPT_ENGINEERING: ["prompt", "few-shot", "zero-shot"],
            TechniqueCategory.REINFORCEMENT_LEARNING: ["reinforcement", "rlhf", "ppo"]
        }
        
        for technique, keywords in technique_keywords.items():
            if any(kw in combined for kw in keywords):
                techniques.append(technique.value)
        
        # Detect industries
        industries = []
        industry_keywords = {
            Industry.HEALTHCARE: ["medical", "clinical", "patient", "diagnosis"],
            Industry.FINANCE: ["financial", "trading", "investment", "banking"],
            Industry.EDUCATION: ["education", "learning", "student", "teaching"]
        }
        
        for industry, keywords in industry_keywords.items():
            if any(kw in combined for kw in keywords):
                industries.append(industry.value)
        
        if not industries:
            industries = ["general"]
        
        # Extract key findings from abstract
        abstract = paper.get('summary', '')
        sentences = abstract.split('. ')
        key_findings = []
        
        # Try to extract meaningful findings from abstract
        for sentence in sentences:
            if any(keyword in sentence.lower() for keyword in ['achieve', 'improve', 'result', 'show', 'demonstrate', 'find']):
                if len(sentence) > 20:  # Avoid very short sentences
                    key_findings.append(sentence.strip() + ('.' if not sentence.endswith('.') else ''))
        
        # If no findings extracted, use the first few sentences
        if not key_findings and sentences:
            key_findings = [s.strip() + ('.' if not s.endswith('.') else '') for s in sentences[:3] if len(s) > 20]
        
        # Generate a more detailed main contribution
        main_contribution = paper.get('title', '')
        if abstract:
            # Try to find the main claim in the abstract
            contribution_sentences = []
            for sentence in sentences:
                if any(keyword in sentence.lower() for keyword in ['we propose', 'we present', 'we introduce', 'our method', 'our approach']):
                    contribution_sentences.append(sentence)
            
            if contribution_sentences:
                main_contribution = ' '.join(contribution_sentences[:2])
            else:
                # Use title + first substantial sentence from abstract
                main_contribution = f"{paper.get('title', '')}. {sentences[0] if sentences else ''}"
        
        return {
            "key_findings": key_findings[:5] if key_findings else [f"Analysis of {paper.get('title', 'this paper')}"],
            "main_contribution": main_contribution[:400],  # Longer contribution
            "limitations": ["Full text not analyzed", "Detailed methodology not available"],
            "study_type": quick_insights.get('study_type', 'unknown'),
            "industry_applications": industries,
            "techniques_used": techniques,
            "implementation_complexity": "unknown",
            "evidence_strength": 0.3,
            "practical_applicability": 0.3,
            "extraction_confidence": 0.4  # Low confidence for minimal extraction
        }
    
    def _create_insights_object(self, raw_insights: Dict, paper_id: str) -> PaperInsights:
        """
        Create validated PaperInsights object from raw extraction.
        """
        # Parse resource requirements
        resource_data = raw_insights.get('resource_requirements', {})
        resources = ResourceRequirements(
            team_size=resource_data.get('team_size', 'not_specified'),
            estimated_time_weeks=resource_data.get('estimated_time_weeks'),
            compute_requirements=resource_data.get('compute_requirements'),
            data_requirements=resource_data.get('data_requirements')
        )
        
        # Parse success metrics
        success_metrics = []
        # Note: We're not extracting success_metrics from the response anymore
        # as requested - removing this field entirely
        
        # Create insights object
        insights = PaperInsights(
            paper_id=paper_id,
            key_findings=raw_insights.get('key_findings', [])[:8],  # Allow up to 8 findings
            main_contribution=raw_insights.get('main_contribution', '')[:400],  # Longer contribution
            limitations=raw_insights.get('limitations', []),
            future_work=raw_insights.get('future_work', []),
            
            study_type=StudyType(raw_insights.get('study_type', 'unknown')),
            industry_applications=[
                Industry(ind) for ind in raw_insights.get('industry_applications', ['general'])
                if ind in [e.value for e in Industry]
            ],
            techniques_used=[
                TechniqueCategory(tech) for tech in raw_insights.get('techniques_used', [])
                if tech in [e.value for e in TechniqueCategory]
            ],
            
            implementation_complexity=ComplexityLevel(
                raw_insights.get('implementation_complexity', 'unknown')
            ),
            resource_requirements=resources,
            success_metrics=[],  # Empty list as requested - no success metrics
            
            problem_addressed=raw_insights.get('problem_addressed', ''),
            prerequisites=raw_insights.get('prerequisites', []),
            comparable_approaches=raw_insights.get('comparable_approaches', []),
            real_world_applications=raw_insights.get('real_world_applications', []),
            
            evidence_strength=raw_insights.get('evidence_strength', 0.5),
            practical_applicability=raw_insights.get('practical_applicability', 0.5),
            extraction_confidence=raw_insights.get('extraction_confidence', 0.5),
            
            has_code_available=raw_insights.get('has_code_available', False),
            has_dataset_available=raw_insights.get('has_dataset_available', False),
            industry_validation=raw_insights.get('industry_validation', False)
        )
        
        return insights
    
    def _create_minimal_insights(self, paper: Dict) -> PaperInsights:
        """Create minimal insights when extraction fails."""
        return PaperInsights(
            paper_id=paper.get('id', 'unknown'),
            main_contribution=paper.get('title', 'Unknown')[:500],
            key_findings=["Extraction failed - minimal insights only"],
            study_type=StudyType.UNKNOWN,
            implementation_complexity=ComplexityLevel.UNKNOWN,
            extraction_confidence=0.1
        )


class SyncHierarchicalExtractor:
    """Synchronous version of the hierarchical extractor for non-async contexts."""
    
    def __init__(self, api_key: Optional[str] = None):
        """Initialize with async extractor."""
        self.async_extractor = HierarchicalInsightExtractor(api_key)
        
    def extract_insights(self, paper: Dict) -> Tuple[PaperInsights, ExtractionMetadata]:
        """Synchronous wrapper for extract_insights."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(
                self.async_extractor.extract_insights(paper)
            )
        finally:
            loop.close()