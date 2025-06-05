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

from core.insight_schema import (
    PaperInsights, StudyType, TechniqueCategory, 
    ComplexityLevel, ResourceRequirements,
    SuccessMetric, ExtractionMetadata
)
from config import Config

logger = logging.getLogger(__name__)


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
            extractor_version="2.0",
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
                
                # Look for quality indicators
                if any(term in excerpt.lower() for term in ['results', 'performance', 'improved', 'reduced']):
                    if len(excerpt) > len(best_excerpt):
                        best_excerpt = excerpt
        
        if best_excerpt and len(best_excerpt) > 1500:
            best_excerpt = best_excerpt[:1500] + "..."
        
        return best_excerpt
    
    def _detect_case_study(self, paper: Dict, sections: Dict) -> bool:
        """
        Detect if this paper contains a case study.
        """
        # Check title and abstract
        text_to_check = f"{paper.get('title', '')} {paper.get('summary', '')}".lower()
        
        case_study_indicators = [
            'case study', 'case-study', 'field study', 'pilot study',
            'deployment', 'production', 'real-world', 'industrial application',
            'company', 'organization', 'enterprise'
        ]
        
        # Check for indicators in title/abstract
        if any(indicator in text_to_check for indicator in case_study_indicators):
            return True
        
        # Check if we found a case study section
        if sections.get('case_study'):
            return True
        
        # Check other sections for case study content
        for section_content in sections.values():
            if any(indicator in section_content.lower() for indicator in case_study_indicators[:4]):
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

Extract comprehensive insights in this JSON format:
{{
    "key_findings": [
        "Detailed finding 1 - Be specific about methods, results, and implications. Include quantitative improvements where mentioned.",
        "Detailed finding 2 - Focus on practical applications and real-world impact. Mention specific techniques or innovations.",
        "Detailed finding 3 - Highlight unique contributions or novel approaches. Include performance metrics if available.",
        "Detailed finding 4 - Discuss implementation details, requirements, or deployment considerations.",
        "Detailed finding 5 - Cover broader implications, lessons learned, or future directions."
    ],  // Extract 5-10 detailed findings. Each should be 1-3 sentences focusing on ACTIONABLE insights.
    
    "limitations": ["Key limitation 1", "Key limitation 2"],
    "future_work": ["Future direction if mentioned"],
    
    "study_type": "{self._infer_study_type(is_case_study, sections)}",
    "techniques_used": ["rag", "fine_tuning", "prompt_engineering"],  // Identify specific techniques
    
    "implementation_complexity": "low|medium|high|very_high",
    "resource_requirements": {{
        "compute_requirements": "Specific if mentioned (e.g., '4 GPUs')",
        "data_requirements": "Specific if mentioned (e.g., '10k examples')"
    }},
    
    "problem_addressed": "What specific problem does this solve?",
    "prerequisites": ["Technical requirements"],
    "real_world_applications": ["Specific use cases mentioned"],
    
    "evidence_strength": 0.7,  // 0-1, based on empirical validation
    "practical_applicability": 0.8,  // 0-1, how ready for implementation
    
    "has_code_available": false,  // Look for GitHub links or code availability
    "has_dataset_available": false,
    "industry_validation": {str(is_case_study).lower()}  // True if tested in industry
}}

{self._get_case_study_instructions() if is_case_study else ''}

IMPORTANT: Focus on extracting PRACTICAL, IMPLEMENTABLE insights that practitioners can use."""

        try:
            response = self.client.messages.create(
                model=Config.LLM_MODEL,
                temperature=Config.LLM_TEMPERATURE,
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            result = json.loads(response.content[0].text)
            return result
            
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
        """Infer study type based on content."""
        if is_case_study:
            return "case_study"
        
        # Check for other study types based on section content
        all_text = ' '.join(sections.values()).lower()
        
        if any(term in all_text for term in ['experiment', 'benchmark', 'evaluation', 'measured']):
            return "empirical"
        elif any(term in all_text for term in ['pilot', 'trial', 'prototype']):
            return "pilot"
        elif any(term in all_text for term in ['survey', 'review', 'analysis of']):
            return "survey"
        elif any(term in all_text for term in ['framework', 'theoretical', 'model']):
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
        
        if not key_findings:
            key_findings = ["Analysis of " + paper.get('title', 'research paper')]
        
        return {
            "key_findings": key_findings[:5],
            "limitations": ["Full analysis not available"],
            "study_type": "case_study" if is_case_study else "unknown",
            "techniques_used": [],
            "implementation_complexity": "unknown",
            "evidence_strength": 0.3,
            "practical_applicability": 0.3,
            "industry_validation": is_case_study
        }
    
    def _create_insights_object(self, raw_insights: Dict, paper_id: str) -> PaperInsights:
        """Create validated PaperInsights object from raw extraction."""
        # Parse resource requirements
        resource_data = raw_insights.get('resource_requirements', {})
        resources = ResourceRequirements(
            compute_requirements=resource_data.get('compute_requirements'),
            data_requirements=resource_data.get('data_requirements')
        )
        
        # Create insights object
        insights = PaperInsights(
            paper_id=paper_id,
            key_findings=raw_insights.get('key_findings', [])[:10],
            limitations=raw_insights.get('limitations', []),
            future_work=raw_insights.get('future_work', []),
            
            study_type=StudyType(raw_insights.get('study_type', 'unknown')),
            techniques_used=[
                TechniqueCategory(tech) for tech in raw_insights.get('techniques_used', [])
                if tech in [e.value for e in TechniqueCategory]
            ],
            
            implementation_complexity=ComplexityLevel(
                raw_insights.get('implementation_complexity', 'unknown')
            ),
            resource_requirements=resources,
            success_metrics=[],  # Not extracting metrics per your requirement
            
            problem_addressed=raw_insights.get('problem_addressed', ''),
            prerequisites=raw_insights.get('prerequisites', []),
            comparable_approaches=raw_insights.get('comparable_approaches', []),
            real_world_applications=raw_insights.get('real_world_applications', []),
            
            evidence_strength=raw_insights.get('evidence_strength', 0.5),
            practical_applicability=raw_insights.get('practical_applicability', 0.5),
            
            has_code_available=raw_insights.get('has_code_available', False),
            has_dataset_available=raw_insights.get('has_dataset_available', False),
            industry_validation=raw_insights.get('industry_validation', False)
        )
        
        return insights
    
    def _create_minimal_insights(self, paper: Dict) -> PaperInsights:
        """Create minimal insights when extraction fails."""
        return PaperInsights(
            paper_id=paper.get('id', 'unknown'),
            key_findings=["Extraction failed - minimal insights only"],
            study_type=StudyType.UNKNOWN,
            implementation_complexity=ComplexityLevel.UNKNOWN,
            extraction_confidence=0.1
        )