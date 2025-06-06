"""
Test the extraction pipeline components.
Run with: python -m pytest tests/test_extraction.py -v
"""

import pytest
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    InsightExtractor,
    PaperInsights,
    StudyType,
    ComplexityLevel,
    TechniqueCategory
)


class TestInsightExtraction:
    """Test the insight extraction system."""
    
    @pytest.fixture
    def sample_paper_minimal(self):
        """Create a minimal test paper."""
        return {
            'id': 'arxiv:2401.00001',
            'title': 'Test Paper: Implementing RAG for Customer Service',
            'summary': 'We present a retrieval-augmented generation system for customer service that improved response accuracy by 35%.',
            'authors': ['Test Author'],
            'published': '2024-01-01',
            'categories': ['cs.AI']
        }
    
    @pytest.fixture
    def sample_paper_full(self):
        """Create a test paper with full text."""
        return {
            'id': 'arxiv:2401.00002',
            'title': 'Fine-tuning LLMs for Medical Diagnosis: A Case Study',
            'summary': '''This paper presents a comprehensive case study on fine-tuning large language models 
                         for medical diagnosis. We achieved 89% accuracy on diagnostic tasks, outperforming 
                         baseline models by 23%. The system was deployed in a hospital setting.''',
            'authors': ['Medical AI Researcher'],
            'published': '2024-01-15',
            'categories': ['cs.AI', 'cs.LG'],
            'full_text': '''
                1. Introduction
                Medical diagnosis using AI has shown tremendous promise. Our work focuses on fine-tuning
                large language models specifically for diagnostic tasks in healthcare settings.
                
                2. Methodology
                We used a dataset of 50,000 medical records and fine-tuned a 7B parameter model using
                LoRA. The training process took 2 weeks on 4 A100 GPUs. Our team of 5 engineers
                implemented custom evaluation metrics.
                
                3. Results
                The fine-tuned model achieved:
                - 89% accuracy on diagnostic tasks
                - 92% precision for common conditions
                - 85% recall for rare diseases
                - 45% reduction in diagnosis time
                
                4. Implementation
                The system requires:
                - 4 GPUs for inference
                - Medical knowledge base integration
                - HIPAA-compliant infrastructure
                
                5. Conclusion
                This case study demonstrates the feasibility of LLMs in medical diagnosis with proper
                fine-tuning and domain adaptation.
            '''
        }
    
    @pytest.fixture
    def extractor(self):
        """Create an extractor instance."""
        return InsightExtractor()
    
    def test_minimal_extraction(self, extractor, sample_paper_minimal):
        """Test extraction with minimal paper data."""
        insights, metadata = extractor.extract_insights(sample_paper_minimal)
        
        # Check insights
        assert isinstance(insights, PaperInsights)
        assert insights.paper_id == '2401.00001'
        assert len(insights.techniques_used) > 0
        assert TechniqueCategory.RAG in insights.techniques_used
        
        # Check metadata
        assert metadata.paper_id == '2401.00001'
        assert metadata.extraction_time_seconds > 0
        assert metadata.api_calls_made >= 1
    
    def test_full_extraction(self, extractor, sample_paper_full):
        """Test extraction with full text."""
        insights, metadata = extractor.extract_insights(sample_paper_full)
        
        # Check insights quality
        assert insights.study_type == StudyType.CASE_STUDY
        assert insights.implementation_complexity in [ComplexityLevel.HIGH, ComplexityLevel.MEDIUM]
        assert TechniqueCategory.FINE_TUNING in insights.techniques_used
        
        # Check extracted details
        assert insights.resource_requirements.compute_requirements is not None
        assert len(insights.key_findings) > 0
        
        # Check confidence
        assert insights.extraction_confidence > 0.6  # Should be high for detailed extraction
    
    def test_section_extraction(self, extractor, sample_paper_full):
        """Test that relevant sections are extracted."""
        # Access the section extraction method
        sections = extractor._extract_relevant_sections(
            sample_paper_full['full_text']
        )
        
        assert 'methodology' in sections
        assert 'results' in sections
        assert 'GPUs' in sections.get('methodology', '')
        assert 'accuracy' in sections.get('results', '')
    
    def test_case_study_detection(self, extractor, sample_paper_full):
        """Test case study detection."""
        # Paper with "case study" in title and hospital deployment
        is_case_study = extractor._detect_case_study(
            sample_paper_full,
            {}  # Empty sections
        )
        
        assert is_case_study == True
    
    def test_error_handling(self, extractor):
        """Test extraction with invalid input."""
        invalid_paper = {
            'id': '',
            'title': '',
            'summary': ''
        }
        
        insights, metadata = extractor.extract_insights(invalid_paper)
        
        # Should return minimal insights without crashing
        assert isinstance(insights, PaperInsights)
        assert insights.extraction_confidence < 0.5
        assert len(metadata.extraction_errors) == 0  # Errors are handled gracefully
    
    def test_complexity_detection(self, extractor):
        """Test complexity level detection."""
        papers = [
            {
                'id': 'test1',
                'title': 'Simple Prompt Engineering for Chatbots',
                'summary': 'We show how simple prompt modifications can improve chatbot responses by 10%.'
            },
            {
                'id': 'test2', 
                'title': 'Distributed Training of 100B Parameter Models',
                'summary': 'We present a complex distributed training system requiring 1000 GPUs and custom infrastructure.'
            }
        ]
        
        complexities = []
        for paper in papers:
            insights, _ = extractor.extract_insights(paper)
            complexities.append(insights.implementation_complexity)
        
        # First paper should be low/medium complexity
        assert complexities[0] in [ComplexityLevel.LOW, ComplexityLevel.MEDIUM]
        
        # Second paper should be high/very high complexity
        assert complexities[1] in [ComplexityLevel.HIGH, ComplexityLevel.VERY_HIGH]


class TestInsightQuality:
    """Test the quality of extracted insights."""
    
    @pytest.fixture
    def extractor(self):
        return InsightExtractor()
    
    def test_insight_completeness(self, extractor):
        """Test that all required fields are populated."""
        paper = {
            'id': 'quality_test',
            'title': 'Complete Analysis of GenAI Implementation',
            'summary': 'A thorough study with metrics, limitations, and future work.',
            'full_text': '''
                We implemented a GenAI system with 25% improvement in efficiency.
                Limitations include high computational cost and data requirements.
                Future work should focus on optimization and broader applications.
            '''
        }
        
        insights, _ = extractor.extract_insights(paper)
        
        # Check required fields
        assert insights.paper_id == 'quality_test'
        assert isinstance(insights.key_findings, list)
        assert isinstance(insights.limitations, list)
        assert isinstance(insights.techniques_used, list)
        assert 0 <= insights.evidence_strength <= 1
        assert 0 <= insights.practical_applicability <= 1
    
    def test_quality_score_calculation(self, extractor):
        """Test quality score calculation."""
        papers = [
            {
                'id': 'high_quality',
                'title': 'Empirical Study: GenAI in Production at Fortune 500',
                'summary': 'Deployed system processing 1M requests daily with measured 40% cost reduction.',
                'full_text': 'Full empirical study with quantitative results and industry validation...'
            },
            {
                'id': 'low_quality',
                'title': 'Theoretical Framework for Future AI',
                'summary': 'We propose a theoretical framework that might work in the future.'
            }
        ]
        
        quality_scores = []
        for paper in papers:
            insights, _ = extractor.extract_insights(paper)
            quality_scores.append(insights.get_quality_score())
        
        # High quality paper should score higher
        assert quality_scores[0] > quality_scores[1]
        assert quality_scores[0] > 0.6  # Good empirical study
        assert quality_scores[1] < 0.5  # Purely theoretical


if __name__ == "__main__":
    # Run basic tests without pytest
    print("Running extraction tests...")
    
    extractor = InsightExtractor()
    
    # Test 1: Basic extraction
    print("\nTest 1: Basic extraction")
    test_paper = {
        'id': 'manual_test',
        'title': 'Testing the Extraction Pipeline',
        'summary': 'This tests if extraction works correctly.'
    }
    
    insights, metadata = extractor.extract_insights(test_paper)
    print(f"✓ Extraction completed in {metadata.extraction_time_seconds:.2f}s")
    print(f"  Study type: {insights.study_type.value}")
    print(f"  Complexity: {insights.implementation_complexity.value}")
    print(f"  Confidence: {insights.extraction_confidence:.2f}")
    
    print("\nAll tests completed!")