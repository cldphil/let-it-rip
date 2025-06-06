"""
Detailed test of the extraction system to verify it's working correctly.
"""

import json
from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))

from core import InsightExtractor, TechniqueCategory

def test_detailed_extraction():
    """Run a detailed extraction test with a more realistic paper."""
    
    print("=== Detailed Extraction Test ===\n")
    
    # Create extractor
    extractor = InsightExtractor()
    
    # Test paper with more content
    test_paper = {
        'id': 'arxiv:2401.12345',
        'title': 'Implementing RAG for Enterprise Customer Service: A Case Study at Fortune 500 Company',
        'summary': '''We present a comprehensive case study of deploying a Retrieval-Augmented Generation (RAG) 
                     system for customer service at a Fortune 500 technology company. The system processes over 
                     100,000 queries daily and achieved a 45% reduction in average response time while maintaining 
                     92% customer satisfaction. Our implementation used a fine-tuned 7B parameter model with a 
                     custom vector database containing 2M technical documents. The system required 4 A100 GPUs 
                     for inference and was deployed using Kubernetes. Key challenges included handling multi-lingual 
                     queries and ensuring GDPR compliance. This production deployment demonstrates the feasibility 
                     of RAG systems in enterprise settings.''',
        'published': '2024-01-15',
        'categories': ['cs.AI', 'cs.CL']
    }
    
    print("Extracting insights from case study paper...")
    insights, metadata = extractor.extract_insights(test_paper)
    
    print(f"\n✓ Extraction completed in {metadata.extraction_time_seconds:.2f}s")
    print(f"  API calls made: {metadata.api_calls_made}")
    print(f"  Estimated cost: ${metadata.estimated_cost_usd:.4f}")
    
    print("\n=== Extracted Insights ===")
    print(f"Study Type: {insights.study_type.value}")
    print(f"Complexity: {insights.implementation_complexity.value}")
    print(f"Confidence: {insights.extraction_confidence:.2f}")
    print(f"Quality Score: {insights.get_quality_score():.2f}")
    
    print(f"\nTechniques Used:")
    for technique in insights.techniques_used:
        print(f"  - {technique.value}")
    
    print(f"\nKey Findings ({len(insights.key_findings)} total):")
    for i, finding in enumerate(insights.key_findings[:3], 1):
        print(f"  {i}. {finding[:100]}...")
    
    if insights.problem_addressed:
        print(f"\nProblem Addressed: {insights.problem_addressed}")
    
    if insights.resource_requirements.compute_requirements:
        print(f"\nCompute Requirements: {insights.resource_requirements.compute_requirements}")
    
    print(f"\nIndustry Validated: {insights.industry_validation}")
    print(f"Evidence Strength: {insights.evidence_strength:.2f}")
    print(f"Practical Applicability: {insights.practical_applicability:.2f}")
    
    # Test with minimal paper
    print("\n\n=== Testing Minimal Paper ===")
    minimal_paper = {
        'id': 'test_minimal',
        'title': 'A Simple Test',
        'summary': 'This is just a test paper with minimal information.'
    }
    
    insights2, metadata2 = extractor.extract_insights(minimal_paper)
    print(f"\n✓ Extraction completed in {metadata2.extraction_time_seconds:.2f}s")
    print(f"  Study Type: {insights2.study_type.value}")
    print(f"  Complexity: {insights2.implementation_complexity.value}")
    print(f"  Confidence: {insights2.extraction_confidence:.2f}")
    print(f"  Techniques found: {len(insights2.techniques_used)}")
    
    print("\n=== Test Complete ===")
    return insights, metadata

if __name__ == "__main__":
    # Check if API key is set
    import os
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("ERROR: Please set ANTHROPIC_API_KEY environment variable")
        print("Export it or add to .env file")
        sys.exit(1)
    
    try:
        insights, metadata = test_detailed_extraction()
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()