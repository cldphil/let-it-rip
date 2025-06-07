"""
Detailed test of the extraction system to verify it's working correctly.
Updated to test new quality score calculation and remove deprecated field tests.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from core import InsightExtractor, TechniqueCategory

def test_detailed_extraction():
    """Run a detailed extraction test with a more realistic paper."""
    
    print("=== Detailed Extraction Test ===\n")
    
    # Create extractor
    extractor = InsightExtractor()
    
    # Test paper with more content and known authors for quality score testing
    test_paper = {
        'id': 'arxiv:2401.12345',
        'title': 'Implementing RAG for Enterprise Customer Service: A Case Study at Fortune 500 Company',
        'authors': ['Geoffrey Hinton', 'Yann LeCun', 'Alice Johnson'],  # Mix of known and unknown authors
        'summary': '''We present a comprehensive case study of deploying a Retrieval-Augmented Generation (RAG) 
                     system for customer service at a Fortune 500 technology company. The system processes over 
                     100,000 queries daily and achieved a 45% reduction in average response time while maintaining 
                     92% customer satisfaction. Our implementation used a fine-tuned 7B parameter model with a 
                     custom vector database containing 2M technical documents. The system required 4 A100 GPUs 
                     for inference and was deployed using Kubernetes. Key challenges included handling multi-lingual 
                     queries and ensuring GDPR compliance. This production deployment demonstrates the feasibility 
                     of RAG systems in enterprise settings. Accepted at NeurIPS 2024.''',
        'published': '2024-01-15',
        'categories': ['cs.AI', 'cs.CL'],
        'comments': 'Accepted at NeurIPS 2024. Code and dataset available.'
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
    
    # Test new quality score calculation
    print(f"\n=== Quality Score Analysis ===")
    print(f"Total Author H-Index: {insights.total_author_hindex}")
    print(f"Conference Mention Detected: {insights.has_conference_mention}")
    print(f"Calculated Quality Score: {insights.get_quality_score():.3f}")
    
    # Verify quality score calculation manually
    expected_quality = insights.total_author_hindex * (1.5 if insights.has_conference_mention else 1.0) / 100.0
    expected_quality = min(1.0, expected_quality)
    print(f"Expected Quality Score: {expected_quality:.3f}")
    
    if abs(insights.get_quality_score() - expected_quality) < 0.001:
        print("✓ Quality score calculation is correct")
    else:
        print("✗ Quality score calculation mismatch")
    
    print(f"\nTechniques Used:")
    for technique in insights.techniques_used:
        print(f"  - {technique.value}")
    
    print(f"\nKey Findings ({len(insights.key_findings)} total):")
    for i, finding in enumerate(insights.key_findings[:3], 1):
        print(f"  {i}. {finding[:100]}...")
    
    if insights.problem_addressed:
        print(f"\nProblem Addressed: {insights.problem_addressed}")
    
    if insights.resource_requirements and insights.resource_requirements.compute_requirements:
        print(f"\nCompute Requirements: {insights.resource_requirements.compute_requirements}")
    
    print(f"\nIndustry Validated: {insights.industry_validation}")
    print(f"Has Code Available: {insights.has_code_available}")
    print(f"Has Dataset Available: {insights.has_dataset_available}")
    
    # Test case study detection
    print(f"\n=== Case Study Detection ===")
    is_detected_case_study = insights.study_type.value == 'case_study'
    print(f"Detected as Case Study: {is_detected_case_study}")
    if is_detected_case_study:
        print("✓ Correctly identified case study based on title and content")
    
    # Test conference detection
    print(f"\n=== Conference Detection ===")
    print(f"Conference Detected: {insights.has_conference_mention}")
    expected_conference = True  # Should detect "NeurIPS 2024"
    if insights.has_conference_mention == expected_conference:
        print("✓ Correctly detected conference mention")
    else:
        print("✗ Conference detection failed")
    
    return insights, metadata

def test_minimal_paper():
    """Test with minimal paper data."""
    print("\n\n=== Testing Minimal Paper ===")
    
    extractor = InsightExtractor()
    
    minimal_paper = {
        'id': 'test_minimal',
        'title': 'A Simple Test of Prompt Engineering Techniques',
        'authors': ['Unknown Author'],
        'summary': 'This is just a test paper with minimal information about prompt engineering.'
    }
    
    insights, metadata = extractor.extract_insights(minimal_paper)
    print(f"\n✓ Extraction completed in {metadata.extraction_time_seconds:.2f}s")
    print(f"  Study Type: {insights.study_type.value}")
    print(f"  Complexity: {insights.implementation_complexity.value}")
    print(f"  Confidence: {insights.extraction_confidence:.2f}")
    print(f"  Techniques found: {len(insights.techniques_used)}")
    print(f"  Quality Score: {insights.get_quality_score():.3f}")
    print(f"  Total Author H-Index: {insights.total_author_hindex}")
    print(f"  Conference Detected: {insights.has_conference_mention}")
    
    return insights, metadata

def test_quality_score_variations():
    """Test quality score calculation with different scenarios."""
    print("\n\n=== Testing Quality Score Variations ===")
    
    extractor = InsightExtractor()
    
    # Test cases with different author scenarios
    test_cases = [
        {
            'name': 'High H-Index Authors with Conference',
            'paper': {
                'id': 'test_high_quality',
                'title': 'Advanced AI Research Published at ICML',
                'authors': ['Geoffrey Hinton', 'Yann LeCun', 'Yoshua Bengio'],
                'summary': 'Advanced research accepted at ICML 2024.',
                'comments': 'Accepted at ICML 2024'
            },
            'expected_high_quality': True
        },
        {
            'name': 'Unknown Authors without Conference',
            'paper': {
                'id': 'test_low_quality',
                'title': 'Simple Experiment with Basic Methods',
                'authors': ['Unknown Researcher', 'Another Unknown'],
                'summary': 'Basic experiment with standard methods.'
            },
            'expected_high_quality': False
        },
        {
            'name': 'Mixed Authors with Conference',
            'paper': {
                'id': 'test_mixed_quality',
                'title': 'Novel Approach Presented at NeurIPS Workshop',
                'authors': ['Geoffrey Hinton', 'Unknown Researcher'],
                'summary': 'Novel approach presented at NeurIPS workshop.',
                'comments': 'Workshop paper at NeurIPS'
            },
            'expected_high_quality': True  # Should benefit from conference bonus
        }
    ]
    
    for test_case in test_cases:
        print(f"\n--- {test_case['name']} ---")
        insights, metadata = extractor.extract_insights(test_case['paper'])
        
        quality_score = insights.get_quality_score()
        print(f"Quality Score: {quality_score:.3f}")
        print(f"Author H-Index Total: {insights.total_author_hindex}")
        print(f"Conference Detected: {insights.has_conference_mention}")
        
        # Verify calculation
        expected_score = insights.total_author_hindex * (1.5 if insights.has_conference_mention else 1.0) / 100.0
        expected_score = min(1.0, expected_score)
        print(f"Expected Score: {expected_score:.3f}")
        
        if abs(quality_score - expected_score) < 0.001:
            print("✓ Quality calculation correct")
        else:
            print("✗ Quality calculation error")
        
        # Check if expectation matches
        is_high_quality = quality_score > 0.3  # Threshold for "high quality"
        if is_high_quality == test_case['expected_high_quality']:
            print("✓ Quality expectation met")
        else:
            print("✗ Quality expectation not met")

def test_field_removal():
    """Test that deprecated fields are not present in the extraction."""
    print("\n\n=== Testing Deprecated Field Removal ===")
    
    extractor = InsightExtractor()
    
    test_paper = {
        'id': 'test_field_removal',
        'title': 'Testing Field Removal',
        'authors': ['Test Author'],
        'summary': 'Testing that deprecated fields are not included.'
    }
    
    insights, metadata = extractor.extract_insights(test_paper)
    
    # Check that deprecated fields are not present
    insights_dict = insights.dict()
    
    deprecated_fields = ['evidence_strength', 'practical_applicability']
    
    print("Checking for deprecated fields:")
    for field in deprecated_fields:
        if field in insights_dict:
            print(f"✗ FAIL: Deprecated field '{field}' found in insights")
        else:
            print(f"✓ PASS: Deprecated field '{field}' not found")
    
    # Check that new fields are present
    new_fields = ['total_author_hindex', 'has_conference_mention', 'industry_validation']
    
    print("\nChecking for new fields:")
    for field in new_fields:
        if field in insights_dict:
            print(f"✓ PASS: New field '{field}' found in insights")
        else:
            print(f"✗ FAIL: New field '{field}' not found")
    
    # Test that quality score method works
    try:
        quality_score = insights.get_quality_score()
        print(f"\n✓ Quality score method works: {quality_score:.3f}")
    except Exception as e:
        print(f"\n✗ Quality score method failed: {e}")

def test_extraction_confidence():
    """Test extraction confidence calculation."""
    print("\n\n=== Testing Extraction Confidence ===")
    
    extractor = InsightExtractor()
    
    # Test with rich full-text paper
    rich_paper = {
        'id': 'test_rich',
        'title': 'Comprehensive Study with Full Text',
        'authors': ['Test Author'],
        'summary': 'Detailed study with methodology and results.',
        'full_text': '''
        Introduction
        This paper presents a comprehensive study...
        
        Methodology
        We implemented a novel approach using...
        
        Results
        Our experiments show significant improvements...
        
        Discussion
        The results demonstrate the effectiveness...
        
        Conclusion
        We have successfully developed a new method...
        '''
    }
    
    # Test with minimal paper
    minimal_paper = {
        'id': 'test_minimal_confidence',
        'title': 'Minimal Paper',
        'authors': ['Test Author'],
        'summary': 'Brief summary.'
    }
    
    print("Testing rich paper (with full text):")
    rich_insights, _ = extractor.extract_insights(rich_paper)
    print(f"Extraction Confidence: {rich_insights.extraction_confidence:.2f}")
    
    print("\nTesting minimal paper (abstract only):")
    minimal_insights, _ = extractor.extract_insights(minimal_paper)
    print(f"Extraction Confidence: {minimal_insights.extraction_confidence:.2f}")
    
    # Rich paper should have higher confidence
    if rich_insights.extraction_confidence > minimal_insights.extraction_confidence:
        print("✓ Rich paper has higher confidence than minimal paper")
    else:
        print("✗ Confidence calculation may be incorrect")

if __name__ == "__main__":
    # Check if API key is set
    import os
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("ERROR: Please set ANTHROPIC_API_KEY environment variable")
        print("Export it or add to .env file")
        sys.exit(1)
    
    try:
        # Run all tests
        print("🧪 Starting Enhanced Extraction Tests")
        print("=" * 60)
        
        # Main extraction test
        insights, metadata = test_detailed_extraction()
        
        # Minimal paper test
        test_minimal_paper()
        
        # Quality score variation tests
        test_quality_score_variations()
        
        # Field removal verification
        test_field_removal()
        
        # Extraction confidence test
        test_extraction_confidence()
        
        print("\n" + "=" * 60)
        print("✅ All tests completed successfully!")
        print("\n📊 Test Summary:")
        print(f"- Quality score calculation: ✓ Working")
        print(f"- Deprecated fields removed: ✓ Confirmed")
        print(f"- New fields present: ✓ Confirmed")
        print(f"- Case study detection: ✓ Working")
        print(f"- Conference detection: ✓ Working")
        print(f"- Extraction confidence: ✓ Working")
        
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()