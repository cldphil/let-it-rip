"""
Test the storage system components.
Run with: python -m pytest tests/test_storage.py -v
"""

import pytest
import json
import tempfile
import shutil
from pathlib import Path
from datetime import datetime

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).parent.parent))

from core import (
    InsightStorage,
    PaperInsights,
    UserContext,
    Industry,
    StudyType,
    ComplexityLevel,
    TechniqueCategory,
    ResourceRequirements,
    SuccessMetric
)


class TestInsightStorage:
    """Test the insight storage system."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        temp_dir = tempfile.mkdtemp()
        storage = InsightStorage(storage_root=temp_dir)
        yield storage
        # Cleanup
        shutil.rmtree(temp_dir)
    
    @pytest.fixture
    def sample_paper(self):
        """Create sample paper data."""
        return {
            'id': 'arxiv:2401.12345',
            'title': 'Test Paper for Storage',
            'authors': ['Test Author'],
            'published': '2024-01-01',
            'categories': ['cs.AI'],
            'summary': 'This is a test abstract.',
            'pdf_url': 'http://example.com/paper.pdf'
        }
    
    @pytest.fixture
    def sample_insights(self):
        """Create sample insights."""
        return PaperInsights(
            paper_id='2401.12345',
            main_contribution='Test contribution for storage system',
            key_findings=['Finding 1', 'Finding 2'],
            limitations=['Limitation 1'],
            study_type=StudyType.EMPIRICAL,
            industry_applications=[Industry.HEALTHCARE, Industry.FINANCE],
            techniques_used=[TechniqueCategory.RAG, TechniqueCategory.FINE_TUNING],
            implementation_complexity=ComplexityLevel.MEDIUM,
            resource_requirements=ResourceRequirements(
                team_size='small_team',
                estimated_time_weeks=8
            ),
            success_metrics=[
                SuccessMetric(
                    metric_name='accuracy',
                    improvement_value=25.5,
                    improvement_unit='percentage'
                )
            ],
            evidence_strength=0.8,
            practical_applicability=0.7,
            extraction_confidence=0.85
        )
    
    def test_store_and_retrieve_paper(self, temp_storage, sample_paper):
        """Test storing and retrieving paper data."""
        # Store paper
        paper_id = temp_storage.store_paper(sample_paper)
        assert paper_id == '2401.12345'
        
        # Retrieve paper
        retrieved = temp_storage.load_paper(paper_id)
        assert retrieved is not None
        assert retrieved['title'] == sample_paper['title']
        assert retrieved['authors'] == sample_paper['authors']
    
    def test_store_and_retrieve_insights(self, temp_storage, sample_insights):
        """Test storing and retrieving insights."""
        # Store insights
        temp_storage.store_insights('2401.12345', sample_insights)
        
        # Retrieve insights
        retrieved = temp_storage.load_insights('2401.12345')
        assert retrieved is not None
        assert retrieved.main_contribution == sample_insights.main_contribution
        assert retrieved.study_type == sample_insights.study_type
        assert len(retrieved.techniques_used) == 2
    
    def test_vector_search(self, temp_storage, sample_paper, sample_insights):
        """Test vector similarity search."""
        # Store multiple papers with insights
        papers_data = [
            ('medical_ai', PaperInsights(
                paper_id='medical_ai',
                main_contribution='AI system for medical diagnosis',
                study_type=StudyType.CASE_STUDY,
                industry_applications=[Industry.HEALTHCARE],
                techniques_used=[TechniqueCategory.FINE_TUNING],
                implementation_complexity=ComplexityLevel.HIGH
            )),
            ('finance_fraud', PaperInsights(
                paper_id='finance_fraud',
                main_contribution='Fraud detection using LLMs',
                study_type=StudyType.EMPIRICAL,
                industry_applications=[Industry.FINANCE],
                techniques_used=[TechniqueCategory.RAG],
                implementation_complexity=ComplexityLevel.MEDIUM
            )),
            ('retail_rec', PaperInsights(
                paper_id='retail_rec',
                main_contribution='Recommendation system for e-commerce',
                study_type=StudyType.PILOT,
                industry_applications=[Industry.RETAIL],
                techniques_used=[TechniqueCategory.PROMPT_ENGINEERING],
                implementation_complexity=ComplexityLevel.LOW
            ))
        ]
        
        # Store all papers
        for paper_id, insights in papers_data:
            temp_storage.store_paper({'id': paper_id, 'title': f'Paper {paper_id}'})
            temp_storage.store_insights(paper_id, insights)
        
        # Search for healthcare papers
        user_context = UserContext(
            industry=Industry.HEALTHCARE,
            use_case_description='Medical diagnosis automation'
        )
        
        results = temp_storage.find_similar_papers(user_context, n_results=3)
        
        # Healthcare paper should rank first
        assert len(results) > 0
        assert results[0]['paper_id'] == 'medical_ai'
        assert results[0]['similarity_score'] > 0.5
    
    def test_statistics(self, temp_storage, sample_paper, sample_insights):
        """Test storage statistics."""
        # Store some data
        temp_storage.store_paper(sample_paper)
        temp_storage.store_insights('2401.12345', sample_insights)
        
        # Get statistics
        stats = temp_storage.get_statistics()
        
        assert stats['total_papers'] == 1
        assert stats['total_insights'] == 1
        assert stats['complexity_distribution'].get('medium', 0) == 1
        assert stats['study_type_distribution'].get('empirical', 0) == 1
    
    def test_metadata_persistence(self, temp_storage, sample_paper):
        """Test SQLite metadata persistence."""
        # Store paper
        paper_id = temp_storage.store_paper(sample_paper)
        
        # Query metadata directly
        cursor = temp_storage.metadata_conn.cursor()
        cursor.execute("SELECT title, authors FROM papers WHERE paper_id = ?", (paper_id,))
        result = cursor.fetchone()
        
        assert result is not None
        assert result['title'] == sample_paper['title']
        assert json.loads(result['authors']) == sample_paper['authors']
    
    def test_clear_storage(self, temp_storage, sample_paper, sample_insights):
        """Test clearing all storage."""
        # Store data
        temp_storage.store_paper(sample_paper)
        temp_storage.store_insights('2401.12345', sample_insights)
        
        # Clear storage
        temp_storage.clear_all()
        
        # Verify cleared
        stats = temp_storage.get_statistics()
        assert stats['total_papers'] == 0
        assert stats['total_insights'] == 0


class TestStorageEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.fixture
    def temp_storage(self):
        """Create temporary storage for testing."""
        temp_dir = tempfile.mkdtemp()
        storage = InsightStorage(storage_root=temp_dir)
        yield storage
        shutil.rmtree(temp_dir)
    
    def test_load_nonexistent_paper(self, temp_storage):
        """Test loading non-existent paper."""
        result = temp_storage.load_paper('nonexistent')
        assert result is None
    
    def test_load_nonexistent_insights(self, temp_storage):
        """Test loading non-existent insights."""
        result = temp_storage.load_insights('nonexistent')
        assert result is None
    
    def test_duplicate_paper_storage(self, temp_storage):
        """Test storing same paper twice."""
        paper = {'id': 'duplicate_test', 'title': 'Test'}
        
        # Store twice
        id1 = temp_storage.store_paper(paper)
        id2 = temp_storage.store_paper(paper)
        
        # Should use same ID
        assert id1 == id2
        
        # Should only have one entry
        stats = temp_storage.get_statistics()
        assert stats['total_papers'] == 1
    
    def test_empty_search(self, temp_storage):
        """Test search with no stored papers."""
        user_context = UserContext(
            industry=Industry.HEALTHCARE,
            use_case_description='Test search'
        )
        
        results = temp_storage.find_similar_papers(user_context)
        assert len(results) == 0


class TestUserContextMatching:
    """Test user context matching functionality."""
    
    @pytest.fixture
    def storage_with_papers(self):
        """Create storage with various papers."""
        temp_dir = tempfile.mkdtemp()
        storage = InsightStorage(storage_root=temp_dir)
        
        # Add diverse papers
        test_papers = [
            {
                'id': 'low_complexity',
                'insights': PaperInsights(
                    paper_id='low_complexity',
                    main_contribution='Simple prompt engineering',
                    implementation_complexity=ComplexityLevel.LOW,
                    resource_requirements=ResourceRequirements(
                        team_size='solo',
                        estimated_time_weeks=2
                    ),
                    evidence_strength=0.8,
                    industry_validation=True
                )
            },
            {
                'id': 'high_complexity',
                'insights': PaperInsights(
                    paper_id='high_complexity',
                    main_contribution='Complex distributed training',
                    implementation_complexity=ComplexityLevel.HIGH,
                    resource_requirements=ResourceRequirements(
                        team_size='large_team',
                        estimated_time_weeks=24
                    ),
                    evidence_strength=0.9,
                    industry_validation=False
                )
            }
        ]
        
        for paper_data in test_papers:
            storage.store_paper({'id': paper_data['id'], 'title': 'Test'})
            storage.store_insights(paper_data['id'], paper_data['insights'])
        
        yield storage
        shutil.rmtree(temp_dir)
    
    def test_budget_constraint_filtering(self, storage_with_papers):
        """Test filtering by budget constraint."""
        # Low budget should exclude high complexity
        user_context = UserContext(
            budget_constraint="low",
            use_case_description="Simple automation"
        )
        
        results = storage_with_papers.find_similar_papers(user_context)
        
        # Should prefer low complexity paper
        for result in results:
            insights = result['insights']
            assert insights.implementation_complexity in [ComplexityLevel.LOW, ComplexityLevel.MEDIUM]
    
    def test_timeline_constraint_filtering(self, storage_with_papers):
        """Test filtering by timeline constraint."""
        # Short timeline should exclude long projects
        user_context = UserContext(
            timeline_weeks=4,
            use_case_description="Quick POC"
        )
        
        results = storage_with_papers.find_similar_papers(user_context)
        
        # Should only include papers with short timelines
        for result in results:
            insights = result['insights']
            if insights.resource_requirements.estimated_time_weeks:
                assert insights.resource_requirements.estimated_time_weeks <= 4
    
    def test_risk_tolerance_filtering(self, storage_with_papers):
        """Test filtering by risk tolerance."""
        # Conservative should prefer validated approaches
        user_context = UserContext(
            risk_tolerance="conservative",
            use_case_description="Production system"
        )
        
        results = storage_with_papers.find_similar_papers(user_context)
        
        # Should prefer high evidence strength
        if results:
            top_result = results[0]
            assert top_result['insights'].evidence_strength >= 0.7


if __name__ == "__main__":
    # Run basic storage tests without pytest
    print("Running storage tests...")
    
    # Create temporary storage
    import tempfile
    temp_dir = tempfile.mkdtemp()
    storage = InsightStorage(storage_root=temp_dir)
    
    # Test 1: Basic storage
    print("\nTest 1: Basic storage operations")
    test_paper = {
        'id': 'manual_test_123',
        'title': 'Manual Test Paper',
        'authors': ['Tester'],
        'summary': 'Testing storage system'
    }
    
    paper_id = storage.store_paper(test_paper)
    print(f"✓ Stored paper with ID: {paper_id}")
    
    retrieved = storage.load_paper(paper_id)
    print(f"✓ Retrieved paper: {retrieved['title']}")
    
    # Test 2: Statistics
    print("\nTest 2: Storage statistics")
    stats = storage.get_statistics()
    print(f"✓ Total papers: {stats['total_papers']}")
    print(f"✓ Total insights: {stats['total_insights']}")
    
    # Cleanup
    shutil.rmtree(temp_dir)
    print("\nAll tests completed!")