import streamlit as st
import json
import os
import re
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from pathlib import Path

from arxiv_fetcher import ArxivGenAIFetcher
from metadata_extractor import MetadataExtractor

# Import new core modules
from core import (
    InsightStorage,
    SyncBatchProcessor,
    SynthesisEngine,
    UserContext,
    Industry,
    TechniqueCategory
)

# Page configuration
st.set_page_config(
    page_title="GenAI Research Metadata Extractor",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .paper-title {
        font-size: 1.2rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .tag-chip {
        background-color: #e1f5fe;
        padding: 0.2rem 0.5rem;
        border-radius: 1rem;
        margin: 0.1rem;
        display: inline-block;
        font-size: 0.8rem;
    }
    .complexity-low { background-color: #c8e6c9; }
    .complexity-medium { background-color: #fff3e0; }
    .complexity-high { background-color: #ffcdd2; }
    .roadmap-phase {
        background-color: #f5f5f5;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize storage and processors
@st.cache_resource
def get_storage():
    """Get or create storage instance."""
    return InsightStorage()

@st.cache_resource  
def get_batch_processor():
    """Get or create batch processor."""
    return SyncBatchProcessor(storage=get_storage())

@st.cache_resource
def get_synthesis_engine():
    """Get or create synthesis engine."""
    return SynthesisEngine(storage=get_storage())

def load_existing_papers():
    """Load existing papers from new storage system."""
    storage = get_storage()
    stats = storage.get_statistics()
    
    # Load all papers with insights
    papers_data = []
    insights_dir = Path("storage/insights")
    
    if insights_dir.exists():
        for insight_file in insights_dir.glob("*_insights.json"):
            paper_id = insight_file.stem.replace("_insights", "")
            
            # Load paper data
            paper_data = storage.load_paper(paper_id)
            if paper_data:
                # Load insights and convert to legacy format
                insights = storage.load_insights(paper_id)
                if insights:
                    # Merge for compatibility
                    paper_data['business_tags'] = {
                        'methodology_type': insights.study_type.value,
                        'industry': [ind.value for ind in insights.industry_applications],
                        'implementation_complexity': insights.implementation_complexity.value,
                        'confidence_score': insights.extraction_confidence
                    }
                    paper_data['quality_score'] = insights.get_quality_score()
                    papers_data.append(paper_data)
    
    return papers_data, stats

def normalize_text(text):
    """Normalize text for better processing."""
    if not text:
        return ""
    
    # Remove excessive whitespace and newlines
    normalized = ' '.join(text.split())
    
    # Remove page markers
    normalized = re.sub(r'--- Page \d+ ---', ' ', normalized)
    
    # Remove common PDF artifacts
    normalized = re.sub(r'\s+', ' ', normalized)
    normalized = re.sub(r'(?<=[.!?])\s*\n\s*(?=[A-Z])', ' ', normalized)
    
    # Remove excessive punctuation
    normalized = re.sub(r'\.{3,}', '...', normalized)
    normalized = re.sub(r'-{2,}', '--', normalized)
    
    # Clean up academic formatting
    normalized = re.sub(r'\b(Fig|Figure|Table|Eq|Equation)\.\s*(\d+)', r'\1 \2', normalized)
    
    # Remove citation markers
    normalized = re.sub(r'\[\d+\]', '', normalized)
    
    # Remove extra spaces around punctuation
    normalized = re.sub(r'\s+([,.;:!?])', r'\1', normalized)
    normalized = re.sub(r'([,.;:!?])\s+', r'\1 ', normalized)
    
    return normalized.strip()

def fetch_new_papers(max_results):
    """Fetch and process new papers using batch processor."""
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Initialize components
    status_text.text("Initializing arXiv fetcher...")
    fetcher = ArxivGenAIFetcher()
    processor = get_batch_processor()
    progress_bar.progress(0.1)
    
    # Fetch papers with full text
    status_text.text("Fetching papers from arXiv...")
    papers = fetcher.fetch_papers(max_results=max_results, include_full_text=True)
    progress_bar.progress(0.3)
    
    if not papers:
        st.error("No papers found. Try again later.")
        return []
    
    # Normalize text in papers
    status_text.text("Normalizing paper text...")
    for paper in papers:
        if paper.get('full_text'):
            paper['full_text'] = normalize_text(paper['full_text'])
    progress_bar.progress(0.4)
    
    # Process with batch processor
    status_text.text("Processing papers with hierarchical extraction...")
    checkpoint_name = f"streamlit_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Process papers
    stats = processor.process_papers(papers, checkpoint_name=checkpoint_name)
    progress_bar.progress(0.9)
    
    status_text.text(f"Complete! Processed {stats['successful']} papers successfully.")
    progress_bar.progress(1.0)
    
    return stats

def display_summary_statistics(papers, storage_stats):
    """Display comprehensive statistics about the papers."""
    st.header("📊 Summary Statistics")
    
    # Basic metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", storage_stats.get('total_papers', 0))
    
    with col2:
        st.metric("Papers with Insights", storage_stats.get('total_insights', 0))
    
    with col3:
        st.metric("Papers with Code", storage_stats.get('papers_with_code', 0))
        
    with col4:
        avg_quality = storage_stats.get('average_quality_score', 0)
        st.metric("Avg Quality Score", f"{avg_quality:.2f}")
    
    # Visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        # Implementation Complexity Distribution
        complexity_data = storage_stats.get('complexity_distribution', {})
        
        if complexity_data:
            fig_complexity = px.pie(
                values=list(complexity_data.values()),
                names=list(complexity_data.keys()),
                title="Implementation Complexity Distribution",
                color_discrete_map={
                    'low': '#4CAF50',
                    'medium': '#FF9800', 
                    'high': '#F44336',
                    'very_high': '#D32F2F',
                    'unknown': '#9E9E9E'
                }
            )
            st.plotly_chart(fig_complexity, use_container_width=True)
    
    with col2:
        # Study Type Distribution
        study_data = storage_stats.get('study_type_distribution', {})
        
        if study_data:
            fig_study = px.pie(
                values=list(study_data.values()),
                names=list(study_data.keys()),
                title="Study Type Distribution",
                color_discrete_map={
                    'case_study': '#4CAF50',
                    'empirical': '#2196F3',
                    'theoretical': '#FF9800', 
                    'pilot': '#9C27B0',
                    'survey': '#795548',
                    'meta_analysis': '#607D8B',
                    'unknown': '#9E9E9E'
                }
            )
            st.plotly_chart(fig_study, use_container_width=True)
    
    # Cost tracking
    if storage_stats.get('total_extraction_cost', 0) > 0:
        st.subheader("💰 Processing Costs")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Extraction Cost", f"${storage_stats['total_extraction_cost']:.2f}")
        with col2:
            cost_per_paper = storage_stats['total_extraction_cost'] / max(storage_stats['total_insights'], 1)
            st.metric("Average Cost per Paper", f"${cost_per_paper:.3f}")

def display_paper_browser(papers):
    """Display interactive paper browser with new insights."""
    if not papers:
        st.warning("No papers to browse.")
        return
    
    st.header("📖 Paper Browser")
    
    # Load storage for detailed insights
    storage = get_storage()
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        complexity_filter = st.selectbox(
            "Filter by Complexity",
            ["All"] + list(set(p.get('business_tags', {}).get('implementation_complexity', 'unknown') for p in papers))
        )
    
    with col2:
        all_methodologies = set()
        for paper in papers:
            methodology = paper.get('business_tags', {}).get('methodology_type', 'unknown')
            all_methodologies.add(methodology)
        
        methodology_filter = st.selectbox(
            "Filter by Methodology",
            ["All"] + sorted(list(all_methodologies))
        )
    
    with col3:
        min_quality = st.slider("Minimum Quality Score", 0.0, 1.0, 0.0, 0.1)
    
    # Apply filters
    filtered_papers = papers
    
    if complexity_filter != "All":
        filtered_papers = [p for p in filtered_papers 
                          if p.get('business_tags', {}).get('implementation_complexity') == complexity_filter]
    
    if methodology_filter != "All":
        filtered_papers = [p for p in filtered_papers 
                          if p.get('business_tags', {}).get('methodology_type') == methodology_filter]
    
    filtered_papers = [p for p in filtered_papers 
                      if p.get('quality_score', 0) >= min_quality]
    
    st.write(f"Showing {len(filtered_papers)} of {len(papers)} papers")
    
    # Paper selection
    if filtered_papers:
        paper_options = [f"{i}: {paper['title'][:80]}..." if len(paper['title']) > 80 else f"{i}: {paper['title']}" 
                        for i, paper in enumerate(filtered_papers)]
        
        selected_paper_idx = st.selectbox("Select a paper to view:", range(len(paper_options)), 
                                         format_func=lambda x: paper_options[x])
        
        if selected_paper_idx is not None:
            display_enhanced_paper_details(filtered_papers[selected_paper_idx], storage)

def display_enhanced_paper_details(paper, storage):
    """Display detailed view with insights."""
    st.subheader("📄 Paper Details")
    
    # Get paper ID
    paper_id = paper.get('id', '').split('/')[-1]
    
    # Load full insights if available
    insights = storage.load_insights(paper_id) if paper_id else None
    
    # Basic information
    st.markdown(f"<div class='paper-title'>{paper.get('title', 'No title')}</div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write("**Authors:**", ", ".join(paper.get('authors', [])))
        st.write("**Published:**", paper.get('published', 'Unknown'))
        st.write("**Categories:**", ", ".join(paper.get('categories', [])))
        if paper.get('pdf_url'):
            st.write("**PDF:**", paper['pdf_url'])
    
    with col2:
        # Quality metrics
        if insights:
            st.metric("Quality Score", f"{insights.get_quality_score():.2f}")
            st.metric("Evidence Strength", f"{insights.evidence_strength:.2f}")
            st.metric("Practical Applicability", f"{insights.practical_applicability:.2f}")
    
    # Key insights
    if insights:
        st.subheader("🔍 Key Insights")
        
        # Main contribution
        if insights.main_contribution:
            st.info(f"**Main Contribution:** {insights.main_contribution}")
        
        # Key insights
        if insights:
            st.write("**Key Findings:**")
            for i, finding in enumerate(insights.key_findings, 1):
                st.write(f"{i}. {finding}")
        
        # Implementation details
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Implementation Details:**")
            st.write(f"- Complexity: {insights.implementation_complexity.value}")
            st.write(f"- Team Size: {insights.resource_requirements.team_size.value}")
            if insights.resource_requirements.estimated_time_weeks:
                st.write(f"- Timeline: {insights.resource_requirements.estimated_time_weeks} weeks")
        
        with col2:
            st.write("**Technical Requirements:**")
            for tech in insights.techniques_used[:5]:
                st.write(f"• {tech.value.replace('_', ' ').title()}")
    
    # Abstract
    st.subheader("📝 Abstract")
    st.write(paper.get('summary', 'No abstract available'))
    
    # Full text if available
    if paper.get('full_text'):
        st.subheader("📖 Full Text")
        
        full_text = paper['full_text']
        st.write(f"**Total length:** {len(full_text):,} characters")
        
        with st.expander("View Full Text", expanded=False):
            st.text_area(
                "Full Paper Content",
                value=full_text,
                height=400,
                help="Complete normalized text extracted from the PDF"
            )

def display_recommendations_tab():
    """Display personalized recommendations tab."""
    st.header("🎯 Personalized Recommendations")
    
    # User context form
    with st.form("user_context_form"):
        st.subheader("Tell us about your needs")
        
        col1, col2 = st.columns(2)
        
        with col1:
            industry = st.selectbox(
                "Your Industry",
                options=[ind.value for ind in Industry],
                format_func=lambda x: x.replace('_', ' ').title()
            )
            
            company_size = st.selectbox(
                "Company Size",
                options=["startup", "small", "medium", "large", "enterprise"]
            )
            
            maturity = st.selectbox(
                "AI Maturity Level",
                options=["greenfield", "pilot_ready", "scaling", "optimizing"],
                format_func=lambda x: x.replace('_', ' ').title()
            )
        
        with col2:
            timeline_weeks = st.number_input(
                "Timeline (weeks)",
                min_value=1,
                max_value=52,
                value=12
            )
            
            budget = st.selectbox(
                "Budget Constraint",
                options=["low", "medium", "high", "unlimited"]
            )
            
            risk_tolerance = st.selectbox(
                "Risk Tolerance",
                options=["conservative", "moderate", "aggressive"]
            )
        
        use_case = st.text_area(
            "Describe your use case",
            placeholder="What problem are you trying to solve with GenAI?"
        )
        
        submitted = st.form_submit_button("Get Recommendations")
    
    if submitted and use_case:
        # Create user context
        user_context = UserContext(
            industry=Industry(industry),
            company_size=company_size,
            maturity_level=maturity,
            timeline_weeks=timeline_weeks,
            budget_constraint=budget,
            risk_tolerance=risk_tolerance,
            use_case_description=use_case
        )
        
        # Get recommendations
        with st.spinner("Analyzing papers and generating recommendations..."):
            engine = get_synthesis_engine()
            recommendations = engine.synthesize_recommendations(user_context)
        
        # Display recommendations
        if recommendations['recommendations']['top_approaches']:
            st.success(f"Found {len(recommendations['recommendations']['top_approaches'])} recommended approaches based on {recommendations['papers_analyzed']} papers!")
            
            # Top approaches
            st.subheader("🏆 Recommended Approaches")
            
            for i, approach in enumerate(recommendations['recommendations']['top_approaches'], 1):
                with st.expander(f"{i}. {approach['approach_name'].replace('_', ' ').title()}", expanded=(i==1)):
                    col1, col2 = st.columns([2, 1])
                    
                    with col1:
                        st.write(f"**Why recommended:** {approach['why_recommended']}")
                        st.write(f"**Complexity:** {approach['complexity']}")
                        st.write(f"**Timeline:** {approach['expected_timeline_weeks']} weeks")
                        st.write(f"**Team Size:** {approach['team_size_required'].replace('_', ' ')}")
                    
                    with col2:
                        st.metric("Confidence", f"{approach['confidence_score']:.0%}")
                    
                    if approach.get('example_implementations'):
                        st.write("**Example Implementations:**")
                        for example in approach['example_implementations'][:2]:
                            st.write(f"• {example['title'][:60]}...")
                            st.caption(f"  → {example['key_insight']}")
            
            # Implementation roadmap
            if recommendations.get('implementation_roadmap'):
                st.subheader("📋 Implementation Roadmap")
                roadmap = recommendations['implementation_roadmap']
                
                st.write(f"**Approach:** {roadmap['approach'].replace('_', ' ').title()}")
                st.write(f"**Total Duration:** {roadmap['total_duration_weeks']} weeks")
                
                # Phases
                for phase in roadmap['phases']:
                    with st.expander(f"Phase {phase['phase_number']}: {phase['name']} ({phase['duration_weeks']} weeks)"):
                        st.write("**Activities:**")
                        for activity in phase['activities']:
                            st.write(f"• {activity}")
                        
                        st.write("**Deliverables:**")
                        for deliverable in phase['deliverables']:
                            st.write(f"✓ {deliverable}")
                        
                        if phase.get('prerequisites'):
                            st.write("**Prerequisites:**")
                            for prereq in phase['prerequisites']:
                                st.write(f"- {prereq}")
            
            # Success factors and pitfalls
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ Success Factors")
                for factor in recommendations['recommendations']['success_factors'][:5]:
                    st.write(f"• {factor}")
            
            with col2:
                st.subheader("⚠️ Common Pitfalls")
                for pitfall in recommendations['recommendations']['common_pitfalls'][:5]:
                    st.write(f"• {pitfall}")
            
        else:
            st.warning("No relevant papers found for your criteria. Try adjusting your requirements.")

def main():
    """Main Streamlit application."""
    st.title("🔬 GenAI Research Metadata Extractor")
    st.markdown("Extract insights from research papers and get personalized implementation recommendations")
    
    # Sidebar
    st.sidebar.title("Controls")
    
    # Load existing data
    existing_papers, storage_stats = load_existing_papers()
    if existing_papers:
        st.sidebar.success(f"Found {len(existing_papers)} papers in storage")
        st.sidebar.info(f"Total extraction cost: ${storage_stats.get('total_extraction_cost', 0):.2f}")
    
    # Fetch new papers section
    st.sidebar.subheader("Fetch New Papers")
    
    max_results = st.sidebar.slider("Number of Papers", min_value=1, max_value=50, value=10)
    
    if st.sidebar.button("🚀 Fetch New Papers", type="primary"):
        st.sidebar.info("Fetching and processing papers... This may take a few minutes.")
        stats = fetch_new_papers(max_results)
        
        if stats['successful'] > 0:
            st.sidebar.success(f"Successfully processed {stats['successful']} papers!")
            st.sidebar.info(f"Cost: ${stats.get('total_cost', 0):.2f}")
            st.rerun()
    
    # Batch processing status
    if st.sidebar.checkbox("Show Batch Processing Status"):
        processor = get_batch_processor()
        checkpoints = processor.list_checkpoints()
        
        if checkpoints:
            st.sidebar.subheader("Recent Batches")
            for cp in checkpoints[:5]:
                st.sidebar.write(f"**{cp['name']}**")
                st.sidebar.write(f"- Papers: {cp['processed_count']}")
                st.sidebar.write(f"- Time: {cp['timestamp'][:16]}")
    
    # Main content tabs
    if existing_papers:
        tab1, tab2, tab3 = st.tabs(["📊 Summary", "📖 Browse Papers", "🎯 Get Recommendations"])
        
        with tab1:
            display_summary_statistics(existing_papers, storage_stats)
        
        with tab2:
            display_paper_browser(existing_papers)
            
        with tab3:
            display_recommendations_tab()
    else:
        st.info("No papers found. Use the sidebar to fetch new papers from arXiv.")
        
        # Show sample workflow
        st.subheader("🎯 How it works:")
        st.markdown("""
        1. **Fetch Papers**: Download GenAI research from arXiv with full text
        2. **Extract Insights**: AI analyzes papers for practical implementation details
        3. **Get Recommendations**: Receive personalized implementation plans based on your needs
        4. **Track Progress**: Monitor extraction costs and processing status
        """)
        
        st.subheader("✨ New Features:")
        st.markdown("""
        - **Hierarchical Extraction**: Smart processing that saves costs by analyzing only relevant sections
        - **Vector Search**: Find similar papers based on your specific use case
        - **Synthesis Engine**: Get actionable recommendations from multiple papers
        - **Implementation Roadmaps**: Step-by-step plans tailored to your constraints
        - **Local Storage**: All data stored locally with SQLite and ChromaDB
        """)

if __name__ == "__main__":
    main()