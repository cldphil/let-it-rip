"""
Streamlit web interface for the GenAI Research Implementation Platform.
Updated to remove deprecated evidence_strength and practical_applicability fields.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
import json
from pathlib import Path

# Import core modules
from core import (
    InsightStorage,
    SyncBatchProcessor,
    SynthesisEngine,
    UserContext
)
from services.arxiv_fetcher import ArxivGenAIFetcher
from config import Config

# Page configuration
st.set_page_config(
    page_title="Research Implementation Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'storage' not in st.session_state:
    st.session_state.storage = InsightStorage()
if 'processor' not in st.session_state:
    st.session_state.processor = SyncBatchProcessor()
if 'synthesis_engine' not in st.session_state:
    st.session_state.synthesis_engine = SynthesisEngine()

# Professional styling
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary-color: #0066FF;
        --secondary-color: #00D4FF;
        --accent-color: #FF6B6B;
        --text-primary: #1A1A1A;
        --text-secondary: #666666;
        --bg-primary: #FFFFFF;
        --bg-secondary: #F8F9FA;
        --border-color: #E5E7EB;
    }
    
    /* Main container styling */
    .main {
        background-color: var(--bg-secondary);
    }
    
    /* Card styling */
    .stExpander {
        background-color: var(--bg-primary);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Metric styling */
    div[data-testid="metric-container"] {
        background-color: var(--bg-primary);
        border: 1px solid var(--border-color);
        padding: 1rem;
        border-radius: 8px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--text-primary);
        font-weight: 600;
    }
    
    /* Buttons */
    .stButton > button {
        background-color: var(--primary-color);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background-color: #0052CC;
        transform: translateY(-1px);
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* Info boxes */
    .info-box {
        background-color: #EBF5FF;
        border-left: 4px solid var(--primary-color);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Success boxes */
    .success-box {
        background-color: #D4EDDA;
        border-left: 4px solid #28A745;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Warning boxes */
    .warning-box {
        background-color: #FFF3CD;
        border-left: 4px solid #FFC107;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Case study badge */
    .case-study-badge {
        background-color: #28A745;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.875rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    
    /* Industry validation badge */
    .validation-badge {
        background-color: #007BFF;
        color: white;
        padding: 0.25rem 0.5rem;
        border-radius: 4px;
        font-size: 0.875rem;
        font-weight: 600;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# Header with gradient
st.markdown("""
<div style="background: linear-gradient(135deg, #0066FF 0%, #00D4FF 100%); padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">Research Implementation Platform</h1>
    <p style="color: white; margin: 0.5rem 0 0 0; opacity: 0.9;">Transform research into actionable insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Dashboard", "📚 Browse Insights", "🎯 Get Recommendations", "⚙️ Settings"]
)

# Dashboard Page
if page == "📊 Dashboard":
    st.header("Platform Overview")
    
    # Always get fresh statistics when dashboard loads
    current_stats = st.session_state.storage.get_statistics()
    
    # Key metrics - updated to remove deprecated fields
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Papers",
            current_stats['total_papers']
        )
    
    with col2:
        # Reputation score metric
        if current_stats.get('total_insights', 0) > 0:
            st.metric(
                "Avg Reputation Score",
                f"{current_stats['average_reputation_score']:.2f}",
                help="Based on author h-index and conference validation"
            )
        else:
            st.metric(
                "Avg Reputation Score",
                "N/A",
                help="Process papers to see reputation scores"
            )

    with col3:
        # Industry validated papers
        if current_stats.get('total_insights', 0) > 0:
            validation_rate = current_stats.get('industry_validated_count', 0) / current_stats['total_insights']
            st.metric(
                "Industry Validated",
                f"{current_stats.get('industry_validated_count', 0)} ({validation_rate:.0%})",
                help="Papers with real-world validation"
            )
        else:
            st.metric(
                "Industry Validated",
                "N/A",
                help="Process papers to see validation status"
            )
    
    with col4:
        # Case studies metric
        if current_stats.get('total_insights', 0) > 0:
            case_study_rate = current_stats.get('case_studies_count', 0) / current_stats['total_insights']
            st.metric(
                "Case Studies",
                f"{current_stats.get('case_studies_count', 0)} ({case_study_rate:.0%})",
                help="Real-world implementation studies"
            )
        else:
            st.metric(
                "Case Studies",
                "N/A",
                help="Process papers to see case study count"
            )
    
    # Additional metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Avg Key Findings",
            f"{current_stats.get('average_key_findings_count', 0):.1f}",
            help="Average number of key findings per paper"
        )
    
    with col2:
        st.metric(
            "Papers with Code",
            current_stats.get('papers_with_code', 0),
            help="Papers with available implementation code"
        )
    
    with col3:
        st.metric(
            "Recent Papers",
            current_stats.get('recent_papers_count', 0),
            help="Papers from the last 2 years"
        )
    
    with col4:
        # Calculate processing rate if we have data
        if current_stats.get('total_papers', 0) > 0 and current_stats.get('total_insights', 0) > 0:
            processing_rate = current_stats['total_insights'] / current_stats['total_papers']
            st.metric(
                "Processing Rate",
                f"{processing_rate:.0%}",
                help="Percentage of papers successfully processed"
            )
        else:
            st.metric(
                "Processing Rate",
                "N/A",
                help="Process papers to see success rate"
            )
    
    # Charts
    st.subheader("Research Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Complexity distribution
        if current_stats['complexity_distribution']:
            fig_complexity = px.bar(
                x=list(current_stats['complexity_distribution'].keys()),
                y=list(current_stats['complexity_distribution'].values()),
                title="Implementation Complexity",
                color_discrete_sequence=['#0066FF'],
                labels={'x': 'Complexity Level', 'y': 'Number of Papers'}
            )
            fig_complexity.update_layout(
                showlegend=False,
                title_font_size=16,
                font_size=12
            )
            st.plotly_chart(fig_complexity, use_container_width=True)
    
    with col2:
        # Study type distribution
        if current_stats['study_type_distribution']:
            fig_study = px.bar(
                x=list(current_stats['study_type_distribution'].keys()),
                y=list(current_stats['study_type_distribution'].values()),
                title="Study Types",
                color_discrete_sequence=['#00D4FF'],
                labels={'x': 'Study Type', 'y': 'Number of Papers'}
            )
            fig_study.update_layout(
                showlegend=False,
                title_font_size=16,
                font_size=12
            )
            st.plotly_chart(fig_study, use_container_width=True)
    
    st.markdown("---")
    st.subheader("Get New Insights")
    
    st.markdown("""
    <div class="info-box">
        <p><strong>Fetch and analyze the latest GenAI papers from arXiv</strong></p>
        <p>This will retrieve recent papers and extract actionable insights using AI analysis.</p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        num_papers = st.number_input(
            "Number of papers to analyze",
            min_value=1,
            max_value=50,
            value=10,
            help="Start with fewer papers to test the system. Each paper costs approximately $0.005 to analyze."
        )
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        get_insights_button = st.button("Get Insights", use_container_width=True, type="primary")
    
    if get_insights_button:
        # Combined fetch and extract process
        with st.spinner(f"Fetching {num_papers} papers from arXiv..."):
            fetcher = ArxivGenAIFetcher()
            papers = fetcher.fetch_papers(
                max_results=num_papers,
                include_full_text=True
            )
            
            if not papers:
                st.error("❌ No papers fetched. Please try again.")
            else:
                st.success(f"✅ Fetched {len(papers)} papers successfully!")
                
                # Immediately process the papers
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                checkpoint_name = f"dashboard_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                
                # Process in batches with progress updates
                total_papers = len(papers)
                batch_stats = {'successful': 0, 'failed': 0, 'total_cost': 0.0, 'total_time': 0.0}
                
                status_text.text("Extracting insights from papers...")
                
                for i in range(0, total_papers, 5):
                    batch = papers[i:i+5]
                    status_text.text(f"Processing papers {i+1} to {min(i+5, total_papers)}...")
                    
                    stats = st.session_state.processor.process_papers(
                        batch,
                        checkpoint_name=checkpoint_name,
                        force_reprocess=False
                    )
                    
                    # Accumulate stats
                    batch_stats['successful'] += stats.get('successful', 0)
                    batch_stats['failed'] += stats.get('failed', 0)
                    batch_stats['total_cost'] += stats.get('total_cost', 0.0)
                    batch_stats['total_time'] += stats.get('total_time', 0.0)
                    
                    progress_bar.progress((i + len(batch)) / total_papers)
                
                progress_bar.progress(1.0)
                status_text.text("Processing complete!")
                
                # Show results
                st.markdown("---")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Papers Processed", batch_stats['successful'], delta=f"+{batch_stats['successful']}")
                
                with col2:
                    st.metric("Failed", batch_stats['failed'])
                
                with col3:
                    st.metric("Processing Cost", f"${batch_stats['total_cost']:.2f}")
                
                with col4:
                    st.metric("Time Taken", f"{batch_stats['total_time']:.1f}s")
                
                st.success("""
                ✅ **Analysis Complete!**  
                Navigate to 'Browse Insights' to explore the extracted insights or 'Get Recommendations' for personalized guidance.
                """)
                
                # Refresh the page to show updated statistics
                st.rerun()

# Browse Insights Page
elif page == "📚 Browse Insights":
    st.header("Browse Research Insights")
    
    # Enhanced filters with better styling - removed deprecated field filters
    st.markdown("#### Filter Research Papers")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        complexity_filter = st.multiselect(
            "Implementation Complexity",
            options=['low', 'medium', 'high', 'very_high', 'unknown'],
            default=['low', 'medium', 'high', 'very_high', 'unknown'],
            help="Filter by how complex the implementation would be"
        )
    
    with col2:
        study_type_filter = st.multiselect(
            "Research Type",
            options=['empirical', 'case_study', 'theoretical', 'pilot', 'survey', 'meta_analysis', 'review', 'unknown'],
            default=['empirical', 'case_study', 'theoretical', 'pilot', 'survey', 'meta_analysis', 'review', 'unknown'],
            help="Filter by type of research study"
        )
    
    with col3:
        # Updated sort options - removed deprecated fields
        sort_by = st.selectbox(
            "Sort Results By",
            options=['reputation_score', 'recency', 'key_findings_count', 'extraction_confidence'],
            index=0,
            help="Choose how to order the results"
        )
    
    # Additional filter row
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_case_studies_only = st.checkbox("Show only case studies", value=False)
    
    with col2:
        show_validated_only = st.checkbox("Show only industry validated", value=False)
    
    with col3:
        show_with_code_only = st.checkbox("Show only papers with code", value=False)
    
    # Get all papers using the storage API
    all_papers = []
    
    try:
        # Get statistics to check if we have any papers
        stats = st.session_state.storage.get_statistics()
        
        # Use the storage API to get papers
        insights_dir = Path("storage/insights")
        
        if insights_dir.exists():
            for insight_file in insights_dir.glob("*.json"):
                if "_insights" in insight_file.name:
                    paper_id = insight_file.stem.replace("_insights", "")
                    
                    try:
                        insights = st.session_state.storage.load_insights(paper_id)
                        paper_data = st.session_state.storage.load_paper(paper_id)
                        
                        if insights and paper_data:
                            # Apply filters
                            if (insights.implementation_complexity.value in complexity_filter and
                                insights.study_type.value in study_type_filter):
                                
                                # Apply case study filter if checked
                                if show_case_studies_only and insights.study_type.value != 'case_study':
                                    continue
                                
                                # Apply validation filter if checked
                                if show_validated_only and not insights.industry_validation:
                                    continue
                                
                                # Apply code availability filter if checked
                                if show_with_code_only and not insights.has_code_available:
                                    continue
                                
                                all_papers.append({
                                    'paper_id': paper_id,
                                    'title': paper_data.get('title', 'Unknown'),
                                    'authors': paper_data.get('authors', []),
                                    'published': paper_data.get('published', ''),
                                    'pdf_url': paper_data.get('pdf_url', ''),
                                    'insights': insights,
                                    'reputation_score': insights.get_reputation_score(),
                                    'recency': paper_data.get('published', '2020'),
                                    'key_findings_count': len(insights.key_findings),
                                    'extraction_confidence': insights.extraction_confidence
                                })
                    except Exception as e:
                        st.warning(f"Error loading paper {paper_id}: {str(e)}")
                        continue
        
        # Sort papers based on selected criteria
        if all_papers:
            if sort_by == 'recency':
                all_papers.sort(key=lambda x: x['published'], reverse=True)
            else:
                all_papers.sort(key=lambda x: x[sort_by], reverse=True)
        
        # Display results
        st.write(f"Found {len(all_papers)} papers matching filters")
        
        if not all_papers:
            st.info("No papers found. Try fetching some papers first or adjusting your filters.")
        else:
            # Display papers with updated metrics
            for paper in all_papers[:50]:  # Show top 50
                with st.expander(f"📄 {paper['title'][:100]}..."):
                    # Add badges for case study and validation status
                    badges_html = ""
                    if paper['insights'].study_type.value == 'case_study':
                        badges_html += '<span class="case-study-badge">CASE STUDY</span> '
                    if paper['insights'].industry_validation:
                        badges_html += '<span class="validation-badge">INDUSTRY VALIDATED</span>'
                    
                    if badges_html:
                        st.markdown(badges_html, unsafe_allow_html=True)
                    
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Authors:** {', '.join(paper['authors'][:3])}")
                        st.write(f"**Published:** {paper['published'][:10] if paper['published'] else 'Unknown'}")
                        st.write(f"**Study Type:** {paper['insights'].study_type.value}")
                        st.write(f"**Complexity:** {paper['insights'].implementation_complexity.value}")
                        
                        techniques = [t.value for t in paper['insights'].techniques_used]
                        if techniques:
                            st.write(f"**Techniques:** {', '.join(techniques)}")
                        
                        # Problem addressed
                        if paper['insights'].problem_addressed:
                            st.write(f"**Problem Addressed:** {paper['insights'].problem_addressed}")
                        
                        # Key findings
                        if paper['insights'].key_findings:
                            st.write(f"**Key Findings ({len(paper['insights'].key_findings)}):**")
                            for i, finding in enumerate(paper['insights'].key_findings[:5], 1):
                                st.write(f"{i}. {finding}")
                        
                        # Real-world applications
                        if paper['insights'].real_world_applications:
                            st.write("**Real-World Applications:**")
                            for app in paper['insights'].real_world_applications[:3]:
                                st.write(f"- {app}")
                        
                        # Show limitations if any
                        if paper['insights'].limitations:
                            st.write("**Limitations:**")
                            for limitation in paper['insights'].limitations[:3]:
                                st.write(f"- {limitation}")
                        
                        # Link to full text
                        if paper.get('pdf_url'):
                            st.markdown(f"📄 [View Full Paper PDF]({paper['pdf_url']})")
                    
                    with col2:
                        # Updated reputation metrics - removed deprecated fields
                        st.metric("Reputation Score", f"{paper['reputation_score']:.2f}")
                        st.metric("Key Findings", paper['key_findings_count'])
                        st.metric("Extraction Confidence", f"{paper['extraction_confidence']:.2f}")
                        
                        # Publication year
                        if paper['published']:
                            try:
                                pub_year = int(paper['published'][:4])
                                st.metric("Year", pub_year)
                            except:
                                st.metric("Year", "Unknown")
                        
                        # Status badges
                        if paper['insights'].has_code_available:
                            st.success("✅ Code Available")
                        if paper['insights'].has_dataset_available:
                            st.success("✅ Dataset Available")
                        if paper['insights'].industry_validation:
                            st.success("✅ Industry Validated")
                            
    except Exception as e:
        st.error(f"Error loading papers: {str(e)}")
        st.info("Try refreshing the page or checking if papers have been processed.")

# Get Recommendations Page
elif page == "🎯 Get Recommendations":
    st.header("Get Personalized Recommendations")
    
    st.markdown("""
    <div class="info-box">
        <p><strong>💡 How it works:</strong> Our AI consultant will analyze research papers and provide strategic recommendations tailored to your business context.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User context form
    with st.form("user_context_form"):
        st.subheader("Your Context")
        
        business_context = st.text_area(
            "Describe your business context, such as company size, industry, team size, budget constraints, and AI maturity level (Optional)",
            placeholder="e.g., We are a mid-size healthcare company looking to improve patient engagement...",
            height=100
        )
        
        # Just looking for ideas option
        just_looking = st.checkbox("Just looking for ideas", value=False)
        
        specific_problems = ""
        if not just_looking:
            specific_problems = st.text_area(
                "Specific problems to solve (Optional)",
                placeholder="e.g., Reduce response time, improve accuracy of technical answers...",
                height=80
            )
        
        submit_button = st.form_submit_button("🔍 Get Recommendations", use_container_width=True)
    
    if submit_button:
        # Create user context with optional fields
        context = UserContext(
            use_case_description=business_context if business_context else "General exploration of GenAI applications",
            specific_problems=[p.strip() for p in specific_problems.split('\n') if p.strip()] if specific_problems else []
        )
        
        # Check if synthesis_engine has interactive mode support
        if hasattr(st.session_state.synthesis_engine, 'synthesize_recommendations'):
            # Get method signature to check if it supports interactive parameter
            import inspect
            sig = inspect.signature(st.session_state.synthesis_engine.synthesize_recommendations)
            if 'interactive' in sig.parameters:
                with st.spinner("Our AI consultant is analyzing research and preparing recommendations..."):
                    synthesis_result = st.session_state.synthesis_engine.synthesize_recommendations(context, interactive=True)
            else:
                with st.spinner("Analyzing research and generating recommendations..."):
                    synthesis_result = st.session_state.synthesis_engine.synthesize_recommendations(context)
        else:
            with st.spinner("Analyzing research and generating recommendations..."):
                synthesis_result = st.session_state.synthesis_engine.synthesize_recommendations(context)
        
        # Store synthesis in session state
        st.session_state.current_synthesis = synthesis_result
        st.session_state.current_context = context
        
        # Check if we have consultant analysis (interactive mode)
        if 'consultant_analysis' in synthesis_result:
            # Display consultant analysis
            st.markdown("---")
            st.markdown("## 📊 Strategic Analysis & Recommendations")
            st.markdown(synthesis_result['consultant_analysis'])
            
            # Show metadata in an expander
            with st.expander("📈 Analysis Details"):
                metadata = synthesis_result.get('metadata', {})
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Papers Analyzed", metadata.get('papers_analyzed', 0))
                    st.metric("Case Studies Included", metadata.get('case_studies_included', 0))
                
                with col2:
                    st.metric("Analysis Date", datetime.now().strftime("%Y-%m-%d"))
                    if 'top_paper_scores' in metadata:
                        st.write("**Top Research Sources:**")
                        for paper in metadata['top_paper_scores'][:3]:
                            case_tag = " [CASE STUDY]" if paper.get('is_case_study') else ""
                            st.write(f"- {paper['title']}{case_tag}")
                            # Updated score components display
                            components = paper.get('components', {})
                            st.write(f"  Reputation: {components.get('reputation', 0):.2f} | "
                                   f"Recency: {components.get('recency', 0):.2f} | "
                                   f"Case Study: {components.get('case_study', 0):.2f} | "
                                   f"Validation: {components.get('validation', 0):.2f}")
            
            # Interactive options
            st.markdown("---")
            st.subheader("🔄 Next Steps")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("📋 Generate Detailed Roadmap", use_container_width=True):
                    st.session_state.show_roadmap = True
                    st.rerun()
                    
            with col2:
                if st.button("🔍 Explore Alternatives", use_container_width=True):
                    st.session_state.show_alternatives = True
                    st.rerun()
                    
            with col3:
                if st.button("💬 Ask Follow-up Question", use_container_width=True):
                    st.session_state.show_followup = True
                    st.rerun()
        else:
            # Standard recommendations display (non-interactive mode)
            st.success(f"✅ Analyzed {synthesis_result.get('papers_analyzed', 0)} papers")
            
            # Top approaches
            st.subheader("🎯 Recommended Approaches")
            
            recommendations = synthesis_result.get('recommendations', {})
            for i, approach in enumerate(recommendations.get('top_approaches', []), 1):
                with st.expander(f"{i}. {approach['approach_name']} (Confidence: {approach['confidence_score']:.0%})"):
                    st.write(f"**Why recommended:** {approach['why_recommended']}")
                    st.write(f"**Complexity:** {approach['complexity']}")
                    
                    st.write("**Example Implementations:**")
                    for example in approach.get('example_implementations', []):
                        st.write(f"- {example['title']}")
                        st.write(f"  *Key insight: {example['key_insight']}*")
            
            # Success factors and pitfalls
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("✅ Success Factors")
                for factor in recommendations.get('success_factors', []):
                    st.write(f"- {factor}")
            
            with col2:
                st.subheader("⚠️ Common Pitfalls")
                for pitfall in recommendations.get('common_pitfalls', []):
                    st.write(f"- {pitfall}")
            
            # Implementation roadmap
            if synthesis_result.get('implementation_roadmap'):
                st.subheader("📋 Implementation Roadmap")
                
                roadmap = synthesis_result['implementation_roadmap']
                st.write(f"**Approach:** {roadmap['approach']}")
                st.write(f"**Total Duration:** {roadmap['total_duration_weeks']} weeks")
                
                # Phase details
                for phase in roadmap.get('phases', []):
                    with st.expander(f"Phase {phase['phase_number']}: {phase['name']} ({phase['duration_weeks']} weeks)"):
                        st.write("**Activities:**")
                        for activity in phase.get('activities', []):
                            st.write(f"- {activity}")
                        
                        st.write("**Deliverables:**")
                        for deliverable in phase.get('deliverables', []):
                            st.write(f"- {deliverable}")
                        
                        if phase.get('prerequisites'):
                            st.write("**Prerequisites:**")
                            for prereq in phase['prerequisites']:
                                st.write(f"- {prereq}")
                
                # Risk mitigation
                st.subheader("🛡️ Risk Mitigation")
                for risk in roadmap.get('risk_mitigation', []):
                    st.write(f"**{risk['risk']}:** {risk['mitigation']}")
            
            # Confidence score
            st.metric("Overall Confidence", f"{synthesis_result.get('confidence_score', 0):.0%}")
    
    # Handle interactive next steps
    if hasattr(st.session_state, 'show_roadmap') and st.session_state.show_roadmap:
        st.markdown("---")
        st.subheader("📋 Implementation Roadmap")
        
        # Check if synthesis engine has the method
        if hasattr(st.session_state.synthesis_engine, 'generate_implementation_roadmap'):
            with st.spinner("Generating detailed implementation roadmap..."):
                roadmap_result = st.session_state.synthesis_engine.generate_implementation_roadmap(
                    st.session_state.current_synthesis,
                    st.session_state.current_context
                )
            
            if 'error' not in roadmap_result:
                st.markdown(roadmap_result['implementation_roadmap'])
                
                # Download button for roadmap
                st.download_button(
                    label="📥 Download Roadmap",
                    data=roadmap_result['implementation_roadmap'],
                    file_name=f"implementation_roadmap_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
            else:
                st.error(f"Failed to generate roadmap: {roadmap_result['error']}")
        else:
            st.info("Detailed roadmap generation is not available. Using the standard roadmap from recommendations.")
        
        # Reset flag
        st.session_state.show_roadmap = False
        
        # Button to go back
        if st.button("⬅️ Back to Recommendations"):
            st.rerun()
    
    # Handle alternatives
    if hasattr(st.session_state, 'show_alternatives') and st.session_state.show_alternatives:
        st.markdown("---")
        st.subheader("🔍 Alternative Approaches")
        
        # Check if synthesis engine has the method
        if hasattr(st.session_state.synthesis_engine, 'explore_alternative_approaches'):
            with st.spinner("Exploring alternative approaches..."):
                # Re-retrieve papers for alternatives
                relevant_papers = st.session_state.storage.find_similar_papers(
                    st.session_state.current_context, 
                    n_results=50
                )
                
                alternatives_result = st.session_state.synthesis_engine.explore_alternative_approaches(
                    relevant_papers,
                    st.session_state.current_context,
                    num_alternatives=3
                )
            
            if 'error' not in alternatives_result:
                st.markdown(alternatives_result['alternative_approaches'])
                
                # Download button for alternatives
                st.download_button(
                    label="📥 Download Alternatives Analysis",
                    data=alternatives_result['alternative_approaches'],
                    file_name=f"alternative_approaches_{datetime.now().strftime('%Y%m%d')}.md",
                    mime="text/markdown"
                )
            else:
                st.error(f"Failed to generate alternatives: {alternatives_result['error']}")
        else:
            st.info("Alternative approaches exploration is not available in the current version.")
        
        # Reset flag
        st.session_state.show_alternatives = False
        
        # Button to go back
        if st.button("⬅️ Back to Recommendations"):
            st.rerun()
    
    # Handle follow-up questions
    if hasattr(st.session_state, 'show_followup') and st.session_state.show_followup:
        st.markdown("---")
        st.subheader("💬 Follow-up Questions")
        
        st.info("🚧 Interactive Q&A functionality is coming soon!")
        st.write("For now, you can:")
        st.write("- Generate a detailed implementation roadmap")
        st.write("- Explore alternative approaches")
        st.write("- Re-run the analysis with different parameters")
        
        # Reset flag
        st.session_state.show_followup = False
        
        # Button to go back
        if st.button("⬅️ Back to Recommendations"):
            st.rerun()

# Settings Page
elif page == "⚙️ Settings":
    st.header("Platform Settings")
    
    # Storage management
    st.subheader("Storage Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            if st.checkbox("I understand this will delete all data"):
                st.session_state.storage.clear_all()
                st.success("✅ All data cleared successfully")
                st.rerun()
    
    with col2:
        if st.button("📊 Export Statistics", use_container_width=True):
            # Get fresh statistics
            export_stats = st.session_state.storage.get_statistics()
            stats_json = json.dumps(export_stats, indent=2)
            st.download_button(
                label="Download Statistics",
                data=stats_json,
                file_name=f"genai_stats_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.rerun()
    
    # Enhanced statistics display
    st.subheader("Current Statistics")
    
    # Get current statistics for display
    current_stats = st.session_state.storage.get_statistics()
    
    # Create a comprehensive statistics table
    stats_data = {
        "Metric": [
            "Total Papers",
            "Total Insights",
            "Average Reputation Score",
            "Average Key Findings Count",
            "Papers with Code",
            "Industry Validated Papers",
            "Case Studies",
            "Recent Papers (Last 2 Years)"
        ],
        "Value": [
            current_stats.get('total_papers', 0),
            current_stats.get('total_insights', 0),
            f"{current_stats.get('average_reputation_score', 0):.2f}",
            f"{current_stats.get('average_key_findings_count', 0):.1f}",
            current_stats.get('papers_with_code', 0),
            current_stats.get('industry_validated_count', 0),
            current_stats.get('case_studies_count', 0),
            current_stats.get('recent_papers_count', 0)
        ]
    }
    
    stats_df = pd.DataFrame(stats_data)
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Configuration
    st.subheader("Configuration")
    
    st.info(f"""
    **Current Configuration:**
    - LLM Model: {Config.LLM_MODEL}
    - Batch Size: {Config.BATCH_SIZE}
    - Max Key Findings: {Config.MAX_KEY_FINDINGS}
    - Recency Weight: {Config.RECENCY_WEIGHT}
    - Reputation Weight: {Config.REPUTATION_WEIGHT}
    """)
    
    # Reputation Score Information
    st.subheader("Reputation Score Information")
    
    st.markdown("""
    <div class="info-box">
        <h4>Reputation Score Calculation</h4>
        <p>The reputation score is automatically calculated based on:</p>
        <ul>
            <li><strong>Author H-Index:</strong> Sum of h-indices for all paper authors</li>
            <li><strong>Conference Validation:</strong> 1.5x multiplier if published at a recognized conference</li>
            <li><strong>Formula:</strong> (total_author_hindex × conference_multiplier) / 100</li>
        </ul>
        <p>This provides an objective measure of research credibility and impact.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # About
    st.subheader("About")
    
    st.markdown("""
    <div class="info-box">
        <h4>GenAI Research Implementation Platform</h4>
        <p>Transform cutting-edge AI research into actionable business insights.</p>
        <p><strong>Version:</strong> 2.0.0</p>
        <p><strong>License:</strong> MIT</p>
        <p><strong>Updates in v2.0:</strong></p>
        <ul>
            <li>Removed subjective evidence metrics</li>
            <li>Enhanced reputation scoring with author h-index</li>
            <li>Improved case study validation</li>
            <li>Better industry validation tracking</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>© 2025 GenAI Research Platform | Enhanced with Objective Reputation Metrics</p>
</div>
""", unsafe_allow_html=True)