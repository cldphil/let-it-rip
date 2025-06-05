"""
Streamlit web interface for the GenAI Research Implementation Platform.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import json
from pathlib import Path

# Import core modules
from core import (
    InsightStorage,
    SyncBatchProcessor,
    SynthesisEngine,
    UserContext,
    TechniqueCategory,
    ComplexityLevel
)
from arxiv_fetcher import ArxivGenAIFetcher
from config import Config

# Page configuration
st.set_page_config(
    page_title="GenAI Research Platform",
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
</style>
""", unsafe_allow_html=True)

# Header with gradient
st.markdown("""
<div style="background: linear-gradient(135deg, #0066FF 0%, #00D4FF 100%); padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">🧬 GenAI Research Platform</h1>
    <p style="color: white; margin: 0.5rem 0 0 0; opacity: 0.9;">Transform research into actionable insights</p>
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Select Page",
    ["📊 Dashboard", "🔍 Fetch Papers", "📚 Browse Insights", "🎯 Get Recommendations", "⚙️ Settings"]
)

# Get current statistics
stats = st.session_state.storage.get_statistics()

# Dashboard Page
if page == "📊 Dashboard":
    st.header("Platform Overview")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Papers",
            stats['total_papers']
        )
    
    with col2:
        st.metric(
            "Insights Extracted",
            stats['total_insights']
        )
    
    with col3:
        st.metric(
            "Avg Quality Score",
            f"{stats['average_quality_score']:.2f}"
        )
    
    with col4:
        st.metric(
            "Total Cost",
            f"${stats['total_extraction_cost']:.2f}"
        )
    
    # Charts
    st.subheader("Research Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Complexity distribution
        if stats['complexity_distribution']:
            fig_complexity = px.pie(
                values=list(stats['complexity_distribution'].values()),
                names=list(stats['complexity_distribution'].keys()),
                title="Implementation Complexity",
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig_complexity, use_container_width=True)
    
    with col2:
        # Study type distribution
        if stats['study_type_distribution']:
            fig_study = px.bar(
                x=list(stats['study_type_distribution'].keys()),
                y=list(stats['study_type_distribution'].values()),
                title="Study Types",
                color_discrete_sequence=['#0066FF']
            )
            st.plotly_chart(fig_study, use_container_width=True)
    
    # Quality metrics
    st.subheader("Quality Metrics")
    
    quality_data = {
        'Metric': ['Quality Score', 'Evidence Strength', 'Practical Applicability', 'Key Findings/Paper'],
        'Average': [
            stats['average_quality_score'],
            stats['average_evidence_strength'],
            stats['average_practical_applicability'],
            stats['average_key_findings_count']
        ]
    }
    
    fig_quality = go.Figure(data=[
        go.Bar(
            x=quality_data['Metric'],
            y=quality_data['Average'],
            marker_color=['#0066FF', '#00D4FF', '#28A745', '#FF6B6B']
        )
    ])
    fig_quality.update_layout(title="Average Quality Metrics Across All Papers")
    st.plotly_chart(fig_quality, use_container_width=True)

# Fetch Papers Page
elif page == "🔍 Fetch Papers":
    st.header("Fetch New Research Papers")
    
    st.markdown("""
    <div class="info-box">
        <p><strong>📌 Note:</strong> This will fetch the latest GenAI papers from arXiv and extract insights using AI.</p>
    </div>
    """, unsafe_allow_html=True)
    
    num_papers = st.number_input(
        "Number of papers to fetch",
        min_value=1,
        max_value=50,
        value=10,
        help="Start with fewer papers to test the system"
    )
    
    if st.button("🚀 Fetch New Papers", use_container_width=True):
        with st.spinner("Fetching papers from arXiv..."):
            fetcher = ArxivGenAIFetcher()
            papers = fetcher.fetch_papers(
                max_results=num_papers,
                include_full_text=False  # Always False now
            )
            
            if papers:
                st.session_state.fetched_papers = papers  # Store in session state
                st.success(f"✅ Fetched {len(papers)} papers successfully!")
            else:
                st.error("❌ No papers fetched. Please try again.")
    
    # Show fetched papers if they exist in session state
    if 'fetched_papers' in st.session_state and st.session_state.fetched_papers:
        papers = st.session_state.fetched_papers
        
        # Show preview
        with st.expander("Preview fetched papers"):
            for i, paper in enumerate(papers[:5]):
                st.write(f"**{i+1}. {paper['title']}**")
                st.write(f"Authors: {', '.join(paper['authors'][:3])}")
                st.write(f"Published: {paper['published'][:10]}")
                st.write("---")
        
        # Process papers
        if st.button("💡 Extract Insights", use_container_width=True):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            checkpoint_name = f"streamlit_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Process in batches with progress updates
            total_papers = len(papers)
            batch_stats = {'successful': 0, 'failed': 0, 'total_cost': 0.0, 'total_time': 0.0}
            
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
            
            st.success(f"""
            ✅ Processing Complete!
            - Successfully processed: {batch_stats['successful']} papers
            - Failed: {batch_stats['failed']} papers
            - Total cost: ${batch_stats['total_cost']:.2f}
            - Time taken: {batch_stats['total_time']:.1f} seconds
            """)
            
            # Clear fetched papers from session state after processing
            del st.session_state.fetched_papers
            
            # Refresh the page to show new data
            st.rerun()

# Browse Insights Page
elif page == "📚 Browse Insights":
    st.header("Browse Research Insights")
    
    # Filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        complexity_filter = st.multiselect(
            "Complexity Level",
            options=['low', 'medium', 'high', 'very_high', 'unknown'],
            default=['low', 'medium']
        )
    
    with col2:
        min_quality = st.slider(
            "Minimum Quality Score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,  # Set to 0 to show all papers by default
            step=0.1
        )
    
    with col3:
        sort_by = st.selectbox(
            "Sort By",
            options=['quality_score', 'evidence_strength', 'practical_applicability', 'recency'],
            index=0
        )
    
    # Get all papers using the storage API
    all_papers = []
    
    try:
        # Get statistics to check if we have any papers
        stats = st.session_state.storage.get_statistics()
        
        if stats['total_insights'] > 0:
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
                                    insights.get_quality_score() >= min_quality):
                                    
                                    all_papers.append({
                                        'paper_id': paper_id,
                                        'title': paper_data.get('title', 'Unknown'),
                                        'authors': paper_data.get('authors', []),
                                        'published': paper_data.get('published', ''),
                                        'pdf_url': paper_data.get('pdf_url', ''),
                                        'insights': insights,
                                        'quality_score': insights.get_quality_score(),
                                        'evidence_strength': insights.evidence_strength,
                                        'practical_applicability': insights.practical_applicability,
                                        'recency': paper_data.get('published', '2020')
                                    })
                        except Exception as e:
                            st.warning(f"Error loading paper {paper_id}: {str(e)}")
                            continue
        
        # Sort papers
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
            # Display papers
            for paper in all_papers[:20]:  # Show top 20
                with st.expander(f"📄 {paper['title'][:100]}..."):
                    col1, col2 = st.columns([3, 1])
                    
                    with col1:
                        st.write(f"**Authors:** {', '.join(paper['authors'][:3])}")
                        st.write(f"**Published:** {paper['published'][:10] if paper['published'] else 'Unknown'}")
                        st.write(f"**Study Type:** {paper['insights'].study_type.value}")
                        st.write(f"**Complexity:** {paper['insights'].implementation_complexity.value}")
                        
                        techniques = [t.value for t in paper['insights'].techniques_used]
                        if techniques:
                            st.write(f"**Techniques:** {', '.join(techniques)}")
                        
                        # Key findings
                        if paper['insights'].key_findings:
                            st.write("**Key Findings:**")
                            for i, finding in enumerate(paper['insights'].key_findings[:5], 1):
                                st.write(f"{i}. {finding}")
                        
                        # Show limitations if any
                        if paper['insights'].limitations:
                            st.write("**Limitations:**")
                            for limitation in paper['insights'].limitations[:3]:
                                st.write(f"- {limitation}")
                        
                        # Link to full text
                        if paper.get('pdf_url'):
                            st.markdown(f"📄 [View Full Paper PDF]({paper['pdf_url']})")
                    
                    with col2:
                        # Quality metrics
                        st.metric("Quality Score", f"{paper['quality_score']:.2f}")
                        st.metric("Evidence", f"{paper['evidence_strength']:.2f}")
                        st.metric("Applicability", f"{paper['practical_applicability']:.2f}")
                        
                        # Badges
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
        
        col1, col2 = st.columns(2)
        
        with col1:
            company_size = st.selectbox(
                "Company Size (Optional)",
                options=["", "startup", "small", "medium", "large", "enterprise"],
                index=0
            )
            
            maturity_level = st.selectbox(
                "AI Maturity Level (Optional)",
                options=["", "greenfield", "pilot_ready", "scaling", "optimizing"],
                index=0
            )
            
        with col2:
            budget = st.selectbox(
                "Budget (Optional)",
                options=["", "low", "medium", "high", "unlimited"],
                index=0
            )
            
            risk_tolerance = st.selectbox(
                "Risk Tolerance (Optional)",
                options=["", "conservative", "moderate", "aggressive"],
                index=0
            )
        
        business_context = st.text_area(
            "Describe your business context (Optional)",
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
            company_size=company_size if company_size else "medium",
            maturity_level=maturity_level if maturity_level else "pilot_ready",
            budget_constraint=budget if budget else None,
            risk_tolerance=risk_tolerance if risk_tolerance else "moderate",
            preferred_techniques=[],  # Removed from UI
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
                    st.metric("Analysis Date", datetime.now().strftime("%Y-%m-%d"))
                
                with col2:
                    if 'top_paper_scores' in metadata:
                        st.write("**Top Research Sources:**")
                        for paper in metadata['top_paper_scores'][:3]:
                            st.write(f"- {paper['title']}")
                            st.write(f"  Score: {paper['score']:.2f}")
            
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
            st.success(f"✅ Analyzed {synthesis_result['papers_analyzed']} papers")
            
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
    
    # Configuration
    st.subheader("Configuration")
    
    st.info(f"""
    **Current Configuration:**
    - LLM Model: {Config.LLM_MODEL}
    - Batch Size: {Config.BATCH_SIZE}
    - Max Key Findings: {Config.MAX_KEY_FINDINGS}
    - Recency Weight: {Config.RECENCY_WEIGHT}
    """)
    
    # About
    st.subheader("About")
    
    st.markdown("""
    <div class="info-box">
        <h4>GenAI Research Implementation Platform</h4>
        <p>Transform cutting-edge AI research into actionable business insights.</p>
        <p><strong>Version:</strong> 1.0.0</p>
        <p><strong>License:</strong> MIT</p>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>© 2025</p>
</div>
""", unsafe_allow_html=True)