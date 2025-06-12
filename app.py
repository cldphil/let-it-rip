"""
Streamlit web interface for the GenAI Research Implementation Platform.
Cloud-optimized version for Streamlit Cloud deployment.
"""

import os
import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import json

# Streamlit Cloud secrets handling
if hasattr(st, 'secrets'):
    for key, value in st.secrets.items():
        os.environ[key] = str(value)

# Import core modules with error handling for deployment
try:
    from core import (
        InsightStorage,
        SyncBatchProcessor,
        SynthesisEngine,
        UserContext
    )
    from config import Config
    from components.datetime_utils import prepare_dataframe_for_streamlit
    
    # Import manual processing if available
    try:
        from core.manual_processing_system import ManualProcessingController
        MANUAL_PROCESSING_AVAILABLE = True
    except ImportError:
        MANUAL_PROCESSING_AVAILABLE = False
        
except ImportError as e:
    st.error(f"Error importing modules: {e}")
    st.error("Please check that all required packages are installed.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Research Implementation Platform",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state with error handling
@st.cache_resource
def initialize_components():
    """Initialize core components with caching for Streamlit Cloud."""
    try:
        storage = InsightStorage()
        processor = SyncBatchProcessor()
        synthesis_engine = SynthesisEngine()
        manual_controller = ManualProcessingController() if MANUAL_PROCESSING_AVAILABLE else None
        
        return storage, processor, synthesis_engine, manual_controller
    except Exception as e:
        st.error(f"Failed to initialize components: {e}")
        return None, None, None, None

# Initialize components
storage, processor, synthesis_engine, manual_controller = initialize_components()

if not storage:
    st.error("Failed to initialize the application. Please check your configuration.")
    st.stop()

# Store in session state
if 'storage' not in st.session_state:
    st.session_state.storage = storage
if 'processor' not in st.session_state:
    st.session_state.processor = processor
if 'synthesis_engine' not in st.session_state:
    st.session_state.synthesis_engine = synthesis_engine
if 'manual_controller' not in st.session_state:
    st.session_state.manual_controller = manual_controller

# Professional styling
st.markdown("""
<style>
    /* Modern color scheme */
    :root {
        --primary-color: #0066FF;
        --secondary-color: #00D4FF;
        --accent-color: #FF6B6B;
        --success-color: #28A745;
        --warning-color: #FFC107;
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
        border-left: 4px solid var(--success-color);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Warning boxes */
    .warning-box {
        background-color: #FFF3CD;
        border-left: 4px solid var(--warning-color);
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    /* Cloud storage indicator */
    .cloud-indicator {
        background-color: #E7F3FF;
        border: 1px solid #B3D9FF;
        border-radius: 6px;
        padding: 0.5rem;
        margin: 0.5rem 0;
        font-size: 0.875rem;
    }
</style>
""", unsafe_allow_html=True)

# Header with gradient
st.markdown("""
<div style="background: linear-gradient(135deg, #0066FF 0%, #00D4FF 100%); padding: 2rem; border-radius: 12px; margin-bottom: 2rem;">
    <h1 style="color: white; margin: 0; font-size: 2.5rem;">Research Implementation Platform</h1>
    <p style="color: white; margin: 0.5rem 0 0 0; opacity: 0.9;">Transform cutting-edge research into actionable insights</p>
</div>
""", unsafe_allow_html=True)

# Environment status check for deployment debugging
with st.expander("System Status", expanded=False):
    st.write("**Environment Check:**")
    
    # Check essential environment variables
    env_status = {
        "Supabase URL": "✅ Set" if os.getenv('SUPABASE_URL') else "❌ Missing",
        "Supabase Key": "✅ Set" if os.getenv('SUPABASE_ANON_KEY') else "❌ Missing", 
        "Anthropic API": "✅ Set" if os.getenv('ANTHROPIC_API_KEY') else "❌ Missing",
        "Manual Processing": "✅ Available" if MANUAL_PROCESSING_AVAILABLE else "❌ Unavailable"
    }
    
    for item, status in env_status.items():
        st.write(f"- {item}: {status}")

# Cloud storage indicator
st.markdown("""
<div class="cloud-indicator">
    ☁️ <strong>Streamlit Cloud Deployment</strong> - Data synced to Supabase
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")

# Dynamic page list based on available features
page_options = ["📊 Dashboard", "🎯 Get Recommendations"]

if MANUAL_PROCESSING_AVAILABLE:
    page_options.append("📅 Manual Processing")

page_options.append("⚙️ Settings")

page = st.sidebar.radio("Select Page", page_options)

# Dashboard Page
if page == "📊 Dashboard":
    st.header("Platform Overview")
    
    # Get statistics with error handling
    try:
        current_stats = st.session_state.storage.get_statistics()
    except Exception as e:
        st.error(f"Error loading statistics: {e}")
        current_stats = {}
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Total Papers",
            current_stats.get('total_papers', 0)
        )

    with col2:
        # Case studies metric
        if current_stats.get('total_papers', 0) > 0:
            case_study_rate = current_stats.get('case_studies_count', 0) / current_stats['total_papers']
            st.metric(
                "Case Studies",
                f"{current_stats.get('case_studies_count', 0)} ({case_study_rate:.0%})",
                help="Real-world implementation studies"
            )
        else:
            st.metric(
                "Case Studies",
                "No data",
                help="Process papers to see case study count"
            )
    
    with col3:
        if current_stats.get('total_papers', 0) > 0:
            code_rate = current_stats.get('papers_with_code', 0) / current_stats['total_papers']
            st.metric(
                "Papers with Code",
                f"{current_stats.get('papers_with_code', 0)} ({code_rate:.0%})",
                help="Papers with available implementation code"
            )
        else:
            st.metric("Papers with Code", "No data")

    with col4:
        st.metric(
            "Recent Papers",
            current_stats.get('recent_papers_count', 0),
            help="Papers from the last 2 years"
        )
    
    # Charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Complexity distribution
        if current_stats.get('complexity_distribution'):
            fig_complexity = px.bar(
                x=list(current_stats['complexity_distribution'].keys()),
                y=list(current_stats['complexity_distribution'].values()),
                title="Implementation Complexity",
                color_discrete_sequence=['#0066FF'],
                labels={'x': '', 'y': ''}
            )
            fig_complexity.update_layout(
                showlegend=False,
                title_font_size=16,
                font_size=12
            )
            st.plotly_chart(fig_complexity, use_container_width=True)
        else:
            st.info("No complexity data available. Process some papers to see distribution.")
    
    with col2:
        # Study type distribution
        if current_stats.get('study_type_distribution'):
            fig_study = px.bar(
                x=list(current_stats['study_type_distribution'].keys()),
                y=list(current_stats['study_type_distribution'].values()),
                title="Study Types",
                color_discrete_sequence=['#0066FF'],
                labels={'x': '', 'y': ''}
            )
            fig_study.update_layout(
                showlegend=False,
                title_font_size=16,
                font_size=12
            )
            st.plotly_chart(fig_study, use_container_width=True)
        else:
            st.info("No study type data available. Process some papers to see distribution.")

# Get Recommendations Page
elif page == "🎯 Get Recommendations":
    st.header("Get Personalized Recommendations")
    
    st.markdown("""
    <div class="info-box">
        <p><strong>💡 How it works:</strong> Our AI consultant analyzes research papers and provides strategic recommendations tailored to your business context.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # User context form
    with st.form("user_context_form"):
        st.subheader("Your Context")
        
        business_context = st.text_area(
            "Describe your business context (optional)",
            placeholder="e.g., We are a mid-size healthcare company looking to improve patient engagement...",
            height=100,
            help="Include company size, industry, team size, budget constraints, and AI maturity level"
        )
        
        specific_problems = st.text_area(
            "Specific problems to solve (optional)",
            placeholder="e.g., Reduce response time, improve accuracy of technical answers...",
            height=80
        )
        
        submit_button = st.form_submit_button("🔍 Get Recommendations", use_container_width=True)
    
    if submit_button:
        # Create user context
        context = UserContext(
            use_case_description=business_context if business_context else "General exploration of GenAI applications",
            specific_problems=[p.strip() for p in specific_problems.split('\n') if p.strip()] if specific_problems else []
        )
        
        # Get recommendations with error handling
        try:
            with st.spinner("Our AI consultant is analyzing research and preparing recommendations..."):
                synthesis_result = st.session_state.synthesis_engine.synthesize_recommendations(context)
            
            # Store synthesis in session state
            st.session_state.current_synthesis = synthesis_result
            st.session_state.current_context = context
            
            # Display results
            if 'consultant_analysis' in synthesis_result:
                st.markdown("---")
                st.markdown("## 📊 Strategic Analysis & Recommendations")
                st.markdown(synthesis_result['consultant_analysis'])
                
                # Show metadata
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
            else:
                st.warning("No recommendations could be generated. This may be due to insufficient data or configuration issues.")
                
        except Exception as e:
            st.error(f"Error generating recommendations: {e}")
            st.info("Please check your configuration and try again.")

# Manual Processing Page (if available)
elif page == "📅 Manual Processing" and MANUAL_PROCESSING_AVAILABLE:
    st.header("Manual Processing Control")
    
    st.info("Manual processing allows you to fetch and process papers from specific date ranges.")
    
    # Simple date selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=(datetime.now() - timedelta(days=7)).date(),
            max_value=datetime.now().date()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date", 
            value=(datetime.now() - timedelta(days=1)).date(),
            max_value=datetime.now().date()
        )
    
    with col3:
        max_papers = st.number_input(
            "Max Papers",
            min_value=1,
            max_value=100,  # Reduced for free tier
            value=20,
            help="Limit papers for free tier performance"
        )
    
    # Convert to datetime
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Validate and process
    try:
        is_valid, error_msg = st.session_state.manual_controller.validate_date_range(start_datetime, end_datetime)
        
        if not is_valid:
            st.error(f"Invalid date range: {error_msg}")
        else:
            if st.button("🔄 Process Papers", use_container_width=True, type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                def simple_progress_callback(message, progress, **kwargs):
                    progress_bar.progress(min(progress / 100, 1.0))
                    status_text.text(message)
                
                try:
                    results = st.session_state.manual_controller.process_date_range(
                        start_datetime,
                        end_datetime,
                        max_papers=max_papers,
                        progress_callback=simple_progress_callback
                    )
                    
                    if results.get('success'):
                        st.success(f"✅ Processed {results['papers_processed']} papers successfully!")
                    else:
                        st.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
                        
                except Exception as e:
                    st.error(f"❌ Unexpected error: {str(e)}")
                    
    except Exception as e:
        st.error(f"Manual processing error: {e}")

# Settings Page
elif page == "⚙️ Settings":
    st.header("Platform Settings")
    
    # System status
    st.subheader("System Status")
    
    try:
        current_stats = st.session_state.storage.get_statistics()
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Storage", "Supabase Cloud")
        
        with col2:
            st.metric("Total Papers", current_stats.get('total_papers', 0))
        
        with col3:
            st.metric("Total Insights", current_stats.get('total_insights', 0))
        
        # Configuration info
        st.subheader("Configuration")
        
        config_info = f"""
        **Current Configuration:**
        - Deployment: Streamlit Cloud
        - LLM Model: {getattr(Config, 'LLM_MODEL', 'Not configured')}
        - Storage: Supabase Cloud
        - Manual Processing: {'Available' if MANUAL_PROCESSING_AVAILABLE else 'Unavailable'}
        """
        
        st.info(config_info)
        
        # Export statistics
        if st.button("📊 Export Statistics"):
            stats_json = json.dumps(current_stats, indent=2, default=str)
            st.download_button(
                label="Download Statistics JSON",
                data=stats_json,
                file_name=f"genai_stats_{datetime.now().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
        
    except Exception as e:
        st.error(f"Error loading settings: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>Research Implementation Platform - Powered by Streamlit Cloud</p>
</div>
""", unsafe_allow_html=True)