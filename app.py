"""
Streamlit web interface for the GenAI Research Implementation Platform.
Cloud-only version with Supabase integration.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime, timedelta
import json

# Import core modules
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
if 'manual_controller' not in st.session_state and MANUAL_PROCESSING_AVAILABLE:
    st.session_state.manual_controller = ManualProcessingController()

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
    
    /* Cost estimate box */
    .cost-estimate {
        background-color: #F8F9FA;
        border: 1px solid var(--border-color);
        border-radius: 8px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Case study badge */
    .case-study-badge {
        background-color: var(--success-color);
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
    <p style="color: white; margin: 0.5rem 0 0 0; opacity: 0.9;">Transform research into actionable insights</p>
</div>
""", unsafe_allow_html=True)

# Cloud storage indicator
st.markdown("""
<div class="cloud-indicator">
    ☁️ <strong>Cloud Storage Active</strong> - Data synced to Supabase
</div>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("Navigation")

# Dynamic page list based on available features
page_options = ["📊 Dashboard", "📚 Browse Insights", "🎯 Get Recommendations"]

if MANUAL_PROCESSING_AVAILABLE:
    page_options.append("📅 Manual Processing")

page_options.append("⚙️ Settings")

page = st.sidebar.radio("Select Page", page_options)

# Dashboard Page
if page == "📊 Dashboard":
    st.header("Platform Overview")
    
    # Always get fresh statistics when dashboard loads
    current_stats = st.session_state.storage.get_statistics()
    
    # Key metrics
    col1, col2, col3 = st.columns(3)
    
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

# Manual Processing Page
elif page == "📅 Manual Processing" and MANUAL_PROCESSING_AVAILABLE:
    st.header("Manual Processing Control")
    
    st.markdown("""
    <div class="info-box">
        <p><strong>Manual Processing</strong> allows you to select specific date ranges, estimate costs, and have full control over which papers to process.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Date Range Selection
    st.subheader("📅 Select Date Range")
    
    # Get available date ranges
    available_ranges = st.session_state.manual_controller.get_available_date_ranges()
    
    # Custom date selection
    st.markdown("**Custom Date Range:**")
    col1, col2 = st.columns(2)
    
    with col1:
        start_date = st.date_input(
            "Start Date",
            value=st.session_state.get('selected_start', datetime.now() - timedelta(days=7)).date(),
            max_value=datetime.now().date()
        )
    
    with col2:
        end_date = st.date_input(
            "End Date",
            value=st.session_state.get('selected_end', datetime.now() - timedelta(days=1)).date(),
            max_value=datetime.now().date()
        )
    
    # Convert to datetime
    start_datetime = datetime.combine(start_date, datetime.min.time())
    end_datetime = datetime.combine(end_date, datetime.max.time())
    
    # Validate date range
    is_valid, error_msg = st.session_state.manual_controller.validate_date_range(start_datetime, end_datetime)
    
    if not is_valid:
        st.error(f"Invalid date range: {error_msg}")
    else:
        # Cost Estimation
        st.subheader("💰 Cost Estimation")
        
        # Maximum papers setting
        col1, col2 = st.columns(2)
        
        with col1:
            max_papers = st.number_input(
                "Maximum papers to process",
                min_value=1,
                max_value=1000,
                value=100,
                help="Limit the number of papers to process from this date range"
            )
        
        with col2:
            skip_existing = st.checkbox(
                "Skip existing papers",
                value=True,
                help="Skip papers that have already been processed"
            )
        
        # Get cost estimate
        estimate = st.session_state.manual_controller.estimate_processing_cost(
            start_datetime, end_datetime, max_papers
        )
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Estimated Papers", estimate['estimated_papers'])
        
        with col2:
            st.metric("Estimated Cost", f"${estimate['estimated_cost_usd']:.2f}")
        
        with col3:
            st.metric("Estimated Time", f"{estimate['estimated_time_minutes']:.1f} min")
        
        with col4:
            st.metric("Days in Range", estimate['days_in_range'])
        
        # Additional estimate details
        if estimate.get('reputation_filter_active'):
            st.info(f"🔍 **Reputation filtering active** (minimum score: {estimate['min_reputation_score']:.2f})")
        
        # Check existing papers
        existing_check = st.session_state.manual_controller.check_existing_papers(start_datetime, end_datetime)
        
        if existing_check['existing_papers'] > 0:
            st.warning(f"⚠️ {existing_check['message']}")
        
        # Processing Controls
        st.subheader("🚀 Start Processing")

        # Processing button
        if st.button("🔄 Process Papers", use_container_width=True, type="primary", disabled=not is_valid):
            
            # Create progress tracking UI containers
            st.markdown("### Processing Progress")
            
            # Create empty containers that will be updated
            phase_container = st.empty()
            progress_bar = st.empty()
            progress_text = st.empty()
            stats_container = st.empty()
            current_activity = st.empty()
            error_container = st.empty()
            
            # Initialize progress tracking state
            progress_state = {
                'phase': 'initializing',
                'overall_progress': 0,
                'papers_found': 0,
                'papers_processed': 0,
                'insights_generated': 0,
                'current_paper': '',
                'start_time': datetime.now(),
                'errors': []
            }
            
            def update_progress_display(state):
                """Update the progress display with current state."""
                
                # Phase indicator
                phase_emojis = {
                    'initializing': '🔧',
                    'fetching': '📡',
                    'processing': '⚙️',
                    'completed': '✅',
                    'error': '❌'
                }
                
                phase_names = {
                    'initializing': 'Initializing',
                    'fetching': 'Fetching Papers',
                    'processing': 'Extracting Insights',
                    'completed': 'Processing Complete',
                    'error': 'Error Occurred'
                }
                
                phase_colors = {
                    'initializing': '#0066FF',
                    'fetching': '#17A2B8',
                    'processing': '#FFC107',
                    'completed': '#28A745',
                    'error': '#DC3545'
                }
                
                emoji = phase_emojis.get(state['phase'], '⚙️')
                name = phase_names.get(state['phase'], 'Processing')
                color = phase_colors.get(state['phase'], '#0066FF')
                
                # Update phase indicator
                phase_container.markdown(f"""
                <div style="background: linear-gradient(135deg, {color} 0%, {color}AA 100%); 
                            padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1.5rem;
                            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                    <h2 style="color: white; margin: 0; font-size: 1.8rem;">{emoji} {name}</h2>
                </div>
                """, unsafe_allow_html=True)
                
                # Update progress bar
                progress_bar.progress(state['overall_progress'] / 100)
                
                # Update progress text
                elapsed = (datetime.now() - state['start_time']).total_seconds()
                if elapsed < 60:
                    time_str = f"{elapsed:.0f}s"
                else:
                    time_str = f"{elapsed/60:.1f}m"
                
                progress_text.markdown(f"""
                **Progress:** {state['overall_progress']:.1f}% | **Elapsed:** {time_str}
                """)
                
                # Update statistics
                with stats_container.container():
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Papers Found", state['papers_found'])
                    
                    with col2:
                        st.metric("Papers Processed", state['papers_processed'])
                    
                    with col3:
                        st.metric("Insights Generated", state['insights_generated'])
                    
                    with col4:
                        if state['papers_processed'] > 0:
                            rate = state['papers_processed'] / (elapsed / 60)
                            st.metric("Rate", f"{rate:.1f}/min")
                        else:
                            st.metric("Rate", "Calculating...")
                
                # Update current activity
                if state['current_paper']:
                    current_activity.info(f"📄 Processing: {state['current_paper'][:100]}...")
                else:
                    current_activity.empty()
                
                # Update errors if any
                if state['errors']:
                    with error_container.expander(f"⚠️ Errors ({len(state['errors'])})"):
                        for error in state['errors'][-3:]:
                            st.error(f"• {error}")
                else:
                    error_container.empty()
            
            # Enhanced progress callback function
            def enhanced_progress_callback(message, progress, **kwargs):
                """Enhanced progress callback that handles detailed progress updates."""
                
                # Update progress state based on message and additional info
                if "Fetching papers" in message:
                    progress_state['phase'] = 'fetching'
                    progress_state['overall_progress'] = min(progress, 15)
                    
                elif "Found" in message and "papers" in message:
                    import re
                    match = re.search(r'(\d+)', message)
                    if match:
                        progress_state['papers_found'] = int(match.group(1))
                    progress_state['phase'] = 'processing'
                    progress_state['overall_progress'] = 15
                    
                elif "Processing paper" in message:
                    import re
                    match = re.search(r'Processing paper (\d+)/(\d+)', message)
                    if match:
                        current, total = int(match.group(1)), int(match.group(2))
                        progress_state['papers_processed'] = current - 1
                        progress_state['overall_progress'] = 15 + (current / total) * 70
                        
                        if kwargs.get('current_paper_title'):
                            progress_state['current_paper'] = kwargs['current_paper_title']
                    
                elif "Extracting insights" in message:
                    if kwargs.get('insights_generated'):
                        progress_state['insights_generated'] = kwargs['insights_generated']
                
                elif "Processing complete" in message or progress >= 100:
                    progress_state['phase'] = 'completed'
                    progress_state['overall_progress'] = 100
                    progress_state['current_paper'] = ''
                    
                elif "Error" in message:
                    progress_state['phase'] = 'error'
                    if kwargs.get('error_details'):
                        progress_state['errors'].append(kwargs['error_details'])
                
                # Handle any additional updates from kwargs
                for key, value in kwargs.items():
                    if key.startswith('progress_'):
                        field = key.replace('progress_', '')
                        progress_state[field] = value
                
                # Update the display
                update_progress_display(progress_state)
            
            # Start processing with enhanced progress tracking
            try:
                # Show initial state
                update_progress_display(progress_state)
                
                # Call the processing function with enhanced callback
                results = st.session_state.manual_controller.process_date_range_enhanced(
                    start_datetime,
                    end_datetime,
                    max_papers=max_papers,
                    skip_existing=skip_existing,
                    progress_callback=enhanced_progress_callback
                )
                
                # Display final results
                if results.get('success'):
                    progress_state['phase'] = 'completed'
                    progress_state['overall_progress'] = 100
                    progress_state['papers_processed'] = results['papers_processed']
                    progress_state['insights_generated'] = results['papers_processed']
                    update_progress_display(progress_state)
                    
                    st.success("✅ Processing completed successfully!")
                    
                    # Success rate calculation
                    if results['papers_found'] > 0:
                        success_rate = results['papers_processed'] / results['papers_found']
                        if success_rate >= 0.9:
                            st.success(f"🎉 Excellent success rate: {success_rate:.1%}")
                        elif success_rate >= 0.7:
                            st.info(f"✅ Good success rate: {success_rate:.1%}")
                        else:
                            st.warning(f"⚠️ Success rate: {success_rate:.1%} - some papers may have failed processing")
                    
                    # Estimate vs Actual comparison
                    with st.expander("📊 Estimate vs Actual Comparison"):
                        estimate_vs_actual = results.get('estimate_vs_actual', {})
                        
                        comparison_data = [
                            {
                                "Metric": "Papers",
                                "Estimated": estimate_vs_actual.get('estimated_papers', 'N/A'),
                                "Actual": estimate_vs_actual.get('actual_papers', 'N/A'),
                                "Accuracy": "N/A"
                            },
                            {
                                "Metric": "Cost",
                                "Estimated": f"${estimate_vs_actual.get('estimated_cost', 0):.2f}",
                                "Actual": f"${estimate_vs_actual.get('actual_cost', 0):.2f}",
                                "Accuracy": "N/A"
                            }
                        ]
                        
                        # Calculate accuracy if we have the data
                        if estimate_vs_actual.get('estimated_papers') and estimate_vs_actual.get('actual_papers'):
                            est_papers = estimate_vs_actual['estimated_papers']
                            act_papers = estimate_vs_actual['actual_papers']
                            if est_papers > 0:
                                accuracy = 100 - abs(est_papers - act_papers) / est_papers * 100
                                comparison_data[0]['Accuracy'] = f"{accuracy:.1f}%"

                        if estimate_vs_actual.get('estimated_cost') and estimate_vs_actual.get('actual_cost'):
                            est_cost = estimate_vs_actual['estimated_cost']
                            act_cost = estimate_vs_actual['actual_cost']
                            if est_cost > 0:
                                accuracy = 100 - abs(est_cost - act_cost) / est_cost * 100
                                comparison_data[1]['Accuracy'] = f"{accuracy:.1f}%"
                        
                        comparison_df = prepare_dataframe_for_streamlit(pd.DataFrame(comparison_data))
                        st.dataframe(comparison_df, use_container_width=True, hide_index=True)
                    
                    # Reputation filtering info
                    reputation_info = results.get('reputation_filtering', {})
                    if reputation_info.get('active'):
                        st.info(f"🎯 Reputation filtering applied (threshold: {reputation_info['threshold']:.2f}) - {reputation_info['papers_stored']} papers stored")
                
                else:
                    progress_state['phase'] = 'error'
                    progress_state['errors'].append(results.get('error', 'Unknown error'))
                    update_progress_display(progress_state)
                    st.error(f"❌ Processing failed: {results.get('error', 'Unknown error')}")
            
            except Exception as e:
                progress_state['phase'] = 'error'
                progress_state['errors'].append(str(e))
                update_progress_display(progress_state)
                st.error(f"❌ Unexpected error: {str(e)}")

# Browse Insights Page - Updated for cloud-only
elif page == "📚 Browse Insights":
    st.header("Browse Research Insights")
    
    # Enhanced filters with better styling
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
    
    # Cloud-only paper retrieval using vector search
    try:
        # Use empty user context to get all papers
        from core.insight_schema import UserContext
        empty_context = UserContext(use_case_description="browse all papers")
        
        # Get papers using vector search with very low threshold to get all
        Config.VECTOR_SIMILARITY_THRESHOLD = 0.0  # Temporarily lower threshold
        all_papers_data = st.session_state.storage.find_similar_papers(
            empty_context, 
            n_results=1000  # Get up to 1000 papers
        )
        Config.VECTOR_SIMILARITY_THRESHOLD = 0.6  # Reset threshold
        
        # Convert to display format and apply filters
        all_papers = []
        for paper_data in all_papers_data:
            insights = paper_data['insights']
            
            # Load paper metadata
            paper_info = st.session_state.storage.load_paper(paper_data['paper_id'])
            if not paper_info:
                continue
            
            # Apply filters
            if (insights.implementation_complexity.value in complexity_filter and
                insights.study_type.value in study_type_filter):
                
                # Apply case study filter if checked
                if show_case_studies_only and insights.study_type.value != 'case_study':
                    continue
                
                # Apply industry validation filter if checked
                if show_validated_only and not insights.industry_validation:
                    continue
                
                # Apply code availability filter if checked
                if show_with_code_only and not insights.has_code_available:
                    continue
                
                all_papers.append({
                    'paper_id': paper_data['paper_id'],
                    'title': paper_info.get('title', 'Unknown'),
                    'authors': paper_info.get('authors', []),
                    'published': paper_info.get('published', ''),
                    'pdf_url': paper_info.get('pdf_url', ''),
                    'insights': insights,
                    'reputation_score': insights.get_reputation_score(),
                    'recency': paper_info.get('published', '2020'),
                    'key_findings_count': len(insights.key_findings),
                    'extraction_confidence': insights.extraction_confidence
                })
        
        # Sort papers based on selected criteria
        if all_papers:
            if sort_by == 'recency':
                all_papers.sort(key=lambda x: x['published'], reverse=True)
            else:
                all_papers.sort(key=lambda x: x[sort_by], reverse=True)
        
        # Display results
        st.write(f"Found {len(all_papers)} papers matching filters")
        
        if not all_papers:
            st.info("No papers found. Try processing some papers first or adjusting your filters.")
        else:
            # Display papers
            for paper in all_papers[:50]:  # Show top 50
                with st.expander(f"📄 {paper['title'][:100]}..."):
                    # Add badges for case study and validation status
                    badges_html = ""
                    if paper['insights'].study_type.value == 'case_study':
                        badges_html += '<span class="case-study-badge">CASE STUDY</span> '
                    if paper['insights'].industry_validation:
                        badges_html += '<span class="validation-badge">INDUSTRY VALIDATED</span> '
                    
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
                        # Reputation metrics
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
        
        # Get recommendations
        with st.spinner("Our AI consultant is analyzing research and preparing recommendations..."):
            synthesis_result = st.session_state.synthesis_engine.synthesize_recommendations(context)
        
        # Store synthesis in session state
        st.session_state.current_synthesis = synthesis_result
        st.session_state.current_context = context
        
        # Check if we have consultant analysis
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
                            # Score components display
                            components = paper.get('components', {})
                            st.write(f"  Reputation: {components.get('reputation', 0):.2f} | "
                                   f"Recency: {components.get('recency', 0):.2f} | "
                                   f"Case Study: {components.get('case_study', 0):.2f}")

# Settings Page
elif page == "⚙️ Settings":
    st.header("Platform Settings")
    
    # Cloud Storage Status
    st.subheader("☁️ Cloud Storage Status")
    
    st.markdown("""
    <div class="success-box">
        <h4>Cloud Storage Active</h4>
        <p>Your data is being synced to Supabase cloud storage.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Cloud storage metrics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Storage Type", "☁️ Supabase")
    
    with col2:
        # Health check button
        if st.button("🔍 Health Check"):
            with st.spinner("Checking system health..."):
                health_status = st.session_state.storage.health_check()
                
                if health_status['healthy']:
                    st.success("✅ System is healthy")
                else:
                    st.error("❌ System health issues detected")
                
                # Show detailed health status
                with st.expander("Health Check Details"):
                    for check_name, check_result in health_status['checks'].items():
                        status_icon = "✅" if check_result['status'] == 'ok' else "❌"
                        st.write(f"{status_icon} **{check_name}**: {check_result['message']}")
    
    with col3:
        st.metric("Min Reputation", f"{Config.MINIMUM_REPUTATION_SCORE:.2f}")
    
    # Storage management
    st.subheader("Storage Management")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("🗑️ Clear All Data", use_container_width=True):
            if st.checkbox("I understand this will delete all data"):
                with st.spinner("Clearing all data..."):
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
            "Case Studies",
            "Recent Papers (Last 2 Years)",
            "Average Extraction Confidence"
        ],
        "Value": [
            current_stats.get('total_papers', 0),
            current_stats.get('total_insights', 0),
            f"{current_stats.get('average_reputation_score', 0):.2f}",
            f"{current_stats.get('average_key_findings_count', 0):.1f}",
            current_stats.get('papers_with_code', 0),
            current_stats.get('case_studies_count', 0),
            current_stats.get('recent_papers_count', 0),
            f"{current_stats.get('average_extraction_confidence', 0):.2f}"
        ]
    }
    
    stats_df = prepare_dataframe_for_streamlit(pd.DataFrame(stats_data))
    st.dataframe(stats_df, use_container_width=True, hide_index=True)
    
    # Configuration
    st.subheader("Configuration")
    
    # Show current configuration
    config_info = f"""
    **Current Configuration:**
    - LLM Model: {Config.LLM_MODEL}
    - Batch Size: {Config.BATCH_SIZE}
    - Max Key Findings: {Config.MAX_KEY_FINDINGS}
    - Cloud Storage: ☁️ Supabase
    - Enable Author Lookup: {'Yes' if Config.ENABLE_AUTHOR_LOOKUP else 'No'}
    - Manual Processing: {'Available' if MANUAL_PROCESSING_AVAILABLE else 'Unavailable'}
    - Vector Model: {Config.VECTOR_EMBEDDING_MODEL}
    """
    
    st.info(config_info)
    
    # Manual Processing Status
    if MANUAL_PROCESSING_AVAILABLE:
        st.subheader("📅 Manual Processing Configuration")
        
        st.markdown("""
        <div class="success-box">
            <h4>Manual Processing Available</h4>
            <p>Date range selection, cost estimation, and processing control are enabled.</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.subheader("📅 Manual Processing Status")
        
        st.markdown("""
        <div class="warning-box">
            <h4>Manual Processing Unavailable</h4>
            <p>Manual processing features are not available. Please ensure the manual processing system is properly installed.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # About
    st.subheader("About")
    
    version_info = """
    **GenAI Research Implementation Platform**
    
    Transform cutting-edge AI research into actionable business insights.
    
    **Version:** 2.2.0 (Cloud-Only)  
    **License:** MIT
    
    **Recent Updates:**
    - Full cloud-only operation with Supabase
    - Enhanced vector search capabilities
    - Improved reputation scoring system
    - Real-time health monitoring
    - Streamlined architecture
    """
    
    st.markdown(f"""
    <div class="info-box">
        {version_info}
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #666; padding: 1rem;">
    <p>© 2025 GenAI Research Platform | Cloud-Only with Supabase</p>
</div>
""", unsafe_allow_html=True)