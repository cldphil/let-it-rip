import streamlit as st
import pandas as pd
from datetime import datetime
import time

class ProcessingProgressTracker:
    """
    Reusable progress tracking component for Streamlit apps.
    Provides detailed progress visualization for paper processing tasks.
    """
    
    def __init__(self, container=None):
        """
        Initialize the progress tracker.
        
        Args:
            container: Streamlit container to render in (optional)
        """
        self.container = container or st
        self.state = {
            'phase': 'idle',
            'overall_progress': 0,
            'papers_found': 0,
            'papers_processed': 0,
            'insights_generated': 0,
            'current_paper': '',
            'start_time': None,
            'errors': [],
            'phase_start_time': None
        }
        
        # Create containers for different sections
        self.phase_container = None
        self.progress_container = None
        self.stats_container = None
        self.details_container = None
    
    def initialize_display(self):
        """Initialize the progress display containers."""
        with self.container:
            self.phase_container = st.empty()
            self.progress_container = st.empty()
            self.stats_container = st.empty()
            self.details_container = st.empty()
    
    def update(self, message="", progress=0, **kwargs):
        """
        Update the progress display.
        
        Args:
            message: Status message to display
            progress: Progress percentage (0-100)
            **kwargs: Additional state updates
        """
        # Update state
        self.state['overall_progress'] = progress
        
        # Update state from kwargs
        for key, value in kwargs.items():
            if key.startswith('progress_'):
                field = key.replace('progress_', '')
                self.state[field] = value
            elif key in self.state:
                self.state[key] = value
        
        # Set start time if not set
        if self.state['start_time'] is None:
            self.state['start_time'] = datetime.now()
        
        # Update phase start time if phase changed
        if kwargs.get('phase') and kwargs['phase'] != self.state.get('phase'):
            self.state['phase_start_time'] = datetime.now()
        
        # Render the display
        self._render_display()
    
    def _render_display(self):
        """Render the progress display components."""
        if not all([self.phase_container, self.progress_container, 
                   self.stats_container, self.details_container]):
            self.initialize_display()
        
        self._render_phase()
        self._render_progress()
        self._render_stats()
        self._render_details()
    
    def _render_phase(self):
        """Render the phase indicator."""
        phase_emojis = {
            'idle': '⏸️',
            'initializing': '🔧',
            'fetching': '📡',
            'processing': '⚙️',
            'completed': '✅',
            'error': '❌'
        }
        
        phase_names = {
            'idle': 'Ready',
            'initializing': 'Initializing',
            'fetching': 'Fetching Papers',
            'processing': 'Extracting Insights',
            'completed': 'Processing Complete',
            'error': 'Error Occurred'
        }
        
        phase_colors = {
            'idle': '#6C757D',
            'initializing': '#0066FF',
            'fetching': '#17A2B8',
            'processing': '#FFC107',
            'completed': '#28A745',
            'error': '#DC3545'
        }
        
        emoji = phase_emojis.get(self.state['phase'], '⚙️')
        name = phase_names.get(self.state['phase'], 'Processing')
        color = phase_colors.get(self.state['phase'], '#0066FF')
        
        with self.phase_container:
            st.markdown(f"""
            <div style="background: linear-gradient(135deg, {color} 0%, {color}AA 100%); 
                        padding: 1.5rem; border-radius: 12px; text-align: center; margin-bottom: 1.5rem;
                        box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <h2 style="color: white; margin: 0; font-size: 1.8rem;">{emoji} {name}</h2>
            </div>
            """, unsafe_allow_html=True)
    
    def _render_progress(self):
        """Render the progress bar and percentage."""
        with self.progress_container:
            # Main progress bar
            st.progress(self.state['overall_progress'] / 100)
            
            # Progress details
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown(f"""
                <div style="text-align: left; font-size: 1.1rem; margin: 0.5rem 0;">
                    Progress: <strong>{self.state['overall_progress']:.1f}%</strong>
                </div>
                """, unsafe_allow_html=True)
            
            with col2:
                # Elapsed time
                if self.state['start_time']:
                    elapsed = (datetime.now() - self.state['start_time']).total_seconds()
                    if elapsed < 60:
                        time_str = f"{elapsed:.0f}s"
                    else:
                        time_str = f"{elapsed/60:.1f}m"
                    
                    st.markdown(f"""
                    <div style="text-align: right; font-size: 1.1rem; margin: 0.5rem 0;">
                        Elapsed: <strong>{time_str}</strong>
                    </div>
                    """, unsafe_allow_html=True)
    
    def _render_stats(self):
        """Render the statistics metrics."""
        with self.stats_container:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                delta_found = None
                if hasattr(self, '_last_papers_found'):
                    delta_found = self.state['papers_found'] - self._last_papers_found
                self._last_papers_found = self.state['papers_found']
                
                st.metric(
                    "Papers Found", 
                    self.state['papers_found'],
                    delta=delta_found if delta_found else None,
                    help="Total papers discovered"
                )
            
            with col2:
                delta_processed = None
                if hasattr(self, '_last_papers_processed'):
                    delta_processed = self.state['papers_processed'] - self._last_papers_processed
                self._last_papers_processed = self.state['papers_processed']
                
                st.metric(
                    "Papers Processed", 
                    self.state['papers_processed'],
                    delta=delta_processed if delta_processed else None,
                    help="Papers analyzed for insights"
                )
            
            with col3:
                delta_insights = None
                if hasattr(self, '_last_insights_generated'):
                    delta_insights = self.state['insights_generated'] - self._last_insights_generated
                self._last_insights_generated = self.state['insights_generated']
                
                st.metric(
                    "Insights Generated", 
                    self.state['insights_generated'],
                    delta=delta_insights if delta_insights else None,
                    help="Successful insight extractions"
                )
            
            with col4:
                # Processing rate
                if (self.state['start_time'] and 
                    self.state['papers_processed'] > 0):
                    elapsed = (datetime.now() - self.state['start_time']).total_seconds()
                    rate = self.state['papers_processed'] / (elapsed / 60)  # per minute
                    st.metric(
                        "Rate",
                        f"{rate:.1f}/min",
                        help="Papers processed per minute"
                    )
                else:
                    st.metric("Rate", "Calculating...", help="Papers processed per minute")
    
    def _render_details(self):
        """Render detailed information and current activity."""
        with self.details_container:
            # Current paper being processed
            if self.state['current_paper'] and self.state['phase'] == 'processing':
                st.markdown(f"""
                <div style="background-color: #E3F2FD; border-left: 4px solid #2196F3; 
                           padding: 1rem; border-radius: 0 6px 6px 0; margin: 1rem 0;">
                    <strong>📄 Currently Processing:</strong><br>
                    <em>{self.state['current_paper'][:120]}{'...' if len(self.state['current_paper']) > 120 else ''}</em>
                </div>
                """, unsafe_allow_html=True)
            
            # Estimated time remaining
            if (self.state['papers_processed'] > 0 and 
                self.state['papers_found'] > 0 and 
                self.state['phase'] == 'processing'):
                
                elapsed = (datetime.now() - self.state['start_time']).total_seconds()
                papers_remaining = self.state['papers_found'] - self.state['papers_processed']
                
                if papers_remaining > 0:
                    avg_time_per_paper = elapsed / self.state['papers_processed']
                    estimated_remaining = papers_remaining * avg_time_per_paper
                    
                    if estimated_remaining < 60:
                        time_str = f"{estimated_remaining:.0f} seconds"
                    elif estimated_remaining < 3600:
                        time_str = f"{estimated_remaining/60:.1f} minutes"
                    else:
                        time_str = f"{estimated_remaining/3600:.1f} hours"
                    
                    st.info(f"⏱️ Estimated time remaining: {time_str}")
            
            # Show recent errors
            if self.state['errors']:
                with st.expander(f"⚠️ Errors ({len(self.state['errors'])})", expanded=False):
                    for i, error in enumerate(self.state['errors'][-3:], 1):  # Show last 3 errors
                        st.error(f"{i}. {error}")
                    
                    if len(self.state['errors']) > 3:
                        st.caption(f"... and {len(self.state['errors']) - 3} more errors")
    
    def complete(self, success=True, message=""):
        """Mark processing as complete."""
        if success:
            self.update(message or "Processing completed successfully!", 100, phase='completed')
        else:
            self.update(message or "Processing failed", 0, phase='error')
    
    def reset(self):
        """Reset the progress tracker to initial state."""
        self.state = {
            'phase': 'idle',
            'overall_progress': 0,
            'papers_found': 0,
            'papers_processed': 0,
            'insights_generated': 0,
            'current_paper': '',
            'start_time': None,
            'errors': [],
            'phase_start_time': None
        }
        
        # Clear tracking variables
        for attr in ['_last_papers_found', '_last_papers_processed', '_last_insights_generated']:
            if hasattr(self, attr):
                delattr(self, attr)


# Example usage function
def create_progress_tracker(container=None):
    """
    Factory function to create a progress tracker.
    
    Args:
        container: Streamlit container to render in
        
    Returns:
        ProcessingProgressTracker instance
    """
    tracker = ProcessingProgressTracker(container)
    tracker.initialize_display()
    return tracker


# Usage example in the streamlit app:
"""
# In your processing code:

# Create progress tracker
progress_tracker = create_progress_tracker()

# Update during processing
progress_tracker.update("Starting fetch...", 5, phase='fetching')
progress_tracker.update("Found 25 papers", 15, papers_found=25, phase='processing')
progress_tracker.update("Processing paper 5/25", 35, 
                        papers_processed=4, insights_generated=4,
                        current_paper="Advanced Transformer Architecture for...")

# Mark as complete
progress_tracker.complete(success=True, message="All papers processed successfully!")
"""