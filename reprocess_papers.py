"""
Reprocess existing papers with the new stringent case study criteria.
This will update the study_type classification for all stored papers.
"""

import json
from pathlib import Path
from datetime import datetime
from core import InsightStorage, InsightExtractor, PaperInsights
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def reprocess_study_types():
    """Reprocess all papers to fix study type classification."""
    
    storage = InsightStorage()
    extractor = InsightExtractor()
    
    # Get all insight files
    insights_dir = Path("storage/insights")
    if not insights_dir.exists():
        print("No insights directory found.")
        return
    
    insight_files = list(insights_dir.glob("*_insights.json"))
    total_papers = len(insight_files)
    
    if total_papers == 0:
        print("No papers found to reprocess.")
        return
    
    print(f"Found {total_papers} papers to reprocess")
    print("=" * 80)
    
    # Statistics
    stats = {
        'total': total_papers,
        'changed': 0,
        'unchanged': 0,
        'errors': 0,
        'before': {
            'case_study': 0,
            'empirical': 0,
            'theoretical': 0,
            'pilot': 0,
            'survey': 0,
            'review': 0,
            'meta_analysis': 0,
            'unknown': 0
        },
        'after': {
            'case_study': 0,
            'empirical': 0,
            'theoretical': 0,
            'pilot': 0,
            'survey': 0,
            'review': 0,
            'meta_analysis': 0,
            'unknown': 0
        }
    }
    
    changes = []
    
    for i, insight_file in enumerate(insight_files, 1):
        paper_id = insight_file.stem.replace("_insights", "")
        
        try:
            # Load existing insights
            with open(insight_file, 'r', encoding='utf-8') as f:
                insights_data = json.load(f)
            
            # Load paper data
            paper_data = storage.load_paper(paper_id)
            if not paper_data:
                logger.warning(f"Paper data not found for {paper_id}")
                stats['errors'] += 1
                continue
            
            # Get current study type
            old_study_type = insights_data.get('study_type', 'unknown')
            stats['before'][old_study_type] = stats['before'].get(old_study_type, 0) + 1
            
            # Re-detect case study with new criteria
            sections = {}
            if 'full_text' in paper_data:
                sections = extractor._extract_relevant_sections(paper_data.get('full_text', ''))
            
            is_case_study = extractor._detect_case_study(paper_data, sections)
            
            # Infer new study type
            new_study_type = extractor._infer_study_type(is_case_study, sections)
            
            # Update if changed
            if new_study_type != old_study_type:
                insights_data['study_type'] = new_study_type
                insights_data['industry_validation'] = is_case_study
                
                # Save updated insights
                with open(insight_file, 'w', encoding='utf-8') as f:
                    json.dump(insights_data, f, indent=2, ensure_ascii=False)
                
                # Update in storage (vector DB and SQLite)
                insights_obj = PaperInsights(**insights_data)
                storage.store_insights(paper_id, insights_obj)
                
                stats['changed'] += 1
                changes.append({
                    'paper_id': paper_id,
                    'title': paper_data.get('title', 'Unknown')[:80],
                    'old_type': old_study_type,
                    'new_type': new_study_type
                })
                
                print(f"\n[{i}/{total_papers}] Changed: {paper_id}")
                print(f"  Title: {paper_data.get('title', 'Unknown')[:80]}...")
                print(f"  {old_study_type} → {new_study_type}")
            else:
                stats['unchanged'] += 1
                if i % 10 == 0:  # Progress indicator
                    print(f"[{i}/{total_papers}] Processing... ({stats['changed']} changed so far)")
            
            stats['after'][new_study_type] = stats['after'].get(new_study_type, 0) + 1
            
        except Exception as e:
            logger.error(f"Error processing {paper_id}: {e}")
            stats['errors'] += 1
    
    # Print summary
    print("\n" + "=" * 80)
    print("REPROCESSING COMPLETE")
    print("=" * 80)
    
    print(f"\nTotal papers processed: {total_papers}")
    print(f"Changed: {stats['changed']}")
    print(f"Unchanged: {stats['unchanged']}")
    print(f"Errors: {stats['errors']}")
    
    print("\n\nStudy Type Distribution BEFORE:")
    for study_type, count in sorted(stats['before'].items()):
        if count > 0:
            print(f"  {study_type:15} {count:4} ({count/total_papers*100:5.1f}%)")
    
    print("\n\nStudy Type Distribution AFTER:")
    for study_type, count in sorted(stats['after'].items()):
        if count > 0:
            print(f"  {study_type:15} {count:4} ({count/total_papers*100:5.1f}%)")
    
    if changes:
        print("\n\nDetailed Changes:")
        print("-" * 80)
        for change in changes[:20]:  # Show first 20 changes
            print(f"\n{change['paper_id']}")
            print(f"  {change['title']}")
            print(f"  {change['old_type']} → {change['new_type']}")
        
        if len(changes) > 20:
            print(f"\n... and {len(changes) - 20} more changes")
    
    # Save change log
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path("storage") / f"reprocess_log_{timestamp}.json"
    with open(log_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'stats': stats,
            'changes': changes
        }, f, indent=2)
    
    print(f"\n\nChange log saved to: {log_file}")
    
    return stats


def preview_changes(limit=10):
    """Preview what changes would be made without actually changing files."""
    
    storage = InsightStorage()
    extractor = InsightExtractor()
    
    insights_dir = Path("storage/insights")
    if not insights_dir.exists():
        print("No insights directory found.")
        return
    
    insight_files = list(insights_dir.glob("*_insights.json"))[:limit]
    
    print(f"Previewing changes for {len(insight_files)} papers...")
    print("=" * 80)
    
    for insight_file in insight_files:
        paper_id = insight_file.stem.replace("_insights", "")
        
        try:
            # Load existing insights
            with open(insight_file, 'r', encoding='utf-8') as f:
                insights_data = json.load(f)
            
            # Load paper data
            paper_data = storage.load_paper(paper_id)
            if not paper_data:
                continue
            
            # Get current study type
            old_study_type = insights_data.get('study_type', 'unknown')
            
            # Re-detect case study
            sections = {}
            if 'full_text' in paper_data:
                sections = extractor._extract_relevant_sections(paper_data.get('full_text', ''))
            
            is_case_study = extractor._detect_case_study(paper_data, sections)
            new_study_type = extractor._infer_study_type(is_case_study, sections)
            
            if new_study_type != old_study_type:
                print(f"\n{paper_data.get('title', 'Unknown')[:80]}...")
                print(f"Would change: {old_study_type} → {new_study_type}")
                print(f"Abstract: {paper_data.get('summary', '')[:200]}...")
            
        except Exception as e:
            print(f"Error previewing {paper_id}: {e}")


if __name__ == "__main__":
    import sys
    import os
    
    # Check if API key is set
    if not os.getenv('ANTHROPIC_API_KEY'):
        print("ERROR: Please set ANTHROPIC_API_KEY environment variable")
        sys.exit(1)
    
    print("Paper Study Type Reprocessor")
    print("This will update the study_type classification using new stringent criteria.")
    print("\nOptions:")
    print("1. Preview changes (first 10 papers)")
    print("2. Reprocess all papers")
    print("3. Exit")
    
    choice = input("\nEnter your choice (1-3): ")
    
    if choice == "1":
        preview_changes()
    elif choice == "2":
        confirm = input("\nThis will update all papers. Continue? (yes/no): ")
        if confirm.lower() == "yes":
            reprocess_study_types()
        else:
            print("Cancelled.")
    else:
        print("Exiting.")