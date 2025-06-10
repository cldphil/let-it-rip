"""
Diagnostic script to show exactly which columns are missing from your Supabase tables.

INTEGRATION INSTRUCTIONS:
1. Save this as diagnose_schema.py in your project root
2. Run: python diagnose_schema.py
3. It will show exactly which columns need to be added
"""

import os
from dotenv import load_dotenv
from supabase import create_client
import json

# Load environment variables
load_dotenv()

def diagnose_schema():
    """Show which columns are present and missing in each table."""
    
    # Initialize Supabase client
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_ANON_KEY')
    
    if not url or not key:
        print("❌ Missing SUPABASE_URL or SUPABASE_ANON_KEY in .env")
        return
    
    supabase = create_client(url, key)
    
    # Define expected columns for each table
    expected_schema = {
        'papers': {
            'id': 'uuid',
            'paper_id': 'text',
            'title': 'text',
            'authors': 'jsonb',
            'summary': 'text',
            'published_date': 'text',
            'arxiv_categories': 'jsonb',
            'pdf_url': 'text',
            'full_text': 'text',
            'comments': 'text',
            'metadata': 'jsonb',
            'stored_timestamp': 'timestamptz'
        },
        'insights': {
            'id': 'uuid',
            'paper_id': 'text',
            'study_type': 'text',
            'techniques_used': 'jsonb',
            'implementation_complexity': 'text',
            'reputation_score': 'numeric',
            'extraction_confidence': 'numeric',
            'has_code': 'boolean',
            'has_dataset': 'boolean',
            'key_findings_count': 'integer',
            'extraction_timestamp': 'timestamptz',
            'total_author_hindex': 'integer',
            'has_conference_mention': 'boolean',
            'key_findings': 'jsonb',
            'limitations': 'jsonb',
            'problem_addressed': 'text',
            'prerequisites': 'jsonb',
            'real_world_applications': 'jsonb',
            'full_insights': 'jsonb'
        },
        'processing_logs': {
            'id': 'uuid',
            'batch_name': 'text',
            'papers_processed': 'integer',
            'papers_failed': 'integer',
            'total_cost': 'numeric',
            'processing_time_seconds': 'numeric',
            'date_range': 'jsonb',
            'success_rate': 'numeric',
            'created_at': 'timestamptz'
        }
    }
    
    print("🔍 Supabase Schema Diagnostic Report")
    print("=" * 60)
    
    for table_name, expected_columns in expected_schema.items():
        print(f"\n📋 Table: {table_name}")
        print("-" * 40)
        
        try:
            # Try to get one row to see actual columns
            result = supabase.table(table_name).select('*').limit(1).execute()
            
            if result.data and len(result.data) > 0:
                # Get actual columns from the result
                actual_columns = set(result.data[0].keys())
                expected_column_names = set(expected_columns.keys())
                
                # Find missing and extra columns
                missing_columns = expected_column_names - actual_columns
                extra_columns = actual_columns - expected_column_names
                
                if missing_columns:
                    print(f"❌ Missing columns ({len(missing_columns)}):")
                    for col in sorted(missing_columns):
                        print(f"   - {col} ({expected_columns[col]})")
                else:
                    print("✅ All expected columns present")
                
                if extra_columns:
                    print(f"ℹ️  Extra columns found ({len(extra_columns)}):")
                    for col in sorted(extra_columns):
                        print(f"   - {col}")
                
                print(f"\n📊 Column count: {len(actual_columns)} present, {len(expected_column_names)} expected")
                
            else:
                # Table is empty, can't check columns this way
                print("⚠️  Table is empty - cannot verify columns")
                print(f"   Expected {len(expected_columns)} columns:")
                for col, dtype in expected_columns.items():
                    print(f"   - {col} ({dtype})")
            
        except Exception as e:
            if 'relation' in str(e) and 'does not exist' in str(e):
                print(f"❌ Table does not exist!")
            else:
                print(f"❌ Error checking table: {e}")
    
    print("\n" + "=" * 60)
    print("📝 SQL to add missing columns:")
    print("=" * 60)
    
    # Generate SQL for missing columns
    print("\n-- Run this SQL in Supabase to add missing columns:")
    print("-- (Already generated in fix_insights_table.sql)")
    print("\nSee fix_insights_table.sql for the complete migration script.")


def test_minimal_insert():
    """Test inserting with only required fields."""
    print("\n\n🧪 Testing Minimal Insert...")
    print("-" * 40)
    
    import uuid
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_ANON_KEY')
    supabase = create_client(url, key)
    
    # Test minimal insights insert
    try:
        minimal_insights = {
            'id': str(uuid.uuid4()),
            'paper_id': 'minimal_test_123',
            'study_type': 'empirical',
            'implementation_complexity': 'medium',
            'extraction_timestamp': '2025-01-01T00:00:00'
        }
        
        result = supabase.table('insights').insert(minimal_insights).execute()
        print("✅ Minimal insights insert successful")
        
        # Clean up
        supabase.table('insights').delete().eq('paper_id', 'minimal_test_123').execute()
        
    except Exception as e:
        print(f"❌ Minimal insights insert failed: {e}")
        error_details = json.loads(str(e).split(':', 1)[1]) if ':' in str(e) else {}
        if 'message' in error_details:
            print(f"   Error message: {error_details['message']}")


if __name__ == "__main__":
    diagnose_schema()
    test_minimal_insert()