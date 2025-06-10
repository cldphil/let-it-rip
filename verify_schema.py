"""
Script to verify your Supabase schema is correct after migration.

INTEGRATION INSTRUCTIONS:
1. Save this as verify_supabase_schema.py in your project root
2. Run: python verify_supabase_schema.py
3. It will check all required columns exist and show any missing ones
"""

import os
from dotenv import load_dotenv
from supabase import create_client

# Load environment variables
load_dotenv()

def verify_schema():
    """Verify that all required columns exist in Supabase tables."""
    
    # Initialize Supabase client
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_ANON_KEY')
    
    if not url or not key:
        print("❌ Missing SUPABASE_URL or SUPABASE_ANON_KEY in .env")
        return False
    
    supabase = create_client(url, key)
    
    # Define expected schema
    expected_schema = {
        'papers': [
            'id', 'paper_id', 'title', 'authors', 'summary', 
            'published_date', 'arxiv_categories', 'pdf_url', 
            'full_text', 'comments', 'metadata', 'stored_timestamp'
        ],
        'insights': [
            'id', 'paper_id', 'study_type', 'techniques_used', 
            'implementation_complexity', 'reputation_score', 
            'extraction_confidence', 'has_code', 'has_dataset', 
            'key_findings_count', 'extraction_timestamp', 
            'total_author_hindex', 'has_conference_mention',
            'key_findings', 'limitations', 'problem_addressed',
            'prerequisites', 'real_world_applications', 'full_insights'
        ],
        'processing_logs': [
            'id', 'batch_name', 'papers_processed', 'papers_failed',
            'total_cost', 'processing_time_seconds', 'date_range',
            'success_rate', 'created_at'
        ],
        'extraction_metadata': [
            'id', 'paper_id', 'paper_uuid', 'extraction_time_seconds',
            'api_calls_made', 'estimated_cost_usd', 'extractor_version',
            'llm_model', 'extraction_timestamp'
        ]
    }
    
    all_good = True
    
    print("🔍 Verifying Supabase Schema\n")
    
    for table_name, expected_columns in expected_schema.items():
        print(f"Checking table: {table_name}")
        
        try:
            # Try to query the table to see if it exists
            result = supabase.table(table_name).select('*').limit(1).execute()
            
            # If we get here, table exists
            print(f"✅ Table '{table_name}' exists")
            
            # Check if we can see the columns from a sample
            if result.data and len(result.data) > 0:
                actual_columns = set(result.data[0].keys())
                missing_columns = set(expected_columns) - actual_columns
                
                if missing_columns:
                    print(f"❌ Missing columns in '{table_name}': {', '.join(missing_columns)}")
                    all_good = False
                else:
                    print(f"✅ All expected columns present in '{table_name}'")
            else:
                # Table is empty, we can't verify columns this way
                print(f"⚠️  Table '{table_name}' is empty - cannot verify columns")
                print(f"   Expected columns: {', '.join(expected_columns)}")
            
        except Exception as e:
            if 'relation' in str(e) and 'does not exist' in str(e):
                print(f"❌ Table '{table_name}' does not exist")
                all_good = False
            else:
                print(f"❌ Error checking table '{table_name}': {e}")
                all_good = False
        
        print()
    
    if all_good:
        print("✅ Schema verification complete - all tables exist!")
        print("\n⚠️  Note: Column verification only works for tables with data.")
        print("Run the SQL migration to ensure all columns exist.")
    else:
        print("❌ Schema issues found. Please run the SQL migration.")
    
    return all_good


def test_insert():
    """Test inserting a dummy record to verify schema."""
    print("\n🧪 Testing insert operations...\n")
    
    import uuid
    
    url = os.getenv('SUPABASE_URL')
    key = os.getenv('SUPABASE_ANON_KEY')
    supabase = create_client(url, key)
    
    # Test papers table
    try:
        import uuid
        test_paper = {
            'id': str(uuid.uuid4()),  # Explicitly provide UUID
            'paper_id': 'test_verify_123',
            'title': 'Schema Verification Test',
            'authors': ['Test Author'],
            'summary': 'Testing schema',
            'published_date': '2025-01-01',
            'arxiv_categories': ['cs.AI'],
            'pdf_url': 'http://example.com/test.pdf',
            'full_text': 'This is a test',
            'comments': 'Test comment',
            'metadata': {'test': True}
        }
        
        result = supabase.table('papers').upsert(test_paper).execute()
        print("✅ Successfully inserted test paper")
        
        # Clean up
        supabase.table('papers').delete().eq('paper_id', 'test_verify_123').execute()
        
    except Exception as e:
        print(f"❌ Failed to insert test paper: {e}")
        return False
    
    # Test insights table
    try:
        test_insights = {
            'id': str(uuid.uuid4()),  # Explicitly provide UUID
            'paper_id': 'test_verify_insights_123',
            'study_type': 'empirical',
            'techniques_used': ['rag', 'fine_tuning'],
            'implementation_complexity': 'medium',
            'reputation_score': 0.75,
            'extraction_confidence': 0.9,
            'has_code': True,
            'has_dataset': False,
            'key_findings_count': 5,
            'extraction_timestamp': '2025-01-01T00:00:00',
            'total_author_hindex': 42,
            'has_conference_mention': True,
            'key_findings': ['Finding 1', 'Finding 2'],
            'limitations': ['Limitation 1'],
            'problem_addressed': 'Test problem',
            'prerequisites': ['Python', 'PyTorch'],
            'real_world_applications': ['Healthcare'],
            'full_insights': {'test': True}
        }
        
        result = supabase.table('insights').upsert(test_insights).execute()
        print("✅ Successfully inserted test insights")
        
        # Clean up
        supabase.table('insights').delete().eq('paper_id', 'test_verify_insights_123').execute()
        
    except Exception as e:
        print(f"❌ Failed to insert test insights: {e}")
        return False
    
    # Test processing_logs table
    try:
        test_log = {
            'batch_name': 'test_verify_batch',
            'papers_processed': 10,
            'papers_failed': 1,
            'total_cost': 0.05,
            'processing_time_seconds': 120,
            'date_range': {'start': '2025-01-01', 'end': '2025-01-07'},
            'success_rate': 0.9
        }
        
        result = supabase.table('processing_logs').insert(test_log).execute()
        print("✅ Successfully inserted test processing log")
        
        # Clean up
        if result.data and len(result.data) > 0:
            log_id = result.data[0]['id']
            supabase.table('processing_logs').delete().eq('id', log_id).execute()
        
    except Exception as e:
        print(f"❌ Failed to insert test processing log: {e}")
        return False
    
    print("\n✅ All insert tests passed!")
    return True


if __name__ == "__main__":
    print("=" * 60)
    print("Supabase Schema Verification Tool")
    print("=" * 60)
    
    # First verify schema structure
    schema_ok = verify_schema()
    
    # Then test inserts
    if schema_ok:
        test_insert()
    else:
        print("\n⚠️  Skipping insert tests due to schema issues")
    
    print("\n" + "=" * 60)
    print("If you see errors above, please run the SQL migration first.")
    print("=" * 60)