#!/usr/bin/env python3
"""
Complete Supabase connection and functionality test for let-it-rip project.
Run this after setting up your Supabase database to verify everything works.
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_environment_variables():
    """Test that required environment variables are set."""
    print("🔧 Testing Environment Variables...")
    
    required_vars = [
        'SUPABASE_URL',
        'SUPABASE_ANON_KEY'
    ]
    
    missing_vars = []
    for var in required_vars:
        if not os.environ.get(var):
            missing_vars.append(var)
    
    if missing_vars:
        print(f"❌ Missing environment variables: {', '.join(missing_vars)}")
        print("💡 Please check your .env file")
        return False
    
    print("✅ All required environment variables are set")
    return True

def test_supabase_import():
    """Test that supabase library is installed."""
    print("\n📦 Testing Supabase Import...")
    
    try:
        from supabase import create_client, Client
        print("✅ Supabase library imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import supabase: {e}")
        print("💡 Install with: pip install supabase")
        return False

def test_basic_connection():
    """Test basic connection to Supabase."""
    print("\n🌐 Testing Basic Connection...")
    
    try:
        from supabase import create_client, Client
        
        url = os.environ.get("SUPABASE_URL")
        key = os.environ.get("SUPABASE_ANON_KEY")
        
        if not url or not key:
            print("❌ Missing SUPABASE_URL or SUPABASE_ANON_KEY")
            return False
        
        supabase: Client = create_client(url, key)
        
        # Test connection by querying papers table
        result = supabase.table('papers').select('*').limit(1).execute()
        
        print(f"✅ Connection successful to {url[:30]}...")
        print(f"✅ Papers table accessible (found {len(result.data)} records)")
        
        return True, supabase
        
    except Exception as e:
        print(f"❌ Connection failed: {e}")
        return False, None

def test_all_tables(supabase):
    """Test that all required tables exist and are accessible."""
    print("\n📋 Testing Database Tables...")
    
    tables = ['papers', 'insights', 'extraction_metadata', 'processing_logs']
    
    for table in tables:
        try:
            result = supabase.table(table).select('*').limit(1).execute()
            print(f"✅ Table '{table}' accessible (found {len(result.data)} records)")
        except Exception as e:
            print(f"❌ Table '{table}' failed: {e}")
            return False
    
    return True

def test_vector_extension(supabase):
    """Test that vector extension and similarity search function work."""
    print("\n🔍 Testing Vector Search...")
    
    try:
        # Test the match_insights function with dummy embedding
        dummy_embedding = [0.1] * 384  # 384-dimensional vector
        
        result = supabase.rpc('match_insights', {
            'query_embedding': dummy_embedding,
            'match_threshold': 0.5,
            'match_count': 5
        }).execute()
        
        print("✅ Vector search function 'match_insights' working")
        print(f"✅ Vector search returned {len(result.data)} results")
        
        return True
        
    except Exception as e:
        print(f"❌ Vector search failed: {e}")
        print("💡 Make sure you enabled the 'vector' extension in Database → Extensions")
        return False

def test_sentence_transformers():
    """Test that sentence-transformers library is available."""
    print("\n🤖 Testing Sentence Transformers...")
    
    try:
        from sentence_transformers import SentenceTransformer
        
        # Try to load the model used by the system
        model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Test embedding generation
        test_text = "This is a test sentence for embedding generation"
        embedding = model.encode([test_text])[0]
        
        print(f"✅ Sentence transformer model loaded successfully")
        print(f"✅ Generated embedding with {len(embedding)} dimensions")
        
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import sentence-transformers: {e}")
        print("💡 Install with: pip install sentence-transformers")
        return False
    except Exception as e:
        print(f"❌ Sentence transformer test failed: {e}")
        return False

def test_insert_and_query(supabase):
    """Test inserting test data and querying it back."""
    print("\n📝 Testing Data Insert/Query...")
    
    try:
        # Insert test paper
        test_paper = {
            'id': 'test_paper_123',
            'title': 'Test Paper for Connection Verification',
            'authors': ['Test Author'],
            'summary': 'This is a test paper to verify Supabase connectivity.'
        }
        
        # Insert paper
        insert_result = supabase.table('papers').upsert(test_paper).execute()
        print("✅ Test paper inserted successfully")
        
        # Query it back
        query_result = supabase.table('papers').select('*').eq('id', 'test_paper_123').execute()
        
        if query_result.data:
            retrieved_paper = query_result.data[0]
            print(f"✅ Test paper retrieved: {retrieved_paper['title']}")
            
            # Clean up - delete test paper
            delete_result = supabase.table('papers').delete().eq('id', 'test_paper_123').execute()
            print("✅ Test paper cleaned up")
            
            return True
        else:
            print("❌ Failed to retrieve test paper")
            return False
            
    except Exception as e:
        print(f"❌ Insert/query test failed: {e}")
        return False

def test_storage_statistics(supabase):
    """Test getting storage statistics."""
    print("\n📊 Testing Storage Statistics...")
    
    try:
        # Get counts from each table
        papers_count = supabase.table('papers').select('id', count='exact').execute()
        insights_count = supabase.table('insights').select('id', count='exact').execute()
        
        print(f"✅ Storage statistics retrieved:")
        print(f"   📄 Papers: {papers_count.count}")
        print(f"   🧠 Insights: {insights_count.count}")
        
        return True
        
    except Exception as e:
        print(f"❌ Storage statistics test failed: {e}")
        return False

def test_anthropic_api():
    """Test Anthropic API key (optional)."""
    print("\n🤖 Testing Anthropic API (optional)...")
    
    anthropic_key = os.environ.get('ANTHROPIC_API_KEY')
    if not anthropic_key:
        print("⚠️  ANTHROPIC_API_KEY not found (this is optional for now)")
        return True
    
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=anthropic_key)
        print("✅ Anthropic API client initialized successfully")
        return True
    except ImportError:
        print("⚠️  Anthropic library not installed (install with: pip install anthropic)")
        return True
    except Exception as e:
        print(f"⚠️  Anthropic API test failed: {e}")
        return True

def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("=" * 60)
    print("🧪 Supabase Connection Test for let-it-rip Project")
    print("=" * 60)
    
    # Track test results
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Environment variables
    total_tests += 1
    if test_environment_variables():
        tests_passed += 1
    else:
        print("\n❌ Cannot continue without environment variables")
        return False
    
    # Test 2: Supabase import
    total_tests += 1
    if test_supabase_import():
        tests_passed += 1
    else:
        print("\n❌ Cannot continue without supabase library")
        return False
    
    # Test 3: Basic connection
    total_tests += 1
    connection_result = test_basic_connection()
    if connection_result[0]:
        tests_passed += 1
        supabase = connection_result[1]
    else:
        print("\n❌ Cannot continue without database connection")
        return False
    
    # Test 4: Database tables
    total_tests += 1
    if test_all_tables(supabase):
        tests_passed += 1
    
    # Test 5: Vector search
    total_tests += 1
    if test_vector_extension(supabase):
        tests_passed += 1
    
    # Test 6: Sentence transformers
    total_tests += 1
    if test_sentence_transformers():
        tests_passed += 1
    
    # Test 7: Data operations
    total_tests += 1
    if test_insert_and_query(supabase):
        tests_passed += 1
    
    # Test 8: Storage statistics
    total_tests += 1
    if test_storage_statistics(supabase):
        tests_passed += 1
    
    # Test 9: Anthropic API (optional)
    total_tests += 1
    if test_anthropic_api():
        tests_passed += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Your Supabase setup is working perfectly!")
        print("✅ Ready to start processing papers!")
        print("\nNext steps:")
        print("1. Save manual_processing_system.py")
        print("2. Run: streamlit run streamlit_app.py")
        print("3. Start processing papers with date range selection!")
        return True
    else:
        print(f"\n⚠️  {total_tests - tests_passed} tests failed")
        print("Please fix the issues above before proceeding")
        
        if tests_passed >= 6:  # Core functionality works
            print("\n💡 Core functionality appears to work")
            print("You can proceed but some features may be limited")
        
        return False

def main():
    """Main function."""
    try:
        success = run_comprehensive_test()
        
        if success:
            print("\n🚀 System is ready for cloud processing!")
        else:
            print("\n🔧 Please fix the issues and run the test again")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n\n⏹️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n💥 Unexpected error during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()