#!/usr/bin/env python3
"""
Test script for the manual processing system.
Run this after setting up Supabase to verify manual processing works.
"""

from pathlib import Path
import sys

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

import os
from datetime import datetime, timedelta
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_imports():
    """Test that all required modules can be imported."""
    print("📦 Testing Imports...")
    
    try:
        from core.manual_processing_system import ManualProcessingController
        print("✅ ManualProcessingController imported successfully")
        return True
    except ImportError as e:
        print(f"❌ Failed to import ManualProcessingController: {e}")
        print("💡 Make sure manual_processing_system.py is in your project root")
        return False

def test_controller_initialization():
    """Test that the controller can be initialized."""
    print("\n🔧 Testing Controller Initialization...")
    
    try:
        from core.manual_processing_system import ManualProcessingController
        controller = ManualProcessingController()
        print("✅ Controller initialized successfully")
        print(f"✅ Storage type: {type(controller.storage).__name__}")
        print(f"✅ Processor type: {type(controller.processor).__name__}")
        print(f"✅ Fetcher type: {type(controller.fetcher).__name__}")
        return controller
    except Exception as e:
        print(f"❌ Controller initialization failed: {e}")
        return None

def test_date_ranges(controller):
    """Test date range functionality."""
    print("\n📅 Testing Date Range Functions...")
    
    try:
        # Test available date ranges
        ranges = controller.get_available_date_ranges()
        print(f"✅ Available date ranges: {len(ranges)} options")
        
        for name, info in ranges.items():
            print(f"   📅 {name}: {info['description']} ({info['estimated_papers']} papers)")
        
        return True
    except Exception as e:
        print(f"❌ Date range test failed: {e}")
        return False

def test_date_validation(controller):
    """Test date validation logic."""
    print("\n✅ Testing Date Validation...")
    
    try:
        # Test valid date range
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        is_valid, error_msg = controller.validate_date_range(start_date, end_date)
        
        if is_valid:
            print("✅ Valid date range passed validation")
        else:
            print(f"❌ Valid date range failed: {error_msg}")
            return False
        
        # Test invalid date range (start after end)
        invalid_start = datetime.now()
        invalid_end = datetime.now() - timedelta(days=1)
        
        is_valid, error_msg = controller.validate_date_range(invalid_start, invalid_end)
        
        if not is_valid:
            print("✅ Invalid date range correctly rejected")
        else:
            print("❌ Invalid date range incorrectly accepted")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Date validation test failed: {e}")
        return False

def test_cost_estimation(controller):
    """Test cost estimation functionality."""
    print("\n💰 Testing Cost Estimation...")
    
    try:
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        estimate = controller.estimate_processing_cost(start_date, end_date, max_papers=10)
        
        print(f"✅ Cost estimation completed:")
        print(f"   📊 Estimated papers: {estimate['estimated_papers']}")
        print(f"   💰 Estimated cost: ${estimate['estimated_cost_usd']}")
        print(f"   ⏱️  Estimated time: {estimate['estimated_time_minutes']} minutes")
        print(f"   📅 Days in range: {estimate['days_in_range']}")
        
        # Verify estimate has required fields
        required_fields = ['estimated_papers', 'estimated_cost_usd', 'estimated_time_minutes', 'days_in_range']
        for field in required_fields:
            if field not in estimate:
                print(f"❌ Missing field in estimate: {field}")
                return False
        
        return True
        
    except Exception as e:
        print(f"❌ Cost estimation test failed: {e}")
        return False

def test_existing_papers_check(controller):
    """Test existing papers check functionality."""
    print("\n🔍 Testing Existing Papers Check...")
    
    try:
        start_date = datetime.now() - timedelta(days=7)
        end_date = datetime.now()
        
        existing_check = controller.check_existing_papers(start_date, end_date)
        
        print(f"✅ Existing papers check completed:")
        print(f"   📊 Existing papers: {existing_check['existing_papers']}")
        print(f"   💡 Recommendation: {existing_check['recommendation']}")
        print(f"   📝 Message: {existing_check['message']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Existing papers check failed: {e}")
        return False

def test_storage_usage(controller):
    """Test storage usage functionality."""
    print("\n📊 Testing Storage Usage...")
    
    try:
        usage = controller.get_storage_usage()
        
        print(f"✅ Storage usage retrieved:")
        print(f"   📄 Total papers: {usage.get('total_papers', 0)}")
        print(f"   🧠 Total insights: {usage.get('total_insights', 0)}")
        print(f"   📊 Avg reputation score: {usage.get('average_reputation_score', 0)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Storage usage test failed: {e}")
        return False

def test_processing_history(controller):
    """Test processing history functionality."""
    print("\n📋 Testing Processing History...")
    
    try:
        history = controller.get_processing_history(limit=5)
        
        print(f"✅ Processing history retrieved: {len(history)} entries")
        
        if history:
            for entry in history[:2]:  # Show first 2 entries
                print(f"   📦 Batch: {entry.get('batch_name', 'Unknown')}")
                print(f"      Papers: {entry.get('papers_processed', 0)}")
                print(f"      Date: {entry.get('created_at', 'Unknown')[:10]}")
        else:
            print("   📝 No processing history found (expected for fresh setup)")
        
        return True
        
    except Exception as e:
        print(f"❌ Processing history test failed: {e}")
        return False

def test_date_range_fetcher():
    """Test the date range fetcher functionality."""
    print("\n🌐 Testing Date Range Fetcher...")
    
    try:
        from core.manual_processing_system import DateRangeArxivFetcher
        
        fetcher = DateRangeArxivFetcher()
        print("✅ DateRangeArxivFetcher initialized successfully")
        
        # Test URL encoding
        test_query = "test query with spaces"
        encoded = fetcher._url_encode(test_query)
        print(f"✅ URL encoding works: '{test_query}' -> '{encoded}'")
        
        # Test query building
        query = fetcher._build_date_range_query("20240101", "20240107", 5)
        print(f"✅ Query building works: {len(query)} characters")
        
        return True
        
    except Exception as e:
        print(f"❌ Date range fetcher test failed: {e}")
        return False

def run_comprehensive_test():
    """Run all tests and provide summary."""
    print("=" * 60)
    print("🧪 Manual Processing System Test")
    print("=" * 60)
    
    # Track test results
    tests_passed = 0
    total_tests = 0
    
    # Test 1: Imports
    total_tests += 1
    if test_imports():
        tests_passed += 1
    else:
        print("\n❌ Cannot continue without successful imports")
        return False
    
    # Test 2: Controller initialization
    total_tests += 1
    controller = test_controller_initialization()
    if controller:
        tests_passed += 1
    else:
        print("\n❌ Cannot continue without controller initialization")
        return False
    
    # Test 3: Date ranges
    total_tests += 1
    if test_date_ranges(controller):
        tests_passed += 1
    
    # Test 4: Date validation
    total_tests += 1
    if test_date_validation(controller):
        tests_passed += 1
    
    # Test 5: Cost estimation
    total_tests += 1
    if test_cost_estimation(controller):
        tests_passed += 1
    
    # Test 6: Existing papers check
    total_tests += 1
    if test_existing_papers_check(controller):
        tests_passed += 1
    
    # Test 7: Storage usage
    total_tests += 1
    if test_storage_usage(controller):
        tests_passed += 1
    
    # Test 8: Processing history
    total_tests += 1
    if test_processing_history(controller):
        tests_passed += 1
    
    # Test 9: Date range fetcher
    total_tests += 1
    if test_date_range_fetcher():
        tests_passed += 1
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    print(f"Tests passed: {tests_passed}/{total_tests}")
    
    if tests_passed == total_tests:
        print("\n🎉 ALL TESTS PASSED!")
        print("✅ Manual processing system is working correctly!")
        print("\nNext steps:")
        print("1. Run: streamlit run streamlit_app.py")
        print("2. Use the manual processing interface")
        print("3. Select date ranges and process papers!")
        return True
    else:
        print(f"\n⚠️  {total_tests - tests_passed} tests failed")
        print("Please fix the issues above before using manual processing")
        
        if tests_passed >= 7:  # Most functionality works
            print("\n💡 Core functionality appears to work")
            print("You can try using the manual processing interface")
        
        return False

def main():
    """Main function."""
    try:
        success = run_comprehensive_test()
        
        if success:
            print("\n🚀 Manual processing system is ready!")
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