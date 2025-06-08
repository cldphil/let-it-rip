#!/usr/bin/env python3
"""
Test script to verify configuration and imports work correctly.
Run this to check if the config.py fix resolves the import issues.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

def test_config_imports():
    """Test that configuration imports work correctly."""
    print("🧪 Testing Configuration Imports")
    print("=" * 50)
    
    try:
        # Test basic config import
        from config import Config
        print("✅ Config class imported successfully")
        
        # Test configuration validation
        print("\n📋 Running configuration validation...")
        Config.validate_config()
        
        # Test storage class import
        print("\n💾 Testing storage class import...")
        storage_class = Config.get_storage_class()
        print(f"✅ Storage class imported: {storage_class.__name__}")
        
        # Test other config methods
        print("\n⚙️  Testing configuration methods...")
        
        api_key = Config.get_active_api_key()
        print(f"🔑 API Key available: {'✅' if api_key else '❌'}")
        
        reputation_config = Config.get_reputation_score_config()
        print(f"📊 Reputation config: {len(reputation_config)} settings")
        
        ranking_config = Config.get_ranking_config()
        print(f"🎯 Ranking config: {len(ranking_config)} settings")
        
        cloud_config = Config.get_cloud_config()
        print(f"☁️  Cloud config: {cloud_config['use_cloud_storage']}")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_core_imports():
    """Test that core module imports work correctly."""
    print("\n🧪 Testing Core Module Imports")
    print("=" * 50)
    
    try:
        # Test core module imports
        from core import (
            InsightStorage,
            SyncBatchProcessor,
            SynthesisEngine,
            UserContext,
            PaperInsights
        )
        print("✅ Core module imports successful")
        
        # Test creating instances
        print("\n🏗️  Testing instance creation...")
        
        storage = InsightStorage()
        print("✅ InsightStorage created")
        
        processor = SyncBatchProcessor()
        print("✅ SyncBatchProcessor created")
        
        synthesis_engine = SynthesisEngine()
        print("✅ SynthesisEngine created")
        
        # Test basic functionality
        print("\n🔍 Testing basic functionality...")
        
        stats = storage.get_statistics()
        print(f"✅ Storage statistics: {stats['total_papers']} papers")
        
        return True
        
    except Exception as e:
        print(f"❌ Core imports test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_optional_imports():
    """Test optional imports that might not be available."""
    print("\n🧪 Testing Optional Imports")
    print("=" * 50)
    
    optional_modules = [
        ('anthropic', 'Anthropic API'),
        ('chromadb', 'ChromaDB vector storage'),
        ('sentence_transformers', 'Sentence transformers'),
        ('streamlit', 'Streamlit web framework'),
        ('supabase', 'Supabase client')
    ]
    
    available_count = 0
    
    for module_name, description in optional_modules:
        try:
            __import__(module_name)
            print(f"✅ {description}: Available")
            available_count += 1
        except ImportError:
            print(f"⚠️  {description}: Not available")
    
    print(f"\n📊 {available_count}/{len(optional_modules)} optional modules available")
    
    return True

def test_manual_processing():
    """Test manual processing import if available."""
    print("\n🧪 Testing Manual Processing")
    print("=" * 50)
    
    try:
        from core.manual_processing_system import ManualProcessingController
        print("✅ Manual processing system available")
        
        controller = ManualProcessingController()
        print("✅ ManualProcessingController created")
        
        # Test basic functionality
        available_ranges = controller.get_available_date_ranges()
        print(f"✅ Available date ranges: {len(available_ranges)}")
        
        return True
        
    except ImportError as e:
        print(f"⚠️  Manual processing not available: {e}")
        return False
    except Exception as e:
        print(f"❌ Manual processing test failed: {e}")
        return False

def main():
    """Run all tests and provide summary."""
    print("🚀 GenAI Research Platform - Configuration Test")
    print("=" * 60)
    
    test_results = []
    
    # Run all tests
    test_results.append(("Configuration", test_config_imports()))
    test_results.append(("Core Imports", test_core_imports()))
    test_results.append(("Optional Imports", test_optional_imports()))
    test_results.append(("Manual Processing", test_manual_processing()))
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Test Summary")
    print("=" * 60)
    
    passed = 0
    total = len(test_results)
    
    for test_name, result in test_results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed! System is ready to use.")
        return True
    else:
        print(f"\n⚠️  {total - passed} tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)