#!/usr/bin/env python3
"""
Installation script for GenAI Research Platform dependencies.
Handles the installation in the correct order to avoid conflicts.
"""

import subprocess
import sys

def install_package(package):
    """Install a package using pip."""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ Successfully installed {package}\n")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ Failed to install {package}\n")
        return False

def main():
    """Install all dependencies in the correct order."""
    print("=" * 60)
    print("GenAI Research Platform - Dependency Installation")
    print("=" * 60)
    print()
    
    # Core dependencies (no conflicts)
    core_packages = [
        "requests>=2.31.0",
        "PyPDF2>=3.0.1", 
        "python-dotenv>=1.0.0",
        "pandas>=2.0.0",
        "plotly>=5.15.0",
        "pydantic>=2.0.0",
        "aiofiles>=23.0.0"
    ]
    
    # Install core packages
    print("Installing core packages...")
    for package in core_packages:
        if not install_package(package):
            print(f"Warning: Failed to install {package}, continuing anyway...")
    
    # Install Anthropic API
    print("Installing Anthropic API...")
    install_package("anthropic>=0.18.0")
    
    # Install PyTorch (required for sentence-transformers)
    print("Installing PyTorch...")
    if not install_package("torch"):
        print("WARNING: PyTorch installation failed.")
        print("You may need to install it manually from https://pytorch.org/")
        response = input("Continue without PyTorch? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)
    
    # Install sentence-transformers
    print("Installing sentence-transformers...")
    if not install_package("sentence-transformers"):
        print("WARNING: sentence-transformers installation failed.")
        print("The system will work but without vector search functionality.")
    
    # Install ChromaDB
    print("Installing ChromaDB...")
    if not install_package("chromadb>=0.4.0"):
        print("WARNING: ChromaDB installation failed.")
        print("The system will work but without vector storage.")
    
    # Install Streamlit last (it has many dependencies)
    print("Installing Streamlit...")
    install_package("streamlit>=1.28.0")
    
    print("\n" + "=" * 60)
    print("Installation Summary")
    print("=" * 60)
    
    # Check what's installed
    installed_packages = []
    failed_packages = []
    
    check_packages = [
        "requests", "PyPDF2", "dotenv", "streamlit", "pandas", 
        "plotly", "anthropic", "chromadb", "torch", 
        "sentence_transformers", "pydantic", "aiofiles"
    ]
    
    for package in check_packages:
        try:
            __import__(package)
            installed_packages.append(package)
        except ImportError:
            failed_packages.append(package)
    
    print(f"\n✓ Successfully installed: {len(installed_packages)} packages")
    for pkg in installed_packages:
        print(f"  - {pkg}")
    
    if failed_packages:
        print(f"\n✗ Failed to install: {len(failed_packages)} packages")
        for pkg in failed_packages:
            print(f"  - {pkg}")
        
        print("\nThe system can still run with limited functionality.")
        print("Core features will work, but some advanced features may be unavailable.")
    else:
        print("\n✓ All packages installed successfully!")
    
    print("\nNext steps:")
    print("1. Make sure your .env file contains ANTHROPIC_API_KEY")
    print("2. Run: python setup.py")
    print("3. Run: python test_local_storage.py")
    print("4. Run: streamlit run streamlit_app.py")

if __name__ == "__main__":
    main()