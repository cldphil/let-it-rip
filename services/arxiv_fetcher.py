import requests
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict
import time
import json
import io
from PyPDF2 import PdfReader

class ArxivGenAIFetcher:
    """
    A class to fetch generative AI research papers from arXiv.
    
    Uses the arXiv API to search for papers related to generative AI
    published in the current year.
    """
    
    def __init__(self):
        """Initialize the fetcher with arXiv API base URL."""
        self.base_url = "http://export.arxiv.org/api/query"
        self.current_year = datetime.now().year
        
    def build_search_query(self, max_results: int = 50) -> str:
        """
        Build search query for generative AI papers from current year.
        
        Args:
            max_results: Maximum number of results to fetch
            
        Returns:
            Formatted query string for arXiv API
        """
        # Search terms for generative AI research
        search_terms = [
            "generative artificial intelligence",
            "generative AI", 
            "large language model",
            "LLM",
            "GPT",
            "diffusion model",
            "generative model",
            "text generation",
            "image generation"
        ]
        
        # Build OR query for search terms
        query_parts = []
        for term in search_terms:
            query_parts.append(f'all:"{term}"')
        
        search_query = " OR ".join(query_parts)
        
        # Add year filter - papers from current year
        year_filter = f"submittedDate:[{self.current_year}0101* TO {self.current_year}1231*]"
        
        full_query = f"({search_query}) AND {year_filter}"
        
        return f"?search_query={requests.utils.quote(full_query)}&start=0&max_results={max_results}&sortBy=submittedDate&sortOrder=descending"
    
    def parse_arxiv_entry(self, entry: ET.Element) -> Dict:
        """
        Parse a single arXiv entry from XML response.
        
        Args:
            entry: XML element representing one paper
            
        Returns:
            Dictionary with paper metadata
        """
        # Handle XML namespaces
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        paper = {}
        
        # Basic metadata
        paper['id'] = entry.find('atom:id', namespaces).text if entry.find('atom:id', namespaces) is not None else ""
        paper['title'] = entry.find('atom:title', namespaces).text.strip() if entry.find('atom:title', namespaces) is not None else ""
        paper['summary'] = entry.find('atom:summary', namespaces).text.strip() if entry.find('atom:summary', namespaces) is not None else ""
        
        # Authors
        authors = []
        for author in entry.findall('atom:author', namespaces):
            name_elem = author.find('atom:name', namespaces)
            if name_elem is not None:
                authors.append(name_elem.text)
        paper['authors'] = authors
        
        # Dates
        published = entry.find('atom:published', namespaces)
        paper['published'] = published.text if published is not None else ""
        
        updated = entry.find('atom:updated', namespaces)
        paper['updated'] = updated.text if updated is not None else ""
        
        # Categories
        categories = []
        for category in entry.findall('atom:category', namespaces):
            term = category.get('term')
            if term:
                categories.append(term)
        paper['categories'] = categories
        
        # PDF link
        for link in entry.findall('atom:link', namespaces):
            if link.get('type') == 'application/pdf':
                paper['pdf_url'] = link.get('href')
                break
        else:
            paper['pdf_url'] = ""
            
        return paper
    
    def extract_pdf_text(self, pdf_url: str) -> str:
        """
        Extract text content from PDF URL without downloading to disk.
        
        Args:
            pdf_url: URL to the PDF file
            
        Returns:
            Extracted text content as string
        """
        try:
            print(f"Extracting text from PDF...")
            
            # Stream PDF content directly into memory
            response = requests.get(pdf_url, stream=True, timeout=30)
            response.raise_for_status()
            
            # Create PDF reader from memory buffer
            pdf_buffer = io.BytesIO(response.content)
            pdf_reader = PdfReader(pdf_buffer)
            
            # Extract text from all pages
            full_text = ""
            for page_num, page in enumerate(pdf_reader.pages):
                try:
                    page_text = page.extract_text()
                    full_text += f"\n--- Page {page_num + 1} ---\n"
                    full_text += page_text
                except Exception as e:
                    print(f"Error extracting page {page_num + 1}: {e}")
                    continue
            
            return full_text.strip()
            
        except requests.RequestException as e:
            print(f"Error downloading PDF {pdf_url}: {e}")
            return ""
        except Exception as e:
            print(f"Error extracting text from PDF {pdf_url}: {e}")
            return ""
    
    def fetch_papers(self, max_results: int = 50, include_full_text: bool = False) -> List[Dict]:
        """
        Fetch generative AI papers from arXiv.
        
        Args:
            max_results: Maximum number of papers to fetch
            include_full_text: Whether to extract full text from PDFs
            
        Returns:
            List of paper dictionaries
        """
        query = self.build_search_query(max_results)
        url = self.base_url + query
        
        print(f"Fetching papers from: {url}")
        print(f"Searching for generative AI papers from {self.current_year}...")
        
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            # Parse XML response
            root = ET.fromstring(response.content)
            
            # Find all entry elements
            entries = root.findall('.//{http://www.w3.org/2005/Atom}entry')
            
            papers = []
            for i, entry in enumerate(entries, 1):
                print(f"Processing paper {i}/{len(entries)}...")
                paper = self.parse_arxiv_entry(entry)
                
                # Extract full text if requested and PDF URL is available
                if include_full_text and paper.get('pdf_url'):
                    paper['full_text'] = self.extract_pdf_text(paper['pdf_url'])
                    # Add text length for reference
                    paper['text_length'] = len(paper.get('full_text', ''))
                
                papers.append(paper)
                
                # Small delay to be respectful to arXiv servers
                if include_full_text:
                    time.sleep(1)
                
            print(f"Successfully processed {len(papers)} papers")
            return papers
            
        except requests.RequestException as e:
            print(f"Error fetching data from arXiv: {e}")
            return []
        except ET.ParseError as e:
            print(f"Error parsing XML response: {e}")
            return []
    
    def save_papers_to_json(self, papers: List[Dict], filename: str = "arxiv_genai_papers.json") -> None:
        """
        Save papers to JSON file in output directory.
        
        Args:
            papers: List of paper dictionaries
            filename: Output filename
        """
        # Ensure output directory exists
        import os
        output_dir = "output"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        filepath = os.path.join(output_dir, filename)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(papers, f, indent=2, ensure_ascii=False)
            print(f"Papers saved to {filepath}")
        except Exception as e:
            print(f"Error saving papers: {e}")
    
    def print_paper_summary(self, papers: List[Dict]) -> None:
        """
        Print a summary of fetched papers.
        
        Args:
            papers: List of paper dictionaries
        """
        if not papers:
            print("No papers found")
            return
            
        print(f"\n=== SUMMARY: {len(papers)} Generative AI Papers from {self.current_year} ===\n")
        
        for i, paper in enumerate(papers[:10], 1):  # Show first 10
            print(f"{i}. {paper['title']}")
            print(f"   Authors: {', '.join(paper['authors'][:3])}{'...' if len(paper['authors']) > 3 else ''}")
            print(f"   Published: {paper['published'][:10]}")
            print(f"   Categories: {', '.join(paper['categories'][:3])}")
            print(f"   Abstract: {paper['summary'][:150]}...")
            print(f"   PDF: {paper['pdf_url']}")
            print()

def main():
    """
    Main function to demonstrate the arXiv fetcher with direct core module usage.
    """
    print("=== arXiv Generative AI Research Fetcher ===")
    print(f"Searching for papers from {datetime.now().year}")
    
    # Initialize fetcher
    fetcher = ArxivGenAIFetcher()
    
    # Import core modules directly (no wrapper needed)
    from core import SyncBatchProcessor, InsightStorage
    
    # Choose whether to include full text
    include_text = input("Include full text extraction? (y/n): ").lower().startswith('y')
    
    # Fetch papers (start with fewer papers if extracting full text)
    max_papers = 5 if include_text else 20
    papers = fetcher.fetch_papers(max_results=max_papers, include_full_text=include_text)
    
    if papers:
        print(f"\n=== PROCESSING {len(papers)} PAPERS WITH ENHANCED EXTRACTION ===")
        
        # Initialize storage and processor directly
        storage = InsightStorage()
        processor = SyncBatchProcessor(storage=storage)
        
        # Process papers in batch
        checkpoint_name = f"main_batch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        stats = processor.process_papers(papers, checkpoint_name=checkpoint_name)
        
        if stats['successful'] > 0:
            print(f"\nSuccessfully processed {stats['successful']} papers!")
            print(f"Total cost: ${stats.get('total_cost', 0):.2f}")
            print(f"Processing time: {stats.get('total_time', 0):.1f} seconds")
            
            # Print enhanced statistics
            storage_stats = storage.get_statistics()
            
            print(f"\n=== ENHANCED STATISTICS ===")
            print(f"Total papers in storage: {storage_stats['total_papers']}")
            print(f"Average quality score: {storage_stats['average_quality_score']:.2f}")
            print(f"Average key findings per paper: {storage_stats['average_key_findings_count']:.1f}")
            print(f"Recent papers (last 2 years): {storage_stats['recent_papers_count']}")
            
            # Show complexity distribution
            print(f"\n=== IMPLEMENTATION COMPLEXITY ===")
            for complexity, count in storage_stats['complexity_distribution'].items():
                print(f"{complexity}: {count} papers")
            
            # Show study type distribution
            print(f"\n=== STUDY TYPES ===")
            for study_type, count in storage_stats['study_type_distribution'].items():
                print(f"{study_type}: {count} papers")
        
        # Print paper summary
        fetcher.print_paper_summary(papers)
        
        # Save papers to output folder
        filename = "arxiv_genai_papers_enhanced_fulltext.json" if include_text else "arxiv_genai_papers_enhanced.json"
        fetcher.save_papers_to_json(papers, filename)
    
    else:
        print("No papers found. Try adjusting search terms or date range.")

if __name__ == "__main__":
    main()