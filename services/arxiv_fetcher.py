"""
Enhanced arXiv fetcher with robust error handling and connection management.
Drop-in replacement for the original ArxivFetcher.
"""

import requests
import xml.etree.ElementTree as ET
import time
import logging
from datetime import datetime
from typing import List, Dict, Optional
from urllib.parse import quote_plus
import random

logger = logging.getLogger(__name__)

class ArxivFetcher:
    """
    Enhanced arXiv fetcher with robust error handling and connection management.
    Drop-in replacement for the original with better reliability.
    """
    
    def __init__(self):
        """Initialize with arXiv API configuration."""
        self.base_url = "http://export.arxiv.org/api/query"
        self.max_retries = 3
        self.base_delay = 2.0  # Base delay between requests
        self.timeout = 30  # Request timeout
        self.max_results_per_request = 100  # Limit to avoid large responses
        
        # Session for connection reuse
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GenAI-Research-Platform/1.0 (Educational Use)',
            'Accept': 'application/atom+xml',
            'Connection': 'keep-alive'
        })
        
        logger.info("Initialized arXiv fetcher with enhanced error handling")
    
    def fetch_papers_date_range(self, start_date: datetime, end_date: datetime,
                               max_results: int = 100, include_full_text: bool = True) -> List[Dict]:
        """
        Fetch papers within a specific date range with robust error handling.
        
        Args:
            start_date: Start date for papers
            end_date: End date for papers
            max_results: Maximum papers to fetch
            include_full_text: Whether to extract full text
            
        Returns:
            List of paper dictionaries
        """
        # Format dates for arXiv API
        start_str = start_date.strftime('%Y%m%d')
        end_str = end_date.strftime('%Y%m%d')
        
        logger.info(f"Fetching papers from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Limit max_results to avoid overwhelming the API
        if max_results > self.max_results_per_request:
            logger.warning(f"Limiting max_results from {max_results} to {self.max_results_per_request}")
            max_results = self.max_results_per_request
        
        try:
            # Build and execute query
            query_url = self._build_query(start_str, end_str, max_results)
            papers = self._fetch_with_retry(query_url)
            
            if not papers:
                logger.warning("No papers found in API response")
                return []
            
            logger.info(f"Successfully fetched {len(papers)} papers from arXiv")
            
            # Extract full text if requested
            if include_full_text:
                papers = self._add_full_text(papers)
            
            return papers
            
        except Exception as e:
            logger.error(f"Failed to fetch papers: {e}")
            return []
    
    def _build_query(self, start_date: str, end_date: str, max_results: int) -> str:
        """Build a robust query URL that's less likely to cause connection issues."""
        
        # Simplified search terms to reduce query complexity
        search_terms = [
            "generative AI",
            "large language model", 
            "LLM",
            "GPT",
            "diffusion model"
        ]
        
        # Build simpler OR query
        search_query = " OR ".join(f'all:"{term}"' for term in search_terms)
        
        # Add date filter
        date_filter = f"submittedDate:[{start_date}* TO {end_date}*]"
        
        # Combine query parts
        full_query = f"({search_query}) AND {date_filter}"
        
        # Build URL with proper encoding
        params = {
            'search_query': full_query,
            'start': '0',
            'max_results': str(max_results),
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        # Manual URL construction to avoid encoding issues
        query_parts = []
        for key, value in params.items():
            encoded_value = quote_plus(value)
            query_parts.append(f"{key}={encoded_value}")
        
        query_string = "&".join(query_parts)
        full_url = f"{self.base_url}?{query_string}"
        
        logger.info(f"Built query URL (length: {len(full_url)})")
        logger.debug(f"Query: {full_query}")
        
        return full_url
    
    def _fetch_with_retry(self, url: str) -> List[Dict]:
        """Fetch data with exponential backoff retry logic."""
        
        for attempt in range(self.max_retries):
            try:
                # Add jitter to delay to avoid thundering herd
                delay = self.base_delay * (2 ** attempt) + random.uniform(0, 1)
                
                if attempt > 0:
                    logger.info(f"Retry attempt {attempt + 1}/{self.max_retries} after {delay:.1f}s delay")
                    time.sleep(delay)
                
                # Make request with timeout
                logger.info(f"Making request to arXiv API (attempt {attempt + 1})")
                response = self.session.get(url, timeout=self.timeout, stream=True)
                
                # Check for HTTP errors
                response.raise_for_status()
                
                # Log response info
                content_length = response.headers.get('content-length', 'unknown')
                logger.info(f"Received response: {response.status_code}, length: {content_length}")
                
                # Parse XML response
                try:
                    # Read content in chunks to handle large responses
                    content = b''
                    for chunk in response.iter_content(chunk_size=8192):
                        content += chunk
                    
                    root = ET.fromstring(content)
                    logger.info("Successfully parsed XML response")
                    
                except ET.ParseError as e:
                    logger.error(f"XML parsing failed: {e}")
                    logger.debug(f"Response content (first 500 chars): {content[:500]}")
                    if attempt == self.max_retries - 1:
                        raise
                    continue
                
                # Extract papers from XML
                papers = self._extract_papers_from_xml(root)
                logger.info(f"Extracted {len(papers)} papers from XML")
                
                return papers
                
            except requests.exceptions.ConnectionError as e:
                logger.warning(f"Connection error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise ConnectionError(f"Failed to connect to arXiv after {self.max_retries} attempts: {e}")
                
            except requests.exceptions.Timeout as e:
                logger.warning(f"Timeout on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise TimeoutError(f"Request timed out after {self.max_retries} attempts: {e}")
                
            except requests.exceptions.HTTPError as e:
                logger.error(f"HTTP error on attempt {attempt + 1}: {e}")
                if e.response.status_code in [429, 503]:  # Rate limit or service unavailable
                    if attempt < self.max_retries - 1:
                        delay = min(60, self.base_delay * (2 ** (attempt + 2)))  # Cap at 60 seconds
                        logger.info(f"Rate limited, waiting {delay}s before retry")
                        time.sleep(delay)
                        continue
                raise
                
            except Exception as e:
                logger.error(f"Unexpected error on attempt {attempt + 1}: {e}")
                if attempt == self.max_retries - 1:
                    raise
        
        return []
    
    def _extract_papers_from_xml(self, root: ET.Element) -> List[Dict]:
        """Extract paper data from XML response."""
        papers = []
        
        # Find all entry elements
        namespaces = {
            'atom': 'http://www.w3.org/2005/Atom',
            'arxiv': 'http://arxiv.org/schemas/atom'
        }
        
        entries = root.findall('.//atom:entry', namespaces)
        logger.info(f"Found {len(entries)} paper entries in XML")
        
        for i, entry in enumerate(entries, 1):
            try:
                paper = self._parse_arxiv_entry(entry, namespaces)
                if paper:
                    papers.append(paper)
                    
                if i % 10 == 0:
                    logger.debug(f"Processed {i}/{len(entries)} papers")
                    
            except Exception as e:
                logger.warning(f"Failed to parse paper {i}: {e}")
                continue
        
        return papers
    
    def _parse_arxiv_entry(self, entry: ET.Element, namespaces: dict) -> Optional[Dict]:
        """Parse a single paper entry from XML."""
        try:
            paper = {}
            
            # Basic metadata
            id_elem = entry.find('atom:id', namespaces)
            paper['id'] = id_elem.text if id_elem is not None else ""
            
            title_elem = entry.find('atom:title', namespaces)
            paper['title'] = title_elem.text.strip() if title_elem is not None else ""
            
            summary_elem = entry.find('atom:summary', namespaces)
            paper['summary'] = summary_elem.text.strip() if summary_elem is not None else ""
            
            # Authors
            authors = []
            for author in entry.findall('atom:author', namespaces):
                name_elem = author.find('atom:name', namespaces)
                if name_elem is not None:
                    authors.append(name_elem.text)
            paper['authors'] = authors
            
            # Dates
            published_elem = entry.find('atom:published', namespaces)
            paper['published'] = published_elem.text if published_elem is not None else ""
            
            updated_elem = entry.find('atom:updated', namespaces)
            paper['updated'] = updated_elem.text if updated_elem is not None else ""
            
            # Categories
            categories = []
            for category in entry.findall('atom:category', namespaces):
                term = category.get('term')
                if term:
                    categories.append(term)
            paper['categories'] = categories
            
            # PDF link
            paper['pdf_url'] = ""
            for link in entry.findall('atom:link', namespaces):
                if link.get('type') == 'application/pdf':
                    paper['pdf_url'] = link.get('href', '')
                    break
            
            # Comments (optional)
            comment_elem = entry.find('arxiv:comment', namespaces)
            if comment_elem is not None:
                paper['comments'] = comment_elem.text
            
            return paper
            
        except Exception as e:
            logger.error(f"Error parsing paper entry: {e}")
            return None
    
    def _add_full_text(self, papers: List[Dict]) -> List[Dict]:
        """Add full text to papers with safe error handling."""
        
        for i, paper in enumerate(papers, 1):
            if not paper.get('pdf_url'):
                logger.debug(f"Paper {i}: No PDF URL available")
                continue
            
            try:
                logger.info(f"Extracting full text for paper {i}/{len(papers)}")
                full_text = self._extract_pdf_text(paper['pdf_url'])
                
                if full_text:
                    paper['full_text'] = full_text
                    paper['text_length'] = len(full_text)
                    logger.debug(f"Paper {i}: Extracted {len(full_text)} characters")
                else:
                    logger.warning(f"Paper {i}: No text extracted from PDF")
                
                # Rate limiting for PDF requests
                if i < len(papers):
                    time.sleep(1)
                    
            except Exception as e:
                logger.warning(f"Paper {i}: PDF extraction failed: {e}")
                paper['full_text'] = ""
                paper['text_length'] = 0
                continue
        
        return papers
    
    def _extract_pdf_text(self, pdf_url: str) -> str:
        """Extract text from PDF with robust error handling."""
        try:
            # Use shorter timeout for PDF downloads
            response = self.session.get(pdf_url, timeout=20, stream=True)
            response.raise_for_status()
            
            # Check content size
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > 20971520: # 20MB limit 
                logger.warning(f"PDF too large: {content_length} bytes")
                return ""
            
            # Read PDF content
            import io
            from PyPDF2 import PdfReader
            
            pdf_content = b''
            for chunk in response.iter_content(chunk_size=8192):
                pdf_content += chunk
                if len(pdf_content) > 20971520:  # 20MB limit
                    logger.warning("PDF download exceeded size limit")
                    break
            
            # Extract text
            pdf_buffer = io.BytesIO(pdf_content)
            pdf_reader = PdfReader(pdf_buffer)
            
            full_text = ""
            max_pages = min(20, len(pdf_reader.pages))  # Limit pages to process
            
            for page_num in range(max_pages):
                try:
                    page_text = pdf_reader.pages[page_num].extract_text()
                    full_text += f"\n--- Page {page_num + 1} ---\n"
                    full_text += page_text
                except Exception as e:
                    logger.debug(f"Error extracting page {page_num + 1}: {e}")
                    continue
            
            return full_text.strip()
            
        except Exception as e:
            logger.warning(f"PDF extraction failed for {pdf_url}: {e}")
            return ""
    
    def _url_encode(self, text: str) -> str:
        """URL encode the query string."""
        return quote_plus(text)
    
    def __del__(self):
        """Clean up session on deletion."""
        if hasattr(self, 'session'):
            self.session.close()