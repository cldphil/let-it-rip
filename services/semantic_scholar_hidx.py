"""
Semantic Scholar API integration for fetching author metrics.
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import json
from pathlib import Path

logger = logging.getLogger(__name__)

class SemanticScholarAPI:
    """
    Client for Semantic Scholar API to fetch author h-index and other metrics.
    
    API Documentation: https://api.semanticscholar.org/
    """
    
    def __init__(self, api_key: Optional[str] = None, cache_dir: str = "storage/author_cache"):
        """
        Initialize Semantic Scholar API client.
        
        Args:
            api_key: Optional API key for higher rate limits
            cache_dir: Directory to cache author data
        """
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key
        
        # Set up caching to avoid repeated API calls
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 10 requests per second without API key
        
        logger.info("Initialized Semantic Scholar API client")
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _get_cache_path(self, author_name: str) -> Path:
        """Get cache file path for an author."""
        # Simple filename sanitization
        safe_name = "".join(c for c in author_name if c.isalnum() or c in "._- ")
        return self.cache_dir / f"{safe_name}.json"
    
    def _load_from_cache(self, author_name: str) -> Optional[Dict]:
        """Load author data from cache if available and recent."""
        cache_path = self._get_cache_path(author_name)
        if cache_path.exists():
            try:
                with open(cache_path, 'r') as f:
                    cached_data = json.load(f)
                
                # Check if cache is recent (within 30 days)
                cache_time = datetime.fromisoformat(cached_data.get('cached_at', '2020-01-01'))
                if (datetime.now() - cache_time).days < 30:
                    return cached_data['author_data']
                    
            except Exception as e:
                logger.warning(f"Failed to load cache for {author_name}: {e}")
        
        return None
    
    def _save_to_cache(self, author_name: str, author_data: Dict):
        """Save author data to cache."""
        cache_path = self._get_cache_path(author_name)
        cache_data = {
            'cached_at': datetime.now().isoformat(),
            'author_data': author_data
        }
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(cache_data, f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to cache data for {author_name}: {e}")

    def get_api_status(self) -> Dict:
        """
        Get current API status including rate limit tracking.
        
        Returns:
            Dict with API status information
        """
        if not hasattr(self, '_api_status'):
            self._api_status = {
                'consecutive_failures': 0,
                'max_consecutive_failures': 3,
                'last_success_time': None,
                'total_requests': 0,
                'total_failures': 0
            }
        
        return self._api_status

    def _update_api_status(self, success: bool):
        """Update API status after a request."""
        if not hasattr(self, '_api_status'):
            self.get_api_status()  # Initialize if needed
        
        self._api_status['total_requests'] += 1
        
        if success:
            self._api_status['consecutive_failures'] = 0
            self._api_status['last_success_time'] = time.time()
        else:
            self._api_status['consecutive_failures'] += 1
            self._api_status['total_failures'] += 1
    
    def search_author(self, author_name: str) -> Optional[Dict]:
        """
        Search for an author by name and return their data including h-index.
        
        Args:
            author_name: Full name of the author
            
        Returns:
            Author data dict with h-index, or None if not found
        """
        # Check cache first
        cached_data = self._load_from_cache(author_name)
        if cached_data:
            logger.info(f"Using cached data for author: {author_name}")
            return cached_data
        
        # Rate limiting
        self._rate_limit()
        
        try:
            # Search endpoint
            search_url = f"{self.base_url}/author/search"
            params = {
                'query': author_name,
                'fields': 'authorId,name,affiliations,paperCount,citationCount,hIndex',
                'limit': 5  # Get top 5 matches
            }
            
            response = requests.get(search_url, params=params, headers=self.headers, timeout=10)
            response.raise_for_status()
            
            data = response.json()
            authors = data.get('data', [])
            
            # Update API status on success
            self._update_api_status(success=True)
            
            if not authors:
                logger.info(f"No authors found for: {author_name}")
                # Cache negative result
                self._save_to_cache(author_name, {'hIndex': 0, 'found': False})
                return None
            
            # Try to find exact match first
            for author in authors:
                if author.get('name', '').lower() == author_name.lower():
                    author['found'] = True
                    self._save_to_cache(author_name, author)
                    return author
            
            # If no exact match, return first result (best match)
            best_match = authors[0]
            best_match['found'] = True
            best_match['approximate_match'] = True
            self._save_to_cache(author_name, best_match)
            
            logger.info(f"Found approximate match for {author_name}: {best_match.get('name')}")
            return best_match
            
        except requests.exceptions.RequestException as e:
            # Update API status on failure
            self._update_api_status(success=False)
            logger.error(f"API request failed for author {author_name}: {e}")
            return None
        except Exception as e:
            self._update_api_status(success=False)
            logger.error(f"Unexpected error searching for author {author_name}: {e}")
            return None
    
    def get_author_hindex(self, author_name: str) -> int:
        """
        Get h-index for a single author.
        
        Args:
            author_name: Author name
            
        Returns:
            h-index value (0 if not found)
        """
        author_data = self.search_author(author_name)
        if author_data and author_data.get('found'):
            return author_data.get('hIndex', 0)
        return 0
    
    def get_paper_total_hindex(self, authors: List[str]) -> Tuple[int, Dict[str, int]]:
        """
        Get total h-index for all authors of a paper.
        
        Args:
            authors: List of author names
            
        Returns:
            Tuple of (total h-index, dict of individual h-indices)
        """
        total_hindex = 0
        individual_hindices = {}
        
        for author in authors:
            h_index = self.get_author_hindex(author)
            individual_hindices[author] = h_index
            total_hindex += h_index
            
            # Small delay between authors to be respectful
            if len(authors) > 1:
                time.sleep(0.1)
        
        logger.info(f"Total h-index for {len(authors)} authors: {total_hindex}")
        return total_hindex, individual_hindices
    
    def detect_conference_mention(self, paper_data: Dict) -> bool:
        """
        Detect if a paper mentions acceptance at a conference/workshop.
        
        Args:
            paper_data: Paper metadata including title, abstract, and optionally full_text
            
        Returns:
            True if conference/workshop mention detected
        """
        # Major AI/ML conferences and workshops
        conferences = [
            # Top-tier AI/ML conferences
            'neurips', 'nips', 'icml', 'iclr', 'aaai', 'ijcai', 'cvpr', 'iccv', 'eccv',
            'acl', 'emnlp', 'naacl', 'coling', 'sigir', 'kdd', 'www', 'icra', 'iros',
            'uai', 'aistats', 'colt', 'interspeech', 'asru', 'icassp',
            
            # Workshops
            'workshop', 'symposium', 'tutorial',
            
            # Specific mentions
            'accepted at', 'accepted to', 'to appear in', 'published in',
            'presented at', 'submission to', 'camera ready', 'conference paper'
        ]
        
        # Combine all text to search
        search_text = ""
        if paper_data.get('title'):
            search_text += paper_data['title'].lower() + " "
        if paper_data.get('summary'):
            search_text += paper_data['summary'].lower() + " "
        if paper_data.get('full_text'):
            # Just check first 2000 chars of full text to avoid false positives
            search_text += paper_data['full_text'][:2000].lower()
        
        # Check for conference mentions
        for conf in conferences:
            if conf in search_text:
                logger.info(f"Detected conference mention: {conf}")
                return True
        
        # Check arXiv comments field if available
        if paper_data.get('comments'):
            comments_lower = paper_data['comments'].lower()
            for conf in conferences:
                if conf in comments_lower:
                    logger.info(f"Detected conference in comments: {conf}")
                    return True
        
        return False


# Convenience function for testing
def test_semantic_scholar_api():
    """Test the Semantic Scholar API with sample queries."""
    api = SemanticScholarAPI()
    
    # Test single author
    print("Testing single author lookup...")
    h_index = api.get_author_hindex("Geoffrey Hinton")
    print(f"Geoffrey Hinton h-index: {h_index}")
    
    # Test multiple authors
    print("\nTesting multiple authors...")
    authors = ["Yann LeCun", "Yoshua Bengio", "Geoffrey Hinton"]
    total, individual = api.get_paper_total_hindex(authors)
    print(f"Total h-index: {total}")
    print(f"Individual h-indices: {individual}")
    
    # Test conference detection
    print("\nTesting conference detection...")
    test_paper = {
        'title': 'A Great Paper Accepted at NeurIPS 2024',
        'summary': 'We present a new method...'
    }
    has_conference = api.detect_conference_mention(test_paper)
    print(f"Conference mention detected: {has_conference}")

    def search_author(self, author_name: str) -> Optional[Dict]:
        """
        Search for an author by name with enhanced timeout handling.
        """
        # Check cache first
        cached_data = self._load_from_cache(author_name)
        if cached_data:
            logger.info(f"Using cached data for author: {author_name}")
            return cached_data
        
        # Rate limiting
        self._rate_limit()
        
        try:
            # Search endpoint with longer timeout
            search_url = f"{self.base_url}/author/search"
            params = {
                'query': author_name,
                'fields': 'authorId,name,affiliations,paperCount,citationCount,hIndex',
                'limit': 5
            }
            
            # Increased timeout to 30 seconds
            response = requests.get(search_url, params=params, headers=self.headers, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            authors = data.get('data', [])
            
            # Update API status on success
            self._update_api_status(success=True)
            
            if not authors:
                logger.info(f"No authors found for: {author_name}")
                # Cache negative result
                self._save_to_cache(author_name, {'hIndex': 0, 'found': False})
                return None
            
            # Try to find exact match first
            for author in authors:
                if author.get('name', '').lower() == author_name.lower():
                    author['found'] = True
                    self._save_to_cache(author_name, author)
                    return author
            
            # If no exact match, return first result
            best_match = authors[0]
            best_match['found'] = True
            best_match['approximate_match'] = True
            self._save_to_cache(author_name, best_match)
            
            logger.info(f"Found approximate match for {author_name}: {best_match.get('name')}")
            return best_match
            
        except requests.exceptions.Timeout:
            # Handle timeout gracefully
            logger.warning(f"API timeout for author {author_name} - continuing without h-index")
            self._update_api_status(success=False)
            # Cache negative result to avoid re-attempting
            self._save_to_cache(author_name, {'hIndex': 0, 'found': False, 'timeout': True})
            return None
            
        except requests.exceptions.RequestException as e:
            # Handle other API errors gracefully
            logger.warning(f"API request failed for author {author_name}: {e}")
            self._update_api_status(success=False)
            # Cache negative result
            self._save_to_cache(author_name, {'hIndex': 0, 'found': False, 'error': str(e)})
            return None
            
        except Exception as e:
            # Handle unexpected errors
            logger.warning(f"Unexpected error searching for author {author_name}: {e}")
            self._update_api_status(success=False)
            return None

if __name__ == "__main__":
    test_semantic_scholar_api()
