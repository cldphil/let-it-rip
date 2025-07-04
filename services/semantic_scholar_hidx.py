"""
Semantic Scholar API integration for fetching author metrics.
Cloud-only version without local file caching.
Enhanced conference detection that prioritizes comments field.
"""

import requests
import time
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)

class SemanticScholarAPI:
    """
    Client for Semantic Scholar API to fetch author h-index and other metrics.
    Cloud-only version with in-memory caching only.
    
    API Documentation: https://api.semanticscholar.org/
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Semantic Scholar API client.
        
        Args:
            api_key: Optional API key for higher rate limits
        """
        self.base_url = "https://api.semanticscholar.org/graph/v1"
        self.headers = {}
        if api_key:
            self.headers["x-api-key"] = api_key
        
        # In-memory cache for current session only
        self.memory_cache = {}
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.2  # 5 requests per second without API key
        
        # API status tracking
        self._api_status = {
            'consecutive_failures': 0,
            'max_consecutive_failures': 3,
            'last_success_time': None,
            'total_requests': 0,
            'total_failures': 0
        }
        
        logger.info("Initialized Semantic Scholar API client (cloud-only, no local caching)")
    
    def _rate_limit(self):
        """Ensure we don't exceed rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        self.last_request_time = time.time()
    
    def _get_from_memory_cache(self, author_name: str) -> Optional[Dict]:
        """Get author data from in-memory cache."""
        return self.memory_cache.get(author_name.lower())
    
    def _save_to_memory_cache(self, author_name: str, author_data: Dict):
        """Save author data to in-memory cache."""
        self.memory_cache[author_name.lower()] = {
            'cached_at': datetime.now(),
            'author_data': author_data
        }
        
        # Limit cache size to prevent memory issues
        if len(self.memory_cache) > 1000:
            # Remove oldest entries
            oldest_key = min(
                self.memory_cache.keys(),
                key=lambda k: self.memory_cache[k]['cached_at']
            )
            del self.memory_cache[oldest_key]

    def get_api_status(self) -> Dict:
        """
        Get current API status including rate limit tracking.
        
        Returns:
            Dict with API status information
        """
        return self._api_status

    def _update_api_status(self, success: bool):
        """Update API status after a request."""
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
        # Check in-memory cache first
        cached_data = self._get_from_memory_cache(author_name)
        if cached_data:
            # Check if cache is recent (within current session)
            cache_age = (datetime.now() - cached_data['cached_at']).total_seconds()
            if cache_age < 3600:  # 1 hour cache for current session
                logger.debug(f"Using cached data for author: {author_name}")
                return cached_data['author_data']
        
        # Check API status before making request
        if self._api_status['consecutive_failures'] >= self._api_status['max_consecutive_failures']:
            logger.warning(f"Skipping author lookup for {author_name} due to consecutive API failures")
            return None
        
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
            
            response = requests.get(
                search_url, 
                params=params, 
                headers=self.headers, 
                timeout=10
            )
            response.raise_for_status()
            
            data = response.json()
            authors = data.get('data', [])
            
            # Update API status on success
            self._update_api_status(success=True)
            
            if not authors:
                logger.info(f"No authors found for: {author_name}")
                # Cache negative result
                negative_result = {'hIndex': 0, 'found': False}
                self._save_to_memory_cache(author_name, negative_result)
                return None
            
            # Try to find exact match first
            for author in authors:
                if author.get('name', '').lower() == author_name.lower():
                    author['found'] = True
                    self._save_to_memory_cache(author_name, author)
                    return author
            
            # If no exact match, return first result (best match)
            best_match = authors[0]
            best_match['found'] = True
            best_match['approximate_match'] = True
            self._save_to_memory_cache(author_name, best_match)
            
            logger.info(f"Found approximate match for {author_name}: {best_match.get('name')}")
            return best_match
            
        except requests.exceptions.Timeout:
            # Handle timeout gracefully
            logger.warning(f"API timeout for author {author_name} - continuing without h-index")
            self._update_api_status(success=False)
            # Cache negative result to avoid re-attempting
            negative_result = {'hIndex': 0, 'found': False, 'timeout': True}
            self._save_to_memory_cache(author_name, negative_result)
            return None
            
        except requests.exceptions.RequestException as e:
            # Handle other API errors gracefully
            logger.warning(f"API request failed for author {author_name}: {e}")
            self._update_api_status(success=False)
            # Cache negative result
            negative_result = {'hIndex': 0, 'found': False, 'error': str(e)}
            self._save_to_memory_cache(author_name, negative_result)
            return None
            
        except Exception as e:
            # Handle unexpected errors
            logger.warning(f"Unexpected error searching for author {author_name}: {e}")
            self._update_api_status(success=False)
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
        Enhanced conference detection that prioritizes the comments field.
        
        The arXiv comments field is the most reliable source for conference
        acceptance information (e.g., "Accepted at NeurIPS 2024").
        
        Args:
            paper_data: Paper metadata including title, abstract, comments, and optionally full_text
            
        Returns:
            True if conference/workshop mention detected
        """
        # Major AI/ML conferences and workshops
        conferences = [
            # Top-tier AI/ML conferences
            'neurips', 'nips', 'icml', 'iclr', 'aaai', 'ijcai', 'cvpr', 'iccv', 'eccv',
            'acl', 'emnlp', 'naacl', 'coling', 'sigir', 'kdd', 'www', 'icra', 'iros',
            'uai', 'aistats', 'colt', 'interspeech', 'asru', 'icassp',
            
            # Additional venues
            'workshop', 'symposium', 'tutorial', 'proceedings',
            
            # Acceptance indicators
            'accepted at', 'accepted to', 'to appear in', 'published in',
            'presented at', 'submission to', 'camera ready', 'conference paper',
            'peer-reviewed', 'peer reviewed', 'under review'
        ]
        
        detection_sources = []
        
        # PRIORITY 1: Check arXiv comments field (most reliable)
        if paper_data.get('comments'):
            comments_text = paper_data['comments'].lower().strip()
            logger.debug(f"Checking comments: '{comments_text}'")
            
            for conf in conferences:
                if conf in comments_text:
                    detection_sources.append(f"comments:'{conf}'")
                    logger.info(f"Conference detected in comments: '{conf}' in '{comments_text}'")
                    
            # Special handling for common comment patterns
            comment_patterns = [
                r'accepted.*(?:neurips|nips|icml|iclr|aaai|ijcai|cvpr)',
                r'(?:neurips|nips|icml|iclr|aaai|ijcai|cvpr).*(?:2024|2023|2022)',
                r'workshop.*(?:neurips|nips|icml|iclr)',
                r'to appear.*conference'
            ]
            
            import re
            for pattern in comment_patterns:
                if re.search(pattern, comments_text, re.IGNORECASE):
                    detection_sources.append(f"comments:pattern")
                    logger.info(f"Conference pattern detected in comments: {pattern}")
                    break
        
        # PRIORITY 2: Check title (papers often include venue in title)
        if paper_data.get('title'):
            title_lower = paper_data['title'].lower()
            for conf in conferences:
                if conf in title_lower:
                    detection_sources.append(f"title:'{conf}'")
                    logger.debug(f"Conference detected in title: {conf}")
        
        # PRIORITY 3: Check abstract/summary
        if paper_data.get('summary'):
            summary_lower = paper_data['summary'].lower()
            for conf in conferences:
                if conf in summary_lower:
                    detection_sources.append(f"abstract:'{conf}'")
                    logger.debug(f"Conference detected in abstract: {conf}")
        
        # PRIORITY 4: Check full text (limited scope to avoid false positives)
        if paper_data.get('full_text'):
            # Only check first 2000 chars to focus on introduction/abstract
            full_text_snippet = paper_data['full_text'][:2000].lower()
            for conf in conferences:
                if conf in full_text_snippet:
                    detection_sources.append(f"fulltext:'{conf}'")
                    logger.debug(f"Conference detected in full text: {conf}")
                    break  # Only report first match from full text
        
        # Log all detection sources for transparency
        if detection_sources:
            logger.info(f"Conference validation detected from: {', '.join(detection_sources)}")
            return True
        
        logger.debug("No conference mentions detected")
        return False
    
    def clear_memory_cache(self):
        """Clear the in-memory cache."""
        self.memory_cache.clear()
        logger.info("Cleared in-memory author cache")
    
    def get_cache_stats(self) -> Dict:
        """Get statistics about the current in-memory cache."""
        return {
            'cached_authors': len(self.memory_cache),
            'api_requests_total': self._api_status['total_requests'],
            'api_failures_total': self._api_status['total_failures'],
            'consecutive_failures': self._api_status['consecutive_failures'],
            'success_rate': (
                (self._api_status['total_requests'] - self._api_status['total_failures']) / 
                max(1, self._api_status['total_requests'])
            )
        }


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
    print("\nTesting enhanced conference detection...")
    
    # Test with comments field (highest priority)
    test_paper_comments = {
        'title': 'A Great Paper',
        'summary': 'We present a new method...',
        'comments': 'Accepted at NeurIPS 2024'
    }
    has_conference = api.detect_conference_mention(test_paper_comments)
    print(f"Conference from comments: {has_conference}")
    
    # Test with title
    test_paper_title = {
        'title': 'ICML Workshop on Advanced Methods',
        'summary': 'We present a new method...'
    }
    has_conference = api.detect_conference_mention(test_paper_title)
    print(f"Conference from title: {has_conference}")
    
    # Show cache stats
    print(f"\nCache stats: {api.get_cache_stats()}")


if __name__ == "__main__":
    test_semantic_scholar_api()