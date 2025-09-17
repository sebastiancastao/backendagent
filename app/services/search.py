"""Search service for company domain discovery."""

import httpx
from typing import List, Dict, Optional
from app.config import settings
from app.models import NormalizedCompanyName
from app.utils.logging import get_logger


logger = get_logger(__name__)


class SearchService:
    """Service for searching company information."""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=30)
    
    async def search_company(self, normalized: NormalizedCompanyName) -> List[Dict[str, str]]:
        """Search for company using available search providers."""
        try:
            # Try SerpAPI if available
            if settings.serpapi_key:
                return await self._search_with_serpapi(normalized)
            
            # Fallback to DuckDuckGo (no API key required)
            return await self._search_with_duckduckgo(normalized)
            
        except Exception as e:
            logger.error(f"Search failed: {e}")
            return []
    
    async def _search_with_serpapi(self, normalized: NormalizedCompanyName) -> List[Dict[str, str]]:
        """Search using SerpAPI."""
        try:
            query = f'"{normalized.original}" official website'
            
            params = {
                'engine': 'google',
                'q': query,
                'api_key': settings.serpapi_key,
                'num': 10
            }
            
            response = await self.http_client.get(
                'https://serpapi.com/search',
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Process organic results
            for result in data.get('organic_results', []):
                results.append({
                    'title': result.get('title', ''),
                    'url': result.get('link', ''),
                    'snippet': result.get('snippet', '')
                })
            
            logger.info(f"SerpAPI returned {len(results)} results for {normalized.original}")
            return results
            
        except Exception as e:
            logger.error(f"SerpAPI search failed: {e}")
            return []
    
    async def _search_with_duckduckgo(self, normalized: NormalizedCompanyName) -> List[Dict[str, str]]:
        """Search using DuckDuckGo Instant Answer API."""
        try:
            query = f'"{normalized.original}" official website'
            
            params = {
                'q': query,
                'format': 'json',
                'no_html': '1',
                'skip_disambig': '1'
            }
            
            response = await self.http_client.get(
                'https://api.duckduckgo.com/',
                params=params
            )
            response.raise_for_status()
            
            data = response.json()
            results = []
            
            # Process results
            if data.get('AbstractURL'):
                results.append({
                    'title': data.get('AbstractSource', ''),
                    'url': data.get('AbstractURL'),
                    'snippet': data.get('Abstract', '')
                })
            
            # Add related topics
            for topic in data.get('RelatedTopics', []):
                if isinstance(topic, dict) and topic.get('FirstURL'):
                    results.append({
                        'title': topic.get('Text', '').split(' - ')[0],
                        'url': topic.get('FirstURL'),
                        'snippet': topic.get('Text', '')
                    })
            
            # If no results, try a simpler search
            if not results:
                results = await self._fallback_search(normalized)
            
            logger.info(f"DuckDuckGo returned {len(results)} results for {normalized.original}")
            return results
            
        except Exception as e:
            logger.error(f"DuckDuckGo search failed: {e}")
            return await self._fallback_search(normalized)
    
    async def _fallback_search(self, normalized: NormalizedCompanyName) -> List[Dict[str, str]]:
        """Fallback search method - generate likely domains."""
        results = []
        
        # Generate common domain patterns
        base_name = normalized.normalized.lower().replace(' ', '')
        patterns = [
            f"{base_name}.com",
            f"{base_name}.net",
            f"{base_name}.org",
            f"www.{base_name}.com",
        ]
        
        # Add variations with common suffixes removed
        if base_name.endswith(('inc', 'corp', 'ltd', 'llc')):
            clean_name = base_name[:-3]
            patterns.extend([
                f"{clean_name}.com",
                f"{clean_name}.net",
                f"www.{clean_name}.com"
            ])
        
        # Add hyphenated versions
        if ' ' in normalized.normalized:
            hyphenated = normalized.normalized.lower().replace(' ', '-')
            patterns.extend([
                f"{hyphenated}.com",
                f"www.{hyphenated}.com"
            ])
        
        for domain in patterns:
            results.append({
                'title': f"{normalized.original} - Official Website",
                'url': f"https://{domain}",
                'snippet': f"Potential official website for {normalized.original}"
            })
        
        logger.info(f"Fallback search generated {len(results)} potential domains")
        return results
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()




