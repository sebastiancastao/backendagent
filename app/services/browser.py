"""Browser service for JavaScript-heavy pages."""

import httpx
from typing import Optional
from app.config import settings
from app.utils.logging import get_logger


logger = get_logger(__name__)


class BrowserService:
    """Service for rendering JavaScript-heavy pages using Browserless."""
    
    def __init__(self):
        self.http_client = httpx.AsyncClient(timeout=60)
    
    async def get_html(self, url: str) -> str:
        """Get rendered HTML from a URL using Browserless."""
        try:
            # Prepare request payload
            payload = {
                "url": url,
                "options": {
                    "waitUntil": "networkidle2",
                    "timeout": 30000
                }
            }
            
            # Make request to Browserless
            browserless_url = f"{settings.browserless_url.rstrip('/')}/content"
            
            response = await self.http_client.post(
                browserless_url,
                json=payload,
                headers=settings.browserless_headers
            )
            
            if response.status_code == 200:
                html = response.text
                logger.info(f"Successfully rendered {url} with Browserless")
                return html
            else:
                logger.warning(f"Browserless returned status {response.status_code} for {url}")
                raise Exception(f"Browserless error: {response.status_code}")
                
        except Exception as e:
            logger.error(f"Browserless rendering failed for {url}: {e}")
            # Fallback to direct HTTP request
            return await self._fallback_request(url)
    
    async def _fallback_request(self, url: str) -> str:
        """Fallback to direct HTTP request if Browserless fails."""
        try:
            response = await self.http_client.get(url)
            response.raise_for_status()
            logger.info(f"Fallback HTTP request successful for {url}")
            return response.text
        except Exception as e:
            logger.error(f"Fallback request failed for {url}: {e}")
            return ""
    
    async def get_screenshot(self, url: str) -> Optional[bytes]:
        """Get screenshot of a page (optional feature)."""
        try:
            payload = {
                "url": url,
                "options": {
                    "type": "png",
                    "fullPage": False,
                    "quality": 80
                }
            }
            
            browserless_url = f"{settings.browserless_url.rstrip('/')}/screenshot"
            
            response = await self.http_client.post(
                browserless_url,
                json=payload,
                headers=settings.browserless_headers
            )
            
            if response.status_code == 200:
                logger.info(f"Successfully captured screenshot of {url}")
                return response.content
            else:
                logger.warning(f"Screenshot failed for {url}: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Screenshot capture failed for {url}: {e}")
            return None
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()




