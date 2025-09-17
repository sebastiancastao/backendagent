"""Configuration settings for the AI Agent Company Data Scraper."""

import os
import base64
import json
from typing import Optional
from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Server
    port: int = 8080
    host: str = "0.0.0.0"
    
    # Google Sheets
    gsheet_id: str = "test_sheet_id"
    google_service_account_json_base64: str = "dGVzdA=="
    
    # AI Services
    openai_api_key: str = ""
    use_ai: bool = True
    
    # Browser Services
    browserless_url: str = "https://chrome.browserless.io"
    browserless_token: Optional[str] = None
    
    # Search Services
    serpapi_key: Optional[str] = None
    
    # DataForSEO API credentials
    dataforseo_login: str = ""
    dataforseo_password: str = ""
    dataforseo_base64: str = ""
    
    # Network
    http_proxy: Optional[str] = None
    
    # Rate Limiting
    rate_limit_per_minute: int = 60
    
    # Scraping
    max_concurrent_jobs: int = 5
    request_timeout: int = 30
    max_retries: int = 3
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = False

    @property
    def google_service_account_info(self) -> dict:
        """Decode base64 service account JSON."""
        try:
            decoded = base64.b64decode(self.google_service_account_json_base64)
            return json.loads(decoded)
        except Exception as e:
            raise ValueError(f"Invalid Google service account JSON: {e}")

    @property
    def browserless_headers(self) -> dict:
        """Headers for Browserless API requests."""
        headers = {"Content-Type": "application/json"}
        if self.browserless_token:
            headers["Authorization"] = f"Bearer {self.browserless_token}"
        return headers


# Global settings instance
settings = Settings()