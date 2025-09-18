"""Configuration settings for the AI Agent Company Data Scraper."""

import os
import base64
import json
from typing import Optional
try:
    from pydantic_settings import BaseSettings
except ImportError:
    from pydantic import BaseSettings


class Settings(BaseSettings):
    """Application settings."""
    
    # Server
    port: int = 8080
    host: str = "0.0.0.0"
    
    # Google Sheets
    gsheet_id: str = "1UGa-MNcQw0q2TO5hgwD4M1haPBtJp57INcVvOcDrR5E"
    google_service_account_json_base64: str = "eyJ0eXBlIjogInNlcnZpY2VfYWNjb3VudCIsICJwcm9qZWN0X2lkIjogInBob25pYy1nb29kcy0zMTcxMTgiLCAicHJpdmF0ZV9rZXlfaWQiOiAiM2ZjODVmMTM4NTFiN2RhZWZiOThkYzFjOWU2MDk4N2JiODdmNGMyYSIsICJwcml2YXRlX2tleSI6ICItLS0tLUJFR0lOIFBSSVZBVEUgS0VZLS0tLS1cbk1JSUV2Z0lCQURBTkJna3Foa2lHOXcwQkFRRUZBQVNDQktnd2dnU2tBZ0VBQW9JQkFRREJRN0lhWlB5YmVwN0RcbkZ5eEpma1Nrdi9SNUREaWpmUnFhM2owYzdRSTBoSDBsYnZBcGJmQkJGSXBaSFNhRUJkVFkwWXQvRXR2QWZVUm1cblE3OENaSjUvNHdSUjdOdHFDa1JHMGlreUkyS09RT1N6Y2lkT0ZIek0ybUZoL09MZkdYQVVFSkhpKzBCSFIvaExcbm5PdVFVQWRPY0hWTGpRdGw3ZzdsdDZwclhhNHFTejQ5cWZPUWhWc1d4dHVhTUJYTXB1b2V3NGxFLzZJeUNsbE1cbmQ1YytCaW1ZaVZQcmozNGdCcUFaaUJHa1V3Z3QvWFJvWHZWYmdCM0F3TGxmQXpSc2FZRUt4OG55TFdTWkMvaDRcbkVPM0N3T1V4dTdpNFBuVTVzR2ZkTExmSVdhM3UrNktqaVRUcmtWV3VnNDA4Z05YQUhEajd6OVpUU051c0REbGxcblRTdzZiZUN0QWdNQkFBRUNnZ0VBRDJRNEg1NEIrQ2Z3ZmowRU5wT0FjcXZPNlMwRFV4SGk0b2s4ZUx5YUVRWCtcbjFDZXlVZyt0WGZiUzBIeTFLbzB2QWExK2h6cEFsMk0zNVUrK2tJN2tNMFI5MHVzaXk1NThYWW96T2JPcEQ3SGZcbldqaUhKTkxBNEZlc0FHL0d2d1I1bStvalB0NTArZzJqUlVlcFdGbVI4TG55dlhlWUpzcXFFN3E5aEhETGhCYWlcblFsNXFjUE4xVTNtNmVIaUNvbXNURDVHNXhCODJLdVk2UnBlQzVmcEE3OXhsb0hidUFmbzNvUVNMeG9vdHNKVERcbmlPMjlQaVF6WkpTU2FoTVdVYStqbnd3TnlSNkQrVERpS1ZMMVlLNnFLU1ZvRVRuRGZaUHN3MEtMMU4wNWY3b3NcbnQ0UlN6MjIxN1dCeXo3MmlZNkdYS0dya3ZCVkZXVUFmRU1uRTdQV0lnUUtCZ1FEOFRRZXZDQkhlZDY4OCtDQ3Ncbml3KzI4MDJLek9hOHExK0t6VWp2L0c0MzYzNm1NVWsyTm1xbit5dUZqZEVlTnN3dEE1Z3FIQS9GUHdlK2x5a0JcbjVxRUJUY0lZWUd0ejVzS0tPWnoxeFc0dnJkeWRvU1pXWTNVZEl3blB0WnJHWlRkR2o2ekxZWEpJMkgvS1hQUHBcbmVSem12Ty81WWxvZnVsUjZjOHFQMnV3VkVRS0JnUURFR1JVQVVrWWVxZzloMWdEZkRzSVBvSXpJK3l3TFFWTjVcbnU5RUhVVzNBRU9kOHEyOURJRzh3NDRjTXRIcHlnd1lMa2RhdFhlaXJBQ051WjdXT0k4bFp0V0w1dm1qWThFNnhcbnVYRzE5STdFc2x1aEY1Zk9zUHZHRzRQV2l2WHJRSGpXQnUwMEZzUStmQ3dhUEV1bVJiQzlYZStab2F1SWMxUmtcbkdBVjBEV0NoM1FLQmdRQzU1NEI3NWpSVWVsZnpVdG13aVo4QXJYSTdqaE9PZmJBZXRIakQ3SHJDVlpHeW42cFVcbmVmQkk2bmY5SnF3cDJUTEZFRnIwM0V4NmlLRUtQSk9JeFFscHpvUHdOa3UyMFJnVGhiUTBIRSsrYmh1YlFuemZcbk50VzZySEMwVGhwSGlaa2JNdzZkcGFYeUt5U3VWYU9jS3hPeXFSRWg3dWg5Ykd5RzlmOFIxTHdVSVFLQmdIYk1cblZ5dTlyQnN4bldZQTlzQXJPYWVyOVA1aEk5cWh0Y2QrMy9CV0JXNGhENHc0YzR3d0h3eDRHcjI0cktHby9NV1BcbmtZV3Y2WitHMHZBMHhnbVpab2NCV3plL2dkZERKZm1IUmZzY2NFMTdYQVZvdktBTUdrdDNLZFNVbE16elh3RVNcbmF5dHVTMjhyWExCOExMeExaZm9pNlYraVVMKzJWcjdZeXB6MjZiN1ZBb0dCQUtIU1oxa1FENDlVNTdiQlFaN05cbmJIRUlHYkJtQXJaTGw0M0F5L2VEQTRyVWlTeW1peGhKbi8xdzVDazFITHFwanBzU3hKNmlCblRNdXVSbytWQnFcbm5JRkFKUzc2UE5tbnZVQnk5MzhyelZtZU5rcFVWOHphM0FOMXB1TzRRQWRxNkpJQlgzM2RPaHk5SU5aSENZY3JcbnJzVXR3YjJkS0daU3VaQWxRT21jSDY2NVxuLS0tLS1FTkQgUFJJVkFURSBLRVktLS0tLVxuIiwgImNsaWVudF9lbWFpbCI6ICJzZWJhc3RpYW5jYXN0YW5vQHBob25pYy1nb29kcy0zMTcxMTguaWFtLmdzZXJ2aWNlYWNjb3VudC5jb20iLCAiY2xpZW50X2lkIjogIjExMzcwNDA3ODExMjE1MjkyNDIwMiIsICJhdXRoX3VyaSI6ICJodHRwczovL2FjY291bnRzLmdvb2dsZS5jb20vby9vYXV0aDIvYXV0aCIsICJ0b2tlbl91cmkiOiAiaHR0cHM6Ly9vYXV0aDIuZ29vZ2xlYXBpcy5jb20vdG9rZW4iLCAiYXV0aF9wcm92aWRlcl94NTA5X2NlcnRfdXJsIjogImh0dHBzOi8vd3d3Lmdvb2dsZWFwaXMuY29tL29hdXRoMi92MS9jZXJ0cyIsICJjbGllbnRfeDUwOV9jZXJ0X3VybCI6ICJodHRwczovL3d3dy5nb29nbGVhcGlzLmNvbS9yb2JvdC92MS9tZXRhZGF0YS94NTA5L3NlYmFzdGlhbmNhc3Rhbm8lNDBwaG9uaWMtZ29vZHMtMzE3MTE4LmlhbS5nc2VydmljZWFjY291bnQuY29tIiwgInVuaXZlcnNlX2RvbWFpbiI6ICJnb29nbGVhcGlzLmNvbSJ9"
    
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