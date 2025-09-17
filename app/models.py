"""Pydantic models for the AI Agent Company Data Scraper."""

from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pydantic import BaseModel, EmailStr, HttpUrl, Field, validator
from enum import Enum


class JobStatus(str, Enum):
    """Job processing status."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class CompanyProfile(BaseModel):
    """Main company profile schema."""
    company_name: str
    official_email: EmailStr
    website: Optional[HttpUrl] = None
    hq_address: Optional[str] = None
    phone: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None
    year_founded: Optional[int] = None
    employee_count: Optional[str] = None  # e.g., "51–200"
    logo_url: Optional[HttpUrl] = None
    country: Optional[str] = None
    city: Optional[str] = None
    socials: Dict[str, Optional[HttpUrl]] = Field(default_factory=dict)
    target_market: Optional[str] = None
    niche: Optional[str] = None
    services_offered: Optional[str] = None
    client_types: Optional[str] = None
    mission_statement: Optional[str] = None
    founding_story: Optional[str] = None
    why_started: Optional[str] = None
    company_values: Optional[str] = None
    # Competitor analysis
    main_keywords: Optional[List[str]] = None
    competitors: Optional[List[str]] = None
    competitor_urls: Optional[List[HttpUrl]] = None
    
    # Customer acquisition and growth strategy
    customer_acquisition_process: Optional[str] = None
    growth_strategies_that_work: Optional[str] = None
    ineffective_strategies: Optional[str] = None
    seo_and_advertising_approach: Optional[str] = None
    
    # Future goals and challenges
    main_business_goals_12_months: Optional[str] = None
    seo_ads_visibility_goals: Optional[str] = None
    current_blocking_factors: Optional[str] = None
    
    # Service and location priorities
    top_3_priority_services: Optional[str] = None
    service_areas_and_regions: Optional[str] = None
    
    # Content planning and Topic Authority Map
    topic_authority_map: Optional[Dict] = None
    content_plan_summary: Optional[str] = None
    
    confidence_per_field: Dict[str, float] = Field(default_factory=dict)

    @validator('year_founded')
    def validate_year_founded(cls, v):
        if v is not None and (v < 1800 or v > datetime.now().year):
            raise ValueError('Year founded must be between 1800 and current year')
        return v

    @validator('socials', pre=True)
    def validate_socials(cls, v):
        if v is None:
            return {}
        # Ensure all social keys are lowercase
        return {k.lower(): url for k, url in v.items()}


class JobRequest(BaseModel):
    """Request model for creating a new scraping job."""
    company_name: str = Field(..., min_length=1, max_length=200)
    official_email: EmailStr
    domain: Optional[str] = None
    competitor_domains: Optional[List[str]] = None
    main_locations: Optional[List[str]] = None


class JobResponse(BaseModel):
    """Response model for job creation and status."""
    id: str
    status: JobStatus
    company_name: str
    official_email: str
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None
    profile: Optional[CompanyProfile] = None


class Candidate(BaseModel):
    """A candidate value for a field with confidence score."""
    value: str
    score: float
    rank: int


class Source(BaseModel):
    """A source of information with URL, snippet, and score."""
    url: HttpUrl
    snippet: str
    score: float


class JobDetailResponse(BaseModel):
    """Detailed response including candidates and sources."""
    id: str
    status: JobStatus
    company_name: str
    official_email: str
    created_at: datetime
    updated_at: datetime
    error: Optional[str] = None
    profile: Optional[CompanyProfile] = None
    candidates: Dict[str, List[Candidate]] = Field(default_factory=dict)
    sources: Dict[str, List[Source]] = Field(default_factory=dict)


class ProfileOverrides(BaseModel):
    """Model for manual profile overrides."""
    website: Optional[HttpUrl] = None
    hq_address: Optional[str] = None
    phone: Optional[str] = None
    industry: Optional[str] = None
    description: Optional[str] = None
    year_founded: Optional[int] = None
    employee_count: Optional[str] = None
    logo_url: Optional[HttpUrl] = None
    country: Optional[str] = None
    city: Optional[str] = None
    socials: Optional[Dict[str, Optional[HttpUrl]]] = None
    target_market: Optional[str] = None
    niche: Optional[str] = None
    services_offered: Optional[str] = None
    client_types: Optional[str] = None
    mission_statement: Optional[str] = None
    founding_story: Optional[str] = None
    why_started: Optional[str] = None
    company_values: Optional[str] = None
    # Competitor analysis
    main_keywords: Optional[List[str]] = None
    competitors: Optional[List[str]] = None
    competitor_urls: Optional[List[HttpUrl]] = None
    
    # Customer acquisition and growth strategy
    customer_acquisition_process: Optional[str] = None
    growth_strategies_that_work: Optional[str] = None
    ineffective_strategies: Optional[str] = None
    seo_and_advertising_approach: Optional[str] = None
    
    # Future goals and challenges
    main_business_goals_12_months: Optional[str] = None
    seo_ads_visibility_goals: Optional[str] = None
    current_blocking_factors: Optional[str] = None
    
    # Service and location priorities
    top_3_priority_services: Optional[str] = None
    service_areas_and_regions: Optional[str] = None

    @validator('year_founded')
    def validate_year_founded(cls, v):
        if v is not None and (v < 1800 or v > datetime.now().year):
            raise ValueError('Year founded must be between 1800 and current year')
        return v


class ExportResponse(BaseModel):
    """Response model for JSON export."""
    profile: CompanyProfile
    sources: Dict[str, List[Source]]
    metadata: Dict[str, str] = Field(default_factory=dict)


class ScrapingConfig(BaseModel):
    """Configuration for scraping behavior."""
    use_ai: bool = True
    max_sources_per_field: int = 5
    confidence_threshold: float = 0.7
    timeout_seconds: int = 30
    max_retries: int = 3
    ai_guided_scraping: bool = True
    max_scraping_iterations: int = 5
    reward_threshold: float = 0.8


class ScrapingDecision(BaseModel):
    """AI decision for what to scrape next."""
    action: str  # "scrape_url", "search_google", "analyze_content", "complete"
    target: str  # URL to scrape or search query
    reason: str  # Why this action was chosen
    expected_fields: List[str]  # What fields this might help find
    confidence: float  # How confident the AI is about this decision


class DataEvaluation(BaseModel):
    """AI evaluation of scraped data quality."""
    field: str
    value: str
    satisfies_question: bool
    confidence_score: float
    reasoning: str
    reward_score: float  # 0-1 scale for learning


class ScrapingStrategy(BaseModel):
    """AI-generated scraping strategy."""
    questions_to_answer: List[str]
    priority_fields: List[str]
    scraping_plan: List[ScrapingDecision]
    success_criteria: Dict[str, float]


# Internal models for scraper
class ScrapingResult(BaseModel):
    """Internal model for scraping results."""
    field: str
    candidates: List[Tuple[str, float]]  # (value, score)
    sources: List[Dict[str, str]]  # {url, snippet, score}


class NormalizedCompanyName(BaseModel):
    """Normalized company name for search."""
    original: str
    normalized: str
    aliases: List[str] = Field(default_factory=list)


class SupportingArticle(BaseModel):
    """Supporting article for content planning."""
    title: str
    keywords: List[str]
    serp_features: List[str]
    intent: str
    priority_score: float


class TopicPillar(BaseModel):
    """Topic pillar for content authority map."""
    topic: str
    intent: str
    seasonality: Optional[str] = None
    cluster_score: float
    pillar_page_h1: str
    supporting_articles: List[SupportingArticle]
    local_entities_sample: List[str] = Field(default_factory=list)
    top_competing_urls: List[str] = Field(default_factory=list)


class TopicAuthorityMap(BaseModel):
    """Complete topic authority map for content planning."""
    niche: str
    location: str
    language: str
    country_code: str
    pillars: List[TopicPillar]
    total_keywords: int
    avg_search_volume: int
    content_gap_opportunities: List[str] = Field(default_factory=list)


class ContentPlanKeyword(BaseModel):
    """Individual keyword data for content planning."""
    keyword: str
    cluster: str
    intent: str
    volume: int
    trend_12m: float
    cpc: float
    competition: float
    has_local_pack: bool
    has_snippet: bool
    paa_count: int
    top_urls: List[str]
    priority_score: float




