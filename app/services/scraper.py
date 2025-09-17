"""Company scraper service with AI reconciliation."""

import re
import json
import asyncio
from typing import Dict, List, Optional, Tuple, Any
from urllib.parse import urljoin, urlparse
import httpx
from bs4 import BeautifulSoup
import trafilatura
from openai import AsyncOpenAI

from app.config import settings
from app.models import JobStatus, CompanyProfile, NormalizedCompanyName
from app.services.sheets_dao import SheetsDAO
from app.services.search import SearchService
from app.services.browser import BrowserService
from app.utils.logging import get_logger
from app.utils.normalization import normalize_company_name


logger = get_logger(__name__)


class CompanyScraper:
    """Main scraper service for company data extraction."""
    
    def __init__(self, sheets_dao: SheetsDAO):
        self.sheets_dao = sheets_dao
        self.search_service = SearchService()
        self.browser_service = BrowserService()
        self.openai_client = AsyncOpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None
        
        # HTTP client with proper headers
        self.http_client = httpx.AsyncClient(
            timeout=settings.request_timeout,
            follow_redirects=True,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }
        )
    
    async def scrape_company(self, job_id: str, company_name: str, official_email: str):
        """Main scraping workflow."""
        try:
            logger.info(f"Starting scrape for job {job_id}: {company_name}")
            
            # Update job status to running
            await self.sheets_dao.update_job(job_id, JobStatus.RUNNING)
            
            # Step 1: Normalize company name
            normalized = normalize_company_name(company_name)
            logger.info(f"Normalized company name: {normalized.normalized}")
            
            # Step 2: Discovery - find official domain
            domain = await self._discover_domain(normalized, official_email)
            if not domain:
                raise ValueError("Could not discover company domain")
            
            logger.info(f"Discovered domain: {domain}")
            
            # Step 3: Crawl and extract data
            extraction_results = await self._extract_company_data(domain, normalized, official_email)
            
            # Step 4: AI reconciliation
            if settings.use_ai and self.openai_client:
                final_profile = await self._ai_reconciliation(extraction_results, normalized, official_email)
            else:
                final_profile = await self._deterministic_reconciliation(extraction_results, normalized, official_email)
            
            # Step 5: Persist results
            await self._persist_results(job_id, final_profile, extraction_results)
            
            # Update job status to completed
            await self.sheets_dao.update_job(job_id, JobStatus.COMPLETED)
            
            logger.info(f"Completed scrape for job {job_id}")
            
        except Exception as e:
            logger.error(f"Scraping failed for job {job_id}: {e}", exc_info=True)
            await self.sheets_dao.update_job(job_id, JobStatus.FAILED, str(e))
            raise
    
    async def _discover_domain(self, normalized: NormalizedCompanyName, official_email: str) -> Optional[str]:
        """Discover the official company domain."""
        try:
            # Try email domain first
            email_domain = official_email.split('@')[1] if '@' in official_email else None
            if email_domain and not self._is_generic_email_domain(email_domain):
                if await self._validate_domain(email_domain):
                    return email_domain
            
            # Search for company domain
            search_results = await self.search_service.search_company(normalized)
            
            for result in search_results:
                domain = self._extract_domain_from_url(result['url'])
                if domain and await self._validate_domain(domain):
                    # Prefer exact matches
                    if self._domain_matches_company(domain, normalized):
                        return domain
            
            # Fallback to first valid domain
            for result in search_results:
                domain = self._extract_domain_from_url(result['url'])
                if domain and await self._validate_domain(domain):
                    return domain
            
            return None
            
        except Exception as e:
            logger.error(f"Domain discovery failed: {e}")
            return None
    
    def _is_generic_email_domain(self, domain: str) -> bool:
        """Check if email domain is generic (gmail, yahoo, etc.)."""
        generic_domains = {
            'gmail.com', 'yahoo.com', 'hotmail.com', 'outlook.com',
            'aol.com', 'icloud.com', 'mail.com', 'protonmail.com'
        }
        return domain.lower() in generic_domains
    
    def _extract_domain_from_url(self, url: str) -> Optional[str]:
        """Extract domain from URL."""
        try:
            parsed = urlparse(url)
            return parsed.netloc.lower().replace('www.', '')
        except:
            return None
    
    async def _validate_domain(self, domain: str) -> bool:
        """Validate that domain is accessible."""
        try:
            url = f"https://{domain}"
            response = await self.http_client.get(url, timeout=10)
            return response.status_code < 400
        except:
            return False
    
    def _domain_matches_company(self, domain: str, normalized: NormalizedCompanyName) -> bool:
        """Check if domain matches company name."""
        domain_parts = domain.replace('-', ' ').replace('.', ' ').split()
        company_words = normalized.normalized.lower().split()
        
        # Check for significant word overlap
        overlap = sum(1 for word in company_words if any(word in part for part in domain_parts))
        return overlap >= min(2, len(company_words) // 2)
    
    async def _extract_company_data(self, domain: str, normalized: NormalizedCompanyName, official_email: str) -> Dict[str, List[Tuple[str, float, str]]]:
        """Extract company data from various sources."""
        extraction_results = {}
        
        # URLs to crawl
        base_url = f"https://{domain}"
        urls_to_crawl = [
            base_url,
            f"{base_url}/about",
            f"{base_url}/about-us",
            f"{base_url}/company",
            f"{base_url}/contact",
            f"{base_url}/contact-us",
        ]
        
        # Crawl each URL
        for url in urls_to_crawl:
            try:
                data = await self._crawl_url(url)
                if data:
                    # Extract structured data
                    structured_data = self._extract_structured_data(data['html'], url)
                    for field, candidates in structured_data.items():
                        if field not in extraction_results:
                            extraction_results[field] = []
                        extraction_results[field].extend(candidates)
                    
                    # Extract from content
                    content_data = await self._extract_from_content(data['text'], url)
                    for field, candidates in content_data.items():
                        if field not in extraction_results:
                            extraction_results[field] = []
                        extraction_results[field].extend(candidates)
                        
            except Exception as e:
                logger.warning(f"Failed to crawl {url}: {e}")
                continue
        
        # Add known data
        extraction_results['company_name'] = [(normalized.original, 1.0, 'input')]
        extraction_results['official_email'] = [(official_email, 1.0, 'input')]
        extraction_results['website'] = [(base_url, 1.0, 'discovered')]
        
        # Deduplicate and score
        for field in extraction_results:
            extraction_results[field] = self._deduplicate_candidates(extraction_results[field])
        
        return extraction_results
    
    async def _crawl_url(self, url: str) -> Optional[Dict[str, str]]:
        """Crawl a URL and return HTML and text content."""
        try:
            # Try direct HTTP first
            response = await self.http_client.get(url)
            html = response.text
            
            # Check if page needs JavaScript
            if self._needs_javascript(html):
                logger.info(f"Using browser for JS-heavy page: {url}")
                html = await self.browser_service.get_html(url)
            
            # Extract clean text
            text = trafilatura.extract(html) or ""
            
            return {
                'html': html,
                'text': text,
                'url': url
            }
            
        except Exception as e:
            logger.warning(f"Failed to crawl {url}: {e}")
            return None
    
    def _needs_javascript(self, html: str) -> bool:
        """Heuristic to determine if page needs JavaScript."""
        if not html:
            return False
        
        # Check for common indicators
        indicators = [
            'React', 'Vue', 'Angular', 'loading...', 'Please enable JavaScript',
            'document.getElementById', 'window.onload'
        ]
        
        html_lower = html.lower()
        return any(indicator.lower() in html_lower for indicator in indicators)
    
    def _extract_structured_data(self, html: str, url: str) -> Dict[str, List[Tuple[str, float, str]]]:
        """Extract data from structured markup (JSON-LD, meta tags, etc.)."""
        results = {}
        
        try:
            soup = BeautifulSoup(html, 'lxml')
            
            # JSON-LD structured data
            json_ld_scripts = soup.find_all('script', type='application/ld+json')
            for script in json_ld_scripts:
                try:
                    data = json.loads(script.string)
                    structured_results = self._parse_json_ld(data, url)
                    for field, candidates in structured_results.items():
                        if field not in results:
                            results[field] = []
                        results[field].extend(candidates)
                except:
                    continue
            
            # Meta tags
            meta_results = self._extract_meta_tags(soup, url)
            for field, candidates in meta_results.items():
                if field not in results:
                    results[field] = []
                results[field].extend(candidates)
            
            # Social links
            social_results = self._extract_social_links(soup, url)
            for field, candidates in social_results.items():
                if field not in results:
                    results[field] = []
                results[field].extend(candidates)
            
        except Exception as e:
            logger.warning(f"Failed to extract structured data from {url}: {e}")
        
        return results
    
    def _parse_json_ld(self, data: Any, url: str) -> Dict[str, List[Tuple[str, float, str]]]:
        """Parse JSON-LD structured data."""
        results = {}
        
        def extract_from_object(obj):
            if isinstance(obj, dict):
                # Organization data
                if obj.get('@type') in ['Organization', 'Corporation', 'Company']:
                    if 'name' in obj:
                        self._add_candidate(results, 'company_name', obj['name'], 0.9, url)
                    if 'description' in obj:
                        self._add_candidate(results, 'description', obj['description'], 0.8, url)
                    if 'url' in obj:
                        self._add_candidate(results, 'website', obj['url'], 0.9, url)
                    if 'logo' in obj:
                        logo_url = obj['logo'] if isinstance(obj['logo'], str) else obj['logo'].get('url')
                        if logo_url:
                            self._add_candidate(results, 'logo_url', logo_url, 0.8, url)
                    
                    # Address
                    if 'address' in obj:
                        addr = obj['address']
                        if isinstance(addr, dict):
                            if 'streetAddress' in addr or 'addressLocality' in addr:
                                full_addr = ', '.join(filter(None, [
                                    addr.get('streetAddress'),
                                    addr.get('addressLocality'),
                                    addr.get('addressRegion'),
                                    addr.get('postalCode'),
                                    addr.get('addressCountry')
                                ]))
                                if full_addr:
                                    self._add_candidate(results, 'hq_address', full_addr, 0.8, url)
                            
                            if 'addressLocality' in addr:
                                self._add_candidate(results, 'city', addr['addressLocality'], 0.8, url)
                            if 'addressCountry' in addr:
                                self._add_candidate(results, 'country', addr['addressCountry'], 0.8, url)
                    
                    # Contact info
                    if 'telephone' in obj:
                        self._add_candidate(results, 'phone', obj['telephone'], 0.8, url)
                    if 'email' in obj:
                        self._add_candidate(results, 'official_email', obj['email'], 0.7, url)
                    
                    # Other fields
                    if 'foundingDate' in obj:
                        try:
                            year = int(obj['foundingDate'][:4])
                            self._add_candidate(results, 'year_founded', str(year), 0.8, url)
                        except:
                            pass
                    
                    if 'numberOfEmployees' in obj:
                        self._add_candidate(results, 'employee_count', str(obj['numberOfEmployees']), 0.7, url)
                
                # Recursively process nested objects
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        extract_from_object(value)
            
            elif isinstance(obj, list):
                for item in obj:
                    extract_from_object(item)
        
        extract_from_object(data)
        return results
    
    def _extract_meta_tags(self, soup: BeautifulSoup, url: str) -> Dict[str, List[Tuple[str, float, str]]]:
        """Extract data from meta tags."""
        results = {}
        
        # Open Graph tags
        og_tags = {
            'og:title': 'company_name',
            'og:description': 'description',
            'og:url': 'website',
            'og:image': 'logo_url',
            'og:site_name': 'company_name'
        }
        
        for og_prop, field in og_tags.items():
            meta = soup.find('meta', property=og_prop) or soup.find('meta', attrs={'name': og_prop})
            if meta and meta.get('content'):
                score = 0.7 if field == 'company_name' else 0.6
                self._add_candidate(results, field, meta['content'], score, url)
        
        # Standard meta tags
        meta_tags = {
            'description': 'description',
            'author': 'company_name'
        }
        
        for meta_name, field in meta_tags.items():
            meta = soup.find('meta', attrs={'name': meta_name})
            if meta and meta.get('content'):
                self._add_candidate(results, field, meta['content'], 0.5, url)
        
        return results
    
    def _extract_social_links(self, soup: BeautifulSoup, url: str) -> Dict[str, List[Tuple[str, float, str]]]:
        """Extract social media links."""
        results = {}
        
        social_patterns = {
            'linkedin': r'linkedin\.com/company/([^/?]+)',
            'twitter': r'twitter\.com/([^/?]+)',
            'facebook': r'facebook\.com/([^/?]+)',
            'instagram': r'instagram\.com/([^/?]+)',
            'youtube': r'youtube\.com/(channel|user|c)/([^/?]+)'
        }
        
        # Find all links
        links = soup.find_all('a', href=True)
        
        for link in links:
            href = link['href']
            for platform, pattern in social_patterns.items():
                if re.search(pattern, href, re.IGNORECASE):
                    field = f'socials.{platform}'
                    self._add_candidate(results, field, href, 0.8, url)
                    break
        
        return results
    
    async def _extract_from_content(self, text: str, url: str) -> Dict[str, List[Tuple[str, float, str]]]:
        """Extract data from page text content using patterns."""
        results = {}
        
        if not text:
            return results
        
        # Phone number patterns
        phone_patterns = [
            r'\+?[\d\s\-\(\)]{10,}',
            r'\(\d{3}\)\s*\d{3}-\d{4}',
            r'\d{3}-\d{3}-\d{4}'
        ]
        
        for pattern in phone_patterns:
            matches = re.findall(pattern, text)
            for match in matches:
                cleaned = re.sub(r'[^\d+]', '', match)
                if len(cleaned) >= 10:
                    self._add_candidate(results, 'phone', match.strip(), 0.6, url)
        
        # Year founded patterns
        year_patterns = [
            r'founded in (\d{4})',
            r'established (\d{4})',
            r'since (\d{4})',
            r'started in (\d{4})'
        ]
        
        for pattern in year_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for year in matches:
                if 1800 <= int(year) <= 2024:
                    self._add_candidate(results, 'year_founded', year, 0.7, url)
        
        # Employee count patterns
        employee_patterns = [
            r'(\d+[\-–]\d+)\s+employees',
            r'(\d+\+)\s+employees',
            r'over (\d+)\s+employees',
            r'more than (\d+)\s+employees'
        ]
        
        for pattern in employee_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                self._add_candidate(results, 'employee_count', match, 0.6, url)
        
        return results
    
    def _add_candidate(self, results: Dict, field: str, value: str, score: float, source: str):
        """Add a candidate value to results."""
        if field not in results:
            results[field] = []
        
        # Clean and validate value
        cleaned_value = value.strip()
        if cleaned_value and len(cleaned_value) < 1000:  # Reasonable length limit
            results[field].append((cleaned_value, score, source))
    
    def _deduplicate_candidates(self, candidates: List[Tuple[str, float, str]]) -> List[Tuple[str, float, str]]:
        """Deduplicate and rank candidates."""
        # Group by value (case-insensitive)
        value_groups = {}
        for value, score, source in candidates:
            key = value.lower().strip()
            if key not in value_groups:
                value_groups[key] = []
            value_groups[key].append((value, score, source))
        
        # Take best candidate from each group
        deduplicated = []
        for group in value_groups.values():
            # Sort by score, take highest
            best = max(group, key=lambda x: x[1])
            deduplicated.append(best)
        
        # Sort by score and take top 5
        deduplicated.sort(key=lambda x: x[1], reverse=True)
        return deduplicated[:5]
    
    async def _ai_reconciliation(self, extraction_results: Dict, normalized: NormalizedCompanyName, official_email: str) -> CompanyProfile:
        """Use AI to reconcile and select best values."""
        try:
            # Prepare candidates for AI
            candidates_for_ai = {}
            for field, candidates in extraction_results.items():
                if candidates and field not in ['company_name', 'official_email']:  # Skip fixed fields
                    candidates_for_ai[field] = [{"value": v, "score": s, "source": src} for v, s, src in candidates[:3]]
            
            # Create AI prompt
            prompt = self._create_reconciliation_prompt(normalized, official_email, candidates_for_ai)
            
            # Call OpenAI
            response = await self.openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert at analyzing company data and selecting the most accurate information."},
                    {"role": "user", "content": prompt}
                ],
                functions=[{
                    "name": "select_company_data",
                    "description": "Select the best values for company profile fields",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "website": {"type": "string"},
                            "hq_address": {"type": "string"},
                            "phone": {"type": "string"},
                            "industry": {"type": "string"},
                            "description": {"type": "string"},
                            "year_founded": {"type": "integer"},
                            "employee_count": {"type": "string"},
                            "logo_url": {"type": "string"},
                            "country": {"type": "string"},
                            "city": {"type": "string"},
                            "socials": {
                                "type": "object",
                                "properties": {
                                    "linkedin": {"type": "string"},
                                    "twitter": {"type": "string"},
                                    "facebook": {"type": "string"},
                                    "instagram": {"type": "string"},
                                    "youtube": {"type": "string"}
                                }
                            }
                        }
                    }
                }],
                function_call={"name": "select_company_data"}
            )
            
            # Parse AI response
            function_call = response.choices[0].message.function_call
            selected_data = json.loads(function_call.arguments)
            
            # Build profile
            profile_data = {
                'company_name': normalized.original,
                'official_email': official_email,
                **selected_data
            }
            
            # Add confidence scores
            confidence_per_field = {}
            for field, value in selected_data.items():
                if value and field in extraction_results:
                    # Find matching candidate
                    for candidate_value, score, _ in extraction_results[field]:
                        if str(value).lower() == candidate_value.lower():
                            confidence_per_field[field] = score
                            break
                    else:
                        confidence_per_field[field] = 0.8  # AI selected, moderate confidence
            
            profile_data['confidence_per_field'] = confidence_per_field
            
            return CompanyProfile(**profile_data)
            
        except Exception as e:
            logger.warning(f"AI reconciliation failed: {e}, falling back to deterministic")
            return await self._deterministic_reconciliation(extraction_results, normalized, official_email)
    
    def _create_reconciliation_prompt(self, normalized: NormalizedCompanyName, official_email: str, candidates: Dict) -> str:
        """Create prompt for AI reconciliation."""
        prompt = f"""
Company: {normalized.original}
Email: {official_email}

I have extracted the following candidate values for company profile fields. Please select the most accurate value for each field, or leave blank if none are suitable.

Candidates:
"""
        
        for field, field_candidates in candidates.items():
            prompt += f"\n{field}:\n"
            for i, candidate in enumerate(field_candidates, 1):
                prompt += f"  {i}. {candidate['value']} (score: {candidate['score']}, source: {candidate['source']})\n"
        
        prompt += """
Please select the best value for each field. Consider:
1. Accuracy and relevance to the company
2. Source reliability
3. Data freshness
4. Consistency across fields

Use the select_company_data function to return your selections.
"""
        
        return prompt
    
    async def _deterministic_reconciliation(self, extraction_results: Dict, normalized: NormalizedCompanyName, official_email: str) -> CompanyProfile:
        """Deterministic reconciliation - pick highest scored candidates."""
        profile_data = {
            'company_name': normalized.original,
            'official_email': official_email
        }
        
        confidence_per_field = {}
        
        # For each field, take the highest scored candidate
        for field, candidates in extraction_results.items():
            if candidates and field not in ['company_name', 'official_email']:
                best_candidate = max(candidates, key=lambda x: x[1])
                value, score, _ = best_candidate
                
                # Clean field name (remove socials. prefix)
                clean_field = field.replace('socials.', '')
                
                if field.startswith('socials.'):
                    if 'socials' not in profile_data:
                        profile_data['socials'] = {}
                    profile_data['socials'][clean_field] = value
                else:
                    # Convert year_founded to int
                    if field == 'year_founded' and value.isdigit():
                        value = int(value)
                    
                    profile_data[field] = value
                
                confidence_per_field[field] = score
        
        profile_data['confidence_per_field'] = confidence_per_field
        
        return CompanyProfile(**profile_data)
    
    async def _persist_results(self, job_id: str, profile: CompanyProfile, extraction_results: Dict):
        """Persist scraping results to Sheets."""
        try:
            # Write profile
            profile_dict = profile.dict()
            confidence_overall = sum(profile.confidence_per_field.values()) / len(profile.confidence_per_field) if profile.confidence_per_field else 0.0
            
            await self.sheets_dao.write_profile(job_id, profile_dict, confidence_overall)
            
            # Write candidates
            candidates_by_field = {}
            for field, candidates in extraction_results.items():
                candidates_by_field[field] = [(value, score) for value, score, _ in candidates]
            
            await self.sheets_dao.write_candidates(job_id, candidates_by_field)
            
            # Write sources
            sources_by_field = {}
            for field, candidates in extraction_results.items():
                sources_by_field[field] = [
                    {"url": source, "snippet": value[:200], "score": score}
                    for value, score, source in candidates
                ]
            
            await self.sheets_dao.write_sources(job_id, sources_by_field)
            
            logger.info(f"Persisted results for job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to persist results for job {job_id}: {e}")
            raise
    
    async def __aenter__(self):
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.http_client.aclose()




