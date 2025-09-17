"""Main FastAPI application."""

import logging
import uuid
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
# Rate limiting imports (install slowapi if needed)
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
    HAS_RATE_LIMITING = True
except ImportError:
    HAS_RATE_LIMITING = False
    Limiter = None

from app.config import settings
from app.models import (
    JobRequest, 
    JobResponse, 
    JobDetailResponse, 
    ProfileOverrides, 
    ExportResponse,
    JobStatus
)
# from app.services.sheets_dao import SheetsDAO
# from app.services.scraper import CompanyScraper
from app.utils.logging import setup_logging, get_logger
try:
    import httpx
    import asyncio
    from bs4 import BeautifulSoup
    import re
    import json
    import openai
    DEPENDENCIES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Some dependencies not available: {e}")
    DEPENDENCIES_AVAILABLE = False
    # Create mock for deployment
    openai = None


# Setup logging
setup_logging()
logger = get_logger(__name__)

# Rate limiter
limiter = Limiter(key_func=get_remote_address) if HAS_RATE_LIMITING else None

# Global instances (commented out for now)
# sheets_dao: SheetsDAO = None
# scraper: CompanyScraper = None

# Simple in-memory storage for demo
scraped_jobs = {}

# Initialize OpenAI client
if DEPENDENCIES_AVAILABLE and settings.openai_api_key:
    openai.api_key = settings.openai_api_key
    openai_client = openai
else:
    openai_client = None

async def call_openai_chat(messages, model="gpt-3.5-turbo", temperature=0.3, max_tokens=500):
    """Wrapper for OpenAI chat completions that works with different versions."""
    try:
        if not openai_client:
            return {"choices": [{"message": {"content": "OpenAI not available"}}]}
        
        # Use older OpenAI API syntax
        response = await openai.ChatCompletion.acreate(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response
    except Exception as e:
        logger.error(f"OpenAI API call failed: {e}")
        return {"choices": [{"message": {"content": "AI analysis unavailable"}}]}

async def scrape_company_simple(job_id: str, company_name: str, official_email: str, domain: str = None, competitor_domains: list = None, main_locations: list = None):
    """AI-Guided intelligent scraping with reward/punishment learning."""
    try:
        logger.info(f"🤖 Starting AI-GUIDED scraping for {company_name}")
        
        # Update job status to running
        scraped_jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.RUNNING,
            "company_name": company_name,
            "official_email": official_email,
            "profile": None,
            "error": None
        }
        
        # Define the questions we need to answer
        target_questions = [
            "Who is your target market and niche?",
            "Why did you start your business? What was your original mission?",
            "What is the company's industry and main services?",
            "Where is the company located (country/city)?",
            "What are the company's core values and founding story?",
            "How do you currently turn strangers into customers?",
            "What has worked best to grow your business so far?",
            "What hasn't worked or felt like a waste of time?",
            "How do you currently use Search Engine Optimization or Search Advertising?",
            "What are your main business goals for the next 12 months?",
            "What specific SEO, Ads, or visibility goals do you have?",
            "What do you believe is blocking you from reaching your goals right now?",
            "What are your top 3 priority services to promote first?",
            "What areas or regions do you serve? What are your top 3 priority locations?"
        ]
        
        # Step 1: Direct website scraping (proven approach for basic data)
        logger.info("🌐 Phase 1: Direct website scraping for basic data...")
        
        # Use provided domain or extract from email
        if domain:
            website_domain = domain.strip()
        else:
            website_domain = official_email.split('@')[1] if '@' in official_email else None
        
        # Ensure proper URL format without duplication
        if website_domain:
            if website_domain.startswith('http://') or website_domain.startswith('https://'):
                website = website_domain
            else:
                website = f"https://{website_domain}"
        else:
            website = None
        logger.info(f"🌐 Using website: {website}")
        
        # Log additional provided information
        if competitor_domains:
            logger.info(f"🏆 Provided competitor domains: {competitor_domains}")
        if main_locations:
            logger.info(f"📍 Provided main locations: {main_locations}")
        
        # Store all scraped content for competitor analysis
        all_scraped_content = ""
        
        # Initialize with basic data
        profile_data = {
            "company_name": company_name,
            "official_email": official_email,
            "website": website,
        }
        
        # Direct scraping for reliable data (social media, basic info)
        basic_candidates = {}
        if website and not any(generic in domain.lower() for generic in ['gmail', 'yahoo', 'hotmail', 'outlook']):
            basic_candidates = await scrape_website_comprehensive(website, company_name)
        
        # Step 2: AI creates strategy for missing complex data
        logger.info("🧠 Phase 2: AI creating strategy for complex business intelligence...")
        strategy = await create_ai_scraping_strategy(company_name, official_email, target_questions)
        
        # Step 3: Execute AI-guided scraping for complex questions
        extracted_data = basic_candidates.copy()
        reward_matrix = {}
        
        for iteration in range(strategy.get('max_iterations', 5)):
            logger.info(f"🔄 AI-guided iteration {iteration + 1}")
            
            # AI decides what to scrape next
            decision = await ai_decide_next_action(company_name, target_questions, extracted_data, reward_matrix)
            
            # Validate decision has required fields
            if not decision.get('action'):
                logger.warning("AI decision missing 'action' field, defaulting to complete")
                decision['action'] = 'complete'
            
            if decision['action'] == 'complete':
                logger.info("🎯 AI decided scraping is complete")
                break
            
            # Execute the AI's decision
            new_data = await execute_scraping_decision(decision)
            
            # AI evaluates the quality of scraped data
            evaluation = await ai_evaluate_scraped_data(target_questions, new_data)
            
            # Update reward matrix for learning
            update_reward_matrix(reward_matrix, decision, evaluation)
            
            # Merge good data
            merge_evaluated_data(extracted_data, new_data, evaluation)
            
            logger.info(f"📊 Iteration {iteration + 1} reward score: {evaluation.get('overall_reward', 0)}")
        
        # Step 3: Merge basic data with AI analysis
        logger.info("🔄 Merging direct scraping with AI analysis...")
        final_profile = merge_basic_and_ai_data(profile_data, basic_candidates, extracted_data)
        
        # Step 4: AI final synthesis and validation
        logger.info("🤖 AI performing final data synthesis...")
        ai_enhanced_profile = await ai_synthesize_final_profile(company_name, official_email, final_profile, target_questions)
        
        # Ensure we don't lose the basic data
        final_profile.update(ai_enhanced_profile)
        
        # Only extract location from address if country/city are completely missing
        if not final_profile.get('country') and not final_profile.get('city'):
            final_profile = extract_location_from_address_fallback(final_profile)
        
        # Step 5: Customer acquisition and growth strategy analysis
        logger.info("📈 Analyzing customer acquisition and growth strategies...")
        growth_strategy_data = await analyze_growth_strategies(website, final_profile, company_name)
        final_profile.update(growth_strategy_data)
        
        # Step 5.5: Future goals and challenges analysis
        logger.info("🎯 Analyzing future goals and challenges...")
        goals_challenges_data = await analyze_goals_and_challenges(website, final_profile, company_name)
        final_profile.update(goals_challenges_data)
        
        # Step 5.7: Service and location priorities analysis
        logger.info("🎯 Analyzing service and location priorities...")
        service_location_data = await analyze_service_and_location_priorities(website, final_profile, company_name, main_locations)
        final_profile.update(service_location_data)
        
        # Step 6: Competitor analysis
        logger.info("🏆 Starting competitor analysis...")
        # Collect content from basic scraping
        if website:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    response = await client.get(website)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        all_scraped_content = soup.get_text()[:5000]  # First 5000 chars
            except:
                all_scraped_content = ""
        
        # Step 6: Get real keywords from DataForSEO using company domain
        logger.info("🔍 Getting real keywords from DataForSEO using company domain...")
        domain_keywords = await get_domain_keywords_dataforseo(website, final_profile.get('city', ''), final_profile.get('country', 'United States'))
        if domain_keywords:
            final_profile['main_keywords'] = domain_keywords[:6]  # Replace AI keywords with real ones
            logger.info(f"✅ Updated main keywords with DataForSEO data: {domain_keywords[:6]}")
        
        competitor_data = await analyze_competitors(website, final_profile, all_scraped_content, competitor_domains)
        final_profile.update(competitor_data)
        
        # Step 7: Topic Authority Map and Content Planning
        logger.info("📝 Creating Topic Authority Map for content planning...")
        
        # Validate and select the best pillar keyword from keywords (starting with top 3)
        main_keywords = final_profile.get('main_keywords', [])
        if main_keywords:
            pillar_keyword = await select_best_pillar_keyword(
                main_keywords,  # Pass all keywords, function will iterate through them
                company_name, 
                final_profile.get('industry', ''),
                final_profile.get('description', ''),
                final_profile.get('services_offered', ''),
                website  # Pass website for fresh scraping
            )
            
            if pillar_keyword:
                logger.info(f"🎯 Selected validated pillar keyword: {pillar_keyword}")
                content_plan_data = await create_focused_topic_authority_map(
                    pillar_keyword,
                    company_name, 
                    final_profile.get('city', ''), 
                    final_profile.get('country', 'United States'),
                    final_profile.get('industry', ''),
                    final_profile.get('description', ''),
                    main_locations or []
                )
                final_profile.update(content_plan_data)
            else:
                logger.warning("No suitable pillar keyword found in main keywords")
                final_profile.update({
                    'topic_authority_map': None,
                    'content_plan_summary': 'No suitable pillar keyword found for content planning'
                })
        else:
            logger.warning("No main keywords available for content planning")
            final_profile.update({
                'topic_authority_map': None,
                'content_plan_summary': 'Content planning requires keyword analysis to be completed first'
            })
        
        # Update job with results
        scraped_jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.COMPLETED,
            "company_name": company_name,
            "official_email": official_email,
            "profile": final_profile,
            "error": None,
            "strategy": strategy,
            "reward_matrix": reward_matrix,
            "iterations": iteration + 1
        }
        
        logger.info(f"🎉 AI-guided scraping completed for {company_name} in {iteration + 1} iterations")
        logger.info(f"📋 Final profile: {final_profile}")
        
    except Exception as e:
        logger.error(f"💥 AI-guided scraping failed for {company_name}: {e}")
        scraped_jobs[job_id] = {
            "id": job_id,
            "status": JobStatus.FAILED,
            "company_name": company_name,
            "official_email": official_email,
            "profile": None,
            "error": str(e)
        }


async def create_ai_scraping_strategy(company_name: str, official_email: str, questions: list) -> dict:
    """AI creates an intelligent scraping strategy."""
    try:
        domain = official_email.split('@')[1] if '@' in official_email else None
        
        prompt = f"""
Company: {company_name}
Email: {official_email}
Domain: {domain}

I need to create an intelligent scraping strategy to answer these specific questions:
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(questions))}

PRIORITY FOCUS: About pages are the MOST valuable for mission, founding story, and company values.

Create a strategic plan considering:
1. PRIORITIZE About/Mission pages (highest value for business questions)
2. What specific search queries would find missing mission/founding data?
3. Which sources contain founding stories and business motivation?
4. How to evaluate if data answers "Why did you start?" and "What's your mission?"

Focus on About pages and mission-related content for the best results.
"""
        
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert web scraping strategist who creates intelligent plans for extracting specific business information."},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "create_scraping_strategy",
                "description": "Create an intelligent scraping strategy",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "priority_urls": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "URLs to scrape in order of priority"
                        },
                        "search_queries": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "Google search queries for missing data"
                        },
                        "field_mapping": {
                            "type": "object",
                            "description": "Which fields each URL/search is likely to contain"
                        },
                        "success_criteria": {
                            "type": "object",
                            "description": "How to evaluate if each question is answered"
                        }
                    }
                }
            }],
            function_call={"name": "create_scraping_strategy"}
        )
        
        function_call = response.choices[0].message.function_call
        strategy = json.loads(function_call.arguments)
        
        logger.info(f"🧠 AI created strategy: {strategy}")
        return strategy
        
    except Exception as e:
        logger.error(f"Failed to create AI strategy: {e}")
        return {"priority_urls": [f"https://{domain}"], "search_queries": [], "field_mapping": {}}


async def ai_decide_next_action(company_name: str, questions: list, current_data: dict, reward_matrix: dict) -> dict:
    """AI decides what to scrape next based on current data and learning."""
    try:
        prompt = f"""
Company: {company_name}
Questions to Answer: {questions}

Current Extracted Data:
{json.dumps(current_data, indent=2)}

Reward Matrix (Learning Data):
{json.dumps(reward_matrix, indent=2)}

Based on the current data and learning from previous attempts, what should I do next to answer the remaining questions?

Analyze:
1. Which questions are still unanswered?
2. What data is still missing?
3. What's the best next action to take?
4. Based on the reward matrix, what strategies have worked well?

Choose the most strategic next action.
"""
        
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an intelligent scraping agent that makes strategic decisions about what to scrape next. Learn from previous successes and failures."},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "decide_next_action",
                "description": "Decide the next scraping action",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "action": {
                            "type": "string",
                            "enum": ["scrape_url", "search_google", "complete"],
                            "description": "Next action to take"
                        },
                        "target": {"type": "string", "description": "URL to scrape or search query"},
                        "reason": {"type": "string", "description": "Why this action was chosen"},
                        "expected_fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Fields this action might help find"
                        },
                        "confidence": {"type": "number", "description": "Confidence in this decision (0-1)"}
                    }
                }
            }],
            function_call={"name": "decide_next_action"}
        )
        
        function_call = response.choices[0].message.function_call
        decision = json.loads(function_call.arguments)
        
        # Ensure required fields exist
        if 'action' not in decision:
            decision['action'] = 'complete'
        if 'target' not in decision:
            decision['target'] = ''
        if 'reason' not in decision:
            decision['reason'] = 'AI decision'
        if 'expected_fields' not in decision:
            decision['expected_fields'] = []
        if 'confidence' not in decision:
            decision['confidence'] = 0.5
        
        logger.info(f"🎯 AI decision: {decision}")
        return decision
        
    except Exception as e:
        logger.error(f"AI decision failed: {e}")
        return {"action": "complete", "target": "", "reason": "Error in AI decision", "expected_fields": [], "confidence": 0}


async def execute_scraping_decision(decision: dict) -> dict:
    """Execute the AI's scraping decision."""
    try:
        if decision['action'] == 'scrape_url':
            return await scrape_url_with_ai_focus(decision['target'], decision['expected_fields'])
        elif decision['action'] == 'search_google':
            return await search_google_with_ai_focus(decision['target'], decision['expected_fields'])
        else:
            return {}
    except Exception as e:
        logger.error(f"Failed to execute decision: {e}")
        return {}


async def scrape_url_with_ai_focus(url: str, expected_fields: list) -> dict:
    """Scrape a URL with focus on specific fields."""
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with httpx.AsyncClient(timeout=15, headers=headers) as client:
            response = await client.get(url)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                candidates = {}
                
                # Extract all data types
                await extract_json_ld_data(soup, candidates, url)
                await extract_meta_data(soup, candidates, url)
                await extract_content_data(soup, candidates, url)
                await extract_social_links(soup, candidates, url)
                await extract_market_and_niche(soup, candidates, url)
                await extract_client_information(soup, candidates, url)
                await extract_mission_and_story(soup, candidates, url)
                await extract_location_data(soup, candidates, url)
                
                logger.info(f"🌐 Scraped {url} - found fields: {list(candidates.keys())}")
                return candidates
            else:
                logger.warning(f"Failed to scrape {url}: HTTP {response.status_code}")
                return {}
                
    except Exception as e:
        logger.error(f"Error scraping {url}: {e}")
        return {}


async def search_google_with_ai_focus(query: str, expected_fields: list) -> dict:
    """Search Google with AI focus on specific fields."""
    try:
        search_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
        
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with httpx.AsyncClient(timeout=15, headers=headers) as client:
            response = await client.get(search_url, headers=headers)
            
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # Extract search results
                results = []
                result_divs = soup.find_all('div', class_=re.compile(r'result', re.I))
                
                for div in result_divs[:3]:  # Top 3 results
                    snippet = div.get_text().strip()
                    if snippet and len(snippet) > 30:
                        results.append(snippet[:500])
                
                if results:
                    candidates = {}
                    for field in expected_fields:
                        candidates[field] = [{"value": result, "score": 0.6, "source": f"Google: {query}"} for result in results]
                    
                    logger.info(f"🔍 Google search found {len(results)} results for: {query}")
                    return candidates
                    
        return {}
        
    except Exception as e:
        logger.error(f"Google search failed: {e}")
        return {}


async def ai_evaluate_scraped_data(questions: list, scraped_data: dict) -> dict:
    """AI evaluates if scraped data satisfies the target questions."""
    try:
        prompt = f"""
Target Questions to Answer:
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(questions))}

Newly Scraped Data:
{json.dumps(scraped_data, indent=2)}

Evaluate this scraped data:
1. Does it help answer any of the target questions?
2. How confident are you in the quality of each data point?
3. Which fields satisfy the questions well vs poorly?
4. What reward score (0-1) should each data extraction get?

Rate each field's usefulness for answering the questions.
"""
        
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert data evaluator who judges if scraped information satisfies business intelligence questions."},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "evaluate_data_quality",
                "description": "Evaluate scraped data quality and relevance",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "field_evaluations": {
                            "type": "object",
                            "description": "Evaluation of each field"
                        },
                        "overall_reward": {"type": "number", "description": "Overall reward score 0-1"},
                        "questions_answered": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Which questions were answered"
                        },
                        "missing_info": {
                            "type": "array", 
                            "items": {"type": "string"},
                            "description": "What information is still needed"
                        }
                    }
                }
            }],
            function_call={"name": "evaluate_data_quality"}
        )
        
        function_call = response.choices[0].message.function_call
        evaluation = json.loads(function_call.arguments)
        
        logger.info(f"📊 AI evaluation: reward={evaluation.get('overall_reward', 0)}")
        return evaluation
        
    except Exception as e:
        logger.error(f"AI evaluation failed: {e}")
        return {"overall_reward": 0, "field_evaluations": {}}


def update_reward_matrix(reward_matrix: dict, decision: dict, evaluation: dict):
    """Update the reward matrix for AI learning."""
    action_key = f"{decision['action']}_{decision.get('target', '')}"
    
    if action_key not in reward_matrix:
        reward_matrix[action_key] = {"attempts": 0, "total_reward": 0, "success_rate": 0}
    
    reward_matrix[action_key]["attempts"] += 1
    reward_matrix[action_key]["total_reward"] += evaluation.get('overall_reward', 0)
    reward_matrix[action_key]["success_rate"] = reward_matrix[action_key]["total_reward"] / reward_matrix[action_key]["attempts"]


def merge_evaluated_data(extracted_data: dict, new_data: dict, evaluation: dict):
    """Merge new data based on AI evaluation."""
    field_evals = evaluation.get('field_evaluations', {})
    
    for field, candidates in new_data.items():
        field_eval = field_evals.get(field, {})
        reward_score = field_eval.get('reward_score', 0.5)
        
        # Only merge data with good reward scores
        if reward_score > 0.6:
            if field not in extracted_data:
                extracted_data[field] = []
            extracted_data[field].extend(candidates)


async def ai_synthesize_final_profile(company_name: str, official_email: str, extracted_data: dict, questions: list) -> dict:
    """AI creates the final profile by synthesizing all extracted data."""
    try:
        prompt = f"""
Company: {company_name}
Email: {official_email}

Questions to Answer:
{chr(10).join(f'{i+1}. {q}' for i, q in enumerate(questions))}

All Extracted Data:
{json.dumps(extracted_data, indent=2)}

Create the final company profile by:
1. Selecting the best data for each field
2. Ensuring all target questions are answered as completely as possible
3. Synthesizing information from multiple sources
4. Providing comprehensive answers to the business questions

Focus especially on answering:
- Who is your target market and niche?
- Why did you start your business? What was your original mission?
"""
        
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert business analyst who creates comprehensive company profiles from extracted data."},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "create_final_profile",
                "description": "Create the final company profile",
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
                        "target_market": {"type": "string"},
                        "niche": {"type": "string"},
                        "services_offered": {"type": "string"},
                        "client_types": {"type": "string"},
                        "mission_statement": {"type": "string"},
                        "founding_story": {"type": "string"},
                        "why_started": {"type": "string"},
                        "company_values": {"type": "string"},
                        "socials": {"type": "object"}
                    }
                }
            }],
            function_call={"name": "create_final_profile"}
        )
        
        function_call = response.choices[0].message.function_call
        final_profile = json.loads(function_call.arguments)
        
        # Add base data
        final_profile["company_name"] = company_name
        final_profile["official_email"] = official_email
        
        # Fix URLs
        final_profile = fix_all_urls(final_profile)
        
        logger.info(f"🎯 AI synthesized final profile")
        return final_profile
        
    except Exception as e:
        logger.error(f"AI synthesis failed: {e}")
        return {"company_name": company_name, "official_email": official_email}


def merge_basic_and_ai_data(base_profile: dict, basic_candidates: dict, ai_data: dict) -> dict:
    """Merge basic scraping data with AI analysis, preserving social media and contact info."""
    merged_profile = base_profile.copy()
    
    # First, add all basic scraped data (social media, contact info, etc.)
    basic_profile = select_best_candidates(basic_candidates)
    logger.info(f"🔄 Basic profile data: {list(basic_profile.keys())}")
    
    # Preserve social media specifically
    if basic_profile.get('socials'):
        logger.info(f"📱 Preserving social media from basic scraping: {basic_profile['socials']}")
        merged_profile['socials'] = basic_profile['socials']
    
    # Add other basic data
    for key, value in basic_profile.items():
        if key != 'socials':  # Handle socials separately
            merged_profile[key] = value
    
    # Then add AI analysis data (but don't override social media)
    ai_profile = select_best_candidates(ai_data)
    logger.info(f"🤖 AI profile data: {list(ai_profile.keys())}")
    
    for key, value in ai_profile.items():
        if key != 'socials':  # Don't override social media from basic scraping
            merged_profile[key] = value
        elif key == 'socials' and not merged_profile.get('socials'):
            # Only add AI social media if we don't have basic social media
            merged_profile[key] = value
    
    logger.info(f"🔄 Final merged profile keys: {list(merged_profile.keys())}")
    
    # Ensure social media is properly formatted
    if merged_profile.get('socials'):
        logger.info(f"📱 Final social media in merged profile: {merged_profile['socials']}")
    else:
        logger.warning("📱 No social media in final merged profile")
    
    return merged_profile


def extract_location_from_address_fallback(profile: dict) -> dict:
    """Only extract location from address as fallback when no other location data exists."""
    address = profile.get('hq_address', '')
    
    if address:
        logger.info(f"📍 Fallback: Extracting location from address: {address}")
        
        # Only basic extraction as fallback
        if 'San Francisco' in address:
            profile['city'] = 'San Francisco'
            profile['country'] = 'United States'
        elif 'New York' in address:
            profile['city'] = 'New York'
            profile['country'] = 'United States'
        elif any(keyword in address for keyword in ['USA', 'United States']):
            profile['country'] = 'United States'
    
    return profile


async def analyze_competitors(website: str, company_profile: dict, scraped_content: str, provided_competitor_domains: list = None) -> dict:
    """Analyze competitors using ChatGPT to extract keywords and search Google SERP."""
    try:
        logger.info("🏆 Starting competitor analysis...")
        
        # Step 1: ChatGPT extracts main keywords from all scraped content
        keyword_prompt = f"""
        Analyze this company's website content and business profile to identify the main keywords that represent their business:

        Company: {company_profile.get('company_name', 'Unknown')}
        Industry: {company_profile.get('industry', 'Unknown')}
        Description: {company_profile.get('description', 'Unknown')}
        Services: {company_profile.get('services_offered', 'Unknown')}
        Target Market: {company_profile.get('target_market', 'Unknown')}

        Website Content Sample: {scraped_content[:2000]}...

        Extract 5-8 main keywords that competitors would also rank for. Focus on:
        1. Primary service/product keywords
        2. Industry-specific terms
        3. Solution-oriented keywords
        4. Target market descriptors

        Return ONLY a JSON list of keywords like: ["keyword1", "keyword2", "keyword3"]
        """
        
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": keyword_prompt}],
            temperature=0.3,
            max_tokens=200
        )
        
        keywords_text = response.choices[0].message.content.strip()
        logger.info(f"🔍 AI extracted keywords: {keywords_text}")
        
        # Parse keywords
        try:
            import json
            keywords = json.loads(keywords_text)
            if not isinstance(keywords, list):
                keywords = keywords_text.replace('[', '').replace(']', '').replace('"', '').split(',')
                keywords = [k.strip() for k in keywords]
        except:
            keywords = keywords_text.replace('[', '').replace(']', '').replace('"', '').split(',')
            keywords = [k.strip() for k in keywords[:6]]  # Limit to 6 keywords
        
        logger.info(f"🎯 Final keywords: {keywords}")
        
        # Step 2: Start with provided competitor domains if available
        competitor_data = {}  # Dictionary to store company -> url mapping
        
        # Add provided competitors first (highest priority)
        if provided_competitor_domains:
            logger.info(f"🎯 Adding provided competitor domains: {provided_competitor_domains}")
            for domain in provided_competitor_domains:
                if domain and len(domain.strip()) > 0:
                    domain_clean = domain.strip()
                    
                    # Ensure proper URL format without duplication
                    if domain_clean.startswith('http://') or domain_clean.startswith('https://'):
                        final_url = domain_clean
                        domain_for_name = domain_clean.replace('https://', '').replace('http://', '').replace('www.', '')
                    else:
                        final_url = f"https://{domain_clean}"
                        domain_for_name = domain_clean.replace('www.', '')
                    
                    # Extract company name from domain
                    company_name = domain_for_name.split('.')[0].replace('-', ' ').replace('_', ' ').title()
                    
                    competitor_data[company_name] = {
                        'url': final_url,
                        'title': f"{company_name} - Provided Competitor",
                        'description': f"User-provided competitor domain: {domain}",
                        'keyword': 'user_provided',
                        'priority': 1.0  # Highest priority
                    }
                    logger.info(f"✅ Added provided competitor: {company_name} -> {final_url}")
        
        # Step 3: Search Google SERP for additional competitors
        
        for keyword in keywords[:5]:  # Limit to top 5 keywords
            logger.info(f"🔍 Searching Google for: {keyword}")
            
            try:
                # Use DataForSEO API for Google search results
                search_results = await search_google_serp_dataforseo(keyword, company_profile.get('industry', ''))
                
                for result in search_results:
                    url = result.get('url', '')
                    title = result.get('title', '')
                    description = result.get('description', '')
                    
                    # Skip the company's own website
                    company_domain = website.replace('https://', '').replace('http://', '').replace('www.', '') if website else ''
                    if company_domain and company_domain in url:
                        continue
                    
                    # Extract and validate competitor
                    competitor_info = extract_competitor_from_result(title, url, description)
                    if competitor_info:
                        company_name = competitor_info['name']
                        company_url = competitor_info['url']
                        
                        # Store the mapping
                        if company_name not in competitor_data:
                            competitor_data[company_name] = {
                                'url': company_url,
                                'title': title,
                                'description': description,
                                'keyword': keyword
                            }
                            logger.info(f"🏆 Found competitor: {company_name} -> {company_url}")
                
            except Exception as e:
                logger.error(f"Error searching for keyword {keyword}: {e}")
                continue
        
        # Step 3: Use ChatGPT to validate competitors and ensure proper matching
        if competitor_data:
            # Create competitor list for validation
            competitor_list = []
            for name, data in competitor_data.items():
                competitor_list.append(f"{name} ({data['url']})")
            
            competitor_validation_prompt = f"""
            I found these potential competitors for {company_profile.get('company_name')}:
            
            Company Industry: {company_profile.get('industry', 'Unknown')}
            Company Services: {company_profile.get('services_offered', 'Unknown')}
            
            Potential Competitors (name and URL):
            {chr(10).join(competitor_list[:10])}
            
            Filter this list to keep only the most relevant direct competitors. Remove:
            - News/media websites (Forbes, TechCrunch, etc.)
            - Social media platforms (Reddit, LinkedIn, etc.)
            - Generic directories or review sites
            - Unrelated businesses
            
            Return ONLY the company names (not URLs) of the top 5 most relevant competitors as a JSON list: ["Company1", "Company2", ...]
            """
            
            validation_response = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": competitor_validation_prompt}],
                temperature=0.3,
                max_tokens=200
            )
            
            validated_competitors_text = validation_response.choices[0].message.content.strip()
            
            try:
                validated_competitors = json.loads(validated_competitors_text)
                if not isinstance(validated_competitors, list):
                    validated_competitors = list(competitor_data.keys())[:5]
            except:
                validated_competitors = list(competitor_data.keys())[:5]
            
            # Get matching URLs for validated competitors
            matched_urls = []
            final_competitors = []
            
            for competitor in validated_competitors[:5]:
                if competitor in competitor_data:
                    final_competitors.append(competitor)
                    matched_urls.append(competitor_data[competitor]['url'])
                    logger.info(f"✅ Validated competitor: {competitor} -> {competitor_data[competitor]['url']}")
            
            logger.info(f"🏆 Final validated competitors: {final_competitors}")
            
            return {
                'main_keywords': keywords[:6],
                'competitors': final_competitors,
                'competitor_urls': matched_urls
            }
        
        return {
            'main_keywords': keywords[:6],
            'competitors': [],
            'competitor_urls': []
        }
        
    except Exception as e:
        logger.error(f"Competitor analysis failed: {e}")
        return {
            'main_keywords': [],
            'competitors': [],
            'competitor_urls': []
        }


async def search_google_serp_dataforseo(keyword: str, industry: str) -> list:
    """Search Google SERP using DataForSEO API for competitor analysis."""
    try:
        from app.config import settings
        import base64
        
        # Construct search query
        query = f"{keyword} {industry} companies"
        
        # DataForSEO API endpoint for Google organic search
        api_url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
        
        # Prepare request body
        request_body = [{
            "keyword": query,
            "location_code": 2840,  # United States
            "language_code": "en",
            "device": "desktop",
            "os": "windows",
            "depth": 10  # Get top 10 results
        }]
        
        # Prepare headers with authentication
        headers = {
            "Authorization": f"Basic {settings.dataforseo_base64}",
            "Content-Type": "application/json"
        }
        
        logger.info(f"🔍 DataForSEO API searching for: {query}")
        
        # Make API request
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                api_url,
                json=request_body,
                headers=headers
            )
            
            if response.status_code == 200:
                data = response.json()
                results = []
                
                if data.get('status_code') == 20000 and data.get('tasks'):
                    task_results = data['tasks'][0].get('result', [])
                    
                    for result in task_results:
                        items = result.get('items', [])
                        
                        for item in items[:10]:  # Top 10 results
                            if item.get('type') == 'organic':
                                url = item.get('url', '')
                                title = item.get('title', '')
                                description = item.get('description', '')
                                
                                if url and title:
                                    results.append({
                                        'url': url,
                                        'title': title,
                                        'description': description,
                                        'position': item.get('rank_group', 0)
                                    })
                
                logger.info(f"🎯 DataForSEO found {len(results)} results for '{query}'")
                return results[:8]  # Return top 8 results
                
            else:
                logger.error(f"DataForSEO API error: {response.status_code} - {response.text}")
                return []
                
    except Exception as e:
        logger.error(f"DataForSEO search failed for {keyword}: {e}")
        return []
    
    return []


def extract_competitor_from_result(title: str, url: str, description: str = '') -> dict:
    """Extract competitor information from search result with proper validation."""
    try:
        from urllib.parse import urlparse
        domain = urlparse(url).netloc.replace('www.', '')
        
        # Skip generic domains and non-competitor sites
        excluded_domains = [
            'wikipedia.org', 'linkedin.com', 'facebook.com', 'twitter.com', 
            'youtube.com', 'crunchbase.com', 'glassdoor.com', 'indeed.com',
            'reddit.com', 'quora.com', 'stackoverflow.com', 'medium.com',
            'forbes.com', 'techcrunch.com', 'businessinsider.com', 'bloomberg.com',
            'cnbc.com', 'reuters.com', 'wsj.com', 'nytimes.com', 'washingtonpost.com',
            'amazon.com', 'ebay.com', 'etsy.com', 'shopify.com',  # Marketplaces (unless they're the target)
            'apple.com', 'google.com', 'microsoft.com',  # Tech giants (unless specific product)
            'capterra.com', 'g2.com', 'trustpilot.com', 'yelp.com',  # Review sites
            'github.com', 'gitlab.com', 'bitbucket.org',  # Code repositories
        ]
        
        if any(excluded in domain for excluded in excluded_domains):
            return None
        
        # Skip URLs that look like articles or blog posts
        if any(indicator in url.lower() for indicator in ['/blog/', '/news/', '/article/', '/press/', '/media/']):
            return None
        
        # Extract company name from multiple sources
        company_name = None
        
        # Method 1: Extract from title (most reliable)
        if title:
            # Look for company name patterns in title
            title_clean = title.replace(' - ', ' | ').replace(' – ', ' | ')
            if ' | ' in title_clean:
                # Often format is "Company Name | Product/Service"
                potential_name = title_clean.split(' | ')[0].strip()
                if len(potential_name) > 2 and len(potential_name) < 40:
                    company_name = potential_name
        
        # Method 2: Extract from domain if title extraction failed
        if not company_name:
            domain_name = domain.split('.')[0]
            # Clean up domain name
            company_name = domain_name.replace('-', ' ').replace('_', ' ').title()
        
        # Validate company name
        if company_name and len(company_name) > 2 and len(company_name) < 40:
            # Additional validation - skip if it looks like a generic term
            generic_terms = ['home', 'about', 'contact', 'blog', 'news', 'search', 'help', 'support']
            if company_name.lower() not in generic_terms:
                return {
                    'name': company_name,
                    'url': url,
                    'domain': domain
                }
            
    except Exception as e:
        logger.error(f"Error extracting competitor info: {e}")
        
    return None


async def analyze_growth_strategies(website: str, company_profile: dict, company_name: str) -> dict:
    """Analyze customer acquisition and growth strategies using AI."""
    try:
        logger.info("📈 Starting growth strategy analysis...")
        
        # Scrape website for growth-related content
        growth_content = ""
        if website:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    response = await client.get(website)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Look for specific sections that might contain growth info
                        growth_sections = []
                        
                        # Find about, case studies, testimonials, pricing, services pages
                        for section in soup.find_all(['div', 'section'], class_=re.compile(r'about|case|testimonial|pricing|service|customer|client|success|process|method', re.I)):
                            growth_sections.append(section.get_text())
                        
                        # Get main content
                        main_content = soup.get_text()
                        growth_content = main_content[:3000]  # First 3000 chars
                        
                        # Add specific sections
                        for section_text in growth_sections[:3]:  # Top 3 relevant sections
                            growth_content += "\n\n" + section_text[:1000]
                            
            except Exception as e:
                logger.error(f"Error scraping growth content: {e}")
        
        # Use ChatGPT to analyze growth strategies
        growth_analysis_prompt = f"""
        Analyze this company's website content to answer these specific business questions:

        Company: {company_name}
        Industry: {company_profile.get('industry', 'Unknown')}
        Description: {company_profile.get('description', 'Unknown')}
        Target Market: {company_profile.get('target_market', 'Unknown')}

        Website Content:
        {growth_content}

        Based on this content, provide insights for these 4 questions:

        1. How do they currently turn strangers into customers?
        2. What has worked best to grow their business so far?
        3. What hasn't worked or felt like a waste of time?
        4. How do they currently use SEO or Search Advertising?

        Return ONLY a JSON object with these exact keys:
        {{
            "customer_acquisition_process": "description of how they convert strangers to customers",
            "growth_strategies_that_work": "what has worked best for growth",
            "ineffective_strategies": "what hasn't worked or was wasteful",
            "seo_and_advertising_approach": "their SEO and search advertising approach"
        }}

        If information is not available, use "Information not available from website content" as the value.
        """
        
        response = await call_openai_chat(
            messages=[{"role": "user", "content": growth_analysis_prompt}],
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=800
        )
        
        analysis_text = response.choices[0].message.content.strip()
        logger.info(f"📊 AI growth analysis: {analysis_text[:200]}...")
        
        try:
            import json
            growth_data = json.loads(analysis_text)
            
            # Validate the response has the expected keys
            expected_keys = ['customer_acquisition_process', 'growth_strategies_that_work', 'ineffective_strategies', 'seo_and_advertising_approach']
            for key in expected_keys:
                if key not in growth_data:
                    growth_data[key] = "Information not available from website content"
            
            logger.info(f"✅ Growth strategy analysis completed")
            return growth_data
            
        except json.JSONDecodeError:
            logger.error("Failed to parse growth analysis JSON")
            return {
                'customer_acquisition_process': "Analysis could not be completed",
                'growth_strategies_that_work': "Analysis could not be completed",
                'ineffective_strategies': "Analysis could not be completed",
                'seo_and_advertising_approach': "Analysis could not be completed"
            }
        
    except Exception as e:
        logger.error(f"Growth strategy analysis failed: {e}")
        return {
            'customer_acquisition_process': "Analysis failed",
            'growth_strategies_that_work': "Analysis failed",
            'ineffective_strategies': "Analysis failed",
            'seo_and_advertising_approach': "Analysis failed"
        }


async def analyze_goals_and_challenges(website: str, company_profile: dict, company_name: str) -> dict:
    """Analyze future goals and challenges using AI."""
    try:
        logger.info("🎯 Starting goals and challenges analysis...")
        
        # Scrape website for goals and future-focused content
        goals_content = ""
        if website:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    response = await client.get(website)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Look for specific sections that might contain goals/roadmap info
                        goals_sections = []
                        
                        # Find about, roadmap, vision, future, plans, investor sections
                        for section in soup.find_all(['div', 'section'], class_=re.compile(r'about|roadmap|vision|future|plan|goal|investor|mission|strategy|growth|expansion', re.I)):
                            goals_sections.append(section.get_text())
                        
                        # Look for specific pages or content
                        for link in soup.find_all('a', href=re.compile(r'about|roadmap|vision|future|plan|investor', re.I)):
                            link_text = link.get_text()
                            if link_text:
                                goals_sections.append(f"Link: {link_text}")
                        
                        # Get main content
                        main_content = soup.get_text()
                        goals_content = main_content[:3000]  # First 3000 chars
                        
                        # Add specific sections
                        for section_text in goals_sections[:3]:  # Top 3 relevant sections
                            goals_content += "\n\n" + section_text[:1000]
                            
            except Exception as e:
                logger.error(f"Error scraping goals content: {e}")
        
        # Use ChatGPT to analyze goals and challenges
        goals_analysis_prompt = f"""
        Analyze this company's website content to infer their future goals and challenges:

        Company: {company_name}
        Industry: {company_profile.get('industry', 'Unknown')}
        Description: {company_profile.get('description', 'Unknown')}
        Target Market: {company_profile.get('target_market', 'Unknown')}
        Current Services: {company_profile.get('services_offered', 'Unknown')}

        Website Content:
        {goals_content}

        Based on this content and typical business patterns for companies in this industry, provide insights for these 3 questions:

        1. What are their likely main business goals for the next 12 months?
        2. What specific SEO, Ads, or visibility goals might they have?
        3. What do you believe is blocking them from reaching their goals right now?

        Consider:
        - Common industry challenges and opportunities
        - Their current stage of business (startup, growth, mature)
        - Market trends in their sector
        - Typical goals for companies of their size/type

        Return ONLY a JSON object with these exact keys:
        {{
            "main_business_goals_12_months": "likely 12-month business objectives",
            "seo_ads_visibility_goals": "probable SEO and advertising goals",
            "current_blocking_factors": "potential challenges blocking growth"
        }}

        If specific information is not available, provide educated insights based on industry patterns.
        """
        
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": goals_analysis_prompt}],
            temperature=0.4,  # Slightly higher for creative insights
            max_tokens=800
        )
        
        analysis_text = response.choices[0].message.content.strip()
        logger.info(f"🎯 AI goals analysis: {analysis_text[:200]}...")
        
        try:
            import json
            goals_data = json.loads(analysis_text)
            
            # Validate the response has the expected keys
            expected_keys = ['main_business_goals_12_months', 'seo_ads_visibility_goals', 'current_blocking_factors']
            for key in expected_keys:
                if key not in goals_data:
                    goals_data[key] = "Goals analysis could not be completed"
            
            logger.info(f"✅ Goals and challenges analysis completed")
            return goals_data
            
        except json.JSONDecodeError:
            logger.error("Failed to parse goals analysis JSON")
            return {
                'main_business_goals_12_months': "Analysis could not be completed",
                'seo_ads_visibility_goals': "Analysis could not be completed",
                'current_blocking_factors': "Analysis could not be completed"
            }
        
    except Exception as e:
        logger.error(f"Goals and challenges analysis failed: {e}")
        return {
            'main_business_goals_12_months': "Analysis failed",
            'seo_ads_visibility_goals': "Analysis failed",
            'current_blocking_factors': "Analysis failed"
        }


async def analyze_service_and_location_priorities(website: str, company_profile: dict, company_name: str, provided_main_locations: list = None) -> dict:
    """Analyze service and location priorities using AI."""
    try:
        logger.info("🎯 Starting service and location priorities analysis...")
        
        # Scrape website for service and location content
        service_location_content = ""
        if website:
            try:
                async with httpx.AsyncClient(timeout=15) as client:
                    response = await client.get(website)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Look for specific sections that might contain service/location info
                        service_sections = []
                        
                        # Find services, pricing, products, locations, coverage areas
                        for section in soup.find_all(['div', 'section'], class_=re.compile(r'service|product|pricing|location|coverage|area|region|city|state|offer|solution', re.I)):
                            service_sections.append(section.get_text())
                        
                        # Look for navigation links that indicate services/locations
                        for link in soup.find_all('a', href=re.compile(r'service|product|pricing|location|area|coverage|region', re.I)):
                            link_text = link.get_text()
                            if link_text and len(link_text) < 100:
                                service_sections.append(f"Service/Location Link: {link_text}")
                        
                        # Look for specific service/location keywords in text
                        main_content = soup.get_text()
                        service_location_content = main_content[:4000]  # First 4000 chars
                        
                        # Add specific sections
                        for section_text in service_sections[:4]:  # Top 4 relevant sections
                            service_location_content += "\n\n" + section_text[:800]
                            
            except Exception as e:
                logger.error(f"Error scraping service/location content: {e}")
        
        # Use ChatGPT to analyze service and location priorities
        service_location_prompt = f"""
        Analyze this company's website content to determine their service and location priorities:

        Company: {company_name}
        Industry: {company_profile.get('industry', 'Unknown')}
        Description: {company_profile.get('description', 'Unknown')}
        Target Market: {company_profile.get('target_market', 'Unknown')}
        Current Services: {company_profile.get('services_offered', 'Unknown')}
        Location: {company_profile.get('city', 'Unknown')}, {company_profile.get('country', 'Unknown')}
        
        {f"User-Provided Main Locations: {', '.join(provided_main_locations)}" if provided_main_locations else ""}

        Website Content:
        {service_location_content}

        Based on this content and typical business patterns, provide insights for these 2 questions:

        1. What are their top 3 priority services to promote first?
        2. What areas or regions do they serve? What are their top 3 priority locations?

        Consider:
        - Most prominently featured services on their website
        - Services mentioned first or most frequently
        - Geographic indicators (cities, states, regions mentioned)
        - Service area coverage patterns
        - Local vs national vs international focus
        {f"- User provided these main locations as context: {', '.join(provided_main_locations)}" if provided_main_locations else ""}

        Return ONLY a JSON object with these exact keys:
        {{
            "top_3_priority_services": "their 3 most important services to promote, based on website prominence and business focus",
            "service_areas_and_regions": "specific areas they serve and their top 3 priority locations, with geographic details"
        }}

        Be specific with locations (mention counties, cities, regions) and services (actual service names, not generic terms).
        """
        
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": service_location_prompt}],
            temperature=0.3,  # Lower temperature for factual analysis
            max_tokens=600
        )
        
        analysis_text = response.choices[0].message.content.strip()
        logger.info(f"🎯 AI service/location analysis: {analysis_text[:200]}...")
        
        try:
            import json
            service_location_data = json.loads(analysis_text)
            
            # Validate the response has the expected keys
            expected_keys = ['top_3_priority_services', 'service_areas_and_regions']
            for key in expected_keys:
                if key not in service_location_data:
                    service_location_data[key] = "Service/location analysis could not be completed"
            
            logger.info(f"✅ Service and location priorities analysis completed")
            return service_location_data
            
        except json.JSONDecodeError:
            logger.error("Failed to parse service/location analysis JSON")
            return {
                'top_3_priority_services': "Analysis could not be completed",
                'service_areas_and_regions': "Analysis could not be completed"
            }
        
    except Exception as e:
        logger.error(f"Service and location priorities analysis failed: {e}")
        return {
            'top_3_priority_services': "Analysis failed",
            'service_areas_and_regions': "Analysis failed"
        }


async def create_topic_authority_map(company_name: str, industry: str, city: str, country: str, main_locations: list) -> dict:
    """Create comprehensive Topic Authority Map using DataForSEO API for local keyword research."""
    try:
        from app.config import settings
        logger.info("📝 Starting Topic Authority Map creation...")
        
        # Step 1: Resolve location and language IDs
        location_data = await get_dataforseo_location_id(city, country)
        if not location_data:
            logger.warning("Could not resolve location for DataForSEO")
            return await create_basic_content_plan(company_name, industry, city)
        
        location_id = location_data['location_id']
        language_id = location_data['language_id']
        target_location = f"{city}, {country}"
        
        logger.info(f"📍 Using DataForSEO location_id: {location_id}, language_id: {language_id}")
        
        # Step 2: Identify the main business keyword using AI and company data
        main_business_keyword = await identify_main_business_keyword(
            company_name, industry, city, 
            final_profile.get('description', ''), 
            final_profile.get('services_offered', ''),
            all_scraped_content
        )
        logger.info(f"🎯 Identified main business keyword: {main_business_keyword}")
        
        # Step 3: Generate focused seed keywords around the main business
        seed_keywords = await generate_focused_seed_keywords(main_business_keyword, company_name, city, main_locations)
        logger.info(f"🌱 Generated focused seed keywords: {seed_keywords}")
        
        # Step 4: Discover related keywords with DataForSEO
        all_keywords_data = []
        for seed in seed_keywords[:5]:  # Limit to top 5 seeds
            keywords_data = await dataforseo_related_keywords(seed, location_id, language_id)
            all_keywords_data.extend(keywords_data)
        
        logger.info(f"🔍 Found {len(all_keywords_data)} total keywords")
        
        # Step 5: Find the highest-volume keyword that best matches the business
        primary_keyword = await find_primary_business_keyword(all_keywords_data, main_business_keyword, final_profile)
        logger.info(f"🎯 Primary business keyword: {primary_keyword}")
        
        # Step 6: Enrich with SERP data for top keywords (prioritizing primary keyword)
        enriched_keywords = []
        
        # Always include primary keyword first
        if primary_keyword:
            primary_kw_data = next((kw for kw in all_keywords_data if kw['keyword'] == primary_keyword), None)
            if primary_kw_data:
                serp_data = await dataforseo_serp_analysis(primary_keyword, target_location, language_id)
                primary_kw_data.update(serp_data)
                primary_kw_data['is_primary'] = True
                enriched_keywords.append(primary_kw_data)
        
        # Add other high-volume keywords
        for keyword_data in all_keywords_data[:49]:  # Top 49 + primary = 50
            if keyword_data['keyword'] != primary_keyword:
                serp_data = await dataforseo_serp_analysis(keyword_data['keyword'], target_location, language_id)
                keyword_data.update(serp_data)
                keyword_data['is_primary'] = False
                enriched_keywords.append(keyword_data)
        
        # Step 7: Cluster keywords by topic using AI (with primary keyword focus)
        topic_clusters = await cluster_keywords_by_topic(enriched_keywords, main_business_keyword, primary_keyword)
        
        # Step 6: Create Topic Authority Map
        topic_authority_map = await build_topic_authority_map(
            niche, target_location, 'en', country, topic_clusters, enriched_keywords
        )
        
        # Step 7: Generate content plan summary
        content_summary = await generate_content_plan_summary(topic_authority_map, company_name)
        
        logger.info("✅ Topic Authority Map creation completed")
        
        return {
            'topic_authority_map': topic_authority_map,
            'content_plan_summary': content_summary
        }
        
    except Exception as e:
        logger.error(f"Topic Authority Map creation failed: {e}")
        # Fallback to basic content plan
        return await create_basic_content_plan(company_name, industry, city)


async def get_dataforseo_location_id(city: str, country: str) -> dict:
    """Get DataForSEO location and language IDs."""
    try:
        from app.config import settings
        
        # DataForSEO locations endpoint
        api_url = "https://api.dataforseo.com/v3/dataforseo_labs/locations_and_languages"
        
        headers = {
            "Authorization": f"Basic {settings.dataforseo_base64}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.get(api_url, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status_code') == 20000:
                    locations = data.get('tasks', [{}])[0].get('result', {}).get('locations', [])
                    
                    # Find best location match
                    best_match = None
                    for location in locations:
                        if city.lower() in location.get('location_name', '').lower() and \
                           country.lower() in location.get('country_name', '').lower():
                            best_match = location
                            break
                    
                    if not best_match:
                        # Fallback to country-level
                        for location in locations:
                            if country.lower() in location.get('country_name', '').lower():
                                best_match = location
                                break
                    
                    if best_match:
                        return {
                            'location_id': best_match.get('location_code'),
                            'language_id': 1033,  # English default
                            'location_name': best_match.get('location_name')
                        }
        
        return None
        
    except Exception as e:
        logger.error(f"DataForSEO location lookup failed: {e}")
        return None


async def identify_main_business_keyword(company_name: str, industry: str, city: str, description: str, services: str, scraped_content: str) -> str:
    """Use AI to identify the main business keyword that best represents what the company does."""
    try:
        logger.info("🎯 Identifying main business keyword...")
        
        business_analysis_prompt = f"""
        Analyze this company's business to identify their PRIMARY business keyword for SEO content strategy:

        Company: {company_name}
        Industry: {industry}
        Location: {city}
        Description: {description}
        Services: {services}
        
        Website Content Sample:
        {scraped_content[:1500]}

        Based on this information, identify the ONE main keyword that:
        1. Best represents what this business actually does (be specific)
        2. Would have the highest search volume potential
        3. Aligns with their core service offering
        4. Is specific enough to build topical authority around

        Examples:
        - For a roofing company: "roof repair" (not just "roofing")
        - For a dental practice: "orthodontist" or "dental implants" (not just "dentist")
        - For a software company: "project management software" (not just "software")

        Return ONLY the main keyword phrase (2-4 words max), nothing else.
        Be precise and specific to their actual business focus.
        """
        
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": business_analysis_prompt}],
            temperature=0.2,  # Low temperature for precise identification
            max_tokens=50
        )
        
        main_keyword = response.choices[0].message.content.strip().lower()
        logger.info(f"🎯 AI identified main business keyword: {main_keyword}")
        
        return main_keyword
        
    except Exception as e:
        logger.error(f"Main business keyword identification failed: {e}")
        return industry.lower()  # Fallback to industry


async def generate_focused_seed_keywords(main_business_keyword: str, company_name: str, city: str, main_locations: list) -> list:
    """Generate focused seed keywords around the main business keyword."""
    seeds = []
    
    # Core business seeds focused on main keyword
    seeds.extend([
        main_business_keyword,
        f"{main_business_keyword} {city}",
        f"{main_business_keyword} near me",
        f"best {main_business_keyword} {city}",
        f"{main_business_keyword} services {city}",
        f"{main_business_keyword} company {city}"
    ])
    
    # Add location-specific seeds for main business
    for location in main_locations[:3]:
        seeds.extend([
            f"{main_business_keyword} {location}",
            f"best {main_business_keyword} {location}",
            f"{main_business_keyword} services {location}"
        ])
    
    # Company-specific seeds
    seeds.extend([
        f"{company_name} {main_business_keyword}",
        f"{company_name} {city}"
    ])
    
    return list(set(seeds))  # Remove duplicates


async def find_primary_business_keyword(keywords_data: list, main_business_keyword: str, company_profile: dict) -> str:
    """Find the highest-volume keyword that best aligns with the business."""
    try:
        logger.info("🔍 Finding primary business keyword from search data...")
        
        # Filter keywords that contain the main business concept
        relevant_keywords = []
        main_business_words = main_business_keyword.split()
        
        for kw_data in keywords_data:
            keyword = kw_data['keyword'].lower()
            volume = kw_data.get('search_volume', 0)
            
            # Check if keyword is relevant to main business
            relevance_score = 0
            for word in main_business_words:
                if word in keyword:
                    relevance_score += 1
            
            # Additional relevance checks
            if any(service_word in keyword for service_word in ['service', 'company', 'repair', 'install', 'maintenance']):
                relevance_score += 0.5
            
            if relevance_score > 0 and volume > 0:
                relevant_keywords.append({
                    'keyword': kw_data['keyword'],
                    'volume': volume,
                    'relevance': relevance_score,
                    'combined_score': volume * relevance_score
                })
        
        # Sort by combined score (volume * relevance)
        relevant_keywords.sort(key=lambda x: x['combined_score'], reverse=True)
        
        if relevant_keywords:
            primary = relevant_keywords[0]['keyword']
            logger.info(f"🎯 Selected primary keyword: {primary} (volume: {relevant_keywords[0]['volume']}, relevance: {relevant_keywords[0]['relevance']})")
            return primary
        
        # Fallback to main business keyword if no high-volume matches
        return main_business_keyword
        
    except Exception as e:
        logger.error(f"Primary keyword selection failed: {e}")
        return main_business_keyword


async def generate_seed_keywords(company_name: str, niche: str, city: str, main_locations: list) -> list:
    """Generate seed keywords for local keyword research."""
    seeds = []
    
    # Core business seeds
    seeds.extend([
        niche,
        f"{niche} {city}",
        f"{niche} near me",
        f"best {niche} {city}",
        f"{niche} services {city}"
    ])
    
    # Add location-specific seeds
    for location in main_locations[:3]:
        seeds.extend([
            f"{niche} {location}",
            f"best {niche} {location}"
        ])
    
    # Company-specific seeds
    seeds.extend([
        company_name,
        f"{company_name} {city}",
        f"{company_name} services"
    ])
    
    return list(set(seeds))  # Remove duplicates


async def dataforseo_related_keywords(seed: str, location_id: int, language_id: int) -> list:
    """Get related keywords from DataForSEO Labs API."""
    try:
        from app.config import settings
        
        api_url = "https://api.dataforseo.com/v3/dataforseo_labs/google/related_keywords/live"
        
        request_body = [{
            "keywords": [seed],
            "location_id": location_id,
            "language_id": language_id,
            "include_serp_info": True,
            "limit": 100
        }]
        
        headers = {
            "Authorization": f"Basic {settings.dataforseo_base64}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(api_url, json=request_body, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status_code') == 20000 and data.get('tasks'):
                    items = data['tasks'][0].get('result', [{}])[0].get('items', [])
                    
                    keywords_data = []
                    for item in items:
                        keywords_data.append({
                            'keyword': item.get('keyword', ''),
                            'search_volume': item.get('search_volume', 0),
                            'cpc': item.get('cpc', 0),
                            'competition': item.get('competition', 0),
                            'trend_12m': item.get('search_volume_trend', {}).get('trend', 0),
                            'seed': seed
                        })
                    
                    logger.info(f"🔍 Found {len(keywords_data)} related keywords for '{seed}'")
                    return keywords_data[:50]  # Top 50
        
        return []
        
    except Exception as e:
        logger.error(f"DataForSEO related keywords failed for {seed}: {e}")
        return []


async def dataforseo_serp_analysis(keyword: str, location: str, language_id: int) -> dict:
    """Analyze SERP features and competition for a keyword."""
    try:
        from app.config import settings
        
        api_url = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
        
        request_body = [{
            "keyword": keyword,
            "location_name": location,
            "language_code": "en",
            "device": "mobile",
            "depth": 20
        }]
        
        headers = {
            "Authorization": f"Basic {settings.dataforseo_base64}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(api_url, json=request_body, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status_code') == 20000 and data.get('tasks'):
                    result = data['tasks'][0].get('result', [{}])[0]
                    
                    # Extract SERP features
                    has_local_pack = False
                    has_snippet = False
                    paa_count = 0
                    top_urls = []
                    
                    items = result.get('items', [])
                    for item in items:
                        if item.get('type') == 'local_pack':
                            has_local_pack = True
                        elif item.get('type') == 'featured_snippet':
                            has_snippet = True
                        elif item.get('type') == 'people_also_ask':
                            paa_count += 1
                        elif item.get('type') == 'organic':
                            url = item.get('url', '')
                            if url and len(top_urls) < 10:
                                top_urls.append(url)
                    
                    return {
                        'has_local_pack': has_local_pack,
                        'has_snippet': has_snippet,
                        'paa_count': paa_count,
                        'top_urls': top_urls
                    }
        
        return {
            'has_local_pack': False,
            'has_snippet': False,
            'paa_count': 0,
            'top_urls': []
        }
        
    except Exception as e:
        logger.error(f"DataForSEO SERP analysis failed for {keyword}: {e}")
        return {
            'has_local_pack': False,
            'has_snippet': False,
            'paa_count': 0,
            'top_urls': []
        }


async def cluster_keywords_by_topic(keywords_data: list, main_business_keyword: str, primary_keyword: str) -> dict:
    """Use AI to cluster keywords by topic and intent."""
    try:
        # Prepare keywords for AI analysis
        keywords_text = []
        for kw_data in keywords_data[:30]:  # Top 30 keywords
            kw = kw_data['keyword']
            volume = kw_data.get('search_volume', 0)
            has_local = kw_data.get('has_local_pack', False)
            keywords_text.append(f"{kw} (vol: {volume}, local: {has_local})")
        
        clustering_prompt = f"""
        Analyze these keywords for a business specializing in "{main_business_keyword}" and cluster them into content topics:

        PRIMARY BUSINESS FOCUS: {primary_keyword} (highest volume + best business alignment)
        MAIN BUSINESS KEYWORD: {main_business_keyword}

        Keywords with search data:
        {chr(10).join(keywords_text)}

        Create 3-5 topic clusters that build topical authority around "{primary_keyword}". Requirements:

        1. FIRST cluster MUST center around the primary keyword: "{primary_keyword}"
        2. Other clusters should support and relate to the main business
        3. Classify search intent (Informational, Local-Commercial, Transactional, Navigational)
        4. Prioritize Local-Commercial intent for immediate business impact
        5. Group keywords by semantic similarity and search intent

        Return ONLY a JSON object like:
        {{
            "clusters": [
                {{
                    "topic": "{primary_keyword.title()} Services",
                    "intent": "Local-Commercial", 
                    "keywords": ["{primary_keyword}", "{primary_keyword} near me"],
                    "pillar_title": "Professional {primary_keyword.title()} Services in [City]",
                    "priority_score": 1.0,
                    "is_primary_cluster": true
                }},
                {{
                    "topic": "Supporting Topic",
                    "intent": "Informational",
                    "keywords": ["related keyword 1", "related keyword 2"],
                    "pillar_title": "Supporting Content Title",
                    "priority_score": 0.8,
                    "is_primary_cluster": false
                }}
            ]
        }}
        """
        
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": clustering_prompt}],
            temperature=0.3,
            max_tokens=1000
        )
        
        clustering_text = response.choices[0].message.content.strip()
        
        try:
            import json
            clusters = json.loads(clustering_text)
            return clusters.get('clusters', [])
        except:
            logger.error("Failed to parse clustering JSON")
            return []
        
    except Exception as e:
        logger.error(f"Keyword clustering failed: {e}")
        return []


async def build_topic_authority_map(niche: str, location: str, language: str, country: str, clusters: list, keywords_data: list) -> dict:
    """Build the complete Topic Authority Map."""
    try:
        pillars = []
        total_volume = 0
        
        # Sort clusters to prioritize primary cluster first
        clusters_sorted = sorted(clusters[:5], key=lambda x: x.get('is_primary_cluster', False), reverse=True)
        
        for cluster in clusters_sorted:  # Primary cluster first, then others
            # Get keywords for this cluster
            cluster_keywords = []
            cluster_volume = 0
            
            for kw_data in keywords_data:
                if any(kw.lower() in kw_data['keyword'].lower() for kw in cluster.get('keywords', [])):
                    cluster_keywords.append(kw_data)
                    cluster_volume += kw_data.get('search_volume', 0)
            
            # Boost primary cluster with additional relevant keywords
            if cluster.get('is_primary_cluster', False):
                # Add any keywords with the primary business focus
                for kw_data in keywords_data:
                    if kw_data.get('is_primary', False) and kw_data not in cluster_keywords:
                        cluster_keywords.append(kw_data)
                        cluster_volume += kw_data.get('search_volume', 0)
            
            # Generate supporting articles with AI
            supporting_articles = await generate_supporting_articles(cluster, cluster_keywords)
            
            pillar = {
                'topic': cluster.get('topic', ''),
                'intent': cluster.get('intent', 'Informational'),
                'seasonality': 'Year-round',  # Could be enhanced with trends data
                'cluster_score': cluster.get('priority_score', 0.5),
                'pillar_page_h1': cluster.get('pillar_title', '').replace('[City]', city),
                'supporting_articles': supporting_articles,
                'local_entities_sample': [],  # Could be enhanced with business listings
                'top_competing_urls': [kw.get('top_urls', [])[:3] for kw in cluster_keywords[:3]]
            }
            
            pillars.append(pillar)
            total_volume += cluster_volume
        
        return {
            'niche': niche,
            'location': location,
            'language': language,
            'country_code': country,
            'pillars': pillars,
            'total_keywords': len(keywords_data),
            'avg_search_volume': total_volume // max(len(pillars), 1),
            'content_gap_opportunities': []
        }
        
    except Exception as e:
        logger.error(f"Topic Authority Map building failed: {e}")
        return {}


async def generate_supporting_articles(cluster: dict, keywords_data: list) -> list:
    """Generate supporting article ideas for a topic cluster."""
    try:
        cluster_keywords = [kw['keyword'] for kw in keywords_data[:10]]
        
        article_prompt = f"""
        Create 3-5 supporting article ideas for this topic cluster:

        Topic: {cluster.get('topic', '')}
        Intent: {cluster.get('intent', '')}
        Keywords: {', '.join(cluster_keywords)}

        Generate specific, actionable article titles that would support the main pillar page. 
        Each article should target 2-3 related keywords and have clear search intent.

        Return ONLY a JSON array like:
        [
            {{
                "title": "How Much Does Emergency Roof Repair Cost in Austin? (2025 Guide)",
                "keywords": ["roof repair cost austin", "emergency roof repair pricing"],
                "serp_features": ["featured_snippet", "paa"],
                "intent": "Local-Commercial",
                "priority_score": 0.8
            }}
        ]
        """
        
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": article_prompt}],
            temperature=0.4,
            max_tokens=800
        )
        
        articles_text = response.choices[0].message.content.strip()
        
        try:
            import json
            articles = json.loads(articles_text)
            return articles[:5] if isinstance(articles, list) else []
        except:
            return []
        
    except Exception as e:
        logger.error(f"Supporting articles generation failed: {e}")
        return []


async def generate_content_plan_summary(topic_authority_map: dict, company_name: str) -> str:
    """Generate a concise content plan summary."""
    try:
        pillars = topic_authority_map.get('pillars', [])
        
        summary_prompt = f"""
        Create a concise content strategy summary for {company_name} based on this Topic Authority Map:

        Location: {topic_authority_map.get('location', '')}
        Niche: {topic_authority_map.get('niche', '')}
        Total Keywords: {topic_authority_map.get('total_keywords', 0)}

        Topic Pillars:
        {chr(10).join([f"- {p.get('topic', '')} ({p.get('intent', '')}) - {len(p.get('supporting_articles', []))} articles" for p in pillars])}

        Create a 2-3 paragraph summary explaining:
        1. The content strategy approach for local topical authority
        2. Key topic pillars to focus on first
        3. Expected timeline and content production recommendations

        Keep it actionable and specific to their business and location.
        """
        
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.4,
            max_tokens=600
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Content plan summary generation failed: {e}")
        return "Content plan summary could not be generated"


async def create_basic_content_plan(company_name: str, industry: str, city: str) -> dict:
    """Create a basic content plan when DataForSEO is unavailable."""
    try:
        basic_prompt = f"""
        Create a basic content strategy for {company_name} in {industry} located in {city}.

        Generate a simple topic authority map with 3 main content pillars and supporting articles.

        Return ONLY a JSON object like:
        {{
            "topic_authority_map": {{
                "niche": "{industry}",
                "location": "{city}",
                "pillars": [
                    {{
                        "topic": "Main Service Topic",
                        "intent": "Local-Commercial",
                        "pillar_page_h1": "Main Service in {city}",
                        "supporting_articles": [
                            {{"title": "Article 1", "keywords": ["keyword1"], "intent": "Informational"}}
                        ]
                    }}
                ]
            }},
            "content_plan_summary": "Basic content strategy overview"
        }}
        """
        
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": basic_prompt}],
            temperature=0.4,
            max_tokens=800
        )
        
        result_text = response.choices[0].message.content.strip()
        
        try:
            import json
            return json.loads(result_text)
        except:
            return {
                'topic_authority_map': {},
                'content_plan_summary': "Basic content plan could not be generated"
            }
        
    except Exception as e:
        logger.error(f"Basic content plan creation failed: {e}")
        return {
            'topic_authority_map': {},
            'content_plan_summary': "Content plan creation failed"
        }


async def create_focused_topic_authority_map(primary_keyword: str, company_name: str, city: str, country: str, industry: str, description: str, main_locations: list) -> dict:
    """Create focused Topic Authority Map using the first main keyword as the primary business focus."""
    try:
        logger.info(f"📝 Creating focused Topic Authority Map for: {primary_keyword}")
        
        # Step 1: Generate content clusters around the primary keyword
        content_clusters = await generate_primary_keyword_clusters(
            primary_keyword, company_name, city, industry, description
        )
        
        # Step 2: Create Topic Authority Map structure
        target_location = f"{city}, {country}" if city else country
        
        topic_authority_map = {
            'niche': primary_keyword,
            'location': target_location,
            'language': 'en',
            'country_code': country[:2].upper() if len(country) > 2 else 'US',
            'pillars': content_clusters,
            'total_keywords': len(content_clusters) * 5,  # Estimate
            'avg_search_volume': 1000,  # Estimate - could be enhanced with DataForSEO
            'content_gap_opportunities': []
        }
        
        # Step 3: Generate actionable content plan summary
        content_summary = await generate_focused_content_summary(
            primary_keyword, company_name, target_location, content_clusters
        )
        
        logger.info("✅ Focused Topic Authority Map creation completed")
        
        return {
            'topic_authority_map': topic_authority_map,
            'content_plan_summary': content_summary
        }
        
    except Exception as e:
        logger.error(f"Focused Topic Authority Map creation failed: {e}")
        return {
            'topic_authority_map': None,
            'content_plan_summary': f"Content planning failed for keyword: {primary_keyword}"
        }


async def generate_primary_keyword_clusters(primary_keyword: str, company_name: str, city: str, industry: str, description: str) -> list:
    """Generate content clusters focused on the primary business keyword."""
    try:
        logger.info(f"🎯 Generating content clusters for primary keyword: {primary_keyword}")
        
        cluster_prompt = f"""
        Create a focused content strategy for {company_name} based on their primary business keyword: "{primary_keyword}"

        Company Details:
        - Industry: {industry}
        - Location: {city}
        - Description: {description}
        - Primary Keyword: {primary_keyword}

        Create 3-4 content pillars that build topical authority around "{primary_keyword}":

        1. PRIMARY PILLAR: Must center on "{primary_keyword}" with Local-Commercial intent
        2. SUPPORTING PILLARS: Related topics that support the main business (2-3 pillars)

        For each pillar, include:
        - Topic name
        - Search intent classification
        - Pillar page H1 title
        - 3-4 supporting article ideas with specific titles

        Return ONLY a JSON array like:
        [
            {{
                "topic": "{primary_keyword.title()} Services",
                "intent": "Local-Commercial",
                "cluster_score": 1.0,
                "pillar_page_h1": "Professional {primary_keyword.title()} Services in {city}",
                "is_primary_cluster": true,
                "supporting_articles": [
                    {{
                        "title": "Emergency {primary_keyword.title()} in {city}: 24/7 Service Guide",
                        "keywords": ["emergency {primary_keyword}", "{primary_keyword} near me"],
                        "serp_features": ["local_pack", "paa"],
                        "intent": "Local-Commercial",
                        "priority_score": 0.9
                    }}
                ]
            }}
        ]

        Focus on LOCAL SEO and immediate business impact. Make titles specific and actionable.
        """
        
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": cluster_prompt}],
            temperature=0.3,
            max_tokens=1200
        )
        
        clusters_text = response.choices[0].message.content.strip()
        
        try:
            import json
            clusters = json.loads(clusters_text)
            logger.info(f"✅ Generated {len(clusters)} content clusters")
            return clusters if isinstance(clusters, list) else []
        except:
            logger.error("Failed to parse content clusters JSON")
            return []
        
    except Exception as e:
        logger.error(f"Primary keyword clusters generation failed: {e}")
        return []


async def generate_focused_content_summary(primary_keyword: str, company_name: str, location: str, clusters: list) -> str:
    """Generate focused content plan summary based on primary keyword."""
    try:
        summary_prompt = f"""
        Create a focused content strategy summary for {company_name} based on their primary keyword: "{primary_keyword}"

        Location: {location}
        Content Pillars Created: {len(clusters)}
        
        Primary Business Focus: {primary_keyword}

        Clusters Overview:
        {chr(10).join([f"- {cluster.get('topic', '')} ({cluster.get('intent', '')}) - {len(cluster.get('supporting_articles', []))} articles" for cluster in clusters])}

        Create a concise 2-3 paragraph summary explaining:
        1. Why "{primary_keyword}" is the perfect foundation for their content strategy
        2. How the pillar content will establish them as the local authority
        3. Specific next steps for content creation and timeline

        Focus on LOCAL SEO impact and business results. Be specific about the competitive advantage.
        """
        
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": summary_prompt}],
            temperature=0.4,
            max_tokens=500
        )
        
        return response.choices[0].message.content.strip()
        
    except Exception as e:
        logger.error(f"Focused content summary generation failed: {e}")
        return f"Content strategy will focus on building topical authority around '{primary_keyword}' for local SEO dominance in {location}."


async def get_domain_keywords_dataforseo(website: str, city: str, country: str) -> list:
    """Get actual keywords that the company domain is ranking for using DataForSEO."""
    try:
        from app.config import settings
        import re
        
        if not website:
            return []
        
        # Extract clean domain from website URL
        domain = website.replace('https://', '').replace('http://', '').replace('www.', '').split('/')[0]
        logger.info(f"🔍 Getting keywords for domain: {domain}")
        
        # DataForSEO Keywords for Site endpoint
        api_url = "https://api.dataforseo.com/v3/dataforseo_labs/google/keywords_for_site/live"
        
        # Get location ID for better targeting
        location_id = 2840  # Default to US
        if country.lower() == 'canada':
            location_id = 2124
        elif country.lower() == 'united kingdom':
            location_id = 2826
        
        request_body = [{
            "target": domain,
            "location_id": location_id,
            "language_id": 1033,  # English
            "limit": 100,
            "order_by": ["search_volume,desc"],  # Order by search volume
            "filters": [
                ["search_volume", ">", 10]  # Only keywords with decent volume
            ]
        }]
        
        headers = {
            "Authorization": f"Basic {settings.dataforseo_base64}",
            "Content-Type": "application/json"
        }
        
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(api_url, json=request_body, headers=headers)
            
            if response.status_code == 200:
                data = response.json()
                
                if data.get('status_code') == 20000 and data.get('tasks'):
                    items = data['tasks'][0].get('result', [{}])[0].get('items', [])
                    
                    # Extract and clean keywords
                    domain_keywords = []
                    for item in items:
                        keyword = item.get('keyword', '').strip()
                        search_volume = item.get('search_volume', 0)
                        position = item.get('position', 100)
                        
                        # Filter for relevant business keywords (not branded)
                        if (keyword and 
                            search_volume > 10 and 
                            position <= 50 and  # Only keywords ranking in top 50
                            len(keyword) > 2 and
                            not keyword.lower().startswith(domain.split('.')[0].lower())):  # Exclude branded terms
                            
                            domain_keywords.append({
                                'keyword': keyword,
                                'volume': search_volume,
                                'position': position
                            })
                    
                    # Sort by search volume and return top keywords
                    domain_keywords.sort(key=lambda x: x['volume'], reverse=True)
                    keywords_list = [kw['keyword'] for kw in domain_keywords[:10]]
                    
                    logger.info(f"🎯 Found {len(keywords_list)} ranking keywords for {domain}")
                    logger.info(f"Top keywords: {keywords_list[:5]}")
                    
                    return keywords_list
        
        logger.warning(f"No keywords found for domain: {domain}")
        return []
        
    except Exception as e:
        logger.error(f"DataForSEO domain keywords failed for {website}: {e}")
        return []


async def select_best_pillar_keyword(candidate_keywords: list, company_name: str, industry: str, description: str, services: str, website: str = None) -> str:
    """Validate keywords through ChatGPT to find the best pillar keyword, using fresh website content."""
    try:
        logger.info(f"🔍 Validating pillar keywords from list of {len(candidate_keywords)} keywords")
        
        # Step 1: Scrape fresh website content for keyword validation
        fresh_website_content = ""
        if website:
            try:
                logger.info("🌐 Scraping fresh website content for keyword validation...")
                async with httpx.AsyncClient(timeout=15) as client:
                    response = await client.get(website)
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Get comprehensive content for analysis
                        fresh_website_content = soup.get_text()[:4000]  # First 4000 chars
                        
                        # Also get specific business-focused sections
                        business_sections = []
                        for section in soup.find_all(['div', 'section'], class_=re.compile(r'about|service|product|solution|offer|what-we-do|business', re.I)):
                            business_sections.append(section.get_text())
                        
                        for section_text in business_sections[:3]:
                            fresh_website_content += "\n\n" + section_text[:800]
                        
                        logger.info(f"✅ Scraped {len(fresh_website_content)} chars of fresh content")
            except Exception as e:
                logger.error(f"Error scraping fresh content: {e}")
        
        # Step 2: Try up to 6 keywords (or all if less than 6)
        max_keywords_to_check = min(6, len(candidate_keywords))
        
        for i, keyword in enumerate(candidate_keywords[:max_keywords_to_check]):
            logger.info(f"🎯 Validating keyword {i+1}/{max_keywords_to_check}: {keyword}")
            
            validation_prompt = f"""
            Evaluate if this keyword is suitable for creating a content pillar for this company based on their actual business:

            Company: {company_name}
            Industry: {industry}
            Description: {description}
            Services: {services}
            
            FRESH WEBSITE CONTENT (for accurate business understanding):
            {fresh_website_content[:2000]}
            
            KEYWORD TO EVALUATE: "{keyword}"
            
            Analyze the fresh website content to understand what this company ACTUALLY does, then evaluate if "{keyword}" aligns with their core business.

            Based on the fresh website content, evaluate if "{keyword}" meets these criteria:

            ✅ GOOD pillar keyword criteria:
            1. Directly mentioned or implied in the website content as a core business focus
            2. Represents what customers would search for to find this company
            3. Specific enough to build multiple supporting articles around
            4. Has clear commercial intent (people searching this want to buy/hire)
            5. Aligns with the company's main value proposition shown on website

            ❌ BAD pillar keyword criteria:
            - Company name or branded terms
            - Generic terms like "services", "solutions", "technology"
            - Keywords not mentioned or related to website content
            - Too narrow (only 1-2 possible articles)
            - Unrelated to the company's actual business focus

            CRITICAL: Compare "{keyword}" against the fresh website content. Does this keyword represent what the company actually focuses on based on their website?

            Evaluate "{keyword}" for {company_name} based on their actual website content:

            Return ONLY a JSON object:
            {{
                "is_suitable": true/false,
                "reasoning": "specific explanation based on website content analysis",
                "pillar_potential": "high/medium/low",
                "website_alignment_score": 0.0-1.0,
                "alternative_suggestion": "if not suitable, suggest a better keyword that matches the website content"
            }}
            """
            
            response = await openai_client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[{"role": "user", "content": validation_prompt}],
                temperature=0.2,  # Low temperature for consistent validation
                max_tokens=300
            )
            
            validation_text = response.choices[0].message.content.strip()
            
            try:
                import json
                validation = json.loads(validation_text)
                
                is_suitable = validation.get('is_suitable', False)
                reasoning = validation.get('reasoning', '')
                pillar_potential = validation.get('pillar_potential', 'low')
                alignment_score = validation.get('website_alignment_score', 0.0)
                
                logger.info(f"🔍 Keyword '{keyword}' validation:")
                logger.info(f"   Suitable: {is_suitable}")
                logger.info(f"   Potential: {pillar_potential}")
                logger.info(f"   Alignment Score: {alignment_score}")
                logger.info(f"   Reasoning: {reasoning}")
                
                # Accept keyword if suitable AND has good alignment with website content
                if is_suitable and pillar_potential in ['high', 'medium'] and alignment_score >= 0.7:
                    logger.info(f"✅ Selected pillar keyword: {keyword} (alignment: {alignment_score})")
                    return keyword
                elif is_suitable and pillar_potential == 'high' and alignment_score >= 0.5:
                    # Accept high potential keywords with decent alignment
                    logger.info(f"✅ Selected high-potential keyword: {keyword} (alignment: {alignment_score})")
                    return keyword
                else:
                    logger.info(f"❌ Keyword '{keyword}' not suitable - alignment: {alignment_score}, potential: {pillar_potential}")
                    continue
                    
            except json.JSONDecodeError:
                logger.error(f"Failed to parse validation JSON for keyword: {keyword}")
                continue
        
        # If none of the checked keywords are suitable, try to get an alternative
        logger.warning(f"None of top {max_keywords_to_check} keywords suitable for pillar content")
        
        # Try to get alternative suggestion from the last validation
        try:
            last_validation = json.loads(validation_text)
            alternative = last_validation.get('alternative_suggestion', '')
            if alternative and len(alternative) > 3:
                logger.info(f"🔄 Using AI alternative suggestion: {alternative}")
                return alternative
        except:
            pass
        
        # Final fallback - use first keyword anyway
        if candidate_keywords:
            logger.info(f"🔄 Fallback: Using first keyword anyway: {candidate_keywords[0]}")
            return candidate_keywords[0]
        
        return None
        
    except Exception as e:
        logger.error(f"Pillar keyword selection failed: {e}")
        return candidate_keywords[0] if candidate_keywords else None


async def scrape_website_comprehensive(website: str, company_name: str) -> dict:
    """Comprehensive website scraping with About page priority (hybrid approach)."""
    try:
        candidates = {}
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        async with httpx.AsyncClient(timeout=15, headers=headers) as client:
            # Priority URLs (About page first!)
            priority_urls = [
                f"{website}/about",
                f"{website}/about-us", 
                f"{website}/company",
                f"{website}/mission",
                f"{website}/story",
                f"{website}/team",
                website  # Homepage last
            ]
            
            for url in priority_urls:
                try:
                    logger.info(f"🔍 Scraping priority URL: {url}")
                    response = await client.get(url)
                    
                    if response.status_code == 200:
                        soup = BeautifulSoup(response.text, 'html.parser')
                        
                        # Extract ALL data types from each page
                        await extract_json_ld_data(soup, candidates, url)
                        await extract_meta_data(soup, candidates, url)
                        await extract_content_data(soup, candidates, url)
                        await extract_social_links(soup, candidates, url)
                        await extract_market_and_niche(soup, candidates, url)
                        await extract_client_information(soup, candidates, url)
                        await extract_mission_and_story(soup, candidates, url)
                        await extract_location_data(soup, candidates, url)
                        
                        logger.info(f"✅ Scraped {url} - found: {list(candidates.keys())}")
                        
                        # Give About pages extra processing time
                        if 'about' in url.lower():
                            logger.info(f"📖 Deep analysis of About page: {url}")
                            await deep_about_page_analysis(soup, candidates, url)
                        
                    else:
                        logger.info(f"⚠️ {url} returned {response.status_code}")
                        
                except Exception as e:
                    logger.info(f"⚠️ Could not access {url}: {e}")
                    continue
                
                # Small delay between requests
                await asyncio.sleep(0.5)
        
        logger.info(f"🎯 Comprehensive scraping completed. Found fields: {list(candidates.keys())}")
        return candidates
        
    except Exception as e:
        logger.error(f"Comprehensive scraping failed: {e}")
        return {}


async def deep_about_page_analysis(soup: BeautifulSoup, candidates: dict, source_url: str):
    """Deep analysis specifically for About pages."""
    try:
        logger.info("📖 Performing deep About page analysis...")
        
        # About pages often have the best mission and story content
        text_content = soup.get_text()
        
        # Enhanced mission extraction for About pages
        mission_keywords = ['mission', 'vision', 'purpose', 'why we exist', 'what we do', 'our goal']
        for keyword in mission_keywords:
            pattern = rf'{keyword}[^.]*?[.!?]{{1,3}}'
            matches = re.findall(pattern, text_content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match) > 20 and len(match) < 300:
                    add_candidate(candidates, 'mission_statement', match.strip(), 0.9, source_url)
        
        # Enhanced founding story extraction
        founding_keywords = ['founded', 'started', 'began', 'history', 'story', 'journey']
        for keyword in founding_keywords:
            pattern = rf'{keyword}[^.]*?[.!?]{{1,5}}'
            matches = re.findall(pattern, text_content, re.IGNORECASE | re.DOTALL)
            for match in matches:
                if len(match) > 30 and len(match) < 400:
                    add_candidate(candidates, 'founding_story', match.strip(), 0.9, source_url)
        
        # Look for team/founder sections
        team_sections = soup.find_all(['section', 'div'], class_=re.compile(r'team|founder|leadership', re.I))
        for section in team_sections:
            team_text = section.get_text()
            if any(word in team_text.lower() for word in ['founded', 'started', 'ceo', 'founder']):
                add_candidate(candidates, 'founding_story', team_text[:300], 0.8, source_url)
        
        logger.info("✅ Deep About page analysis completed")
        
    except Exception as e:
        logger.warning(f"Deep About analysis failed: {e}")


async def extract_json_ld_data(soup: BeautifulSoup, candidates: dict, source_url: str):
    """Extract data from JSON-LD structured markup."""
    json_ld_scripts = soup.find_all('script', type='application/ld+json')
    
    for script in json_ld_scripts:
        try:
            data = json.loads(script.string)
            if isinstance(data, list):
                data = data[0] if data else {}
            
            if isinstance(data, dict) and data.get('@type') in ['Organization', 'Corporation', 'Company']:
                # Company name
                if 'name' in data:
                    add_candidate(candidates, 'company_name', data['name'], 0.9, source_url)
                
                # Description
                if 'description' in data:
                    add_candidate(candidates, 'description', data['description'], 0.8, source_url)
                
                # Website
                if 'url' in data:
                    add_candidate(candidates, 'website', data['url'], 0.9, source_url)
                
                # Logo
                if 'logo' in data:
                    logo_url = data['logo'] if isinstance(data['logo'], str) else data['logo'].get('url')
                    if logo_url:
                        add_candidate(candidates, 'logo_url', logo_url, 0.8, source_url)
                
                # Address
                if 'address' in data:
                    addr = data['address']
                    if isinstance(addr, dict):
                        full_address = []
                        if 'streetAddress' in addr:
                            full_address.append(addr['streetAddress'])
                        if 'addressLocality' in addr:
                            full_address.append(addr['addressLocality'])
                            add_candidate(candidates, 'city', addr['addressLocality'], 0.8, source_url)
                        if 'addressRegion' in addr:
                            full_address.append(addr['addressRegion'])
                        if 'postalCode' in addr:
                            full_address.append(addr['postalCode'])
                        if 'addressCountry' in addr:
                            add_candidate(candidates, 'country', addr['addressCountry'], 0.8, source_url)
                        
                        if full_address:
                            add_candidate(candidates, 'hq_address', ', '.join(full_address), 0.8, source_url)
                
                # Contact info
                if 'telephone' in data:
                    add_candidate(candidates, 'phone', data['telephone'], 0.8, source_url)
                
                # Founded date
                if 'foundingDate' in data:
                    try:
                        year = int(data['foundingDate'][:4])
                        add_candidate(candidates, 'year_founded', str(year), 0.8, source_url)
                    except:
                        pass
                
                # Employee count
                if 'numberOfEmployees' in data:
                    add_candidate(candidates, 'employee_count', str(data['numberOfEmployees']), 0.7, source_url)
                
        except Exception as e:
            logger.warning(f"Failed to parse JSON-LD: {e}")


async def extract_meta_data(soup: BeautifulSoup, candidates: dict, source_url: str):
    """Extract data from meta tags."""
    # Open Graph tags
    og_mappings = {
        'og:title': 'company_name',
        'og:description': 'description',
        'og:url': 'website',
        'og:image': 'logo_url',
        'og:site_name': 'company_name'
    }
    
    for og_prop, field in og_mappings.items():
        meta = soup.find('meta', property=og_prop) or soup.find('meta', attrs={'name': og_prop})
        if meta and meta.get('content'):
            score = 0.7 if field == 'company_name' else 0.6
            add_candidate(candidates, field, meta['content'], score, source_url)
    
    # Standard meta tags
    meta_desc = soup.find('meta', attrs={'name': 'description'})
    if meta_desc and meta_desc.get('content'):
        add_candidate(candidates, 'description', meta_desc['content'], 0.6, source_url)


async def extract_content_data(soup: BeautifulSoup, candidates: dict, source_url: str):
    """Extract data from page content using Beautiful Soup."""
    # Get all text content
    text_content = soup.get_text()
    
    # Extract phone numbers
    phone_patterns = [
        r'\+?1?[-.\s]?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}',
        r'\+[0-9]{1,3}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,4}[-.\s]?[0-9]{1,9}'
    ]
    
    for pattern in phone_patterns:
        matches = re.findall(pattern, text_content)
        for match in matches:
            cleaned = re.sub(r'[^\d+]', '', match)
            if len(cleaned) >= 10:
                add_candidate(candidates, 'phone', match.strip(), 0.6, source_url)
    
    # Extract year founded
    year_patterns = [
        r'founded in (\d{4})',
        r'established (\d{4})',
        r'since (\d{4})',
        r'started in (\d{4})',
        r'©\s*(\d{4})',
        r'copyright\s*(\d{4})'
    ]
    
    for pattern in year_patterns:
        matches = re.findall(pattern, text_content, re.IGNORECASE)
        for year in matches:
            if 1800 <= int(year) <= 2024:
                add_candidate(candidates, 'year_founded', year, 0.7, source_url)
    
    # Extract employee count
    employee_patterns = [
        r'(\d+[\-–]\d+)\s+employees',
        r'(\d+\+)\s+employees',
        r'over (\d+)\s+employees',
        r'more than (\d+)\s+employees',
        r'(\d+)-(\d+)\s+people'
    ]
    
    for pattern in employee_patterns:
        matches = re.findall(pattern, text_content, re.IGNORECASE)
        for match in matches:
            if isinstance(match, tuple):
                match = f"{match[0]}-{match[1]}"
            add_candidate(candidates, 'employee_count', match, 0.6, source_url)
    
    # Look for industry keywords in specific sections
    industry_sections = soup.find_all(['section', 'div'], class_=re.compile(r'about|industry|business|company', re.I))
    for section in industry_sections:
        section_text = section.get_text()
        # Simple industry detection
        industry_keywords = {
            'technology': ['software', 'tech', 'digital', 'IT', 'computer'],
            'finance': ['financial', 'banking', 'investment', 'capital'],
            'healthcare': ['health', 'medical', 'pharmaceutical', 'biotech'],
            'retail': ['retail', 'commerce', 'shopping', 'store'],
            'manufacturing': ['manufacturing', 'industrial', 'production'],
            'consulting': ['consulting', 'advisory', 'services']
        }
        
        for industry, keywords in industry_keywords.items():
            if any(keyword in section_text.lower() for keyword in keywords):
                add_candidate(candidates, 'industry', industry.title(), 0.5, source_url)
    
    # Enhanced country and city extraction
    await extract_location_data(soup, candidates, source_url)


async def extract_social_links(soup: BeautifulSoup, candidates: dict, source_url: str):
    """Enhanced social media link extraction."""
    try:
        # Enhanced social patterns
        social_patterns = {
            'linkedin': [
                r'linkedin\.com/company/([^/?#\s]+)',
                r'linkedin\.com/in/([^/?#\s]+)',
                r'www\.linkedin\.com/company/([^/?#\s]+)'
            ],
            'twitter': [
                r'twitter\.com/([^/?#\s]+)',
                r'x\.com/([^/?#\s]+)',
                r'www\.twitter\.com/([^/?#\s]+)'
            ],
            'facebook': [
                r'facebook\.com/([^/?#\s]+)',
                r'www\.facebook\.com/([^/?#\s]+)',
                r'fb\.com/([^/?#\s]+)'
            ],
            'instagram': [
                r'instagram\.com/([^/?#\s]+)',
                r'www\.instagram\.com/([^/?#\s]+)'
            ],
            'youtube': [
                r'youtube\.com/(channel|user|c)/([^/?#\s]+)',
                r'www\.youtube\.com/(channel|user|c)/([^/?#\s]+)',
                r'youtu\.be/([^/?#\s]+)'
            ]
        }
        
        # Find all links in the page
        links = soup.find_all('a', href=True)
        
        # Also check for social media sections specifically
        social_sections = soup.find_all(['section', 'div', 'footer'], 
                                       class_=re.compile(r'social|follow|connect', re.I))
        
        all_links = links.copy()
        for section in social_sections:
            all_links.extend(section.find_all('a', href=True))
        
        for link in all_links:
            href = link.get('href', '').strip()
            
            # Skip empty or invalid links
            if not href or href.startswith('#') or href.startswith('mailto:'):
                continue
            
            for platform, patterns in social_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, href, re.IGNORECASE):
                        # Clean up the URL
                        if href.startswith('//'):
                            href = f"https:{href}"
                        elif not href.startswith('http'):
                            href = f"https://{href}"
                        
                        # Validate it's a real social media URL
                        if platform in href.lower():
                            add_candidate(candidates, f'socials.{platform}', href, 0.8, source_url)
                            logger.info(f"📱 Found {platform}: {href}")
                            break
                
                # Break after finding first match for this platform
                if any(f'socials.{platform}' in candidates for pattern in patterns if re.search(pattern, href, re.IGNORECASE)):
                    break
        
        # Also check for social media icons/images with alt text
        social_imgs = soup.find_all('img', alt=re.compile(r'linkedin|twitter|facebook|instagram|youtube', re.I))
        for img in social_imgs:
            parent_link = img.find_parent('a')
            if parent_link and parent_link.get('href'):
                href = parent_link['href']
                alt_text = img.get('alt', '').lower()
                
                for platform in social_patterns.keys():
                    if platform in alt_text:
                        if not href.startswith('http'):
                            href = f"https://{href}"
                        add_candidate(candidates, f'socials.{platform}', href, 0.7, source_url)
                        logger.info(f"📱 Found {platform} via image: {href}")
                        break
        
        logger.info(f"📱 Social media extraction completed for {source_url}")
        
    except Exception as e:
        logger.warning(f"Social media extraction failed: {e}")


async def extract_market_and_niche(soup: BeautifulSoup, candidates: dict, source_url: str):
    """Extract target market and niche information."""
    
    # Look for specific sections that describe target market
    market_sections = soup.find_all(['section', 'div', 'p', 'h1', 'h2', 'h3'], 
                                   class_=re.compile(r'target|market|audience|customer|client|solution|service', re.I))
    
    market_texts = []
    for section in market_sections:
        text = section.get_text().strip()
        if text and len(text) > 20:  # Meaningful content
            market_texts.append(text)
    
    # Look for "for" statements that indicate target market
    all_text = soup.get_text()
    
    # Extract target market patterns
    target_patterns = [
        r'for ([\w\s,]+) (?:companies|businesses|organizations|teams|professionals)',
        r'helps? ([\w\s,]+) (?:to|with|by)',
        r'designed for ([\w\s,]+)',
        r'built for ([\w\s,]+)',
        r'serving ([\w\s,]+)',
        r'solutions for ([\w\s,]+)',
        r'platform for ([\w\s,]+)'
    ]
    
    for pattern in target_patterns:
        matches = re.findall(pattern, all_text, re.IGNORECASE)
        for match in matches:
            if len(match.strip()) > 3 and len(match.strip()) < 100:
                add_candidate(candidates, 'target_market', match.strip(), 0.7, source_url)
    
    # Look for niche/specialization keywords
    niche_sections = soup.find_all(['section', 'div'], 
                                  class_=re.compile(r'specializ|focus|expert|niche|unique', re.I))
    
    for section in niche_sections:
        text = section.get_text()
        if text and len(text) > 20:
            add_candidate(candidates, 'niche', text.strip()[:200], 0.6, source_url)
    
    # Extract services offered
    service_keywords = ['services', 'solutions', 'products', 'offerings', 'capabilities']
    for keyword in service_keywords:
        service_sections = soup.find_all(['section', 'div', 'ul'], 
                                        class_=re.compile(keyword, re.I))
        for section in service_sections:
            text = section.get_text().strip()
            if text and len(text) > 30:
                add_candidate(candidates, 'services_offered', text[:300], 0.6, source_url)


async def extract_client_information(soup: BeautifulSoup, candidates: dict, source_url: str):
    """Extract client information and analyze patterns."""
    
    # Look for client/customer sections
    client_sections = soup.find_all(['section', 'div'], 
                                   class_=re.compile(r'client|customer|testimonial|case.?stud|portfolio|customer', re.I))
    
    client_data = []
    
    for section in client_sections:
        # Look for company names in client sections
        company_elements = section.find_all(['img', 'div', 'span', 'p'], 
                                           attrs={'alt': True, 'title': True, 'data-company': True})
        
        for element in company_elements:
            company_name = (element.get('alt') or element.get('title') or 
                          element.get('data-company') or element.get_text()).strip()
            if company_name and len(company_name) > 2 and len(company_name) < 50:
                client_data.append(company_name)
        
        # Look for logos (often client logos)
        logos = section.find_all('img', src=True)
        for logo in logos:
            alt_text = logo.get('alt', '').strip()
            if alt_text and 'logo' in alt_text.lower():
                # Extract company name from alt text
                company_name = alt_text.replace('logo', '').replace('Logo', '').strip()
                if company_name and len(company_name) > 2:
                    client_data.append(company_name)
    
    # Look for "trusted by" or "used by" sections
    trust_patterns = [
        r'trusted by:?\s*([^.]+)',
        r'used by:?\s*([^.]+)',
        r'clients include:?\s*([^.]+)',
        r'customers include:?\s*([^.]+)',
        r'partners include:?\s*([^.]+)'
    ]
    
    page_text = soup.get_text()
    for pattern in trust_patterns:
        matches = re.findall(pattern, page_text, re.IGNORECASE)
        for match in matches:
            client_data.append(match.strip())
    
    # If we found client data, store it for AI analysis
    if client_data:
        client_list = ', '.join(set(client_data[:10]))  # Top 10 unique clients
        add_candidate(candidates, 'client_types', client_list, 0.8, source_url)
        
        # Also look for common patterns in client names to infer target market
        client_text = ' '.join(client_data).lower()
        
        # Industry patterns from client names
        if any(word in client_text for word in ['bank', 'financial', 'capital', 'fund']):
            add_candidate(candidates, 'target_market', 'Financial Services Companies', 0.7, source_url)
        elif any(word in client_text for word in ['startup', 'tech', 'software', 'app']):
            add_candidate(candidates, 'target_market', 'Technology Companies and Startups', 0.7, source_url)
        elif any(word in client_text for word in ['retail', 'store', 'shop', 'commerce']):
            add_candidate(candidates, 'target_market', 'Retail and E-commerce Companies', 0.7, source_url)
        elif any(word in client_text for word in ['healthcare', 'medical', 'hospital', 'pharma']):
            add_candidate(candidates, 'target_market', 'Healthcare Organizations', 0.7, source_url)
        elif any(word in client_text for word in ['enterprise', 'corporation', 'fortune']):
            add_candidate(candidates, 'target_market', 'Enterprise and Large Corporations', 0.7, source_url)


async def extract_location_data(soup: BeautifulSoup, candidates: dict, source_url: str):
    """Extract headquarters address, city, and country - focus on actual location data."""
    
    all_text = soup.get_text()
    
    # 1. Look for structured address data in specific sections
    location_sections = []
    
    # Find contact/about sections
    for section in soup.find_all(['div', 'section', 'footer'], class_=re.compile(r'contact|address|location|about|footer|office', re.I)):
        location_sections.append(section.get_text())
    
    # Find specific address elements
    for elem in soup.find_all('address'):
        location_sections.append(elem.get_text())
    
    # 2. Extract real addresses (must have street numbers/names)
    real_address_patterns = [
        r'(\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr|Boulevard|Blvd|Lane|Ln|Way|Circle|Cir|Court|Ct|Place|Pl))[^,\n]*,\s*[A-Z][a-z]+[^,\n]*,\s*[A-Z]{2}(?:\s+\d{5})?)',
        r'(\d+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*(?:\s+(?:Street|St|Avenue|Ave|Road|Rd|Drive|Dr))[^,\n]*,\s*[A-Z][a-z]+[^,\n]*)',
    ]
    
    for section_text in location_sections + [all_text]:
        for pattern in real_address_patterns:
            matches = re.findall(pattern, section_text, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 20:  # Real addresses are longer
                    add_candidate(candidates, 'hq_address', match.strip(), 0.95, source_url)
                    logger.info(f"📍 Found real address: {match.strip()}")
    
    # 3. Extract cities from address context only
    city_from_address_patterns = [
        r'\d+[^,]+,\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*[A-Z]{2}',  # From full address
        r'(?:headquarters|office|located|based)(?:\s+(?:in|at))?\s*([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?),\s*(?:CA|NY|TX|FL|WA|IL|PA|OH|GA|NC|MI|NJ|VA|MA|IN|AZ|TN|MO|MD|WI|MN|CO|AL|SC|LA|KY|OR|OK|CT|UT|IA|NV|AR|MS|KS|NM|NE|WV|ID|HI|NH|ME|MT|RI|DE|SD|ND|AK|VT|WY)',
    ]
    
    for section_text in location_sections:
        for pattern in city_from_address_patterns:
            matches = re.findall(pattern, section_text, re.IGNORECASE)
            for match in matches:
                city = match.strip()
                if len(city) > 2 and len(city) < 25:
                    # Validate it's actually a city name
                    if not any(word in city.lower() for word in ['street', 'avenue', 'road', 'drive', 'suite', 'floor', 'building']):
                        add_candidate(candidates, 'city', city, 0.9, source_url)
                        logger.info(f"📍 Found city from address context: {city}")
    
    # 4. Extract country from clear indicators
    country_indicators = {
        r'(?:headquarters|office|located|based)(?:\s+(?:in|at))?\s*[^,\n]*(?:United States|USA|US)': 'United States',
        r'(?:headquarters|office|located|based)(?:\s+(?:in|at))?\s*[^,\n]*Canada': 'Canada',
        r'(?:headquarters|office|located|based)(?:\s+(?:in|at))?\s*[^,\n]*(?:United Kingdom|UK)': 'United Kingdom',
        r'(?:headquarters|office|located|based)(?:\s+(?:in|at))?\s*[^,\n]*Australia': 'Australia',
    }
    
    for pattern, country in country_indicators.items():
        if re.search(pattern, all_text, re.IGNORECASE):
            add_candidate(candidates, 'country', country, 0.85, source_url)
            logger.info(f"📍 Found country from context: {country}")
    
    # 5. Look for major cities with high confidence
    major_cities_context = {
        'San Francisco': 'United States',
        'New York City': 'United States',
        'New York': 'United States',
        'Los Angeles': 'United States',
        'Chicago': 'United States',
        'Boston': 'United States',
        'Seattle': 'United States',
        'Austin': 'United States',
        'Toronto': 'Canada',
        'Vancouver': 'Canada',
        'London': 'United Kingdom',
        'Berlin': 'Germany',
        'Paris': 'France',
        'Sydney': 'Australia',
    }
    
    for section_text in location_sections:
        for city, country in major_cities_context.items():
            # Only if mentioned in location context
            if re.search(rf'(?:headquarters|office|located|based)(?:\s+(?:in|at))?\s*{re.escape(city)}', section_text, re.IGNORECASE):
                add_candidate(candidates, 'city', city, 0.95, source_url)
                add_candidate(candidates, 'country', country, 0.95, source_url)
                logger.info(f"📍 Found major city in location context: {city}, {country}")
    
    # 6. Look for ZIP codes to infer US location
    zip_pattern = r'\b(\d{5}(?:-\d{4})?)\b'
    zip_matches = re.findall(zip_pattern, all_text)
    if zip_matches:
        # If we find US ZIP codes, likely US-based
        add_candidate(candidates, 'country', 'United States', 0.7, source_url)
        logger.info(f"📍 Found ZIP codes, inferring US location")


async def extract_mission_and_story(soup: BeautifulSoup, candidates: dict, source_url: str):
    """Extract mission statement, founding story, and why the company was started."""
    
    all_text = soup.get_text()
    
    # Look for mission statement sections
    mission_sections = soup.find_all(['section', 'div', 'p', 'h1', 'h2', 'h3'], 
                                    class_=re.compile(r'mission|vision|purpose|goal|objective|value', re.I))
    
    for section in mission_sections:
        text = section.get_text().strip()
        if text and len(text) > 20 and len(text) < 500:
            # Check if it contains mission-like language
            if any(keyword in text.lower() for keyword in ['mission', 'vision', 'purpose', 'believe', 'goal', 'aim']):
                add_candidate(candidates, 'mission_statement', text, 0.8, source_url)
    
    # Look for founding story patterns
    founding_patterns = [
        r'(?:founded|started|began|launched)\s+(?:in\s+\d{4}\s+)?(?:by|when|to|because|with the idea|with a vision|with the goal)[^.]{20,300}',
        r'(?:our story|company story|how it started|the beginning|it all started)[^.]{20,300}',
        r'(?:why we started|why we exist|what drives us|our purpose)[^.]{20,300}',
        r'(?:the idea|the concept|the vision)\s+(?:was born|came to life|started|began)[^.]{20,300}'
    ]
    
    for pattern in founding_patterns:
        matches = re.findall(pattern, all_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            cleaned_story = match.strip()
            if len(cleaned_story) > 30:
                add_candidate(candidates, 'founding_story', cleaned_story, 0.7, source_url)
    
    # Look for "why" statements
    why_patterns = [
        r'(?:why we do this|why we exist|why we started|why we built|why we created)[^.]{20,300}',
        r'(?:we believe|we think|we know|we understand)\s+(?:that|in)[^.]{20,300}',
        r'(?:our mission is|our goal is|our purpose is|we aim to|we strive to)[^.]{20,300}',
        r'(?:we wanted to|we needed to|we set out to|we decided to)[^.]{20,300}'
    ]
    
    for pattern in why_patterns:
        matches = re.findall(pattern, all_text, re.IGNORECASE | re.DOTALL)
        for match in matches:
            cleaned_why = match.strip()
            if len(cleaned_why) > 30:
                add_candidate(candidates, 'why_started', cleaned_why, 0.7, source_url)
    
    # Look for values sections
    values_sections = soup.find_all(['section', 'div', 'ul'], 
                                   class_=re.compile(r'value|principle|belief|culture|core', re.I))
    
    for section in values_sections:
        text = section.get_text().strip()
        if text and len(text) > 30 and len(text) < 400:
            if any(keyword in text.lower() for keyword in ['value', 'principle', 'believe', 'culture', 'core']):
                add_candidate(candidates, 'company_values', text, 0.6, source_url)
    
    # Look for specific mission statement indicators
    mission_indicators = soup.find_all(text=re.compile(r'mission|vision|purpose|manifesto', re.I))
    for indicator in mission_indicators:
        parent = indicator.parent
        if parent:
            # Get the next few elements that might contain the mission
            mission_text = parent.get_text().strip()
            if len(mission_text) > 50 and len(mission_text) < 400:
                add_candidate(candidates, 'mission_statement', mission_text, 0.7, source_url)
    
    # Look for founder quotes or stories
    founder_sections = soup.find_all(['section', 'div', 'blockquote'], 
                                    class_=re.compile(r'founder|ceo|quote|testimonial', re.I))
    
    for section in founder_sections:
        text = section.get_text().strip()
        if text and len(text) > 40 and len(text) < 500:
            # Check if it contains founding-related language
            if any(keyword in text.lower() for keyword in ['founded', 'started', 'began', 'vision', 'dream', 'idea']):
                add_candidate(candidates, 'founding_story', text, 0.6, source_url)


def add_candidate(candidates: dict, field: str, value: str, score: float, source: str):
    """Add a candidate value to the candidates dict."""
    if field not in candidates:
        candidates[field] = []
    
    # Clean and validate value
    cleaned_value = value.strip()
    
    # Fix URL formatting for URL fields (prevent double https://)
    if field in ['website', 'logo_url'] or field.startswith('socials.'):
        if cleaned_value:
            # Skip if already has protocol
            if cleaned_value.startswith(("http://", "https://", "mailto:")):
                pass  # Keep as is
            elif cleaned_value.startswith("//"):
                cleaned_value = f"https:{cleaned_value}"
            elif "." in cleaned_value:  # Looks like a domain
                cleaned_value = f"https://{cleaned_value}"
    
    if cleaned_value and len(cleaned_value) < 500:  # Reasonable length limit
        candidates[field].append({
            "value": cleaned_value,
            "score": score,
            "source": source
        })


async def reconcile_with_openai(company_name: str, official_email: str, candidates: dict) -> dict:
    """Use OpenAI to select the best values from candidates."""
    try:
        # Prepare candidates for AI (top 3 per field)
        ai_candidates = {}
        for field, field_candidates in candidates.items():
            if field_candidates:
                # Sort by score and take top 3
                sorted_candidates = sorted(field_candidates, key=lambda x: x['score'], reverse=True)[:3]
                ai_candidates[field] = sorted_candidates
        
        # Create enhanced prompt for target market analysis
        prompt = f"""
Company: {company_name}
Email: {official_email}

I have extracted the following candidate values for company profile fields. Please analyze this data and provide comprehensive insights about the company's target market and niche.

Extracted Data:
"""
        
        for field, field_candidates in ai_candidates.items():
            prompt += f"\n{field}:\n"
            for i, candidate in enumerate(field_candidates, 1):
                prompt += f"  {i}. {candidate['value']} (confidence: {candidate['score']:.2f})\n"
        
        prompt += """

SPECIAL FOCUS: Complete Business Intelligence Analysis
Please analyze and provide insights for:

1. TARGET MARKET & NICHE:
   - WHO is their target market (analyze client types, services, positioning)
   - WHAT is their niche or specialization
   - WHAT services they offer
   - If client data is available, analyze what these clients have in common

2. MISSION & FOUNDING STORY:
   - WHY did they start this business? (founding motivation)
   - WHAT is their mission statement or purpose?
   - HOW did the company begin? (founding story)
   - WHAT are their core values?

3. LOCATION INTELLIGENCE:
   - WHERE are they located? (country and city)
   - Look for headquarters, office locations, or address information

For mission_statement: Extract or infer their core mission and purpose
For founding_story: Identify how and when the company was started
For why_started: Determine the original motivation for starting the business
For company_values: Identify their core principles and beliefs
For target_market: Specific customer segments they serve
For niche: Their unique market position and specialization

Return comprehensive, insightful analysis in the JSON format.
"""
        
        # Call OpenAI with function calling
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert at analyzing company data and selecting the most accurate information from multiple sources."},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "select_company_data",
                "description": "Select the best values for company profile fields",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "website": {"type": "string", "description": "Company website URL"},
                        "hq_address": {"type": "string", "description": "Headquarters address"},
                        "phone": {"type": "string", "description": "Phone number"},
                        "industry": {"type": "string", "description": "Industry sector"},
                        "description": {"type": "string", "description": "Company description"},
                        "year_founded": {"type": "integer", "description": "Year company was founded"},
                        "employee_count": {"type": "string", "description": "Number of employees"},
                        "logo_url": {"type": "string", "description": "Logo image URL"},
                        "country": {"type": "string", "description": "Country"},
                        "city": {"type": "string", "description": "City"},
                        "target_market": {"type": "string", "description": "Target market and customer segments they serve"},
                        "niche": {"type": "string", "description": "Company's unique niche or specialization"},
                        "services_offered": {"type": "string", "description": "Main services or products offered"},
                        "client_types": {"type": "string", "description": "Types of clients they serve and what they have in common"},
                        "mission_statement": {"type": "string", "description": "Company's mission statement or purpose"},
                        "founding_story": {"type": "string", "description": "Story of how and why the company was founded"},
                        "why_started": {"type": "string", "description": "The reason why the founders started this business"},
                        "company_values": {"type": "string", "description": "Core company values and principles"},
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
        
        logger.info(f"🤖 OpenAI selected: {selected_data}")
        return selected_data
        
    except Exception as e:
        logger.error(f"OpenAI reconciliation failed: {e}")
        return select_best_candidates(candidates)


def select_best_candidates(candidates: dict) -> dict:
    """Deterministic selection of best candidates with proper social media handling."""
    result = {}
    
    for field, field_candidates in candidates.items():
        if field_candidates:
            # Sort by score and take the best
            best = max(field_candidates, key=lambda x: x['score'])
            
            if field.startswith('socials.'):
                social_key = field.split('.')[1]
                if 'socials' not in result:
                    result['socials'] = {}
                result['socials'][social_key] = best['value']
                logger.info(f"📱 Adding social media: {social_key} = {best['value']}")
            else:
                # Convert year_founded to int
                if field == 'year_founded' and str(best['value']).isdigit():
                    result[field] = int(best['value'])
                else:
                    result[field] = best['value']
    
    # Log final social media for debugging
    if result.get('socials'):
        logger.info(f"📱 Final social media in result: {result['socials']}")
    
    return result


def fix_all_urls(profile_data: dict) -> dict:
    """Ensure all URLs in profile data have proper protocols."""
    fixed_data = profile_data.copy()
    
    # Fix main URL fields
    url_fields = ['website', 'logo_url']
    for field in url_fields:
        if fixed_data.get(field):
            url = str(fixed_data[field]).strip()
            # Skip if already has protocol
            if url.startswith(("http://", "https://", "mailto:")):
                fixed_data[field] = url
            elif url and "." in url:
                if url.startswith("//"):
                    fixed_data[field] = f"https:{url}"
                elif url.startswith("www."):
                    fixed_data[field] = f"https://{url}"
                elif "." in url:
                    fixed_data[field] = f"https://{url}"
                else:
                    # If it doesn't look like a URL, remove it
                    fixed_data[field] = None
    
    # Fix social URLs
    if fixed_data.get("socials"):
        socials = fixed_data["socials"].copy()
        for platform, url in socials.items():
            if url:
                url = str(url).strip()
                if not url.startswith(("http://", "https://")):
                    if url.startswith("//"):
                        socials[platform] = f"https:{url}"
                    elif "." in url:
                        socials[platform] = f"https://{url}"
                    else:
                        # If it doesn't look like a URL, remove it
                        socials[platform] = None
        fixed_data["socials"] = socials
    
    return fixed_data


def check_missing_fields(profile: dict) -> list:
    """Check which critical fields are missing from the profile."""
    critical_fields = [
        'industry', 'hq_address', 'mission_statement', 'why_started', 
        'founding_story', 'company_values', 'target_market', 'niche'
    ]
    
    missing = []
    for field in critical_fields:
        if not profile.get(field) or profile.get(field) == "":
            missing.append(field)
    
    return missing


async def search_google_for_missing_data(company_name: str, missing_fields: list) -> dict:
    """Search Google for missing company information."""
    try:
        search_results = {}
        
        # Create specific search queries for missing fields
        search_queries = {
            'industry': f'"{company_name}" industry sector business type',
            'hq_address': f'"{company_name}" headquarters address location office',
            'mission_statement': f'"{company_name}" mission statement purpose vision',
            'why_started': f'"{company_name}" founded why started founding story',
            'founding_story': f'"{company_name}" company history founding story founded',
            'company_values': f'"{company_name}" company values principles culture',
            'target_market': f'"{company_name}" target market customers clients serves',
            'niche': f'"{company_name}" specialization niche market position'
        }
        
        async with httpx.AsyncClient(timeout=15) as client:
            for field in missing_fields:
                if field in search_queries:
                    query = search_queries[field]
                    logger.info(f"🔍 Searching Google for {field}: {query}")
                    
                    # Use DuckDuckGo search (no API key required)
                    search_url = f"https://duckduckgo.com/html/?q={query.replace(' ', '+')}"
                    
                    try:
                        headers = {
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
                        }
                        response = await client.get(search_url, headers=headers)
                        
                        if response.status_code == 200:
                            soup = BeautifulSoup(response.text, 'html.parser')
                            
                            # Extract search result snippets
                            results = []
                            result_divs = soup.find_all('div', class_=re.compile(r'result|snippet', re.I))
                            
                            for div in result_divs[:3]:  # Top 3 results
                                snippet = div.get_text().strip()
                                if snippet and len(snippet) > 30:
                                    results.append(snippet[:500])  # Limit length
                            
                            if results:
                                search_results[field] = results
                                logger.info(f"✅ Found {len(results)} results for {field}")
                            
                    except Exception as e:
                        logger.warning(f"Failed to search for {field}: {e}")
                        
                    # Add small delay between searches
                    await asyncio.sleep(1)
        
        return search_results
        
    except Exception as e:
        logger.error(f"Google search failed: {e}")
        return {}


async def analyze_google_results_with_ai(company_name: str, official_email: str, current_profile: dict, google_data: dict) -> dict:
    """Use OpenAI to analyze Google search results and fill missing fields."""
    try:
        # Create prompt with Google search results
        prompt = f"""
Company: {company_name}
Email: {official_email}

Current Profile Data:
{json.dumps(current_profile, indent=2)}

I searched Google for missing information and found these results:

"""
        
        for field, results in google_data.items():
            prompt += f"\n{field.upper()} - Google Search Results:\n"
            for i, result in enumerate(results, 1):
                prompt += f"  Result {i}: {result}\n"
        
        prompt += """

Please analyze these Google search results and provide the missing information for the company profile.

SPECIFIC INSTRUCTIONS:
- For industry: Determine the specific industry/sector from search results
- For hq_address: Extract the headquarters address if found
- For mission_statement: Identify or infer the company's mission and purpose
- For why_started: Determine why the founders started this business
- For founding_story: Extract the founding story and company history
- For company_values: Identify core values and principles
- For target_market: Determine who their customers are
- For niche: Identify their market specialization

Only include fields where you found reliable information from the search results.
Return ONLY the missing fields that you can confidently determine from the Google results.
"""
        
        # Call OpenAI to analyze Google results
        response = await openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert business analyst who can extract company information from web search results and provide accurate business intelligence."},
                {"role": "user", "content": prompt}
            ],
            functions=[{
                "name": "analyze_missing_company_data",
                "description": "Analyze Google search results to fill missing company profile fields",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "industry": {"type": "string", "description": "Industry sector"},
                        "hq_address": {"type": "string", "description": "Headquarters address"},
                        "mission_statement": {"type": "string", "description": "Company mission statement"},
                        "why_started": {"type": "string", "description": "Why the business was started"},
                        "founding_story": {"type": "string", "description": "Company founding story"},
                        "company_values": {"type": "string", "description": "Core company values"},
                        "target_market": {"type": "string", "description": "Target market"},
                        "niche": {"type": "string", "description": "Company niche"}
                    }
                }
            }],
            function_call={"name": "analyze_missing_company_data"}
        )
        
        # Parse AI response
        function_call = response.choices[0].message.function_call
        enhanced_data = json.loads(function_call.arguments)
        
        logger.info(f"🤖 OpenAI enhanced with Google data: {enhanced_data}")
        return enhanced_data
        
    except Exception as e:
        logger.error(f"Failed to analyze Google results: {e}")
        return {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # global sheets_dao, scraper
    
    logger.info("Starting AI Agent Company Data Scraper...")
    
    # Initialize services (commented out for now)
    # sheets_dao = SheetsDAO()
    # await sheets_dao.initialize()
    # scraper = CompanyScraper(sheets_dao)
    
    logger.info("Application started successfully (without Google Sheets)")
    
    yield
    
    logger.info("Shutting down application...")


# Create FastAPI app
app = FastAPI(
    title="AI Agent Company Data Scraper",
    description="Scrape and enrich company data using AI agents",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add rate limiting if available
if HAS_RATE_LIMITING and limiter:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


@app.middleware("http")
async def add_request_id(request: Request, call_next):
    """Add request ID to all requests."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    
    # Add to response headers
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    
    return response


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler."""
    request_id = getattr(request.state, "request_id", "unknown")
    logger.error(f"Unhandled exception in request {request_id}: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error",
            "request_id": request_id
        }
    )


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "healthy", "service": "AI Agent Company Data Scraper"}


@app.get("/debug/jobs")
async def debug_jobs():
    """Debug endpoint to see all jobs in memory."""
    return {"jobs": scraped_jobs, "count": len(scraped_jobs)}

@app.get("/health")
async def health_check():
    """Detailed health check."""
    try:
        # Test Sheets connection (commented out for now)
        # await sheets_dao.health_check()
        
        return {
            "status": "healthy" if DEPENDENCIES_AVAILABLE else "limited",
            "dependencies": DEPENDENCIES_AVAILABLE,
            "services": {
                "api": "running",
                "ai": "enabled" if openai_client else "disabled",
                "dataforseo": "enabled" if settings.dataforseo_base64 else "disabled",
                "note": "AI Agent Company Data Scraper API"
            }
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "error": str(e)
            }
        )


@app.post("/jobs", response_model=JobResponse)
async def create_job(
    request: Request,
    job_request: JobRequest, 
    background_tasks: BackgroundTasks
):
    """Create a new scraping job."""
    try:
        # Start real scraping (without Google Sheets for now)
        import uuid
        from datetime import datetime
        
        job_id = str(uuid.uuid4())
        now = datetime.utcnow()
        
        # Start background scraping task
        background_tasks.add_task(
            scrape_company_simple,
            job_id=job_id,
            company_name=job_request.company_name,
            official_email=str(job_request.official_email),
            domain=job_request.domain,
            competitor_domains=job_request.competitor_domains,
            main_locations=job_request.main_locations
        )
        
        logger.info(f"Real scraping job {job_id} created for company: {job_request.company_name}")
        
        return JobResponse(
            id=job_id,
            status=JobStatus.QUEUED,
            company_name=job_request.company_name,
            official_email=str(job_request.official_email),
            created_at=now,
            updated_at=now,
            error=None
        )
        
        # Original code (commented out):
        # job_id = await sheets_dao.create_job(
        #     company_name=job_request.company_name,
        #     official_email=str(job_request.official_email)
        # )
        # background_tasks.add_task(
        #     scraper.scrape_company,
        #     job_id=job_id,
        #     company_name=job_request.company_name,
        #     official_email=str(job_request.official_email)
        # )
        # job = await sheets_dao.read_job(job_id)
        # return JobResponse(**job)
        
    except Exception as e:
        logger.error(f"Failed to create job: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}", response_model=JobDetailResponse)
async def get_job(job_id: str):
    """Get job status and results."""
    try:
        # Get real job data from in-memory storage
        from datetime import datetime
        
        logger.info(f"Job status request for {job_id}")
        
        job_data = scraped_jobs.get(job_id)
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        # Convert profile to CompanyProfile if available
        profile = None
        if job_data.get("profile"):
            try:
                profile_dict = job_data["profile"].copy()
                logger.info(f"Converting profile to CompanyProfile: {profile_dict}")
                
                # Fix URL validation issues
                if profile_dict.get("website") and not profile_dict["website"].startswith(("http://", "https://")):
                    profile_dict["website"] = f"https://{profile_dict['website']}"
                
                if profile_dict.get("logo_url") and not profile_dict["logo_url"].startswith(("http://", "https://")):
                    profile_dict["logo_url"] = f"https://{profile_dict['logo_url']}"
                
                # Fix social URLs
                if profile_dict.get("socials"):
                    socials = profile_dict["socials"].copy()
                    for platform, url in socials.items():
                        if url and not url.startswith(("http://", "https://")):
                            socials[platform] = f"https://{url}"
                    profile_dict["socials"] = socials
                
                profile = CompanyProfile(**profile_dict)
            except Exception as e:
                logger.warning(f"Failed to create CompanyProfile: {e}")
                # Return the raw profile data if validation fails
                profile = job_data["profile"]
        
        return JobDetailResponse(
            id=job_data["id"],
            status=job_data["status"],
            company_name=job_data["company_name"],
            official_email=job_data["official_email"],
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            error=job_data.get("error"),
            profile=profile,
            candidates={},
            sources={}
        )
        
        # Original code (commented out):
        # job = await sheets_dao.read_job(job_id)
        # if not job:
        #     raise HTTPException(status_code=404, detail="Job not found")
        # profile = None
        # if job["status"] == JobStatus.COMPLETED:
        #     profile = await sheets_dao.read_profile(job_id)
        # candidates = await sheets_dao.read_candidates(job_id)
        # sources = await sheets_dao.read_sources(job_id)
        # return JobDetailResponse(**job, profile=profile, candidates=candidates, sources=sources)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/jobs/{job_id}/finalise", response_model=JobResponse)
async def finalise_job(job_id: str, overrides: ProfileOverrides):
    """Apply manual overrides and finalise the job."""
    try:
        # Mock response for testing (Google Sheets integration commented out)
        from datetime import datetime
        
        logger.info(f"Mock finalise job {job_id} with overrides")
        
        return JobResponse(
            id=job_id,
            status=JobStatus.COMPLETED,
            company_name="Mock Company",
            official_email="test@example.com",
            created_at=datetime.utcnow(),
            updated_at=datetime.utcnow(),
            error=None
        )
        
        # Original code (commented out):
        # profile = await sheets_dao.read_profile(job_id)
        # if not profile:
        #     raise HTTPException(status_code=404, detail="Profile not found")
        # profile_dict = profile.dict() if hasattr(profile, 'dict') else profile
        # override_dict = overrides.dict(exclude_unset=True)
        # for field, value in override_dict.items():
        #     if value is not None:
        #         profile_dict[field] = value
        #         if 'confidence_per_field' not in profile_dict:
        #             profile_dict['confidence_per_field'] = {}
        #         profile_dict['confidence_per_field'][field] = 1.0
        # await sheets_dao.write_profile(job_id, profile_dict, 1.0)
        # await sheets_dao.update_job(job_id, JobStatus.COMPLETED)
        # job = await sheets_dao.read_job(job_id)
        # return JobResponse(**job)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to finalise job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/jobs/{job_id}/export.json", response_model=ExportResponse)
async def export_job(job_id: str):
    """Export final JSON with sources."""
    try:
        # Get real scraped data
        from datetime import datetime
        
        logger.info(f"Export job {job_id}")
        
        job_data = scraped_jobs.get(job_id)
        if not job_data:
            raise HTTPException(status_code=404, detail="Job not found")
        
        if not job_data.get("profile"):
            raise HTTPException(status_code=404, detail="Profile not found")
        
        # Fix URL validation for export
        profile_dict = job_data["profile"].copy()
        if profile_dict.get("website") and not profile_dict["website"].startswith(("http://", "https://")):
            profile_dict["website"] = f"https://{profile_dict['website']}"
        
        if profile_dict.get("logo_url") and not profile_dict["logo_url"].startswith(("http://", "https://")):
            profile_dict["logo_url"] = f"https://{profile_dict['logo_url']}"
        
        if profile_dict.get("socials"):
            socials = profile_dict["socials"].copy()
            for platform, url in socials.items():
                if url and not url.startswith(("http://", "https://")):
                    socials[platform] = f"https://{url}"
            profile_dict["socials"] = socials
        
        profile = CompanyProfile(**profile_dict)
        
        return ExportResponse(
            profile=profile,
            sources={},
            metadata={
                "job_id": job_id,
                "scraped_at": datetime.utcnow().isoformat(),
                "status": job_data["status"],
                "note": "Real scraping data (in-memory storage)"
            }
        )
        
        # Original code (commented out):
        # job = await sheets_dao.read_job(job_id)
        # if not job:
        #     raise HTTPException(status_code=404, detail="Job not found")
        # profile = await sheets_dao.read_profile(job_id)
        # if not profile:
        #     raise HTTPException(status_code=404, detail="Profile not found")
        # sources = await sheets_dao.read_sources(job_id)
        # metadata = {
        #     "job_id": job_id,
        #     "scraped_at": job["updated_at"].isoformat() if isinstance(job["updated_at"], object) else str(job["updated_at"]),
        #     "status": job["status"]
        # }
        # return ExportResponse(profile=profile, sources=sources, metadata=metadata)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to export job {job_id}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        reload=True
    )
