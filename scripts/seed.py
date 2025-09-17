#!/usr/bin/env python3
"""Seed script to test the AI Agent Company Data Scraper with sample companies."""

import asyncio
import sys
import os
from pathlib import Path

# Add the app directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.services.sheets_dao import SheetsDAO
from app.services.scraper import CompanyScraper
from app.config import settings
from app.utils.logging import setup_logging, get_logger


# Setup logging
setup_logging()
logger = get_logger(__name__)


SAMPLE_COMPANIES = [
    {
        "company_name": "Stripe Inc.",
        "official_email": "support@stripe.com",
        "description": "Online payment processing platform"
    },
    {
        "company_name": "Shopify Inc.",
        "official_email": "help@shopify.com",
        "description": "E-commerce platform for online stores"
    },
    {
        "company_name": "Zoom Video Communications",
        "official_email": "support@zoom.us",
        "description": "Video conferencing and communications platform"
    }
]


async def seed_companies():
    """Create seed jobs for sample companies."""
    try:
        logger.info("Starting seed script...")
        
        # Initialize services
        sheets_dao = SheetsDAO()
        await sheets_dao.initialize()
        
        scraper = CompanyScraper(sheets_dao)
        
        logger.info(f"Creating {len(SAMPLE_COMPANIES)} sample jobs...")
        
        job_ids = []
        
        for company in SAMPLE_COMPANIES:
            try:
                # Create job
                job_id = await sheets_dao.create_job(
                    company_name=company["company_name"],
                    official_email=company["official_email"]
                )
                
                job_ids.append({
                    "id": job_id,
                    "company": company["company_name"]
                })
                
                logger.info(f"Created job {job_id} for {company['company_name']}")
                
                # Start scraping (this will run in background)
                asyncio.create_task(
                    scraper.scrape_company(
                        job_id=job_id,
                        company_name=company["company_name"],
                        official_email=company["official_email"]
                    )
                )
                
            except Exception as e:
                logger.error(f"Failed to create job for {company['company_name']}: {e}")
        
        logger.info("Seed jobs created successfully!")
        logger.info("Job IDs:")
        for job in job_ids:
            logger.info(f"  {job['company']}: {job['id']}")
        
        logger.info("\nYou can monitor these jobs by:")
        logger.info(f"1. Checking your Google Sheet: https://docs.google.com/spreadsheets/d/{settings.gsheet_id}")
        logger.info("2. Using the API endpoints:")
        for job in job_ids:
            logger.info(f"   GET /jobs/{job['id']}")
        
        # Wait a bit for jobs to start processing
        logger.info("\nWaiting 30 seconds for jobs to start processing...")
        await asyncio.sleep(30)
        
        # Check job statuses
        logger.info("\nJob statuses:")
        for job in job_ids:
            try:
                job_data = await sheets_dao.read_job(job["id"])
                if job_data:
                    logger.info(f"  {job['company']}: {job_data['status']}")
                else:
                    logger.info(f"  {job['company']}: Not found")
            except Exception as e:
                logger.error(f"  {job['company']}: Error reading status - {e}")
        
        logger.info("\nSeed script completed!")
        
    except Exception as e:
        logger.error(f"Seed script failed: {e}", exc_info=True)
        sys.exit(1)


async def cleanup_jobs():
    """Clean up all jobs (for testing purposes)."""
    try:
        logger.info("Cleaning up all jobs...")
        
        sheets_dao = SheetsDAO()
        await sheets_dao.initialize()
        
        # This would require implementing a cleanup method in SheetsDAO
        logger.warning("Cleanup not implemented - manually clear your Google Sheet if needed")
        
    except Exception as e:
        logger.error(f"Cleanup failed: {e}", exc_info=True)


async def test_connection():
    """Test connection to Google Sheets and external services."""
    try:
        logger.info("Testing connections...")
        
        # Test Sheets connection
        sheets_dao = SheetsDAO()
        await sheets_dao.initialize()
        await sheets_dao.health_check()
        logger.info("✓ Google Sheets connection successful")
        
        # Test OpenAI connection (if configured)
        if settings.openai_api_key:
            from openai import AsyncOpenAI
            client = AsyncOpenAI(api_key=settings.openai_api_key)
            try:
                await client.models.list()
                logger.info("✓ OpenAI connection successful")
            except Exception as e:
                logger.warning(f"✗ OpenAI connection failed: {e}")
        else:
            logger.info("- OpenAI not configured (USE_AI=false mode)")
        
        # Test Browserless connection (if configured)
        if settings.browserless_token:
            import httpx
            async with httpx.AsyncClient() as client:
                try:
                    response = await client.get(f"{settings.browserless_url}/")
                    if response.status_code < 400:
                        logger.info("✓ Browserless connection successful")
                    else:
                        logger.warning(f"✗ Browserless returned {response.status_code}")
                except Exception as e:
                    logger.warning(f"✗ Browserless connection failed: {e}")
        else:
            logger.info("- Browserless not configured")
        
        logger.info("Connection tests completed!")
        
    except Exception as e:
        logger.error(f"Connection test failed: {e}", exc_info=True)
        sys.exit(1)


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python seed.py [seed|cleanup|test]")
        print("  seed    - Create sample company jobs")
        print("  cleanup - Clean up all jobs")
        print("  test    - Test connections")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == "seed":
        asyncio.run(seed_companies())
    elif command == "cleanup":
        asyncio.run(cleanup_jobs())
    elif command == "test":
        asyncio.run(test_connection())
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()



