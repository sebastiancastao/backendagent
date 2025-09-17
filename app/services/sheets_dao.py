"""Google Sheets Data Access Object."""

import uuid
import asyncio
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

import gspread
from google.oauth2.service_account import Credentials

from app.config import settings
from app.models import JobStatus, CompanyProfile, Candidate, Source
from app.utils.logging import get_logger


logger = get_logger(__name__)


class SheetsDAO:
    """Data Access Object for Google Sheets operations."""
    
    def __init__(self):
        self.client = None
        self.spreadsheet = None
        self._initialized = False
    
    async def initialize(self):
        """Initialize the Google Sheets client."""
        try:
            # Setup credentials
            scopes = [
                'https://www.googleapis.com/auth/spreadsheets',
                'https://www.googleapis.com/auth/drive'
            ]
            
            credentials = Credentials.from_service_account_info(
                settings.google_service_account_info,
                scopes=scopes
            )
            
            # Create client
            self.client = gspread.authorize(credentials)
            
            # Open spreadsheet
            self.spreadsheet = self.client.open_by_key(settings.gsheet_id)
            
            # Initialize sheets structure
            await self._initialize_sheets()
            
            self._initialized = True
            logger.info("Google Sheets client initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Google Sheets client: {e}")
            raise
    
    async def _initialize_sheets(self):
        """Initialize required sheet tabs if they don't exist."""
        try:
            # Define required sheets with headers
            required_sheets = {
                'Jobs': [
                    'job_id', 'company_name', 'official_email', 'status', 
                    'created_at', 'updated_at', 'error'
                ],
                'Profiles': [
                    'job_id', 'company_name', 'official_email', 'website', 
                    'hq_address', 'phone', 'industry', 'description', 
                    'year_founded', 'employee_count', 'logo_url', 'country', 
                    'city', 'social_linkedin', 'social_twitter', 'social_facebook', 
                    'social_instagram', 'social_youtube', 'confidence_overall'
                ],
                'Candidates': [
                    'job_id', 'field', 'value', 'score', 'rank'
                ],
                'Sources': [
                    'job_id', 'field', 'url', 'snippet', 'score'
                ]
            }
            
            existing_sheets = {ws.title for ws in self.spreadsheet.worksheets()}
            
            for sheet_name, headers in required_sheets.items():
                if sheet_name not in existing_sheets:
                    logger.info(f"Creating sheet: {sheet_name}")
                    worksheet = self.spreadsheet.add_worksheet(
                        title=sheet_name, 
                        rows=1000, 
                        cols=len(headers)
                    )
                    # Add headers
                    worksheet.append_row(headers)
                else:
                    # Verify headers exist
                    worksheet = self.spreadsheet.worksheet(sheet_name)
                    existing_headers = worksheet.row_values(1)
                    if not existing_headers or existing_headers != headers:
                        logger.info(f"Updating headers for sheet: {sheet_name}")
                        worksheet.clear()
                        worksheet.append_row(headers)
            
        except Exception as e:
            logger.error(f"Failed to initialize sheets structure: {e}")
            raise
    
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception)
    )
    async def _execute_sheets_operation(self, operation):
        """Execute a sheets operation with retry logic."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, operation)
    
    async def health_check(self) -> bool:
        """Check if Sheets connection is healthy."""
        try:
            if not self._initialized:
                await self.initialize()
            
            # Simple operation to test connection
            await self._execute_sheets_operation(
                lambda: self.spreadsheet.worksheet('Jobs').get('A1')
            )
            return True
            
        except Exception as e:
            logger.error(f"Sheets health check failed: {e}")
            raise
    
    async def create_job(self, company_name: str, official_email: str) -> str:
        """Create a new scraping job."""
        job_id = str(uuid.uuid4())
        now = datetime.utcnow().isoformat()
        
        try:
            worksheet = self.spreadsheet.worksheet('Jobs')
            
            row_data = [
                job_id,
                company_name,
                official_email,
                JobStatus.QUEUED,
                now,
                now,
                ''  # error
            ]
            
            await self._execute_sheets_operation(
                lambda: worksheet.append_row(row_data)
            )
            
            logger.info(f"Created job {job_id} for {company_name}")
            return job_id
            
        except Exception as e:
            logger.error(f"Failed to create job: {e}")
            raise
    
    async def update_job(self, job_id: str, status: JobStatus, error: Optional[str] = None):
        """Update job status and error."""
        try:
            worksheet = self.spreadsheet.worksheet('Jobs')
            
            # Find the job row
            job_row = await self._find_job_row(job_id)
            if not job_row:
                raise ValueError(f"Job {job_id} not found")
            
            # Update status, timestamp, and error
            now = datetime.utcnow().isoformat()
            updates = [
                {
                    'range': f'D{job_row}',  # status column
                    'values': [[status]]
                },
                {
                    'range': f'F{job_row}',  # updated_at column
                    'values': [[now]]
                }
            ]
            
            if error:
                updates.append({
                    'range': f'G{job_row}',  # error column
                    'values': [[error]]
                })
            
            await self._execute_sheets_operation(
                lambda: worksheet.batch_update(updates)
            )
            
            logger.info(f"Updated job {job_id} status to {status}")
            
        except Exception as e:
            logger.error(f"Failed to update job {job_id}: {e}")
            raise
    
    async def _find_job_row(self, job_id: str) -> Optional[int]:
        """Find the row number for a given job ID."""
        try:
            worksheet = self.spreadsheet.worksheet('Jobs')
            job_ids = await self._execute_sheets_operation(
                lambda: worksheet.col_values(1)  # job_id column
            )
            
            for i, cell_job_id in enumerate(job_ids[1:], start=2):  # Skip header
                if cell_job_id == job_id:
                    return i
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find job row for {job_id}: {e}")
            raise
    
    async def read_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Read job details."""
        try:
            worksheet = self.spreadsheet.worksheet('Jobs')
            job_row = await self._find_job_row(job_id)
            
            if not job_row:
                return None
            
            row_data = await self._execute_sheets_operation(
                lambda: worksheet.row_values(job_row)
            )
            
            if len(row_data) < 6:
                return None
            
            return {
                'id': row_data[0],
                'company_name': row_data[1],
                'official_email': row_data[2],
                'status': row_data[3],
                'created_at': datetime.fromisoformat(row_data[4]) if row_data[4] else None,
                'updated_at': datetime.fromisoformat(row_data[5]) if row_data[5] else None,
                'error': row_data[6] if len(row_data) > 6 and row_data[6] else None
            }
            
        except Exception as e:
            logger.error(f"Failed to read job {job_id}: {e}")
            raise
    
    async def write_profile(self, job_id: str, profile: Dict[str, Any], confidence_overall: float):
        """Write or update company profile."""
        try:
            worksheet = self.spreadsheet.worksheet('Profiles')
            
            # Check if profile exists
            existing_row = await self._find_profile_row(job_id)
            
            # Prepare row data
            socials = profile.get('socials', {})
            row_data = [
                job_id,
                profile.get('company_name', ''),
                profile.get('official_email', ''),
                str(profile.get('website', '')) if profile.get('website') else '',
                profile.get('hq_address', ''),
                profile.get('phone', ''),
                profile.get('industry', ''),
                profile.get('description', ''),
                str(profile.get('year_founded', '')) if profile.get('year_founded') else '',
                profile.get('employee_count', ''),
                str(profile.get('logo_url', '')) if profile.get('logo_url') else '',
                profile.get('country', ''),
                profile.get('city', ''),
                str(socials.get('linkedin', '')) if socials.get('linkedin') else '',
                str(socials.get('twitter', '')) if socials.get('twitter') else '',
                str(socials.get('facebook', '')) if socials.get('facebook') else '',
                str(socials.get('instagram', '')) if socials.get('instagram') else '',
                str(socials.get('youtube', '')) if socials.get('youtube') else '',
                confidence_overall
            ]
            
            if existing_row:
                # Update existing profile
                await self._execute_sheets_operation(
                    lambda: worksheet.update(f'A{existing_row}:S{existing_row}', [row_data])
                )
            else:
                # Add new profile
                await self._execute_sheets_operation(
                    lambda: worksheet.append_row(row_data)
                )
            
            logger.info(f"Wrote profile for job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to write profile for job {job_id}: {e}")
            raise
    
    async def _find_profile_row(self, job_id: str) -> Optional[int]:
        """Find the row number for a profile by job ID."""
        try:
            worksheet = self.spreadsheet.worksheet('Profiles')
            job_ids = await self._execute_sheets_operation(
                lambda: worksheet.col_values(1)  # job_id column
            )
            
            for i, cell_job_id in enumerate(job_ids[1:], start=2):  # Skip header
                if cell_job_id == job_id:
                    return i
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to find profile row for {job_id}: {e}")
            raise
    
    async def read_profile(self, job_id: str) -> Optional[CompanyProfile]:
        """Read company profile."""
        try:
            worksheet = self.spreadsheet.worksheet('Profiles')
            profile_row = await self._find_profile_row(job_id)
            
            if not profile_row:
                return None
            
            row_data = await self._execute_sheets_operation(
                lambda: worksheet.row_values(profile_row)
            )
            
            if len(row_data) < 19:
                return None
            
            # Build profile dict
            profile_data = {
                'company_name': row_data[1],
                'official_email': row_data[2],
                'website': row_data[3] if row_data[3] else None,
                'hq_address': row_data[4] if row_data[4] else None,
                'phone': row_data[5] if row_data[5] else None,
                'industry': row_data[6] if row_data[6] else None,
                'description': row_data[7] if row_data[7] else None,
                'year_founded': int(row_data[8]) if row_data[8] and row_data[8].isdigit() else None,
                'employee_count': row_data[9] if row_data[9] else None,
                'logo_url': row_data[10] if row_data[10] else None,
                'country': row_data[11] if row_data[11] else None,
                'city': row_data[12] if row_data[12] else None,
                'socials': {
                    'linkedin': row_data[13] if row_data[13] else None,
                    'twitter': row_data[14] if row_data[14] else None,
                    'facebook': row_data[15] if row_data[15] else None,
                    'instagram': row_data[16] if row_data[16] else None,
                    'youtube': row_data[17] if row_data[17] else None,
                }
            }
            
            return CompanyProfile(**profile_data)
            
        except Exception as e:
            logger.error(f"Failed to read profile for job {job_id}: {e}")
            raise
    
    async def write_candidates(self, job_id: str, candidates_by_field: Dict[str, List[Tuple[str, float]]]):
        """Write candidate values for each field."""
        try:
            worksheet = self.spreadsheet.worksheet('Candidates')
            
            # Clear existing candidates for this job
            await self._clear_job_data(worksheet, job_id)
            
            # Prepare batch data
            batch_data = []
            for field, candidates in candidates_by_field.items():
                for rank, (value, score) in enumerate(candidates, 1):
                    batch_data.append([job_id, field, value, score, rank])
            
            if batch_data:
                await self._execute_sheets_operation(
                    lambda: worksheet.append_rows(batch_data)
                )
            
            logger.info(f"Wrote {len(batch_data)} candidates for job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to write candidates for job {job_id}: {e}")
            raise
    
    async def write_sources(self, job_id: str, sources_by_field: Dict[str, List[Dict[str, Any]]]):
        """Write source information for each field."""
        try:
            worksheet = self.spreadsheet.worksheet('Sources')
            
            # Clear existing sources for this job
            await self._clear_job_data(worksheet, job_id)
            
            # Prepare batch data
            batch_data = []
            for field, sources in sources_by_field.items():
                for source in sources:
                    batch_data.append([
                        job_id,
                        field,
                        source.get('url', ''),
                        source.get('snippet', ''),
                        source.get('score', 0.0)
                    ])
            
            if batch_data:
                await self._execute_sheets_operation(
                    lambda: worksheet.append_rows(batch_data)
                )
            
            logger.info(f"Wrote {len(batch_data)} sources for job {job_id}")
            
        except Exception as e:
            logger.error(f"Failed to write sources for job {job_id}: {e}")
            raise
    
    async def _clear_job_data(self, worksheet, job_id: str):
        """Clear existing data for a job ID from a worksheet."""
        try:
            # Get all job IDs in column A
            job_ids = await self._execute_sheets_operation(
                lambda: worksheet.col_values(1)
            )
            
            # Find rows to delete (in reverse order to avoid index issues)
            rows_to_delete = []
            for i, cell_job_id in enumerate(job_ids[1:], start=2):  # Skip header
                if cell_job_id == job_id:
                    rows_to_delete.append(i)
            
            # Delete rows in reverse order
            for row in reversed(rows_to_delete):
                await self._execute_sheets_operation(
                    lambda: worksheet.delete_rows(row)
                )
            
        except Exception as e:
            logger.error(f"Failed to clear job data for {job_id}: {e}")
            # Don't raise - this is a cleanup operation
    
    async def read_candidates(self, job_id: str) -> Dict[str, List[Candidate]]:
        """Read candidate values for a job."""
        try:
            worksheet = self.spreadsheet.worksheet('Candidates')
            all_data = await self._execute_sheets_operation(
                lambda: worksheet.get_all_records()
            )
            
            candidates_by_field = {}
            for row in all_data:
                if row['job_id'] == job_id:
                    field = row['field']
                    if field not in candidates_by_field:
                        candidates_by_field[field] = []
                    
                    candidate = Candidate(
                        value=row['value'],
                        score=float(row['score']),
                        rank=int(row['rank'])
                    )
                    candidates_by_field[field].append(candidate)
            
            # Sort by rank
            for field in candidates_by_field:
                candidates_by_field[field].sort(key=lambda x: x.rank)
            
            return candidates_by_field
            
        except Exception as e:
            logger.error(f"Failed to read candidates for job {job_id}: {e}")
            return {}
    
    async def read_sources(self, job_id: str) -> Dict[str, List[Source]]:
        """Read source information for a job."""
        try:
            worksheet = self.spreadsheet.worksheet('Sources')
            all_data = await self._execute_sheets_operation(
                lambda: worksheet.get_all_records()
            )
            
            sources_by_field = {}
            for row in all_data:
                if row['job_id'] == job_id:
                    field = row['field']
                    if field not in sources_by_field:
                        sources_by_field[field] = []
                    
                    source = Source(
                        url=row['url'],
                        snippet=row['snippet'],
                        score=float(row['score'])
                    )
                    sources_by_field[field].append(source)
            
            # Sort by score (highest first)
            for field in sources_by_field:
                sources_by_field[field].sort(key=lambda x: x.score, reverse=True)
            
            return sources_by_field
            
        except Exception as e:
            logger.error(f"Failed to read sources for job {job_id}: {e}")
            return {}




