"""Simple deployment-ready version of the AI Agent Company Data Scraper."""

from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
import uuid
from typing import Optional
from pydantic import BaseModel, EmailStr

# Simple models for deployment
class JobRequest(BaseModel):
    company_name: str
    official_email: EmailStr
    domain: Optional[str] = None

class JobResponse(BaseModel):
    id: str
    status: str
    company_name: str
    official_email: str
    created_at: datetime
    message: str

# Create FastAPI app
app = FastAPI(
    title="AI Agent Company Data Scraper",
    description="AI-powered company data scraping and analysis",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Simple in-memory storage
jobs = {}

@app.get("/")
async def root():
    """Root endpoint."""
    return {"message": "AI Agent Company Data Scraper API", "status": "running"}

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "message": "AI Agent Company Data Scraper API",
        "timestamp": datetime.utcnow().isoformat(),
        "version": "1.0.0"
    }

@app.post("/jobs", response_model=JobResponse)
async def create_job(job_request: JobRequest, background_tasks: BackgroundTasks):
    """Create a new company analysis job."""
    job_id = str(uuid.uuid4())
    
    # Create job entry
    job = {
        "id": job_id,
        "status": "completed",  # For demo, mark as completed immediately
        "company_name": job_request.company_name,
        "official_email": str(job_request.official_email),
        "domain": job_request.domain,
        "created_at": datetime.utcnow(),
        "profile": {
            "company_name": job_request.company_name,
            "official_email": str(job_request.official_email),
            "website": f"https://{job_request.domain}" if job_request.domain else None,
            "industry": "Technology",
            "description": f"{job_request.company_name} is a technology company providing innovative solutions.",
            "main_keywords": ["technology", "solutions", "services"],
            "target_market": "Businesses and enterprises",
            "content_plan_summary": f"Content strategy for {job_request.company_name} will focus on their technology solutions and target market.",
            "message": "This is a demo version. Full AI analysis requires environment setup."
        }
    }
    
    jobs[job_id] = job
    
    return JobResponse(
        id=job_id,
        status="completed",
        company_name=job_request.company_name,
        official_email=str(job_request.official_email),
        created_at=datetime.utcnow(),
        message="Demo analysis completed"
    )

@app.get("/jobs/{job_id}")
async def get_job(job_id: str):
    """Get job details."""
    if job_id not in jobs:
        return JSONResponse(
            status_code=404,
            content={"detail": "Job not found"}
        )
    
    return jobs[job_id]

@app.post("/jobs/{job_id}/finalise")
async def finalise_job(job_id: str, overrides: dict = None):
    """Finalise job with overrides."""
    if job_id not in jobs:
        return JSONResponse(
            status_code=404,
            content={"detail": "Job not found"}
        )
    
    job = jobs[job_id]
    if overrides:
        job["profile"].update(overrides)
    
    return {"message": "Job finalised successfully"}

@app.get("/jobs/{job_id}/export.json")
async def export_job(job_id: str):
    """Export job as JSON."""
    if job_id not in jobs:
        return JSONResponse(
            status_code=404,
            content={"detail": "Job not found"}
        )
    
    return {
        "profile": jobs[job_id]["profile"],
        "metadata": {
            "exported_at": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
