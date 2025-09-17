"""Tests for Pydantic models."""

import pytest
from datetime import datetime
from pydantic import ValidationError
from app.models import (
    CompanyProfile, 
    JobRequest, 
    JobResponse, 
    JobStatus,
    ProfileOverrides
)


class TestCompanyProfile:
    """Test CompanyProfile model."""
    
    def test_valid_profile(self):
        """Test valid profile creation."""
        profile = CompanyProfile(
            company_name="Acme Corp",
            official_email="contact@acme.com",
            website="https://acme.com",
            year_founded=2000,
            socials={"linkedin": "https://linkedin.com/company/acme"}
        )
        
        assert profile.company_name == "Acme Corp"
        assert profile.official_email == "contact@acme.com"
        assert profile.website == "https://acme.com"
        assert profile.year_founded == 2000
        assert profile.socials["linkedin"] == "https://linkedin.com/company/acme"
    
    def test_invalid_email(self):
        """Test invalid email validation."""
        with pytest.raises(ValidationError):
            CompanyProfile(
                company_name="Acme Corp",
                official_email="invalid-email",
            )
    
    def test_invalid_year_founded(self):
        """Test invalid year_founded validation."""
        with pytest.raises(ValidationError):
            CompanyProfile(
                company_name="Acme Corp",
                official_email="contact@acme.com",
                year_founded=1700  # Too old
            )
        
        with pytest.raises(ValidationError):
            CompanyProfile(
                company_name="Acme Corp",
                official_email="contact@acme.com",
                year_founded=2030  # Future year
            )
    
    def test_optional_fields(self):
        """Test that optional fields can be None."""
        profile = CompanyProfile(
            company_name="Acme Corp",
            official_email="contact@acme.com"
        )
        
        assert profile.website is None
        assert profile.hq_address is None
        assert profile.phone is None
        assert profile.socials == {}
        assert profile.confidence_per_field == {}
    
    def test_socials_normalization(self):
        """Test that social keys are normalized to lowercase."""
        profile = CompanyProfile(
            company_name="Acme Corp",
            official_email="contact@acme.com",
            socials={"LinkedIn": "https://linkedin.com/company/acme"}
        )
        
        assert "linkedin" in profile.socials
        assert "LinkedIn" not in profile.socials


class TestJobRequest:
    """Test JobRequest model."""
    
    def test_valid_request(self):
        """Test valid job request."""
        request = JobRequest(
            company_name="Acme Corp",
            official_email="contact@acme.com"
        )
        
        assert request.company_name == "Acme Corp"
        assert request.official_email == "contact@acme.com"
    
    def test_empty_company_name(self):
        """Test empty company name validation."""
        with pytest.raises(ValidationError):
            JobRequest(
                company_name="",
                official_email="contact@acme.com"
            )
    
    def test_long_company_name(self):
        """Test company name length validation."""
        with pytest.raises(ValidationError):
            JobRequest(
                company_name="A" * 201,  # Too long
                official_email="contact@acme.com"
            )


class TestJobResponse:
    """Test JobResponse model."""
    
    def test_valid_response(self):
        """Test valid job response."""
        now = datetime.utcnow()
        response = JobResponse(
            id="test-id",
            status=JobStatus.COMPLETED,
            company_name="Acme Corp",
            official_email="contact@acme.com",
            created_at=now,
            updated_at=now
        )
        
        assert response.id == "test-id"
        assert response.status == JobStatus.COMPLETED
        assert response.error is None


class TestProfileOverrides:
    """Test ProfileOverrides model."""
    
    def test_valid_overrides(self):
        """Test valid profile overrides."""
        overrides = ProfileOverrides(
            website="https://newsite.com",
            industry="Technology",
            year_founded=2010
        )
        
        assert overrides.website == "https://newsite.com"
        assert overrides.industry == "Technology"
        assert overrides.year_founded == 2010
    
    def test_partial_overrides(self):
        """Test that overrides can be partial."""
        overrides = ProfileOverrides(
            website="https://newsite.com"
        )
        
        assert overrides.website == "https://newsite.com"
        assert overrides.industry is None
        assert overrides.year_founded is None
    
    def test_invalid_year_override(self):
        """Test invalid year validation in overrides."""
        with pytest.raises(ValidationError):
            ProfileOverrides(
                year_founded=1700
            )


if __name__ == "__main__":
    pytest.main([__file__])



