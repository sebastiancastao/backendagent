"""Tests for company name normalization utilities."""

import pytest
from app.utils.normalization import normalize_company_name, clean_extracted_text


class TestNormalization:
    """Test company name normalization."""
    
    def test_basic_normalization(self):
        """Test basic company name normalization."""
        result = normalize_company_name("Acme Corporation")
        assert result.original == "Acme Corporation"
        assert result.normalized == "acme"
        assert "acme" in result.aliases or len(result.aliases) == 0
    
    def test_legal_suffix_removal(self):
        """Test removal of legal suffixes."""
        cases = [
            ("Microsoft Corp.", "microsoft"),
            ("Apple Inc.", "apple"),
            ("Google LLC", "google"),
            ("Amazon.com, Inc.", "amazon com"),
            ("Tesla Motors Ltd", "tesla motors"),
        ]
        
        for input_name, expected in cases:
            result = normalize_company_name(input_name)
            assert result.normalized == expected
    
    def test_punctuation_cleanup(self):
        """Test punctuation and whitespace cleanup."""
        result = normalize_company_name("AT&T Inc.")
        assert result.normalized == "at t"
        
        result = normalize_company_name("Johnson & Johnson")
        assert result.normalized == "johnson johnson"
    
    def test_acronym_generation(self):
        """Test acronym generation for multi-word companies."""
        result = normalize_company_name("International Business Machines")
        assert "ibm" in result.aliases
        
        result = normalize_company_name("General Electric Company")
        assert "gec" in result.aliases
    
    def test_single_word_company(self):
        """Test handling of single word companies."""
        result = normalize_company_name("Tesla")
        assert result.normalized == "tesla"
        assert len(result.aliases) == 0  # No acronym for single word
    
    def test_common_word_filtering(self):
        """Test filtering of common words."""
        result = normalize_company_name("The Walt Disney Company")
        # Should generate alias without "the" and "company"
        assert any("walt disney" in alias for alias in result.aliases)


class TestTextCleaning:
    """Test text cleaning utilities."""
    
    def test_whitespace_cleanup(self):
        """Test excessive whitespace removal."""
        text = "This    has   too   much    whitespace"
        result = clean_extracted_text(text)
        assert result == "This has too much whitespace"
    
    def test_prefix_removal(self):
        """Test removal of common prefixes."""
        text = "About Acme Corporation - We are a leading company"
        result = clean_extracted_text(text)
        assert not result.startswith("About")
        
        text = "Welcome to Acme Corp"
        result = clean_extracted_text(text)
        assert not result.startswith("Welcome to")
    
    def test_suffix_removal(self):
        """Test removal of legal suffixes."""
        text = "Acme Corporation Inc."
        result = clean_extracted_text(text)
        assert not result.endswith("Inc.")
    
    def test_empty_text_handling(self):
        """Test handling of empty or None text."""
        assert clean_extracted_text("") == ""
        assert clean_extracted_text(None) == ""
        assert clean_extracted_text("   ") == ""


if __name__ == "__main__":
    pytest.main([__file__])



