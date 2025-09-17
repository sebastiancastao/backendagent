"""Company name normalization utilities."""

import re
from typing import List
from app.models import NormalizedCompanyName


def normalize_company_name(company_name: str) -> NormalizedCompanyName:
    """Normalize company name for better search results."""
    original = company_name.strip()
    
    # Remove common legal suffixes
    legal_suffixes = [
        r'\b(inc|incorporated)\b\.?',
        r'\b(corp|corporation)\b\.?',
        r'\b(ltd|limited)\b\.?',
        r'\b(llc)\b\.?',
        r'\b(co|company)\b\.?',
        r'\b(plc)\b\.?',
        r'\b(sa|s\.a\.)\b',
        r'\b(gmbh)\b\.?',
        r'\b(ag)\b\.?',
        r'\b(nv|n\.v\.)\b',
        r'\b(bv|b\.v\.)\b',
        r'\b(srl)\b\.?',
        r'\b(spa|s\.p\.a\.)\b',
        r'\b(ab)\b\.?',
        r'\b(as)\b\.?',
        r'\b(oy)\b\.?',
        r'\b(oyj)\b\.?'
    ]
    
    normalized = original.lower()
    
    # Remove legal suffixes
    for suffix_pattern in legal_suffixes:
        normalized = re.sub(suffix_pattern, '', normalized, flags=re.IGNORECASE)
    
    # Clean up punctuation and whitespace
    normalized = re.sub(r'[^\w\s]', ' ', normalized)
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    
    # Generate common aliases
    aliases = _generate_aliases(normalized)
    
    return NormalizedCompanyName(
        original=original,
        normalized=normalized,
        aliases=aliases
    )


def _generate_aliases(normalized: str) -> List[str]:
    """Generate common aliases for a company name."""
    aliases = []
    
    # Add acronym if multiple words
    words = normalized.split()
    if len(words) > 1:
        # Create acronym
        acronym = ''.join(word[0] for word in words if word)
        if len(acronym) > 1:
            aliases.append(acronym)
    
    # Add variations without common words
    common_words = {'the', 'and', 'of', 'for', 'group', 'international', 'global', 'worldwide'}
    filtered_words = [word for word in words if word not in common_words]
    
    if filtered_words != words:
        aliases.append(' '.join(filtered_words))
    
    # Add single word version if multiple words
    if len(words) > 1:
        # Try to find the main brand word (usually the first or longest)
        main_word = max(words, key=len) if words else normalized
        if len(main_word) > 2:
            aliases.append(main_word)
    
    # Remove duplicates and empty strings
    aliases = list(set(alias for alias in aliases if alias and alias != normalized))
    
    return aliases


def clean_extracted_text(text: str) -> str:
    """Clean extracted text for better processing."""
    if not text:
        return ""
    
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove common prefixes/suffixes that don't add value
    text = re.sub(r'^(about\s+|welcome\s+to\s+)', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+(inc|corp|ltd|llc)\.?$', '', text, flags=re.IGNORECASE)
    
    return text.strip()


def extract_domain_variations(domain: str) -> List[str]:
    """Generate domain variations for testing."""
    if not domain:
        return []
    
    # Remove protocol and www
    clean_domain = domain.lower()
    clean_domain = re.sub(r'^https?://', '', clean_domain)
    clean_domain = re.sub(r'^www\.', '', clean_domain)
    clean_domain = clean_domain.split('/')[0]  # Remove path
    
    variations = [
        clean_domain,
        f"www.{clean_domain}",
        f"https://{clean_domain}",
        f"https://www.{clean_domain}"
    ]
    
    # Add common subdomain variations
    base_domain = clean_domain
    if '.' in base_domain:
        name_part = base_domain.split('.')[0]
        domain_part = '.'.join(base_domain.split('.')[1:])
        
        variations.extend([
            f"en.{base_domain}",
            f"www.{name_part}.{domain_part}",
            f"about.{base_domain}",
            f"company.{base_domain}"
        ])
    
    return list(set(variations))




