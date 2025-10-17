# modules/security_checks/__init__.py
"""
Security Check Registry

Auto-discovers and registers all security checks.
This module provides a plugin-based architecture for security checks.
"""

from typing import Dict, Type, List
from modules.security_checks.base import AbstractSecurityCheck
from modules.security_checks.headers import SecurityHeadersCheck
from modules.security_checks.tls import TLSCheck
from modules.security_checks.cookies import CookieSecurityCheck
from modules.security_checks.cors import CORSSecurityCheck
from modules.security_checks.rate_limit import RateLimitCheck

# ✅ Import existing optional checks
try:
    from modules.security_checks.zero_day_detector import ZeroDayDetector
    ZERO_DAY_AVAILABLE = True
except ImportError:
    ZERO_DAY_AVAILABLE = False
    ZeroDayDetector = None

try:
    from modules.security_checks.sql_injection import SQLInjectionCheck
    SQL_INJECTION_AVAILABLE = True
except ImportError:
    SQL_INJECTION_AVAILABLE = False
    SQLInjectionCheck = None

try:
    from modules.security_checks.xss_testing import XSSCheck
    XSS_AVAILABLE = True
except ImportError:
    XSS_AVAILABLE = False
    XSSCheck = None

# ✅ NEW: Import 4 new modules
try:
    from modules.security_checks.csrf_testing import CSRFCheck
    CSRF_AVAILABLE = True
except ImportError:
    CSRF_AVAILABLE = False
    CSRFCheck = None

try:
    from modules.security_checks.api_security import APISecurityCheck
    API_SECURITY_AVAILABLE = True
except ImportError:
    API_SECURITY_AVAILABLE = False
    APISecurityCheck = None

try:
    from modules.security_checks.auth_testing import AuthSecurityCheck
    AUTH_TESTING_AVAILABLE = True
except ImportError:
    AUTH_TESTING_AVAILABLE = False
    AuthSecurityCheck = None

try:
    from modules.security_checks.file_upload import FileUploadSecurityCheck
    FILE_UPLOAD_AVAILABLE = True
except ImportError:
    FILE_UPLOAD_AVAILABLE = False
    FileUploadSecurityCheck = None


# ✅ Plugin registry
CHECK_REGISTRY: Dict[str, Type[AbstractSecurityCheck]] = {
    "headers": SecurityHeadersCheck,
    "cookies": CookieSecurityCheck,
    "tls": TLSCheck,
    "cors": CORSSecurityCheck,
    "rate_limit": RateLimitCheck,
}

# Add optional checks if available
if ZERO_DAY_AVAILABLE:
    CHECK_REGISTRY["zero_day"] = ZeroDayDetector

if SQL_INJECTION_AVAILABLE:
    CHECK_REGISTRY["sql_injection"] = SQLInjectionCheck

if XSS_AVAILABLE:
    CHECK_REGISTRY["xss"] = XSSCheck

# ✅ Add new modules
if CSRF_AVAILABLE:
    CHECK_REGISTRY["csrf"] = CSRFCheck

if API_SECURITY_AVAILABLE:
    CHECK_REGISTRY["api_security"] = APISecurityCheck

if AUTH_TESTING_AVAILABLE:
    CHECK_REGISTRY["auth_testing"] = AuthSecurityCheck

if FILE_UPLOAD_AVAILABLE:
    CHECK_REGISTRY["file_upload"] = FileUploadSecurityCheck


# ✅ Tier presets (UPDATED)
CHECK_TIERS = {
    "basic": [
        "headers", "cookies", "tls"
    ],
    "standard": [
        "headers", "cookies", "tls", "cors", "rate_limit"
    ],
    "advanced": [
        "headers", "cookies", "tls", "cors", "rate_limit",
    ] + (["zero_day"] if ZERO_DAY_AVAILABLE else []),
    "penetration": [
        "headers", "cookies", "tls", "cors", "rate_limit",
    ] + (["zero_day"] if ZERO_DAY_AVAILABLE else []) \
      + (["sql_injection"] if SQL_INJECTION_AVAILABLE else []) \
      + (["xss"] if XSS_AVAILABLE else []) \
      + (["csrf"] if CSRF_AVAILABLE else []),
    "comprehensive": [  # ✅ NEW: Full security audit tier
        "headers", "cookies", "tls", "cors", "rate_limit",
    ] + (["zero_day"] if ZERO_DAY_AVAILABLE else []) \
      + (["sql_injection"] if SQL_INJECTION_AVAILABLE else []) \
      + (["xss"] if XSS_AVAILABLE else []) \
      + (["csrf"] if CSRF_AVAILABLE else []) \
      + (["api_security"] if API_SECURITY_AVAILABLE else []) \
      + (["auth_testing"] if AUTH_TESTING_AVAILABLE else []) \
      + (["file_upload"] if FILE_UPLOAD_AVAILABLE else []),
    "all": list(CHECK_REGISTRY.keys()),
}


def get_enabled_checks(enabled: List[str]) -> List[Type[AbstractSecurityCheck]]:
    """
    Get list of enabled check classes.
    
    Args:
        enabled: List of check names to enable
    
    Returns:
        List of check classes
    """
    return [CHECK_REGISTRY[name] for name in enabled if name in CHECK_REGISTRY]


def get_checks_by_tier(tier: str = "standard") -> List[str]:
    """
    Get check names by tier.
    
    Args:
        tier: "basic", "standard", "advanced", "penetration", "comprehensive", "all"
    
    Returns:
        List of check names
    """
    return CHECK_TIERS.get(tier, CHECK_TIERS["standard"])


def get_all_check_names() -> List[str]:
    """Get list of all available check names."""
    return list(CHECK_REGISTRY.keys())


def validate_check_names(check_names: List[str]) -> tuple[List[str], List[str]]:
    """
    Validate check names against registry.
    
    Returns:
        Tuple of (valid_names, invalid_names)
    """
    valid = []
    invalid = []
    
    for name in check_names:
        if name in CHECK_REGISTRY:
            valid.append(name)
        else:
            invalid.append(name)
    
    return valid, invalid


__all__ = [
    "AbstractSecurityCheck",
    "CHECK_REGISTRY",
    "CHECK_TIERS",
    "get_enabled_checks",
    "get_checks_by_tier",
    "get_all_check_names",
    "validate_check_names",
    # Export check classes
    "SecurityHeadersCheck",
    "CookieSecurityCheck",
    "TLSCheck",
    "CORSSecurityCheck",
    "RateLimitCheck",
]

# Add optional classes to __all__ if available
if ZERO_DAY_AVAILABLE:
    __all__.append("ZeroDayDetector")
if SQL_INJECTION_AVAILABLE:
    __all__.append("SQLInjectionCheck")
if XSS_AVAILABLE:
    __all__.append("XSSCheck")
if CSRF_AVAILABLE:
    __all__.append("CSRFCheck")
if API_SECURITY_AVAILABLE:
    __all__.append("APISecurityCheck")
if AUTH_TESTING_AVAILABLE:
    __all__.append("AuthSecurityCheck")
if FILE_UPLOAD_AVAILABLE:
    __all__.append("FileUploadSecurityCheck")
