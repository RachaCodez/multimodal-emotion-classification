"""
Security utilities for input sanitization and validation.
"""

import re
from typing import Optional


def sanitize_text(text: str, max_length: int = 10000) -> str:
    """
    Sanitize text input by removing potentially dangerous content.

    Args:
        text: Input text to sanitize
        max_length: Maximum allowed length

    Returns:
        Sanitized text string
    """
    if not text:
        return ""

    # Truncate to max length
    text = text[:max_length]

    # Remove null bytes
    text = text.replace('\x00', '')

    # Remove control characters except newlines and tabs
    text = ''.join(char for char in text if char.isprintable() or char in '\n\r\t')

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename to prevent directory traversal attacks.

    Args:
        filename: Original filename

    Returns:
        Sanitized filename
    """
    if not filename:
        return "unknown"

    # Remove path components
    filename = filename.split('/')[-1].split('\\')[-1]

    # Remove dangerous characters
    filename = re.sub(r'[^\w\s\-\.]', '', filename)

    # Remove leading dots (hidden files)
    filename = filename.lstrip('.')

    # Limit length
    if len(filename) > 255:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:250] + ('.' + ext if ext else '')

    return filename or "unknown"


def validate_email(email: str) -> bool:
    """
    Validate email format.

    Args:
        email: Email address to validate

    Returns:
        True if valid, False otherwise
    """
    if not email:
        return False

    # Basic email regex
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return bool(re.match(pattern, email)) and len(email) <= 254


def validate_username(username: str) -> tuple[bool, Optional[str]]:
    """
    Validate username format.

    Args:
        username: Username to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not username:
        return False, "Username cannot be empty"

    if len(username) < 3:
        return False, "Username must be at least 3 characters"

    if len(username) > 50:
        return False, "Username must be less than 50 characters"

    if not re.match(r'^[a-zA-Z0-9_-]+$', username):
        return False, "Username can only contain letters, numbers, underscores, and hyphens"

    return True, None


def validate_password(password: str) -> tuple[bool, Optional[str]]:
    """
    Validate password strength.

    Args:
        password: Password to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not password:
        return False, "Password cannot be empty"

    if len(password) < 8:
        return False, "Password must be at least 8 characters"

    if len(password) > 128:
        return False, "Password must be less than 128 characters"

    # Check for at least one letter and one number
    has_letter = any(c.isalpha() for c in password)
    has_number = any(c.isdigit() for c in password)

    if not (has_letter and has_number):
        return False, "Password must contain at least one letter and one number"

    return True, None


def is_safe_redirect_url(url: str, allowed_hosts: Optional[list] = None) -> bool:
    """
    Check if a redirect URL is safe (prevent open redirect vulnerabilities).

    Args:
        url: URL to check
        allowed_hosts: List of allowed hostnames

    Returns:
        True if safe, False otherwise
    """
    if not url:
        return False

    # Only allow relative URLs or URLs from allowed hosts
    if url.startswith('/'):
        # Prevent protocol-relative URLs
        if url.startswith('//'):
            return False
        return True

    # If absolute URL, check against allowed hosts
    if allowed_hosts:
        from urllib.parse import urlparse
        parsed = urlparse(url)
        return parsed.netloc in allowed_hosts

    # By default, don't allow absolute URLs
    return False
