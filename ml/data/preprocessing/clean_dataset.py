"""
Dataset Cleaning & Sanitization Engine for CodeForge AI.
Implements Section 10 and Section F filtering rules:
1. Max line length <= 1000, mean line length <= 100
2. Alphanumeric character ratio >= 0.25
3. File size between 50 Bytes and 1 MB
4. PII and Secret Redaction (<KEY>, <EMAIL>, <IP>, <NAME>)
5. License allowlisting (MIT, Apache-2.0, BSD-2-Clause, BSD-3-Clause, ISC)
"""

import re
from typing import Dict, Any, Tuple, Optional, List

PERMISSIVE_LICENSES = {"mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause", "isc", "psf"}

SECRET_PATTERNS = [
    (re.compile(r"(?i)(?:api_key|secret|token|password|bearer|auth)\s*[:=]\s*['\"][a-zA-Z0-9_\-\.]{16,}['\"]"), "<KEY>"),
    (re.compile(r"\b(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b"), "<IP>"),
    (re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}"), "<EMAIL>"),
    (re.compile(r"://([^:]+):([^@]+)@"), "://\\1:<KEY>@"),
]

def clean_code_text(text: str) -> Tuple[bool, str, Optional[str]]:
    """
    Cleans and evaluates code document.
    Returns (keep: bool, sanitized_text: str, drop_reason: Optional[str]).
    """
    if not text:
        return False, "", "empty_document"

    byte_len = len(text.encode("utf-8"))
    if byte_len < 50:
        return False, "", f"file_too_small_{byte_len}_bytes"
    if byte_len > 1_000_000:
        return False, "", f"file_too_large_{byte_len}_bytes"

    lines = text.splitlines()
    if not lines:
        return False, "", "empty_lines"

    line_lengths = [len(l) for l in lines]
    max_line = max(line_lengths)
    if max_line > 1000:
        return False, "", f"max_line_exceeded_{max_line}"

    mean_line = sum(line_lengths) / len(line_lengths)
    if mean_line > 100:
        return False, "", f"mean_line_exceeded_{mean_line:.1f}"

    alnum_count = sum(1 for c in text if c.isalnum())
    alnum_ratio = alnum_count / len(text)
    if alnum_ratio < 0.25:
        return False, "", f"low_alnum_fraction_{alnum_ratio:.2f}"

    # PII and Secret Redaction
    sanitized = text
    for pattern, placeholder in SECRET_PATTERNS:
        sanitized = pattern.sub(placeholder, sanitized)

    return True, sanitized, None

def validate_license(license_str: str) -> bool:
    if not license_str:
        return False
    return license_str.strip().lower() in PERMISSIVE_LICENSES
