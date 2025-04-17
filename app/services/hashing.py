# app/utils/file_utils.py

import hashlib


def compute_sha256(content: bytes) -> str:
    """Returns SHA256 hash of the file content."""
    return hashlib.sha256(content).hexdigest()
