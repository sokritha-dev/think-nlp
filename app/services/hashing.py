# app/utils/file_utils.py

import hashlib

import pandas as pd


def compute_sha256(content: bytes) -> str:
    """Returns SHA256 hash of the file content."""
    return hashlib.sha256(content).hexdigest()


def fast_hash_sample(series: pd.Series, sample_size: int = 10) -> str:
    # Take sample first (less memory), then drop nulls
    sample = series.head(sample_size).dropna()
    combined = "|".join(map(str, sample))
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
