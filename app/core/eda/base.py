from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any
import pandas as pd


class EDAAnalyzer(ABC):
    """Port: produce EDA summaries from a lemmatized DataFrame."""

    @abstractmethod
    def analyze(self, df: pd.DataFrame) -> Dict[str, Any]: ...
