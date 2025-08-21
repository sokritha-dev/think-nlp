from __future__ import annotations
from typing import List, Any
import json
import ast


def ensure_token_list(x: Any) -> List[str]:
    # Accept list[str], JSON string, or Python literal string (legacy CSV)
    if isinstance(x, list):
        return [str(t) for t in x]
    if isinstance(x, str):
        try:
            v = json.loads(x)
            if isinstance(v, list):
                return [str(t) for t in v]
        except Exception:
            pass
        try:
            v = ast.literal_eval(x)
            if isinstance(v, list):
                return [str(t) for t in v]
        except Exception:
            return [x]
    return [] if x is None else [str(x)]
