from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class SpecialCleanConfig:
    remove_special: bool = True
    remove_numbers: bool = True
    remove_emoji: bool = True
