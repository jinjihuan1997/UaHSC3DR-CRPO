"""Semantic transmission mode definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List


@dataclass(frozen=True)
class SemanticMode:
    mode_id: str
    low_scale: float
    bg_quality: int
    roi_quality: int
    metadata_bits: int = 256


def get_default_modes() -> List[SemanticMode]:
    return [
        SemanticMode("M0", low_scale=0.125, bg_quality=20, roi_quality=80),
        SemanticMode("M1", low_scale=0.125, bg_quality=30, roi_quality=85),
        SemanticMode("M2", low_scale=0.25, bg_quality=30, roi_quality=90),
        SemanticMode("M3", low_scale=0.25, bg_quality=40, roi_quality=95),
        SemanticMode("M4", low_scale=0.50, bg_quality=50, roi_quality=95),
    ]


def sort_modes_by_quality(modes: Iterable[SemanticMode]) -> List[SemanticMode]:
    return sorted(
        modes,
        key=lambda mode: (mode.low_scale, mode.bg_quality, mode.roi_quality),
    )

