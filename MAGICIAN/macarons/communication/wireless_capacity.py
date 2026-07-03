"""Digital wireless link capacity helpers."""

from __future__ import annotations

import math
from typing import Optional


def capacity_bits_per_slot(
    bandwidth_hz: Optional[float] = None,
    snr_db: Optional[float] = None,
    slot_time_s: Optional[float] = None,
    fixed_capacity_bits: Optional[float] = None,
) -> float:
    if fixed_capacity_bits is not None:
        return float(fixed_capacity_bits)
    if bandwidth_hz is None or snr_db is None or slot_time_s is None:
        raise ValueError(
            "Provide fixed_capacity_bits or all of bandwidth_hz, snr_db, and slot_time_s."
        )
    snr_linear = 10.0 ** (float(snr_db) / 10.0)
    return float(bandwidth_hz) * float(slot_time_s) * math.log2(1.0 + snr_linear)

