"""Shared analysis constants for the resting-state reliability pipeline."""

from __future__ import annotations

import re
from typing import Iterable


RESTING_TASKS = ("EyesOpenNoTask", "EyesClosedNoTask")

BAND_LIMITS = {
    "delta": (1, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "beta": (13, 30),
    "gamma": (30, 45),
    "full": (1, 45),
}

# The five canonical bands are the only validated end-to-end analysis bands.
# ``full`` remains a known definition so it can be enabled deliberately after
# its S2-S5 behavior has been validated.
CANONICAL_BANDS = ("delta", "theta", "alpha", "beta", "gamma")
SUPPORTED_BANDS = CANONICAL_BANDS
DEFAULT_BANDS = CANONICAL_BANDS
KNOWN_BANDS = tuple(BAND_LIMITS)
SUPPORTED_BAND_PATTERN = "|".join(re.escape(band) for band in SUPPORTED_BANDS)


def get_band_freq(band: str) -> tuple[float, float]:
    """Return frequency limits for a known EEG band."""
    try:
        return BAND_LIMITS[band]
    except KeyError as error:
        supported = ", ".join(KNOWN_BANDS)
        raise ValueError(
            f"Unknown band {band!r}; expected one of: {supported}"
        ) from error


def validate_supported_bands(bands: Iterable[str]) -> None:
    """Fail when a requested band is not enabled for the full pipeline."""
    unsupported = sorted(set(bands) - set(SUPPORTED_BANDS))
    if unsupported:
        enabled = ", ".join(SUPPORTED_BANDS)
        raise ValueError(
            f"Unsupported analysis bands {unsupported}; currently enabled: "
            f"{enabled}"
        )
