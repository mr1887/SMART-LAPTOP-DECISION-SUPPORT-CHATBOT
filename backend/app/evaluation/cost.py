"""Utilities for estimating Gemini API token cost during experiments.

Pricing is intentionally centralized here so the benchmark records both token
usage and a reproducible USD estimate. Rates below are the paid-tier standard
prices used for the study on 2026-10-20 (USD per 1M tokens).

They can be overridden without code changes via environment variables:
- GEMINI_COST_<MODEL>_INPUT_PER_1M_USD
- GEMINI_COST_<MODEL>_OUTPUT_PER_1M_USD

Example for gemini-3.8-flash:
GEMINI_COST_GEMINI_3_8_FLASH_INPUT_PER_1M_USD=0.75
GEMINI_COST_GEMINI_3_8_FLASH_OUTPUT_PER_1M_USD=3.75
"""

from __future__ import annotations

import os
import re
from typing import Dict, Optional


# Paid-tier standard text-token pricing used by this project on 2026-10-20.
# Gemini 3.8/3.6 Flash had introductory pricing through 2026-12-31.
DEFAULT_PRICING_USD_PER_1M: Dict[str, Dict[str, float]] = {
    "gemini-3.8-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.6-flash": {"input": 0.75, "output": 3.75},
    "gemini-3.5-flash-lite": {"input": 0.30, "output": 2.50},
}


def _env_model_key(model: str) -> str:
    """Convert model ID to an environment-variable-safe uppercase key."""
    return re.sub(r"[^A-Z0-9]+", "_", str(model).upper()).strip("_")


def get_model_pricing(model: Optional[str]) -> Optional[Dict[str, float]]:
    """Return input/output USD-per-1M pricing for a model.

    Environment overrides take precedence over the built-in study snapshot.
    Returns None when no complete pricing is available.
    """
    if not model:
        return None

    model = str(model)
    defaults = DEFAULT_PRICING_USD_PER_1M.get(model, {})
    env_key = _env_model_key(model)

    input_raw = os.getenv(
        f"GEMINI_COST_{env_key}_INPUT_PER_1M_USD",
        str(defaults.get("input", "")),
    )
    output_raw = os.getenv(
        f"GEMINI_COST_{env_key}_OUTPUT_PER_1M_USD",
        str(defaults.get("output", "")),
    )

    try:
        input_rate = float(input_raw)
        output_rate = float(output_raw)
    except (TypeError, ValueError):
        return None

    if input_rate < 0 or output_rate < 0:
        return None

    return {"input": input_rate, "output": output_rate}


def estimate_gemini_cost_usd(
    model: Optional[str],
    input_tokens: int = 0,
    output_tokens: int = 0,
    thinking_tokens: int = 0,
) -> Optional[float]:
    """Estimate paid-tier Gemini API cost in USD.

    Gemini output pricing includes thinking tokens, so billable output is:
        output_tokens + thinking_tokens

    Returns None if pricing for the model is unavailable.
    """
    pricing = get_model_pricing(model)
    if pricing is None:
        return None

    input_tokens = max(0, int(input_tokens or 0))
    output_tokens = max(0, int(output_tokens or 0))
    thinking_tokens = max(0, int(thinking_tokens or 0))

    cost = (
        input_tokens * pricing["input"]
        + (output_tokens + thinking_tokens) * pricing["output"]
    ) / 1_000_000.0

    return round(cost, 10)
