"""Cost-estimation tool for the PickMyStack CostAgent.

Estimates a rough monthly total cost of ownership (TCO) for running an AI
feature: model token spend plus hosting. Prices are approximate USD and move
often; treat the output as relative guidance rather than a quote.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from unchained import tool

# model name -> (USD per 1M input tokens, USD per 1M output tokens)
MODEL_PRICING = {
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "claude-3-5-sonnet": (3.00, 15.00),
    "claude-3-5-haiku": (0.80, 4.00),
    "llama-3.1-8b": (0.0, 0.0),
    "llama-3.1-70b": (0.0, 0.0),
    "mistral": (0.0, 0.0),
    "qwen-2.5": (0.0, 0.0),
}

# hosting option -> (fixed USD per month, human label)
HOSTING = {
    "managed_api": (0.0, "Managed API (pay per token only)"),
    "serverless": (20.0, "Serverless container (scale-to-zero)"),
    "container": (60.0, "Always-on small container"),
    "self_host_gpu": (250.0, "Self-hosted GPU instance"),
}


def _closest_model(name: str) -> str:
    """Match a free-text model name to the closest known pricing key.

    Tries an exact match first, then falls back to the *longest* known key
    that is a substring match either way. Longest-match matters: without it,
    a plain first-match scan would resolve "gpt-4o-mini" to "gpt-4o" (since
    "gpt4o" is a substring of "gpt4omini" and "gpt-4o" happens to be defined
    first in MODEL_PRICING), silently pricing it ~16x too high.
    """
    normalized = (name or "").lower().replace(" ", "").replace("_", "").replace("-", "")
    if not normalized:
        return "gpt-4o-mini"
    for key in MODEL_PRICING:
        if key.replace("-", "") == normalized:
            return key
    contains_input = [k for k in MODEL_PRICING if k.replace("-", "") in normalized]
    if contains_input:
        return max(contains_input, key=len)
    input_contains = [k for k in MODEL_PRICING if normalized in k.replace("-", "")]
    if input_contains:
        return max(input_contains, key=len)
    return "gpt-4o-mini"


@tool
def estimate_cost(
    model: str,
    monthly_requests: int,
    avg_input_tokens: int = 800,
    avg_output_tokens: int = 400,
    hosting: str = "managed_api",
) -> str:
    """Estimate the monthly cost of an AI feature for a given model and volume.

    Returns a breakdown of token spend and hosting so the total cost of
    ownership is clear. Use realistic monthly_requests and token sizes.
    """
    model_key = _closest_model(model)
    in_price, out_price = MODEL_PRICING[model_key]
    hosting_key = hosting if hosting in HOSTING else "managed_api"
    hosting_cost, hosting_label = HOSTING[hosting_key]

    input_tokens = monthly_requests * avg_input_tokens
    output_tokens = monthly_requests * avg_output_tokens
    token_cost = (input_tokens / 1_000_000) * in_price + (output_tokens / 1_000_000) * out_price

    self_hosted = in_price == 0.0 and out_price == 0.0
    if self_hosted and hosting_key == "managed_api":
        # An open-weight model still needs somewhere to run.
        hosting_cost, hosting_label = HOSTING["self_host_gpu"]

    total = token_cost + hosting_cost
    lines = [
        f"Cost estimate for '{model_key}' at {monthly_requests:,} requests/month:",
        f"  Tokens: {input_tokens:,} in + {output_tokens:,} out",
        f"  Token spend: ${token_cost:,.2f}/mo"
        + ("  (open-weight: no token fees)" if self_hosted else ""),
        f"  Hosting: ${hosting_cost:,.2f}/mo  ({hosting_label})",
        f"  ESTIMATED TOTAL: ${total:,.2f}/month  (~${total * 12:,.2f}/year)",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    print(estimate_cost("gpt-4o-mini", 50_000))
    print()
    print(estimate_cost("llama-3.1-8b", 50_000))
