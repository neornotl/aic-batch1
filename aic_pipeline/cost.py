"""Token and cost estimator for local/Terra/Sonnet/Sol query-time calls."""

from __future__ import annotations

import argparse
import json


MODEL_PRICES = {
    "terra": {"input": 0.30, "output": 1.50},
    "sonnet5": {"input": 0.25, "output": 1.50},
    "sol": {"input": 0.25, "output": 1.50},
    "gpt56-sol": {"input": 0.25, "output": 1.50},
    "luna": {"input": 0.03, "output": 0.15},  # approximate for Pateway Luna
}


def estimate(query_count: int, candidates: int = 10, input_tokens_per_candidate: int = 180, output_tokens: int = 80,
             input_price_per_million: float = 0.30, output_price_per_million: float = 1.50, calls_per_query: int = 1) -> dict:
    input_tokens = query_count * calls_per_query * (120 + candidates * input_tokens_per_candidate)
    output_total = query_count * calls_per_query * output_tokens
    return {
        "queries": query_count,
        "calls": query_count * calls_per_query,
        "input_tokens": input_tokens,
        "output_tokens": output_total,
        "input_cost": input_tokens / 1_000_000 * input_price_per_million,
        "output_cost": output_total / 1_000_000 * output_price_per_million,
        "total_cost": input_tokens / 1_000_000 * input_price_per_million + output_total / 1_000_000 * output_price_per_million,
    }


def estimate_by_model(query_count: int, candidates: int = 10, model: str = "terra", calls_per_query: int = 1) -> dict:
    prices = MODEL_PRICES.get(model.lower(), MODEL_PRICES["terra"])
    return estimate(query_count, candidates, input_price_per_million=prices["input"],
                    output_price_per_million=prices["output"], calls_per_query=calls_per_query)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", type=int)
    parser.add_argument("--candidates", type=int, default=10)
    parser.add_argument("--calls-per-query", type=int, default=1)
    parser.add_argument("--model", choices=list(MODEL_PRICES.keys()), default="terra")
    parser.add_argument("--input-price", type=float, default=0.30)
    parser.add_argument("--output-price", type=float, default=1.50)
    args = parser.parse_args()
    if args.model in MODEL_PRICES:
        result = estimate_by_model(args.queries, args.candidates, args.model, args.calls_per_query)
    else:
        result = estimate(args.queries, args.candidates, input_price_per_million=args.input_price,
                           output_price_per_million=args.output_price, calls_per_query=args.calls_per_query)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()