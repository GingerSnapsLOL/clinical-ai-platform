#!/usr/bin/env python3
"""
M1 demo: call gateway-api /v1/ask and print the aggregated JSON response.

Usage:
  python scripts/demo_m1.py
  python scripts/demo_m1.py --payload examples/ask_request.json
  python scripts/demo_m1.py --url http://localhost:8000
"""
import argparse
import json
import sys
from pathlib import Path

try:
    import httpx
except ImportError:
    print("Error: httpx required. Run: pip install httpx")
    sys.exit(1)

DEFAULT_URL = "http://localhost:8000"
DEFAULT_PAYLOAD = {
    "mode": "strict",
    "note_text": "55-year-old male with hypertension. BP 148/92.",
    "question": "What is the cardiovascular risk profile?",
    "user_context": {"lang": "en"},
}


def main() -> int:
    parser = argparse.ArgumentParser(description="Demo M1: call gateway-api /v1/ask")
    parser.add_argument(
        "--url",
        default=DEFAULT_URL,
        help=f"Gateway API base URL (default: {DEFAULT_URL})",
    )
    parser.add_argument(
        "--payload",
        type=Path,
        help="Path to JSON payload (default: built-in)",
    )
    args = parser.parse_args()

    if args.payload:
        if not args.payload.exists():
            print(f"Error: payload file not found: {args.payload}")
            return 1
        with open(args.payload) as f:
            payload = json.load(f)
    else:
        payload = DEFAULT_PAYLOAD

    url = f"{args.url.rstrip('/')}/v1/ask"
    print(f"POST {url}")
    print(json.dumps(payload, indent=2), "\n")

    try:
        r = httpx.post(url, json=payload, timeout=30.0)
        r.raise_for_status()
        data = r.json()
        print("Response:")
        print(json.dumps(data, indent=2, default=str))
        return 0
    except httpx.HTTPStatusError as e:
        print(f"HTTP {e.response.status_code}: {e.response.text}")
        return 1
    except httpx.RequestError as e:
        print(f"Request failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
