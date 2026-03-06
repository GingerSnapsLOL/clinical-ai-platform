#!/usr/bin/env bash
# M1 demo: call gateway-api /v1/ask and print the aggregated JSON response.
# Usage: ./scripts/demo_m1.sh [payload.json]
# Default payload: examples/ask_request.json
set -e

URL="${GATEWAY_URL:-http://localhost:8000}"
PAYLOAD="${1:-examples/ask_request.json}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PAYLOAD_PATH="$ROOT_DIR/$PAYLOAD"

echo "POST $URL/v1/ask"
echo "Payload: $PAYLOAD_PATH"
echo ""
curl -sS -X POST "$URL/v1/ask" \
  -H "Content-Type: application/json" \
  -d @"$PAYLOAD_PATH" | python3 -m json.tool
