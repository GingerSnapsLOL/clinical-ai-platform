import json
from textwrap import indent

import requests


def pretty_print_ask_response(data: dict) -> None:
    print("\n=== ANSWER ===")
    print(data.get("answer", ""))

    print("\n=== CITATIONS ===")
    citations = data.get("citations") or []
    if not citations:
        print("(none)")
    else:
        for c in citations:
            print(f"- {c.get('source_id')} ({c.get('title')})")

    print("\n=== TOP SOURCES ===")
    sources = data.get("sources") or []
    for s in sources[:3]:
        print(f"\nSource ID: {s.get('source_id')}")
        if s.get("title"):
            print(f"Title: {s['title']}")
        if s.get("score") is not None:
            print(f"Score: {s['score']}")
        snippet = s.get("snippet") or ""
        if snippet:
            print("Snippet:")
            print(indent(snippet.strip(), "  "))

    print("\n=== ENTITIES ===")
    entities = data.get("entities") or []
    if not entities:
        print("(none)")
    else:
        for e in entities:
            print(
                f"- {e.get('type')}: '{e.get('text')}' "
                f"(span {e.get('start')}-{e.get('end')}, conf={e.get('confidence')})"
            )

    print("\n=== RISK ===")
    risk = data.get("risk_block") or data.get("risk")
    if not risk:
        print("(none)")
    else:
        print(f"Label: {risk.get('label')}, score: {risk.get('score')}")
        explanation = risk.get("explanation") or []
        if explanation:
            print("Top contributing factors:")
            for feat in explanation[:5]:
                print(f"- {feat.get('feature')}: {feat.get('contribution')}")


def main() -> None:
    gateway_url = "http://localhost:8000/v1/ask"

    payload = {
        "note_text": "55-year-old patient with long-standing hypertension and type 2 diabetes, "
        "on ACE inhibitor and statin, occasional chest discomfort on exertion.",
        "question": "What is this patient's cardiovascular risk and what monitoring is recommended?",
    }

    print(f"POST {gateway_url}")
    resp = requests.post(gateway_url, json=payload, timeout=60)
    print(f"Status: {resp.status_code}")

    resp.raise_for_status()
    data = resp.json()

    pretty_print_ask_response(data)


if __name__ == "__main__":
    main()

