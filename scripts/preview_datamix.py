#!/usr/bin/env python3
"""Preview records from data/processed/datamix.jsonl."""

from datasets import load_dataset
from collections import Counter
from pathlib import Path

IN_PATH = Path("data/processed/datamix.jsonl")

def main():
    if not IN_PATH.exists() or IN_PATH.stat().st_size == 0:
        print(f"No records to preview: {IN_PATH} is missing or empty.")
        return

    ds = load_dataset("json", data_files={"train": "data/processed/datamix.jsonl"})
    train = ds["train"]

    print("Total:", len(train))
    print("\nSamples:")
    for i in range(min(3, len(train))):
        print(train[i])

    c_source = Counter(train["source"])
    c_type = Counter(train["doc_type"])

    print("\nBy source:", c_source)
    print("By doc_type:", c_type)

if __name__ == "__main__":
    main()