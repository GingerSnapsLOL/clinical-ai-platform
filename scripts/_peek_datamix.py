import json
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/datamix.jsonl"
with open(path, "r", encoding="utf-8", errors="replace") as f:
    for i, line in enumerate(f):
        if i >= 3:
            break
        print("LINE", i, "len", len(line))
        obj = json.loads(line)
        print("keys:", sorted(obj.keys()))
        for k in sorted(obj.keys())[:12]:
            v = obj[k]
            s = str(v)[:300].replace("\n", " ")
            print(f"  {k}: {s}")
        print("---")
