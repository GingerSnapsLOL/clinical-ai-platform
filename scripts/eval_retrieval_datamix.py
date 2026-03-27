from pathlib import Path
import orjson
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient

COLLECTION = "clinical_docs"
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EVAL_PATH = Path("examples/retrieval_eval.jsonl")

def read_jsonl(path: Path):
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)

def main():
    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(url=QDRANT_URL)

    total = 0
    hit_source = 0
    hit_keyword = 0

    for item in read_jsonl(EVAL_PATH):
        total += 1
        qvec = model.encode(item["query"], normalize_embeddings=True).tolist()

        hits = client.search(
            collection_name=COLLECTION,
            query_vector=qvec,
            limit=5,
        )

        texts = [h.payload.get("text", "") for h in hits]
        sources = [h.payload.get("source", "") for h in hits]

        if item["expected_source"] in sources:
            hit_source += 1

        joined = " ".join(texts).lower()
        if any(kw.lower() in joined for kw in item["expected_keywords"]):
            hit_keyword += 1

    print(f"Queries: {total}")
    print(f"Source hit@5: {hit_source/total:.3f}")
    print(f"Keyword hit@5: {hit_keyword/total:.3f}")

if __name__ == "__main__":
    main()