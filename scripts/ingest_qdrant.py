#!/usr/bin/env python3
"""Ingest processed dataset into Qdrant."""

from pathlib import Path
import hashlib
import orjson
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

COLLECTION = "clinical_docs"
QDRANT_URL = "http://localhost:6333"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
IN_PATH = Path("data/processed/datamix.jsonl")

def stable_int_id(text: str) -> int:
    return int(hashlib.md5(text.encode("utf-8")).hexdigest()[:12], 16)

def read_jsonl(path: Path):
    with path.open("rb") as f:
        for line in f:
            line = line.strip()
            if line:
                yield orjson.loads(line)

def batched(iterable, batch_size=64):
    batch = []
    for x in iterable:
        batch.append(x)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch

def main():
    if not IN_PATH.exists() or IN_PATH.stat().st_size == 0:
        raise FileNotFoundError(
            f"Missing or empty dataset: {IN_PATH}. Run scripts/make_datamix.py first."
        )

    model = SentenceTransformer(EMBED_MODEL)
    client = QdrantClient(url=QDRANT_URL)

    dim = model.get_sentence_embedding_dimension()

    client.recreate_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )

    records = list(read_jsonl(IN_PATH))

    for batch in tqdm(list(batched(records, 64))):
        texts = [r["text"] for r in batch]
        vectors = model.encode(texts, normalize_embeddings=True).tolist()

        points = []
        for rec, vec in zip(batch, vectors):
            pid = stable_int_id(rec["id"])
            payload = {k: v for k, v in rec.items() if k != "text"}
            # Keep both keys for compatibility across retrieval/eval code paths.
            payload["doc_id"] = rec.get("doc_id") or rec.get("id")
            payload["text"] = rec["text"]

            points.append(
                PointStruct(
                    id=pid,
                    vector=vec,
                    payload=payload,
                )
            )

        client.upsert(collection_name=COLLECTION, points=points)

    print("Done.")

if __name__ == "__main__":
    main()