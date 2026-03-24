import os
import re
import uuid
from typing import Any, Dict, List

from fastapi import FastAPI
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import CrossEncoder, SentenceTransformer

from services.shared.logging_util import set_trace_id, structured_log_middleware
from services.shared.schemas_v1 import (
    HealthResponse,
    PassageItem,
    RetrieveRequest,
    RetrieveResponse,
)

COLLECTION_NAME = "clinical_docs"
VECTOR_SIZE = 384
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
CHUNK_MIN_TOKENS = 300
CHUNK_MAX_TOKENS = 500

_embed_model: SentenceTransformer | None = None
_rerank_model: CrossEncoder | None = None
_qdrant_client: QdrantClient | None = None


# ---------------------------------------------------------------------------
# Ingest schemas
# ---------------------------------------------------------------------------
class IngestDocument(BaseModel):
    doc_id: str
    text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    documents: List[IngestDocument]


class IngestResponse(BaseModel):
    chunks_inserted: int


def embed_text(text: str) -> list[float]:
    """Embed text with the loaded model. Returns 384-dim vector (CPU only)."""
    global _embed_model
    if _embed_model is None:
        raise RuntimeError("Embedding model not loaded")
    vec = _embed_model.encode(text, convert_to_numpy=True)
    return vec.tolist()


def _get_qdrant() -> QdrantClient:
    global _qdrant_client
    if _qdrant_client is None:
        raise RuntimeError("Qdrant client not initialized")
    return _qdrant_client


def init_qdrant_collection() -> None:
    """On startup: create collection 'clinical_docs' if it does not exist."""
    global _qdrant_client
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", "6333"))
    _qdrant_client = QdrantClient(host=host, port=port)

    if not _qdrant_client.collection_exists(COLLECTION_NAME):
        _qdrant_client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=VECTOR_SIZE, distance=Distance.COSINE),
        )


def _count_tokens(text: str) -> int:
    """Count tokens using model tokenizer if available, else approximate (~4 chars/token)."""
    if _embed_model is None:
        return max(1, len(text) // 4)
    try:
        tok = _embed_model.tokenizer
        return len(tok.encode(text, add_special_tokens=False))
    except AttributeError:
        return max(1, len(text) // 4)


def _chunk_text(text: str, doc_id: str, metadata: Dict[str, Any]) -> List[tuple[int, str, Dict[str, Any]]]:
    """Chunk text into 300–500 token segments. Returns list of (chunk_index, chunk_text, payload)."""
    if not text or not text.strip():
        return []

    sentences = re.split(r"(?<=[.!?])\s+", text)
    if not sentences:
        sentences = [text]

    chunks: List[tuple[int, str, Dict[str, Any]]] = []
    current: List[str] = []
    current_tokens = 0
    chunk_index = 0

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        n = _count_tokens(sent)

        if current_tokens + n > CHUNK_MAX_TOKENS and current:
            chunk_text = " ".join(current)
            payload = {"text": chunk_text, "doc_id": doc_id, **metadata}
            chunks.append((chunk_index, chunk_text, payload))
            chunk_index += 1
            current = []
            current_tokens = 0

        current.append(sent)
        current_tokens += n

        if current_tokens >= CHUNK_MIN_TOKENS:
            chunk_text = " ".join(current)
            payload = {"text": chunk_text, "doc_id": doc_id, **metadata}
            chunks.append((chunk_index, chunk_text, payload))
            chunk_index += 1
            current = []
            current_tokens = 0

    if current:
        chunk_text = " ".join(current)
        payload = {"text": chunk_text, "doc_id": doc_id, **metadata}
        chunks.append((chunk_index, chunk_text, payload))

    return chunks


def _stable_point_uuid(doc_id: str, chunk_index: int) -> str:
    """Return deterministic UUID accepted by Qdrant for point IDs."""
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc_id}::{chunk_index}"))


app = FastAPI(title="Retrieval Service", version="0.1.0")
app.add_middleware(structured_log_middleware("retrieval-service"))


def _load_models() -> None:
    """Load embedding and cross-encoder models once at startup (CPU only)."""
    global _embed_model, _rerank_model
    _embed_model = SentenceTransformer(EMBED_MODEL, device="cpu")
    _rerank_model = CrossEncoder(RERANK_MODEL, device="cpu")


@app.on_event("startup")
async def startup() -> None:
    _load_models()
    init_qdrant_collection()


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(service="retrieval-service")


@app.post("/v1/retrieve", response_model=RetrieveResponse)
async def retrieve(request: RetrieveRequest) -> RetrieveResponse:
    """
    Real retrieval: embed query, search Qdrant, return top_k passages.
    Response passages map to sources (source_id, snippet/text, score, metadata).
    """
    set_trace_id(request.trace_id)

    query_vector = embed_text(request.query)
    client = _get_qdrant()
    hits = client.search(
        collection_name=COLLECTION_NAME,
        query_vector=query_vector,
        limit=request.top_k,
    )

    passages: List[PassageItem] = []
    for h in hits:
        payload = h.payload or {}
        source_id = payload.get("doc_id", str(h.id))
        text = payload.get("text", "")

        # Always expose key metadata fields
        title = payload.get("title") or f"Document {source_id}"
        source = payload.get("source") or "unknown"

        extra_metadata = {k: v for k, v in payload.items() if k not in ("text", "doc_id")}
        metadata = {
            "doc_id": source_id,
            "title": title,
            "source": source,
            **extra_metadata,
        }
        passages.append(
            PassageItem(
                source_id=source_id,
                text=text,
                score=float(h.score) if h.score is not None else 0.0,
                metadata=metadata,
            )
        )

    # Deduplicate by passage text (case-insensitive), keep highest-scoring (order already by score desc)
    seen: set[str] = set()
    unique_passages: List[PassageItem] = []
    for p in passages:
        key = p.text.strip().lower()
        if key not in seen:
            seen.add(key)
            unique_passages.append(p)

    # Optional cross-encoder reranking (query + candidate passages)
    global _rerank_model
    if request.rerank and _rerank_model is not None and unique_passages:
        pairs = [(request.query, p.text) for p in unique_passages]
        scores = _rerank_model.predict(pairs)

        ranked = [
            PassageItem(
                source_id=p.source_id,
                text=p.text,
                score=float(s),
                metadata=p.metadata,
            )
            for p, s in zip(unique_passages, scores)
        ]
        ranked.sort(key=lambda x: x.score, reverse=True)
        reranked_passages = ranked
    else:
        reranked_passages = unique_passages

    # Finally, limit returned passages:
    # - If reranking is enabled, always return top 3 after reranking
    # - Otherwise, respect top_n (fallback to all)
    if request.rerank:
        reranked_passages = reranked_passages[:3]
    else:
        top_n = getattr(request, "top_n", None)
        if isinstance(top_n, int) and top_n > 0:
            reranked_passages = reranked_passages[:top_n]

    return RetrieveResponse(
        trace_id=request.trace_id,
        passages=reranked_passages,
    )


@app.post("/v1/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest) -> IngestResponse:
    """
    Chunk documents (300–500 tokens), compute embeddings, insert into Qdrant.
    Returns number of inserted chunks.
    """
    client = _get_qdrant()
    points: List[PointStruct] = []

    for doc in request.documents:
        for chunk_index, chunk_text, payload in _chunk_text(doc.text, doc.doc_id, doc.metadata):
            # Stable point ID for idempotent ingestion: re-ingesting same doc overwrites
            point_id = _stable_point_uuid(doc.doc_id, chunk_index)
            embedding = embed_text(chunk_text)
            points.append(
                PointStruct(
                    id=point_id,
                    vector=embedding,
                    payload=payload,
                )
            )

    if points:
        client.upsert(collection_name=COLLECTION_NAME, points=points)

    return IngestResponse(chunks_inserted=len(points))


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("RETRIEVAL_SERVICE_PORT", "8040"))
    uvicorn.run("app.main:app", host="0.0.0.0", port=port, reload=True)

