import re
from typing import List

def clean_text(text: str) -> str:
    text = text or ""
    text = re.sub(r"\s+", " ", text).strip()
    return text

def chunk_text(text: str, chunk_size: int = 2200, overlap: int = 250) -> List[str]:
    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text] if text else []

    chunks = []
    start = 0
    n = len(text)

    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end]

        if end < n:
            last_break = max(chunk.rfind(". "), chunk.rfind("\n"), chunk.rfind("; "))
            if last_break > int(chunk_size * 0.6):
                end = start + last_break + 1
                chunk = text[start:end]

        chunk = clean_text(chunk)
        if chunk:
            chunks.append(chunk)

        if end >= n:
            break

        start = max(end - overlap, start + 1)

    return chunks