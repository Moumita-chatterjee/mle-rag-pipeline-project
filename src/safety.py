import re
from pathlib import Path

from src.ingestion import Chunk, load_chunks

INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(prior|previous|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(prior|previous|above)\s+instructions", re.IGNORECASE),
    re.compile(r"reveal\s+(the\s+)?(system\s+prompt|api\s+key)", re.IGNORECASE),
]


def is_suspicious(text: str) -> bool:
    """True when the text contains an instruction aimed at the assistant."""
    return any(pattern.search(text) for pattern in INJECTION_PATTERNS)


def filter_evidence(
    evidence: list[tuple[float, Chunk]],
) -> tuple[list[tuple[float, Chunk]], list[str]]:
    """Split evidence into safe chunks and the IDs of flagged chunks."""
    safe = []
    flagged = []
    for score, chunk in evidence:
        if is_suspicious(chunk.text):
            flagged.append(chunk.chunk_id)
        else:
            safe.append((score, chunk))
    return safe, flagged


if __name__ == "__main__":
    chunks = load_chunks(Path("data/knowledge_base"))
    for chunk in chunks:
        if is_suspicious(chunk.text):
            print("FLAGGED:", chunk.chunk_id)