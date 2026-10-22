import re

from src.ingestion import Chunk


ABSTAIN_MESSAGE = (
    "I can't answer this from the approved knowledge base. "
    "Please escalate it to a colleague."
)

CITATION_PATTERN = re.compile(r"\[([a-z0-9-]+)\]")

def generate_answer(evidence: list[tuple[float, Chunk]]) -> str:
    """Build an extractive answer: each evidence paragraph followed by its citation."""
    if not evidence:
        return ABSTAIN_MESSAGE
    lines = [f"{chunk.text} [{chunk.document_id}]" for _, chunk in evidence]
    return "\n".join(lines)

def find_citation_problems(answer: str, evidence: list[tuple[float, Chunk]]) -> list[str]:
    """List problems with the answer's citations; an empty list means they are valid."""
    cited = CITATION_PATTERN.findall(answer)
    if not cited:
        return ["answer has no citations"]
    allowed = {chunk.document_id for _, chunk in evidence}
    return [f"[{doc_id}] was not in the retrieved evidence" for doc_id in cited if doc_id not in allowed]

