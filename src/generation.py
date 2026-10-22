import re
import sys

from src.embedding import load_model
from src.ingestion import Chunk
from src.retrieval import open_client, retrieve
from src.safety import filter_evidence

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

if __name__ == "__main__":
    question = " ".join(sys.argv[1:]) or "How do I get a refund for a duplicate payment?"
    client = open_client()
    try:
        result = retrieve(client, load_model(), question)
        safe, flagged = filter_evidence(result.evidence)
        print(f"Question: {question}")
        print(f"Top score: {result.top_score:.3f}  abstain: {result.abstain}")
        print(f"Flagged chunks: {flagged}")
        print()
        print(ABSTAIN_MESSAGE if result.abstain else generate_answer(safe))
        fake_answer = "Refunds always take 3 days. [privacy-data] [refund-policy-2024]"
        print("\nCitation check on a fake answer:")
        print(find_citation_problems(fake_answer, safe))
    finally:
        client.close()