import json
from dataclasses import dataclass
from pathlib import Path

from src.embedding import load_model
from src.generation import CITATION_PATTERN
from src.ingestion import Chunk
from src.retrieval import open_client, retrieve

CALIBRATION_PATH = Path("data/evaluation/calibration.json")

@dataclass(frozen=True)
class EvalQuestion:
    id: str
    question: str
    expected_sources: list[str]
    required_terms: list[str]
    should_abstain: bool

def load_questions(path: Path) -> list[EvalQuestion]:
    """Load an evaluation question set from a JSON file."""
    records = json.loads(path.read_text(encoding="utf-8"))
    return [EvalQuestion(**record) for record in records]

def document_ranking(retrieved: list[tuple[float, Chunk]]) -> list[str]:
    """Document IDs in rank order, each listed once."""
    ranking = []
    for _, chunk in retrieved:
        if chunk.document_id not in ranking:
            ranking.append(chunk.document_id)
    return ranking

def recall_at_k(ranking: list[str], expected: list[str]) -> float:
    """Share of expected documents that appear in the ranking."""
    return sum(doc in ranking for doc in expected) / len(expected)

def reciprocal_rank(ranking: list[str], expected: list[str]) -> float:
    """1 / rank of the first expected document, or 0 if none was retrieved."""
    for rank, doc in enumerate(ranking, start=1):
        if doc in expected:
            return 1 / rank
    return 0.0

def normalize(text: str) -> str:
    """Lowercase, unify hyphens, and collapse whitespace for fair text matching."""
    return " ".join(text.lower().replace("\u2011", "-").split())

def answer_relevance(answer_text: str, required_terms: list[str]) -> float:
    """Share of required terms that appear in the answer."""
    text = normalize(answer_text)
    return sum(normalize(term) in text for term in required_terms) / len(required_terms)

def faithfulness(answer_text: str, evidence: list[tuple[float, Chunk]]) -> float:
    """Share of answer lines that appear word for word in the evidence."""
    claims = [
        normalize(CITATION_PATTERN.sub("", line))
        for line in answer_text.splitlines()
        if line.strip()
    ]
    sources = [normalize(chunk.text) for _, chunk in evidence]
    supported = sum(any(claim in source for source in sources) for claim in claims)
    return supported / len(claims)


if __name__ == "__main__":
    questions = load_questions(CALIBRATION_PATH)
    client = open_client()
    try:
        model = load_model()
        correct = 0
        for q in questions:
            result = retrieve(client, model, q.question)
            expected = "ABSTAIN" if q.should_abstain else "ANSWER"
            got = "ABSTAIN" if result.abstain else "ANSWER"
            ok = expected == got
            correct += ok
            print(f"{q.id}  expected={expected:7}  got={got:7}  top={result.top_score:.3f}  {'OK' if ok else 'WRONG'}")
        print(f"\nDecision accuracy: {correct}/{len(questions)}")
    finally:
        client.close()