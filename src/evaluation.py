import json
from dataclasses import dataclass
from pathlib import Path

from src.embedding import load_model
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