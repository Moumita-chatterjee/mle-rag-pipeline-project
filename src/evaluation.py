import json
import sys
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI
from qdrant_client import QdrantClient
from sentence_transformers import SentenceTransformer

from src.pipeline import answer_question
from src.retrieval import open_client
from src.embedding import load_model
from src.generation import CITATION_PATTERN, load_llm
from src.ingestion import Chunk


#CALIBRATION_PATH = Path("data/evaluation/calibration.json")
DATA_FOLDER = Path("data/evaluation")
REPORTS_FOLDER = Path("reports")

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

def evaluate_question(
    client: QdrantClient,
    model: SentenceTransformer,
    q: EvalQuestion,
    llm: tuple[OpenAI, str] | None = None,
) -> dict:
    """Run one question through the pipeline and score it."""
    answer = answer_question(client, model, q.question, llm)
    ranking = document_ranking(answer.retrieved)
    safe = [(s, c) for s, c in answer.retrieved if c.chunk_id not in answer.flagged]

    row = {
        "id": q.id,
        "mode": answer.mode,
        "should_abstain": q.should_abstain,
        "abstained": answer.abstained,
        "decision_correct": answer.abstained == q.should_abstain,
        "top_score": round(answer.top_score, 3),
        "expected": q.expected_sources,
        "retrieved": ranking,
        "flagged": answer.flagged,
        "recall": None,
        "reciprocal_rank": None,
        "relevance": None,
        "faithfulness": None,
        "answer": answer.text,
    }
    if not q.should_abstain:
        row["recall"] = recall_at_k(ranking, q.expected_sources)
        row["reciprocal_rank"] = reciprocal_rank(ranking, q.expected_sources)
        row["relevance"] = 0.0 if answer.abstained else answer_relevance(answer.text, q.required_terms)
        if not answer.abstained and answer.mode == "extractive":
            row["faithfulness"] = faithfulness(answer.text, safe)
    return row

def mean(values: list) -> float | None:
    """Average of the values that are not None."""
    scored = [v for v in values if v is not None]
    return sum(scored) / len(scored) if scored else None

def summarize(rows: list[dict]) -> dict:
    """Average each metric over the questions it applies to."""
    return {
        "questions": len(rows),
        "recall_at_3": mean([r["recall"] for r in rows]),
        "mrr": mean([r["reciprocal_rank"] for r in rows]),
        "answer_relevance": mean([r["relevance"] for r in rows]),
        "faithfulness": mean([r["faithfulness"] for r in rows]),
        "decision_accuracy": mean([float(r["decision_correct"]) for r in rows]),
    }

def show(value: float | None) -> str:
    """Format a score with two decimals, or '-' when it was not rated."""
    return "-" if value is None else f"{value:.2f}"


if __name__ == "__main__":
    # questions = load_questions(CALIBRATION_PATH)
    # client = open_client()
    # try:
    #     model = load_model()
    #     correct = 0
    #     for q in questions:
    #         result = retrieve(client, model, q.question)
    #         expected = "ABSTAIN" if q.should_abstain else "ANSWER"
    #         got = "ABSTAIN" if result.abstain else "ANSWER"
    #         ok = expected == got
    #         correct += ok
    #         print(f"{q.id}  expected={expected:7}  got={got:7}  top={result.top_score:.3f}  {'OK' if ok else 'WRONG'}")
    #     print(f"\nDecision accuracy: {correct}/{len(questions)}")
    # finally:
    #     client.close()

    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    set_name = sys.argv[1] if len(sys.argv) > 1 else "calibration"
    use_llm = "--llm" in sys.argv

    questions = load_questions(DATA_FOLDER / f"{set_name}.json")
    client = open_client()
    try:
        model = load_model()
        llm = load_llm() if use_llm else None
        rows = [evaluate_question(client, model, q, llm) for q in questions]
    finally:
        client.close()

    print(f"{'id':6} {'decision':9} {'top':>5} {'recall':>6} {'RR':>5} {'relev':>5} {'faith':>5}")
    for r in rows:
        decision = ("ABSTAIN" if r["abstained"] else "ANSWER") + ("" if r["decision_correct"] else "!")
        print(
            f"{r['id']:6} {decision:9} {r['top_score']:5.3f} {show(r['recall']):>6}"
            f"{show(r['reciprocal_rank']):>5} {show(r['relevance']):>5} {show(r['faithfulness']):>5}"
        )

    summary = summarize(rows)
    print()
    for name, value in summary.items():
        print(f"{name:18} {value if name == 'questions' else show(value)}")

    REPORTS_FOLDER.mkdir(exist_ok=True)
    output = REPORTS_FOLDER / f"results_{set_name}_{rows[0]['mode']}.json"
    output.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print(f"\nSaved {output}")